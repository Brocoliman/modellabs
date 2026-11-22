from __future__ import annotations
import torch
import torch.nn as nn

import time
from collections.abc import Generator
from rich.console import Console
from rich.live import Live
from rich.text import Text

from .config import *
from .model import Qwen3
from .weights import hf_to_local_format

from ._vendor.tokenizer import Qwen3Tokenizer

class SampleBatch:
    def __init__(self, *, full_ids: torch.Tensor, T_in: torch.Tensor, T_out: torch.Tensor, finish_reasons: list[str],
                 chatmodel: ChatModel):
        self.full_ids = full_ids
        self.T_in = T_in
        self.T_out = T_out
        self.finish_reasons = finish_reasons
        self.tokenizer = chatmodel.tokenizer
        
        # Ensure inputs have same batch size
        assert self.full_ids.shape[0] == self.T_in.shape[0] == self.T_out.shape[0] == len(finish_reasons)
        
        # Extract output_ids
        # Right now assumes all prompts are the same (one batch)
        assert (self.T_in[0] == self.T_in).all()
        self.output_ids = self.full_ids[:, self.T_in[0]:] 
        
        # Response mask - only tag output tokens including EOS: tokens[T_in[b]-1:T_in[b]-1+T_out[b]]
        B, T = self.full_ids.shape
        self.response_mask = torch.zeros([B, T-1], device=self.full_ids.device)  # (B, T-1)
        t_idx = torch.arange(T-1, device=self.full_ids.device)[None, :]  # (1, T)
        starts = (T_in - 1)[:, None]  # (B, 1)
        ends = (T_in - 1 + T_out)[:, None]  # (B, 1)
        self.response_mask[(t_idx >= starts) & (t_idx < ends)] = 1
    
    def decode_all(self) -> list[str]:
        return self._decode(self.full_ids, self.T_out+self.T_in)

    def decode_output(self) -> list[str]:
        return self._decode(self.output_ids, self.T_out)
        
    def _decode(self, output_ids: torch.Tensor, end_idx: torch.Tensor) -> list[str]:
        output_ids_list = [
            # Remove the last token (which is EOS) if it terminated
            output_ids[b, :end_idx[b]-1 if self.finish_reasons[b] == 'FINISHED' else end_idx[b]].tolist() 
            for b in range(output_ids.shape[0])
        ]
        outputs = [self.tokenizer.decode(output_row) for output_row in output_ids_list]  # Recall that output_ids is a list and each row is a full sample output
        return outputs

class ChatModel:
    def __init__(self, spec, config):
        self.spec = ModelSpec(spec)
        self.config = InferenceConfig(config, spec)
        self.device = torch.device(self.config.device)
        self.load()
    
    def load(self):
        # Load tokenizer & weights
        self.tokenizer = Qwen3Tokenizer(tokenizer_file_path=self.config.tokenizer_dir)
        self.model = Qwen3(self.spec)
        self.model.load_state_dict(hf_to_local_format(self.spec, self.config.root_dir), strict=True)
        self.model.to(self.device, dtype=self.spec.dtype)
        
    ##### Display functions (str in, str out)
    
    # str prompt -> str output
    def generate_text2text(self, prompt: str, **config) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output = self.samples(input_ids, batches=1, **config).decode_all()[0]
        return output

    # str prompt -> stdout
    def stream_stdout(self, prompt: str, debug=True, **config):
        max_new_tokens = config.get('max_new_tokens', self.config.max_new_tokens)
        
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        output_stream = ""
        time_start = time.perf_counter()
        with Live(refresh_per_second=10, transient=True, vertical_overflow="crop") as live:
            gen_stream = self.generate(input_tensor, **config)
            n = 1
            while True:
                try:
                    tok_batch, finished = next(gen_stream)
                    assert tok_batch.shape[0] == 1, "stream_stdout does not accept multi-sample"
                except StopIteration as e:
                    finish_reason = e.value
                    break
                output_stream += self.tokenizer.decode(tok_batch.squeeze(0).tolist())
                elapsed = time.perf_counter()-time_start
                live.update(
                    Text("====================\n"
                        f"Tokens: {n}/{max_new_tokens}\n"
                        f"Time elapsed: {elapsed}\n"
                        f"Tokens/s: {n/elapsed:02f}\n"
                        "====================\n"
                            f"{output_stream}"
                            if debug else output_stream
                            )
                )
                n += 1
        Console().print(output_stream)

    ##### Layer 2: tokstream-level
    # Operates on top of Layer 1 and only cares about the actual token sequence
    # Handles EOS trimming and conversion to string
    
    # Main wrapper on top of ChatModel.generate
    # input ids -> output ids, lengths, finish reasons
    def samples(self, input_ids: list[int], batches=1, **config) -> SampleBatch:
        eos_token_id = config.get('eos_token_id', self.config.eos_token_id)
        
        # (B, T_in)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).repeat(batches, 1).to(self.device)
        # (B, 0->T_out)
        output_ids = torch.empty([batches, 0], dtype=torch.long, device=self.device)
        
        gen_stream = self.generate(input_tensor, **config)
        while True:
            try:
                # (B, 1), (B, 1)
                tok_batch, finished = next(gen_stream)
            except StopIteration as e:
                finish_reason = e.value
                break
            # Add to output_ids
            output_ids = torch.cat([output_ids, tok_batch], dim=1)
        
        # Generate T_out (B, )
        is_eos = (output_ids == eos_token_id) # (B, T_out)
        first_eos = is_eos.int().argmax(dim=1) # (B,)
        T_out = torch.where(is_eos.any(dim=1), first_eos + 1, output_ids.shape[1])  # include first EOS
        # Generate T_in (B, )
        T_in = torch.full([batches], len(input_ids), device=self.device)
        # Assemble full ids
        full_ids = torch.cat([input_tensor, output_ids], dim=1)
        
        return SampleBatch(full_ids=full_ids, T_in=T_in, T_out=T_out, finish_reasons=finish_reason, chatmodel=self)

    ##### Layer 1: tokprobability-level
    # Funcs operate on top of Layer 0 (directly calls self.model())
    # Cares about the next token's probability distribution
    
    # Base function for inference, handling caching, batches, exit
    # (B, T) -> T_resp x [(B, 1), (B, 1)], B
    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, **config) -> Generator[torch.Tensor, None, None]:
        # token_ids to be shape [B, T]
        B, T = token_ids.shape
        finished = torch.zeros((B,), dtype=torch.bool, device=token_ids.device)
        
        # inference parameters
        max_new_tokens = config.get('max_new_tokens', self.config.max_new_tokens)
        cached = config.get('cached', self.config.cached)
        temperature = config.get('temperature', self.config.temperature)
        eos_token_id = config.get('eos_token_id', self.config.eos_token_id)
        nucleus_top_p = config.get('nucleus_top_p', self.config.nucleus_top_p)
        T_max = max_new_tokens + token_ids.shape[1]  # required for the KV cache size
        
        self.model.eval()
        prefill = True
        for step in range(max_new_tokens):
            # Forward prog
            if prefill and cached:
                # Prefill
                # token_ids:    (B, T, V)
                # out:          (B, T, V)
                # caches:       (n_layers, B, n_groups, T_max, d_head)
                out, K_cache, V_cache = self.model(token_ids, T_max=T_max)
                prefill = False
            elif prefill and not cached:
                out, K_cache, V_cache = self.model(token_ids, use_cache=False)
            else:
                # Decode
                # last token [B, 1, V] -> out, caches
                out, K_cache, V_cache = self.model(token_ids[:,-1:], 
                    T_max=T_max, T_cached=T_cached, K_cache=K_cache, V_cache=V_cache)  
            
            # Sample
            # sampled_token_id: (B, 1)
            out = out[:, -1]  # last token (B, V)
            if temperature == 0:
                sampled_token_id = torch.argmax(out, dim=1, keepdim=True)
            else:
                # Logits -> probs
                probs = torch.softmax(out / temperature, dim=1)  # (B, V)
                # Require top-p sampling? e.g. keep only tokens that are within top 80%
                if nucleus_top_p is None:
                    sampled_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    sampled_token_id = self._sample_nucleus(probs, nucleus_top_p=nucleus_top_p)
            
            # Output handling
            # anything that was not finished yet, if it printed EOS it is finished
            # updated the finished (B, ) | comp (B, 1 -> B,) -> (B, )
            finished = finished | (sampled_token_id == eos_token_id).squeeze(1)
            sampled_token_id = torch.where(finished.unsqueeze(1), eos_token_id, sampled_token_id)
            # (B, 1), (B, 1)
            yield sampled_token_id, finished.unsqueeze(1)
            
            # Update cache pointer
            T_cached = token_ids.shape[1]
            token_ids = torch.cat([token_ids, sampled_token_id], dim=1)
            
            # Exit condition
            if torch.all(finished): break
        
        finish_reason = ['FINISHED' if i else 'EXCEEDED MAX TOKENS' for i in finished ]
        return finish_reason
    
    # token ids -> conditional probability of next token
    # (B, T) -> (B, T-1)
    def logprobs(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(token_ids, use_cache=False)  # (B, T, V)
        logprobs = torch.log_softmax(logits.float(), dim=-1)[:, :-1]  # (B, T-1, V); need fp32
        posterior_ids = token_ids[:, 1:]  # (B, T-1)
        
        # The intuition
        # match at each batch & token, posterior_ids[b][t] is the index to get
        # for which v in logprobs[b][t][v]
        # logprobs: (B, T-1, V)
        # posterior_ids: (B, T-1) -> (B, T-1, 1)
        token_logprobs = logprobs.gather(dim=-1, index=posterior_ids.unsqueeze(-1)).squeeze(-1)
        # # Same as below
        # token_logprobs = torch.empty([B, T-1])
        # for b in range(B):
        #     for t in range(T-1): 
        #         condprobs[b, t] = logprobs[b, t, posterior_ids[b, t]]
        
        return token_logprobs

    # Find the nucleus and sample from it instead of using only multinomial
    def _sample_nucleus(self, probs: torch.tensor, *, nucleus_top_p: int) -> torch.tensor:
        # Collect nucleus
        probs_sorted, sorted_idx = torch.sort(probs, dim=1, descending=True)
        probs_acc = torch.cumsum(probs_sorted, dim=1)
        keep_mask = (probs_acc-probs_sorted < nucleus_top_p)  # keep if cumprob before it is < p
        probs_nucleus_sorted = torch.where(keep_mask, probs_sorted, torch.zeros_like(probs))
        probs_nucleus = torch.zeros_like(probs).scatter(1, sorted_idx, probs_nucleus_sorted)
        # Sample nucleus
        sampled_token_id = torch.multinomial(probs_nucleus, num_samples=1)
        return sampled_token_id
        

if __name__ == "__main__":
    chatmodel = ChatModel(SPEC_QWEN06B, CONFIG_QWEN06_BASE)
    
    import sys
    prompt = ' '.join(sys.argv[1:]) if len(sys.argv)>1 else "Explain large language models: "
    print(chatmodel.generate_text2text(prompt))
