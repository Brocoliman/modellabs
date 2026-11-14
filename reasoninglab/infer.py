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


class ChatModel:
    def __init__(self, spec, config):
        self.device = torch.device("mps")
        self.spec = ModelSpec(spec)
        self.config = InferenceConfig(config, spec)
        self.load()
    
    def load(self):
        # Load tokenizer & weights
        self.tokenizer = Qwen3Tokenizer(tokenizer_file_path=self.config.tokenizer_dir)
        self.model = Qwen3(self.spec)
        self.model.load_state_dict(hf_to_local_format(self.spec, self.config.root_dir), strict=True)
        self.model.to(self.device, dtype=self.spec.dtype)

    def generate_text2text(self, prompt: str, **config) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)  # [1, T]
        output = list()
        gen_stream = self.generate(input_tensor, **config)
        while True:
            try:
                tok_batch, finished = next(gen_stream)
                assert tok_batch.shape[0] == 1, "stream_stdout does not accept multi-sample"
            except StopIteration as e:
                finish_reason = e.value
                break
            output += tok_batch.squeeze(0).tolist()
        output = self.tokenizer.decode(output)
        return output, finish_reason

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

    def _logprobs(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
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
        
    @torch.inference_mode()
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
            # sampled_token_id: [B, 1]
            out = out[:, -1]  # last token [B, V]
            if temperature == 0:
                sampled_token_id = torch.argmax(out, dim=1, keepdim=True)
            else:
                # Logits -> probs
                probs = torch.softmax(out / temperature, dim=1)  # [B, V]
                # Require top-p sampling? e.g. keep only tokens that are within top 80%
                if nucleus_top_p is None:
                    sampled_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    sampled_token_id = self._sample_nucleus(probs, nucleus_top_p=nucleus_top_p)
            
            # Output handling
            # anything that was not finished yet, if it printed EOS it is finished
            # updated the finished (B, ) | comp (B, 1 -> B,)
            finished = finished | (sampled_token_id == eos_token_id).squeeze(1)
            sampled_token_id = torch.where(finished.unsqueeze(1), eos_token_id, sampled_token_id)
            yield sampled_token_id, finished
            
            # Update cache pointer
            T_cached = token_ids.shape[1]
            assert sampled_token_id.shape == (B, 1)
            token_ids = torch.cat([token_ids, sampled_token_id], dim=1)
            
            # Exit condition
            if torch.all(finished): break
        
        finish_reason = ['FINISHED' if i else 'EXCEEDED MAX TOKENS' for i in finished ]
        return finish_reason
    
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
    chatmodel.stream_stdout(prompt)

r"""
uv run python -m reasoninglab.infer 'In quadrilateral $ABCD$, $\angle B$ is a right angle, diagonal $\overline{AC}$ is perpendicular to $\overline{CD}$, $AB=18$, $BC=21$, and $CD=14$. Find the perimeter of $ABCD$. The answer is '
uv run reasoninglab/main.py 'We assume that $F$ is a field. Let $V$ be a vector space of finite-dimension over $F$ and let $S$ be a subset of $V$. We define the annihilator of $S$ by \[S^\circ = \{f\in V^*\ \vert \ S\subset \ker f\}~.\] Prove that $S^\circ$ is a subspace of $V^*$.'
"""
