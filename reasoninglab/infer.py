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

EOS_LITERAL = '\x01'

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
        output = ""
        gen_stream = self.generate(input_tensor, **config)
        while True:
            try:
                tok = next(gen_stream)
            except StopIteration as e:
                finish_reason = e.value
                break
            output += self.tokenizer.decode(tok.squeeze(0).tolist())
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
                    tok = next(gen_stream)
                except StopIteration as e:
                    finish_reason = e.value
                    break
                output_stream += self.tokenizer.decode(tok.squeeze(0).tolist())
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

    @torch.inference_mode()
    def generate(self, token_ids: torch.Tensor, **config) -> Generator[torch.Tensor, None, None]:
        max_new_tokens = config.get('max_new_tokens', self.config.max_new_tokens)
        cached = config.get('cached', self.config.cached)
        temperature = config.get('temperature', self.config.temperature)
        eos_token_id = config.get('eos_token_id', self.config.eos_token_id)
        nucleus_top_p = config.get('nucleus_top_p', self.config.nucleus_top_p)
        
        T_max = max_new_tokens + token_ids.shape[1]
        
        self.model.eval()
        for token_num in range(max_new_tokens):
            if token_num == 0 or not cached:
                # Prefill
                out, K_cache, V_cache = self.model(token_ids, T_max=T_max)
            else:
                # Decode
                # [B, 1, V]
                out, K_cache, V_cache = self.model(token_ids[:,-1:], 
                    T_max=T_max, T_cached=T_cached, K_cache=K_cache, V_cache=V_cache)  
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
                    # Collect nucleus
                    probs_sorted, sorted_idx = torch.sort(probs, dim=1, descending=True)
                    probs_acc = torch.cumsum(probs_sorted, dim=1)
                    keep_mask = (probs_acc <= nucleus_top_p)
                    if torch.all(keep_mask == False): keep_mask[:,0] = True # ensure at least one token in
                    probs_nucleus_sorted = torch.where(keep_mask, probs_sorted, torch.zeros_like(probs))
                    probs_nucleus = torch.zeros_like(probs).scatter(1, sorted_idx, probs_nucleus_sorted)
                    sampled_token_id = torch.multinomial(probs_nucleus, num_samples=1)
            if torch.all(sampled_token_id == eos_token_id): 
                return 'STOPPED'
            T_cached = token_ids.shape[1]
            token_ids = torch.cat([token_ids, sampled_token_id], dim=1)
            yield sampled_token_id
        return 'EXCEEDED MAX TOKENS'

if __name__ == "__main__":
    chatmodel = ChatModel(SPEC_QWEN06B, CONFIG_QWEN06_BASE)
    
    import sys
    prompt = ' '.join(sys.argv[1:]) if len(sys.argv)>1 else "Explain large language models: "
    chatmodel.stream_stdout(prompt)

r"""
uv run python -m reasoninglab.infer 'In quadrilateral $ABCD$, $\angle B$ is a right angle, diagonal $\overline{AC}$ is perpendicular to $\overline{CD}$, $AB=18$, $BC=21$, and $CD=14$. Find the perimeter of $ABCD$. The answer is '
uv run reasoninglab/main.py 'We assume that $F$ is a field. Let $V$ be a vector space of finite-dimension over $F$ and let $S$ be a subset of $V$. We define the annihilator of $S$ by \[S^\circ = \{f\in V^*\ \vert \ S\subset \ker f\}~.\] Prove that $S^\circ$ is a subspace of $V^*$.'
"""
