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
        self.model.to(self.device)

    def generate_text2text(self, prompt: str, **config) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)  # [1, T]
        output_stream = ""
        for output_token_id in self.generate(input_tensor, **config):
            output_stream += self.tokenizer.decode(output_token_id.squeeze(0).tolist())
        return output_stream

    def stream_stdout(self, prompt: str, debug=True, **config):
        max_new_tokens = config.get('max_new_tokens') or self.config.max_new_tokens
        
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        output_stream = ""
        time_start = time.perf_counter()
        with Live(refresh_per_second=10, transient=True, vertical_overflow="crop") as live:
            for n, tok in enumerate(self.generate(input_tensor, **config), 1):
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
        Console().print(output_stream)

    @torch.inference_mode()
    def generate(self, token_ids: torch.Tensor, **config) -> Generator[torch.Tensor, None, None]:
        max_new_tokens = config.get('max_new_tokens') or self.config.max_new_tokens
        cached = config.get('cached') or self.config.cached
        temperature = config.get('temperature') or self.config.temperature
        eos_token_id = config.get('eos_token_id') or self.config.eos_token_id
        
        self.model.eval()
        for token_num in range(max_new_tokens):
            if token_num == 0 or not cached:
                # Prefill
                out, cache = self.model(token_ids)
            else:
                # Decode
                assert cache[0][0].shape[2] == T_cached
                out, cache = self.model(token_ids[:,-1:], cache=cache, T_cached=T_cached)  # [B, 1, V]
            out = out[:, -1]  # last token [B, V]
            probs = torch.softmax(out / temperature, 1)
            sampled_token_id = torch.multinomial(probs, 1)
            if torch.all(sampled_token_id == eos_token_id): break
            T_cached = token_ids.shape[1]
            token_ids = torch.cat([token_ids, sampled_token_id], dim=1)
            yield sampled_token_id

if __name__ == "__main__":
    chatmodel = ChatModel(SPEC_QWEN06B, CONFIG_QWEN06_BASE)
    
    import sys
    prompt = ' '.join(sys.argv[1:]) if len(sys.argv)>1 else "Explain large language models: "
    chatmodel.stream_stdout(prompt)

r"""
uv run python -m reasoninglab.infer 'In quadrilateral $ABCD$, $\angle B$ is a right angle, diagonal $\overline{AC}$ is perpendicular to $\overline{CD}$, $AB=18$, $BC=21$, and $CD=14$. Find the perimeter of $ABCD$. The answer is '
uv run reasoninglab/main.py 'We assume that $F$ is a field. Let $V$ be a vector space of finite-dimension over $F$ and let $S$ be a subset of $V$. We define the annihilator of $S$ by \[S^\circ = \{f\in V^*\ \vert \ S\subset \ker f\}~.\] Prove that $S^\circ$ is a subspace of $V^*$.'
"""
