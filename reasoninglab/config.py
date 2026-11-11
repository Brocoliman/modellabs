from pathlib import Path
HERE = Path(__file__).resolve().parent

import torch


# ===== Inference configs

CONFIG_QWEN06_BASE = {
    "hf_name": "Qwen/Qwen3-0.6B-Base",
    'root_dir': HERE / "qwen3" / "hf-0.6B-base",
    'tokenizer_dir': HERE / "qwen3" / "hf-0.6B-base" / 'tokenizer.json',
    'eos_token_id': 151643,
    'max_new_tokens': 5000, 
    'temperature': 0.6,
    'cached': True,
    'nucleus_top_p': 0.9,
}

CONFIG_QWEN06 = {
    "hf_name": "Qwen/Qwen3-0.6B",
    'root_dir': HERE / "qwen3" / "hf-0.6B",
    'tokenizer_dir': HERE / "qwen3" / "hf-0.6B" / 'tokenizer.json',
    'eos_token_id': 151645,
    'max_new_tokens': 5000, 
    'temperature': 0.6,
    'cached': True,
    'nucleus_top_p': 0.9,
}

# ===== Model architectures

SPEC_QWEN06B = { # 0.6B model architecture
    "vocab_size": 151936,
    "tokens": 40960,
    "d_model": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "n_groups": 8,
    "d_linear": 3072,
    "d_head": 128,
    "rope_base": 1000000.,
    "dtype": torch.bfloat16
}

# ===== Benchmark paths

PATH_MATH500 = HERE / 'data' / 'math500_test.jsonl'
PATH_MMLU = HERE / 'data' / 'mmlu_dev.jsonl'

# ===== 
class InferenceConfig:
    def __init__(self, config, spec=None):
        self.arch = ModelSpec(spec) if spec else None
        self.hf_name = config['hf_name']
        self.root_dir = config['root_dir']
        self.tokenizer_dir = config['tokenizer_dir']
        self.eos_token_id = config['eos_token_id']
        self.max_new_tokens = config['max_new_tokens']
        self.temperature = config['temperature']
        self.cached = config['cached']
        self.nucleus_top_p = config['nucleus_top_p']

class ModelSpec:
    def __init__(self, spec):
        self.vocab_size = spec["vocab_size"]
        self.tokens = spec["tokens"]
        self.d_model = spec["d_model"]
        self.n_heads = spec["n_heads"]
        self.n_layers = spec["n_layers"]
        self.n_groups = spec["n_groups"]
        self.d_linear = spec["d_linear"]
        self.d_head = spec["d_head"]
        self.rope_base = spec["rope_base"]
        self.dtype = spec["dtype"]
        
        # Validate parameters
        assert self.n_heads % self.n_groups == 0, \
            f"Number of groups must divide number of heads"
        assert self.d_head % 2 == 0, \
            f"For RoPE, d_head must be divisible by 2"
            
        self.group_size = self.n_heads // self.n_groups
            