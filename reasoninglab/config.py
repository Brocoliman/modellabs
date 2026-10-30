import torch

CONFIG_QWEN06B = { # 0.6B model
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

class ModelConfig:
    def __init__(self, config):
        self.vocab_size = config["vocab_size"]
        self.tokens = config["tokens"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.n_groups = config["n_groups"]
        self.d_linear = config["d_linear"]
        self.d_head = config["d_head"]
        self.rope_base = config["rope_base"]
        self.dtype = config["dtype"]
        
        # Validate parameters
        assert self.n_heads % self.n_groups == 0, \
            f"Number of groups must divide number of heads"
        assert self.d_head % 2 == 0, \
            f"For RoPE, d_head must be divisible by 2"
            
        self.group_size = self.n_heads // self.n_groups
            