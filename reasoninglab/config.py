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
    def __init__(self, **kwargs):
        self.vocab_size = kwargs["vocab_size"]
        self.tokens = kwargs["tokens"]
        self.d_model = kwargs["d_model"]
        self.n_heads = kwargs["n_heads"]
        self.n_layers = kwargs["n_layers"]
        self.n_groups = kwargs["n_groups"]
        self.d_linear = kwargs["d_linear"]
        self.d_head = kwargs["d_head"]
        self.rope_base = kwargs["rope_base"]
        self.dtype = kwargs["dtype"]
        
        # Validate parameters
        assert self.n_heads % self.n_groups == 0, \
            f"Number of groups must divide number of heads"
            
        self.group_size = self.n_heads // self.n_groups
            