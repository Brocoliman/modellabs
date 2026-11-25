from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
HERE = Path(__file__).resolve().parent

import torch


# ===== Benchmark paths

PATH_DATA = HERE / 'data'

# ===== Configurations

@dataclass
class RLConfig:
    """Parameters for training RLVR
    """
    pass
@dataclass
class EvalConfig:
    """Parameters for benchmark evaluation
    These are directly used by Runner
    """
    consistency_k: int = 1
    task_range: Iterable = None
    out_path: Path = None

@dataclass
class CheckpointConfig:
    """Description of the checkpoint (model) you are loading
    Protocol for loading checkpoints, from and to
    """
    hf_name: str
    root_dir: Path
    tokenizer_dir: Path
    eos_token_id: int
    device: str = 'mps'
    
@dataclass
class SamplingConfig:
    """How to sample
    Affects ChatModel.generate (ChatModel.samples)
    """
    temperature: float = 0.6
    nucleus_top_p: float = 0.9
    max_new_tokens: int = 5000
    cached: bool = True

@dataclass
class ModelSpec:
    """The model architecture
    """
    vocab_size: int
    tokens: int
    d_model: int
    n_heads: int
    n_layers: int
    n_groups: int
    d_linear: int
    d_head: int
    rope_base: float
    dtype: torch.dtype
        
    def __post_init__(self):
        # Validate parameters
        assert self.n_heads % self.n_groups == 0, \
            f"Number of groups must divide number of heads"
        assert self.d_head % 2 == 0, \
            f"For RoPE, d_head must be divisible by 2"
            
        self.group_size = self.n_heads // self.n_groups


# ===== Predefined configs

QWEN06_BASE = CheckpointConfig(
    hf_name = "Qwen/Qwen3-0.6B-Base",
    root_dir = HERE / "qwen3" / "hf-0.6B-base",
    tokenizer_dir = HERE / "qwen3" / "hf-0.6B-base" / 'tokenizer.json',
    eos_token_id=151643
)

QWEN06 = CheckpointConfig(
    hf_name = "Qwen/Qwen3-0.6B",
    root_dir = HERE / "qwen3" / "hf-0.6B",
    tokenizer_dir = HERE / "qwen3" / "hf-0.6B" / 'tokenizer.json',
    eos_token_id=151645
)

SPEC_QWEN06 = ModelSpec(
    vocab_size = 151936,
    tokens = 40960,
    d_model = 1024,
    n_heads = 16,
    n_layers = 28,
    n_groups = 8,
    d_linear = 3072,
    d_head = 128,
    rope_base = 1000000.,
    dtype = torch.bfloat16
)

SAMPLER_QWEN06 = SamplingConfig()

EVALUATOR_QWEN06 = EvalConfig()