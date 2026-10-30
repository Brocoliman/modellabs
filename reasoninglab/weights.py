from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .config import ModelConfig, PATH_QWEN06B

local_dir = PATH_QWEN06B['root']

hf = {}
for f in Path(local_dir).glob("*.safetensors"):   # 0.6B is one file; 4B is split across several
    hf.update(load_file(f))

def hf_to_local_format(config: ModelConfig) -> dict:
    out = dict()
    out['embedding.weight'] = hf['model.embed_tokens.weight']
    out['unembedding.weight'] = hf['model.embed_tokens.weight']
    out['norm.weight'] = hf['model.norm.weight']
    for i in range(config.n_layers):
        for matrix in ('q','k','v','o'):
            out[f'blocks.{i}.gqa.W{matrix}.weight'] = hf[f'model.layers.{i}.self_attn.{matrix}_proj.weight']
        for norm in ('q','k'):
            out[f'blocks.{i}.gqa.{norm}norm.weight'] = hf[f'model.layers.{i}.self_attn.{norm}_norm.weight']
        out[f'blocks.{i}.norm1.weight'] = hf[f'model.layers.{i}.input_layernorm.weight']
        out[f'blocks.{i}.norm2.weight'] = hf[f'model.layers.{i}.post_attention_layernorm.weight']
        for fcn_no, fcn_name in enumerate(('gate', 'up', 'down')):
            out[f'blocks.{i}.fcn{fcn_no+1}.weight'] = hf[f'model.layers.{i}.mlp.{fcn_name}_proj.weight']
    return out

if __name__ == "__main__":
    load_dotenv()
    path = snapshot_download("Qwen/Qwen3-0.6B-Base",
                            allow_patterns=["*.safetensors", "*.json"],
                            local_dir=local_dir)