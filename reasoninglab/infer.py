import torch
import torch.nn as nn

from .config import ModelConfig, CONFIG_QWEN06B, PATH_QWEN06B
from .model import Qwen3
from .tokenizer import Qwen3Tokenizer
from .weights import hf_to_local_format

"""
BPE TOKENIZER:
set before training, it's a rule playbook
- numbers are split into single digits
- find common adjacent letters, put into token, repeat until we get target # of tokens in total
- insert special tokens
"""

device = torch.device("mps")

# Load tokenizer & weights
tokenizer = Qwen3Tokenizer(tokenizer_file_path=PATH_QWEN06B['tokenizer'])
config = ModelConfig(CONFIG_QWEN06B)
model = Qwen3(config)
model.load_state_dict(hf_to_local_format(config), strict=True)
model.to(device)

def chat(prompt: str):
    input_ids = tokenizer.encode(prompt)  # list[str]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    output = generate(model, input_tensor, max_new_tokens=5000, eos_token_id=151643)
    output = tokenizer.decode(output.squeeze(0).tolist())
    return output

@torch.inference_mode()
def generate(model: nn.Module, token_ids: torch.Tensor, *, max_new_tokens: int = 500, 
             eos_token_id: torch.Tensor=None, temperature=1e-5):
    model.eval()
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1] # [N*, d] -> last token [d]
        probs = torch.softmax(out / temperature, 1) # [d]
        sampled_token_id = torch.multinomial(probs, 1) # [1]
        if torch.all(sampled_token_id == eos_token_id): break
        token_ids = torch.cat([token_ids, sampled_token_id], dim=1)
    return token_ids

if __name__ == "__main__":
    import sys
    prompt = ' '.join(sys.argv[1:]) if len(sys.argv)>1 else "Explain large language models: "
    print(chat(prompt))

r"""
uv run python -m reasoninglab.infer 'In quadrilateral $ABCD$, $\angle B$ is a right angle, diagonal $\overline{AC}$ is perpendicular to $\overline{CD}$, $AB=18$, $BC=21$, and $CD=14$. Find the perimeter of $ABCD$. The answer is '
uv run reasoninglab/main.py 'We assume that $F$ is a field. Let $V$ be a vector space of finite-dimension over $F$ and let $S$ be a subset of $V$. We define the annihilator of $S$ by \[S^\circ = \{f\in V^*\ \vert \ S\subset \ker f\}~.\] Prove that $S^\circ$ is a subspace of $V^*$.'
"""
