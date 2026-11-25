# ModelLabs

Long term foundational model playground to focus on developing and improving models from various domains. 

**Status:** early. Structure and direction are still forming.

## Areas
Status: **Active** = working on now · **Next** = after current work · **Later** = parked

**LLMs**
- [Active] RL / GRPO / SFT + LoRA — `reasoninglab/`
- [Later] Tool assisted workflows
- [Later] Post-transformer architectures

**World models**
- [Next] Latent (JEPA-style) world models on a 2D physics sim
- [Next] Merging world models trained on separate rules (task arithmetic; latent-space merging)

**Science of DL**
- [Next] What RL changes: weight-delta rank, which layers move, CKA / linear probes before vs after

**Generative models**
- [Later] Diffusion / flow matching

**Hardware & efficiency**
- [Later] GPU / kernel design (CUDA → Triton → FlashAttention fwd)
- [Later] Local serving

## Principles
- Code is hand written
- Core components are implemented from scratch at each layer of abstraction I'm focusing on