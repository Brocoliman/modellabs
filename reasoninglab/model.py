import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelSpec


class Qwen3(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.vocab_size = self.spec.vocab_size
        self.d_model = self.spec.d_model
        self.d_head = self.spec.d_head
        self.n_layers = self.spec.n_layers
        self.n_groups = self.spec.n_groups
        self.rope_base = self.spec.rope_base
        self.tokens = self.spec.tokens
        
        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model) # d_model x V
        self.blocks = nn.ModuleList([
            TransformerBlock(self.spec) for _ in range(self.n_layers)
        ])
        self.norm = nn.RMSNorm([self.d_model], eps=1e-6)
        self.unembedding = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.unembedding.weight = self.embedding.weight  # for Qwen they share
        
        # RoPE
        freq = self.rope_base ** (torch.arange(0, self.d_head, 2) / self.d_head)  # 0, 2, 4, ..., d_head
        angles = torch.outer(torch.arange(0, self.tokens), 1.0/freq)  # (T, d_head/2)
        angles = torch.cat([angles, angles], dim=1)  # (T, d_head)
        self.register_buffer("rope_cos", torch.cos(angles), persistent=False)
        self.register_buffer("rope_sin", torch.sin(angles), persistent=False)
    
    def forward(self, token_ids, *, T_max: int, T_cached: int = 0, K_cache = None, V_cache = None): # (B, T) -> (B, T)
        # Embed
        x = self.embedding(token_ids)  # (B, T, d_model)
        
        # Generate mask
        B, T, _ = x.shape
        # lower including diagonal; the ones to KEEP
        tri = torch.ones([T, T], dtype=torch.bool, device=x.device).tril()
        fill = torch.ones([T,T_cached], dtype=torch.bool, device=x.device)
        mask = torch.cat([fill, tri], dim=1)
        
        """
        T=2, T_cached=3
        tri: 2x2
         10
         11
        fill: 2x3
         111
         111
        mask: 2x5
         11110
         11111
        """
        
        # Need updated cache
        # (n_layers, B, n_groups, T_max, d_head)
        if K_cache is None: K_cache = torch.empty([self.n_layers, B, self.n_groups, T_max,self.d_head], dtype=self.spec.dtype, device=x.device)
        if V_cache is None: V_cache = torch.empty([self.n_layers, B, self.n_groups, T_max,self.d_head], dtype=self.spec.dtype, device=x.device)
        
        # (B, n_groups, T_max, d_head)
        # Propagate through blocks
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask, 
                rope_cos=self.rope_cos, rope_sin=self.rope_sin, 
                K_cache=K_cache[i], V_cache=V_cache[i], T_cached=T_cached)
        x = self.norm(x) 
        logits = self.unembedding(x) # (B, T, V)
        return logits, K_cache, V_cache


class TransformerBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        
        # Layers
        self.norm1 = nn.RMSNorm([self.spec.d_model], eps=1e-6)
        self.gqa = GQA(spec)
        self.norm2 = nn.RMSNorm([self.spec.d_model], eps=1e-6)
        self.fcn1 = nn.Linear(self.spec.d_model, self.spec.d_linear, bias=False)
        self.fcn2 = nn.Linear(self.spec.d_model, self.spec.d_linear, bias=False)
        self.fcn3 = nn.Linear(self.spec.d_linear, self.spec.d_model, bias=False)
    
    def forward(self, x, *, mask, rope_cos, rope_sin, 
                K_cache, V_cache, T_cached: int = 0):  # (B, T, d_model)
        #if cache is not None: assert type(cache) == tuple or type(cache) == list, (cache, type(cache))
        pre_block = x
        x = self.norm1(x)
        x = self.gqa(x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin, 
                                       K_cache=K_cache, V_cache=V_cache, T_cached=T_cached)
        x = x + pre_block
        x_rms2 = self.norm2(x)
        x = self.fcn3(F.silu(self.fcn1(x_rms2)) * self.fcn2(x_rms2)) + x
        return x
    
    
class GQA(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.d_model = self.spec.d_model
        self.d_head = self.spec.d_head
        self.n_heads = self.spec.n_heads
        self.n_groups = self.spec.n_groups
        self.group_size = self.spec.group_size
        
        # Matrices and mixing layers and norms
        self.Wq = nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)
        self.Wk = nn.Linear(self.d_model, self.d_head * self.n_groups, bias=False)
        self.Wv = nn.Linear(self.d_model, self.d_head * self.n_groups, bias=False)
        self.Wo = nn.Linear(self.d_head * self.n_heads, self.d_model, bias=False)
        self.qnorm = nn.RMSNorm(self.d_head, eps=1e-6)
        self.knorm = nn.RMSNorm(self.d_head, eps=1e-6)
    
    def _rotate(self, x, *, rope_sin, rope_cos, offset=0):  # (B, n_heads, T, d_head)
        T = x.shape[2]
        x1, x2 = x[..., :self.d_head//2], x[..., self.d_head//2:]
        x90 = torch.cat([-x2, x1], dim=-1)
        out = x * rope_cos[offset : offset + T] + x90 * rope_sin[offset : offset + T]
        return out
        
    def forward(self, x, *, mask, rope_cos, rope_sin, K_cache, V_cache, T_cached: int = 0):  # (B, T, d_model) -> (B, T, d_model)
        B, T, _ = x.shape
        
        Q = self.Wq(x)  # (B, T, d_head * n_heads)
        K = self.Wk(x)  # (B, T, d_head * n_groups)
        V = self.Wv(x)  # (B, T, d_head * n_groups)
        
        # Q   -> (B, T, n_heads, d_head)  -> (B, n_heads, T, d_head)
        # K,V -> (B, T, n_groups, d_head) -> (B, n_groups, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = K.view(B, T, self.n_groups, self.d_head).permute(0, 2, 1, 3)
        V = V.view(B, T, self.n_groups, self.d_head).permute(0, 2, 1, 3)
        
        # QK Norm
        Q = self.qnorm(Q)
        K = self.knorm(K)
        
        # RoPE
        Q = self._rotate(Q, rope_cos=rope_cos, rope_sin=rope_sin, offset=T_cached)
        K = self._rotate(K, rope_cos=rope_cos, rope_sin=rope_sin, offset=T_cached)
        
        # Cache
        # (B, n_groups, T_max, d_head)
        K_cache[:, :, T_cached:T_cached+T] = K
        V_cache[:, :, T_cached:T_cached+T] = V
        
        # # Copy over via repeat_interleave
        # # In Qwen's format, (H1,H2,H3,H4) wants (K1,K1,K2,K2) or contiguous groups along the heads
        # # K,V -> (B, n_heads, T+T_cached, d_head)
        # K_full = K.repeat_interleave(self.group_size, dim=1)
        # V_full = V.repeat_interleave(self.group_size, dim=1)
        
        # # Calc (Replaced by flashattention below)
        # # K -> (B, n_heads, d_head, T+T_cache)
        # K_T = K_full.permute(0, 1, 3, 2)
        # A = (Q @ K_T) / (self.config.d_head ** 0.5)  # (B, n_heads, T, T+T_cache)
        # A = A.masked_fill(mask, float('-inf')) # set all noncausal attention pairs to none
        # A = torch.softmax(A, dim=3) @ V_full  # (B, n_heads, T+T_cache, d_head)
        
        # Attention
        # (B, n_heads, T_max, d_head)
        A = F.scaled_dot_product_attention(
            Q, K_cache[:, :, :T_cached+T], V_cache[:, :, :T_cached+T], mask, enable_gqa=True)  
        
        # Mix the heads
        A = A.permute(0, 2, 1, 3)   # (B, T, n_heads, d_head)
        A = A.reshape(B, T, -1)     # (B, T, n_heads * d_head) not view because permute removed contiguity
        A = self.Wo(A)              # (B, T, d_model)
        
        return A
