import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class Qwen3(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Layers
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model) # d_model x V
        self.blocks = nn.ModuleList([
            TransformerBlock(self.config) for _ in self.config.n_layers
        ])
        self.final_rmsnorm = nn.RMSNorm([self.config.d_model])
        self.out_fcn = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
    
    def forward(self, in_oh: torch.Tensor): # (B, T, V) -> (B, T, V)
        # Embed
        x = self.embedding(in_oh)  # (B, T, d_model)
        
        # Generate mask
        T = x.shape[1]
        mask = torch.tril(torch.ones([T, T]), diagonal=0, dtype=bool) # lower including diagonal; the ones to KEEP
        
        # Propagate through blocks
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_rmsnorm(x) 
        logits = self.out_fcn(x) # (B, T, V)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Layers
        self.rms1 = nn.RMSNorm([self.config.d_model])
        self.gqa = GQA(config)
        self.rms2 = nn.RMSNorm([self.config.d_model])
        self.fcn1 = nn.Linear(self.config.d_model, self.config.d_linear, bias=False)
        self.fcn2 = nn.Linear(self.config.d_model, self.config.d_linear, bias=False)
        self.fcn3 = nn.Linear(self.config.d_linear, self.config.d_model, bias=False)
    
    def forward(self, x, mask):  # (B, T, d_model)
        x = self.gqa(self.rms1(x), mask) + x
        x_rms2 = self.rms2(x)
        x = self.fcn3(F.silu(self.fcn1(x_rms2)) * self.fcn2(x_rms2)) + x
        return x
    

class GQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Matrices and layers and norms
        self.Wq = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.Wk = nn.Linear(self.config.d_model, self.config.d_head * self.config.n_groups, bias=False)
        self.Wv = nn.Linear(self.config.d_model, self.config.d_head * self.config.n_groups, bias=False)
        self.out_proj = nn.Linear(self.config.d_head * self.config.n_heads, self.config.d_model, bias=False)
        self.qnorm = nn.RMSNorm(self.config.d_head)
        self.knorm = nn.RMSNorm(self.config.d_head)
    
    def forward(self, x, mask):  # (B, T, d_model) -> (B, T, d_model)
        B, T, _ = x.shape
        
        Q = self.Wq(x)  # (B, T, d_head * n_heads)
        K = self.Wk(x)  # (B, T, d_head * n_groups)
        V = self.Wv(x)  # (B, T, d_head * n_groups)
        
        # Q   -> (B, T, n_heads, d_head)  -> (B, n_heads, T, d_head)
        # K,V -> (B, T, n_groups, d_head) -> (B, n_groups, T, d_head)
        Q = Q.view(B, T, self.config.n_heads, self.config.d_head).permute(0, 2, 1, 3)
        K = K.view(B, T, self.config.n_groups, self.config.d_head).permute(0, 2, 1, 3)
        V = V.view(B, T, self.config.n_groups, self.config.d_head).permute(0, 2, 1, 3)
        
        # QK Norm
        Q = self.qnorm(Q)
        K = self.knorm(K)
        
        # RoPE
        
        # Cache
        
        # Copy over via repeat_interleave
        # In Qwen's format, (H1,H2,H3,H4) wants (K1,K1,K2,K2) or contiguous groups along the heads
        # K,V -> (B, n_heads, T, d_head)
        K_full = K.repeat_interleave(self.config.group_size, dim=1)
        V_full = V.repeat_interleave(self.config.group_size, dim=1)
        
        # # Calc (Replaced by flashattention below)
        # # K -> (B, n_heads, d_head, T)
        # K_T = K_full.permute(0, 1, 3, 2)
        # A = (Q @ K_T) / (self.config.d_head ** 0.5)  # (B, n_heads, T, T)
        # A = A.masked_fill(mask, float('-inf')) # set all noncausal attention pairs to none
        # A = torch.softmax(A, dim=3) @ V_full  # (B, n_heads, T, d_head)
        
        # calc
        A = F.scaled_dot_product_attention(Q, K_full, V_full, mask)  # (B, n_heads, T, d_head)
        
        # Mix the heads
        A = A.permute(0, 2, 3, 1)   # (B, T, d_head, n_heads)
        A = A.reshape(B, T, -1)     # (B, T, d_head * n_heads) not view because permute removed contiguity
        A = self.out_proj(A)        # (B, T, d_model)
        
        return A
