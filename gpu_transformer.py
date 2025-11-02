# Author: Amjed Farooq Afzal.
# With inspiration from:  https://github.com/qualcomm-ai-research/geometric-algebra-transformer


import torch
import torch.nn as nn
from gpu_building_blocks import (
    MVLayerNorm, MVMultiHeadAttention, MVFeedforwardBlock, MVLinear
)

# -------------------------
# Transformer Block 
# -------------------------
class MVTransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.norm1 = MVLayerNorm(mode="mask")
        self.attn  = MVMultiHeadAttention(num_heads)
        self.norm2 = MVLayerNorm(mode="mask")   
        self.ff    = MVFeedforwardBlock()

    def forward(self, x):        
        res = x
        x   = self.norm1(x)
        x   = self.attn(x)
        x   = x + res
        
        pre_norm = x                 
        x        = self.norm2(x)     

        # Build a GLOBAL normalized reference: pure +/- I. 
        ref_ps   = x[..., 15].mean(dim=1, keepdim=True)        # [B,1]
        ref_sign = torch.where(ref_ps >= 0, ref_ps.new_tensor(1.0),
                                         ref_ps.new_tensor(-1.0))
        ref_I    = torch.zeros_like(x.mean(dim=1, keepdim=True))  # [B,1,16]
        ref_I[..., 15] = ref_sign  # +/- I only. 

        delta = self.ff(x, reference=ref_I)

        return delta + pre_norm
    
    
# -------------------------
# Multiple tokens (multivectors) per atom (n channels per atom)
# -------------------------
class MVTokenExpander(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        assert channels >= 1
        self.channels = channels
        # Single MVLinear expands 1 -> channels
        self.proj = MVLinear(in_channels=1, out_channels=channels)

    def forward(self, x):  # x: [B, N, 16]
        if self.channels == 1:
            return x
        y = self.proj(x)                 # [B, N, channels, 16]
        B, N, C, D = y.shape
        return y.reshape(B, N * C, D)    # [B, N*channels, 16]

# -------------------------
# Transformer (stack of blocks + expander)
# -------------------------
class MVTransformer(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, channels_per_atom: int = 1):
        super().__init__()
        self.channels_per_atom = channels_per_atom
        self.expander = MVTokenExpander(channels_per_atom) if channels_per_atom > 1 else None
        self.layers = nn.ModuleList([MVTransformerBlock(num_heads) for _ in range(num_layers)])

    def forward(self, x):  # x: [B, N, 16]
        if self.expander is not None:
            x = self.expander(x)  # [B, N*channels_per_atom, 16]
        for layer in self.layers:
            x = layer(x)
        return x
