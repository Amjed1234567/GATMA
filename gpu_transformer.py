import torch
import torch.nn as nn
from gpu_building_blocks import (
    MVLayerNorm, MVMultiHeadAttention, MVFeedforwardBlock, MVLinear
)

# -------------------------
# Transformer Block 
# -------------------------
# gpu_transformer.py

import torch
import torch.nn as nn
from gpu_building_blocks import (
    MVLayerNorm, MVMultiHeadAttention, MVFeedforwardBlock, MVLinear
)

# gpu_transformer.py
class MVTransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.norm1 = MVLayerNorm()
        self.attn  = MVMultiHeadAttention(num_heads)
        self.norm2 = MVLayerNorm()
        self.ff    = MVFeedforwardBlock()

    def forward(self, x):
        # Attention + residual
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + res

        # Feedforward + residual (pass pre-norm res as reference)
        res = x                        # <- pre-norm for the FF path
        x = self.norm2(x)              # normalized stream (not used inside FF anymore)
        x = self.ff(x, reference=res)  # FF works entirely on `res`
        x = x + res
        return x

# -------------------------
# Multiple tokens (multivectors) per atom (n channels per atom)
# -------------------------
class MVTokenExpander(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        assert channels >= 1
        self.channels = channels
        # Single MVLinear expands 1 → channels
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
"""
# -------------------------
# Run file directly: quick shape check
# -------------------------
if __name__ == "__main__":
    mvt = MVTransformer(num_layers=3, num_heads=2, channels_per_atom=3)
    print(mvt)

    B, N = 2, 5
    x = torch.randn(B, N, 16)
    y = mvt(x)
    print("in :", x.shape)
    print("out:", y.shape)  # expect [2, 5*3, 16]
"""