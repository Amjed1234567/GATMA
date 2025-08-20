import torch
import torch.nn as nn
from gpu_building_blocks import (
    MVLayerNorm, MVMultiHeadAttention, MVFeedforwardBlock
)
import constants
from multivector import Multivector

class MVTransformerBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.norm1 = MVLayerNorm()
        self.attn = MVMultiHeadAttention(num_heads)
        self.norm2 = MVLayerNorm()
        self.ff = MVFeedforwardBlock()

    def forward(self, x):
        # Attention block with residual
        x_res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + x_res

        # Feedforward block with residual
        x_res = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + x_res

        return x


class MVTransformer(nn.Module):
    def __init__(self, num_layers, num_heads):
        """
        Args:
            num_layers (int): Number of transformer blocks.
            num_heads (int): Number of heads in multi-head attention.
        """
        super().__init__()
        self.layers = nn.ModuleList([MVTransformerBlock(num_heads) for _ in range(num_layers)])

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Shape (batch_size, num_tokens, len(constants.components))
        
        Returns:
            torch.tensor: Shape (batch_size, num_tokens, len(constants.components))
        """
        for layer in self.layers:
            x = layer(x)
        return x
