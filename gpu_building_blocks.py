import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
from batch_operations import geometric_product_batch, equi_join_batch


# This is the linear layer.
class MVLinear(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # One small [Cout x Cin] matrix per equivariant coefficient
        self.w = nn.Parameter(torch.randn(out_channels, in_channels, 5))  # grade projections 0..4
        self.v = nn.Parameter(torch.randn(out_channels, in_channels, 4))  # e0 * grade 0..3
        self.scalar_bias = nn.Parameter(torch.zeros(out_channels))        # only on scalar output

        self.register_buffer("grade_masks", constants.grade_masks)                # [5,16]
        self.register_buffer("e0_geometric_products", constants.e0_geometric_products)  # [16,16]

    def forward(self, x):
        """
        Args:
        x (torch.Tensor):
            - shape [B, N, 16]          (interpreted as Cin=1), or
            - shape [B, N, Cin, 16]     with Cin == self.in_channels.

        Returns:
        torch.Tensor:
            - shape [B, N, Cout, 16]    when out_channels > 1, or
            - shape [B, N, 16]          when out_channels == 1 (channel dim squeezed),
          where Cout == self.out_channels.
        """
        # x can be [B,N,16] or [B,N,Cin,16]
        if x.dim() == 3:
            x = x.unsqueeze(2)   # [B,N,1,16]

        B, N, Cin, D = x.shape
        assert Cin == self.in_channels, f"expected Cin={self.in_channels}, got {Cin}"
        # ---- First sum: \sum_k w_k * <x>_k
        # Broadcast masks: [5,16] -> [1,1,1,5,16]
        gm = self.grade_masks.view(1,1,1,5,D)
        xk = (x.unsqueeze(3) * gm)          # [B,N,Cin,5,16]
        # einsum over in-channels with per-grade matrices W_k[o,i]
        y1 = 0
        for k in range(5):
            Wk = self.w[..., k]             # [Cout, Cin]
            y1 = y1 + torch.einsum('bnid,oi->bnod', xk[:, :, :, k, :], Wk)  # [B,N,Cout,16]
        # add scalar bias to scalar component
        y1[:, :, :, 0] = y1[:, :, :, 0] + self.scalar_bias.view(1,1,-1)

        # ---- Second sum: \sum_j v_j * <x e0>_j, j=0..3
        xe0 = x @ self.e0_geometric_products.T    # [B,N,Cin,16]
        xj  = (xe0.unsqueeze(3) * gm[:, :, :, :4])  # keep grades 0..3 -> [B,N,Cin,4,16]
        y2 = 0
        for j in range(4):
            Vj = self.v[..., j]             # [Cout, Cin]
            y2 = y2 + torch.einsum('bnid,oi->bnod', xj[:, :, :, j, :], Vj)

        y = y1 + y2  # [B,N,Cout,16]

        if self.out_channels == 1:
            y = y.squeeze(2)  # [B,N,16]
        return y



# This is the gated gelu layer.
class MVGatedGelu(nn.Module):
    def __init__(self):
        super().__init__()
        # The .view(...) is necessary, or else we can't multiply with tensor x later.  
        self.register_buffer("scalar_mask", constants.grade_masks[0].view(1, 1, len(constants.components)))
               
    def forward(self, x):
        """
        Args:
            x (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components))

        Returns:
            torch.tensor: Shape: (batch_size, num_tokens, len(constants.components))
        """
        # dim=-1 is the last dimension (multivector components).
        # keepdim=True makes sure that the shape is (batch_size, num_tokens, 1).
        x_scalar = (x * self.scalar_mask).sum(dim=-1, keepdim=True)
        gated = F.gelu(x_scalar)
        
        return gated * x
    
    
    
# E(3)-equivariant LayerNorm for multivectors in G3,0,1.
# Normalizes each token's multivector using only non-e0 components.
class MVLayerNorm(nn.Module):    
    def __init__(self, eps=1e-5):
        super().__init__()
        # eps is used to avoid norm becomming zero.
        self.eps = eps   
        # Masking out the e0 blades.
        self.register_buffer("mask", constants.non_e0_mask)
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).

        Returns:
            torch.tensor: Shape: (batch_size, num_tokens, len(constants.components)).
        """
        # The tensor inner product with itself (without e0 blades).        
        inner = ((x**2) * self.mask).sum(dim=-1, keepdim=True)  # [B, N, 1]        
        norm  = torch.sqrt(inner + self.eps)                    # [B, N, 1]
        
        return x / norm


class MVAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = MVLinear()
        self.key = MVLinear()
        self.value = MVLinear()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        # Fixed parameters. Will not change during training. 
        self.register_buffer("mask", constants.non_e0_mask)
        
        # For scaling. 
        eff_dim = int(constants.non_e0_mask.sum().item())  # = 8
        base = 1.0 / math.sqrt(eff_dim)
        # The scale is learnable, positive and near base.
        self.log_scale = nn.Parameter(torch.tensor(0.0))  
        self.register_buffer("base_scale", torch.tensor(base, dtype=torch.float32))
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).

        Returns:
            (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).
        """
        q = self.query(x); k = self.key(x); v = self.value(x)
        q_masked = q * self.mask; k_masked = k * self.mask
        wanted_dtype = torch.bfloat16  # or torch.float16 
        q_masked = q_masked.to(wanted_dtype)
        k_masked = k_masked.to(wanted_dtype)
        v        = v.to(wanted_dtype)
        attn_logits = torch.bmm(q_masked, k_masked.transpose(1, 2))

        scale = self.base_scale * torch.exp(self.log_scale)
        attn_logits = attn_logits * scale

        # --- Distance-aware bias (low-peak-memory, differentiable) ---
        comps = constants.components
        ix, iy, iz = comps.index('e0e1e2'), comps.index('e0e1e3'), comps.index('e0e2e3')

        # Positions from trivector coords in the *current-layer* input x
        R = torch.stack([x[..., ix], x[..., iy], x[..., iz]], dim=-1)            # [B, N, 3]
        R = R.to(attn_logits.dtype)  # keep bf16/fp16 if it run under AMP

        # r2: per-token squared norm, shaped to broadcast over rows/cols of [B,N,N]
        r2 = (R * R).sum(dim=-1, keepdim=True)  # [B, N, 1]

        # attn_logits -= gamma * r2 (row bias) and -= gamma * r2^T (col bias)
        attn_logits.add_(-self.gamma * r2)                  # broadcast across columns
        attn_logits.add_(-self.gamma * r2.transpose(1, 2))  # broadcast across rows

        # attn_logits += 2*gamma * (R @ R^T) without allocating a separate [B,N,N]
        # We can’t pass a tensor alpha to baddbmm, so scale the inputs instead:
        # Let R_s = sqrt(2*gamma) * R, then R_s @ R_s^T = 2*gamma * (R @ R^T).
        g = torch.sqrt(torch.clamp(2.0 * self.gamma, min=0.0)).to(R.dtype)
        R_s = R * g
        # https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html
        attn_logits = torch.baddbmm(attn_logits, R_s, R_s.transpose(1, 2),
                                    beta=1.0, alpha=1.0)
        # --- End distance-aware bias ---



        # build key mask from input (non-zero tokens)
        key_mask = (x.abs().sum(dim=-1) > 0)  # [B, N] True = real token
        # ~ is just a bitwise NOT in PyTorch. For bool tensors, 
        # it computes the logical NOT. 
        # https://docs.pytorch.org/docs/stable/generated/torch.bitwise_not.html
        attn_logits = attn_logits.masked_fill(~key_mask.unsqueeze(1), -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)
        
        return torch.bmm(attn_weights, v)


class MVMultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        # A head is a full attention mechanism acting on all multivectors,
        # projecting multivectors into Q, K, V.
        self.heads = nn.ModuleList([MVAttentionHead() for _ in range(num_heads)])
        self.output_proj = nn.Linear(num_heads * len(constants.components), 
                                     len(constants.components))
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).

        Returns:
            (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).
        """
        
        B, T, D = x.shape
        H = len(self.heads)
        outs = []
        for h in self.heads:
            outs.append(h(x))             # each is [B,T,16]
        multihead_outputs = torch.stack(outs, dim=-2).reshape(B, T, H*D)  # [B,T,H*16]
        
        # Cast activations to match the Linear weight’s dtype (usually float32)
        multihead_outputs = multihead_outputs.to(self.output_proj.weight.dtype)
        
        return self.output_proj(multihead_outputs)




# This is the bilinear layer.
class MVGeometricBilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, reference):
        """
        Args:
            x, y, reference: torch.Tensor of shape (batch_size, num_tokens, 
            len(constants.components))
        Returns:
            torch.Tensor: Concatenation of geometric product and E(3)-equivariant join. 
            Shape: (batch_size, num_tokens, 2*len(constants.components))
        """
        gp = geometric_product_batch(x, y)           
        join = equi_join_batch(x, y, reference)      

        return torch.cat([gp, join], dim=-1)         
    
    
# Putting the layers together to form a block. 
class MVFeedforwardBlock(nn.Module):    
    def __init__(self):
        super().__init__()
        self.norm = MVLayerNorm()
        self.linear1 = MVLinear()
        self.gelu = MVGatedGelu()
        self.bilinear = MVGeometricBilinear()        
        # With two grade-wise equivariant maps:
        self.linear2_gp   = MVLinear()  # For the geometric product half
        self.linear2_join = MVLinear()  # For the equivariant join half
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        x_proj = self.linear1(x_norm)
        x_gated = self.gelu(x_proj)

        # Pass original + gated into bilinear interaction
        x_bilinear = self.bilinear(x_norm, x_gated, x_norm)

        n = len(constants.components)  # 16
        gp_part   = x_bilinear[..., :n]
        join_part = x_bilinear[..., n:]

        # Apply grade-wise linear per half and sum
        x_out = self.linear2_gp(gp_part) + self.linear2_join(join_part)

        x_out = self.dropout(x_out)

        return x_out + residual        
        