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
        if x.dim() == 3:
            x = x.unsqueeze(2)   # [B,N,1,16]

        B, N, Cin, D = x.shape
        assert Cin == self.in_channels

        # ---- First sum: w_k * <x>_k  
        gm = self.grade_masks.view(1,1,1,5,D)
        xk = (x.unsqueeze(3) * gm)          # [B,N,Cin,5,16]
        y1 = 0
        for k in range(5):
            Wk = self.w[..., k]             # [Cout, Cin]
            y1 = y1 + torch.einsum('bnid,oi->bnod', xk[:, :, :, k, :], Wk)

        # scalar bias on scalar component
        y1[:, :, :, 0] = y1[:, :, :, 0] + self.scalar_bias.view(1,1,-1)

        # ---- Second sum: v_j * <e0 x>_j,  j=0..3   <-- LEFT multiply by e0
        e0 = torch.zeros(D, dtype=x.dtype, device=x.device)
        e0_idx = constants.components.index('e0')
        e0[e0_idx] = 1.0

        x_flat   = x.reshape(B, N*Cin, D)                     # [B, N*Cin, 16]
        e0_flat  = e0.view(1,1,D).expand(B, N*Cin, D)        # [B, N*Cin, 16]
        e0x_flat = geometric_product_batch(e0_flat, x_flat)  # [B, N*Cin, 16]  <-- e0 * x
        e0x      = e0x_flat.view(B, N, Cin, D)               # [B, N, Cin, 16]

        # keep grades 0..3
        xj = (e0x.unsqueeze(3) * gm[:, :, :, :4])            # [B,N,Cin,4,16]
        y2 = 0
        for j in range(4):
            Vj = self.v[..., j]                              # [Cout, Cin]
            y2 = y2 + torch.einsum('bnid,oi->bnod', xj[:, :, :, j, :], Vj)

        y = y1 + y2
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
class MVLayerNorm(nn.Module):
    def __init__(self, eps=1e-5, mode: str = "mask"):
        """
        mode:
          - "mask": ℝ⁸ dot over non-e0 blades (paper's description)
          - "signed": invariant PGA inner product with grade signs (indefinite), |·| before sqrt
        """
        super().__init__()
        self.eps  = eps
        self.mode = mode
        # non-e0 mask (ℝ⁸)
        self.register_buffer("non_e0_mask", constants.non_e0_mask)  # [1,1,16]
        # signed metric (+1 for 1,e1,e2,e3,e1e2e3; -1 for e1e2,e1e3,e2e3; 0 for any e0 blade)
        metric = torch.tensor(
            [ +1, 0, +1, +1, +1, 0, 0, 0, -1, -1, -1, 0, 0, +1, 0, 0 ],
            dtype=torch.float32).view(1,1,-1)
        self.register_buffer("metric_signs", metric)

    def forward(self, x):
        if self.mode == "mask":
            inner = ((x ** 2) * self.non_e0_mask).sum(dim=-1, keepdim=True)
            norm  = torch.sqrt(inner + self.eps)
        else:  # "signed"
            inner = (x * x * self.metric_signs).sum(dim=-1, keepdim=True)   # can be <0
            norm  = torch.sqrt(torch.abs(inner) + self.eps)
        return x / norm



class MVAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = MVLinear()
        self.key = MVLinear()
        self.value = MVLinear()
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # --- metric signs for invariant inner product (G3,0,1) ---
        metric = torch.tensor([
            +1,  # 1
             0,  # e0
            +1,  # e1
            +1,  # e2
            +1,  # e3
             0,  # e0e1
             0,  # e0e2
             0,  # e0e3
            -1,  # e1e2
            -1,  # e1e3
            -1,  # e2e3
             0,  # e0e1e2
             0,  # e0e1e3
            +1,  # e1e2e3
             0,  # e0e2e3
             0,  # e0e1e2e3
        ], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("metric_signs", metric)
        # ---------------------------------------------------------------

        self.register_buffer("mask", constants.non_e0_mask)
        eff_dim = int(constants.non_e0_mask.sum().item())
        base = 1.0 / math.sqrt(eff_dim)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("base_scale", torch.tensor(base, dtype=torch.float32))

    def forward(self, x):
        q = self.query(x); k = self.key(x); v = self.value(x)
        wanted_dtype = x.dtype
        q = q.to(wanted_dtype); k = k.to(wanted_dtype); v = v.to(wanted_dtype)

        # --- invariant inner product <q,k> with grade signs ---
        q_signed = q * self.metric_signs                 # [B,N,16]
        attn_logits = torch.bmm(q_signed, k.transpose(1, 2))  # [B,N,N]
        # -----------------------------------------------------------

        scale = self.base_scale * torch.exp(self.log_scale)
        attn_logits = attn_logits * scale

        # distance-aware term 
        comps = constants.components
        ix, iy, iz = comps.index('e0e1e2'), comps.index('e0e1e3'), comps.index('e0e2e3')
        R = torch.stack([x[..., ix], x[..., iy], x[..., iz]], dim=-1).to(wanted_dtype)  # [B,N,3]
        r2 = (R * R).sum(dim=-1, keepdim=True)
        attn_logits.add_(-self.gamma * r2)
        attn_logits.add_(-self.gamma * r2.transpose(1, 2))
        g = torch.sqrt(torch.clamp(2.0 * self.gamma, min=0.0)).to(R.dtype)
        attn_logits = torch.baddbmm(attn_logits, R * g, (R * g).transpose(1, 2), beta=1.0, alpha=1.0)

        key_mask = (x.abs().sum(dim=-1) > 0)
        mask_val = -1e4 if attn_logits.dtype == torch.float16 else -1e9
        attn_logits = attn_logits.masked_fill(~key_mask.unsqueeze(1), mask_val)
        attn_logits = torch.clamp(attn_logits, min=-1e4, max=1e4)

        # numerically stable softmax: subtract max, compute in float32, cast back
        max_per_row = attn_logits.max(dim=-1, keepdim=True).values
        attn_logits = attn_logits - max_per_row
        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(attn_logits.dtype)

        return torch.bmm(attn_weights, v)


class MVMultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([MVAttentionHead() for _ in range(num_heads)])
        # Collapse heads equivariantly: [B, N, H, 16] -> [B, N, 16]
        self.output_proj = MVLinear(in_channels=num_heads, out_channels=1)

    def forward(self, x):
        B, T, D = x.shape
        outs = [h(x) for h in self.heads]           # each: [B, T, 16]
        y = torch.stack(outs, dim=2)                # [B, T, H, 16]
        y = self.output_proj(y)                     # [B, T, 16] (channel dim squeezed)
        return y


# --- Add toggles at top of file (temporary debug flags) ---
DEBUG_DISABLE_JOIN = False
DEBUG_DISABLE_GP   = False
# ----------------------------------------------------------

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
        gp = geometric_product_batch(x, y) if not DEBUG_DISABLE_GP else torch.zeros_like(x)
        join = equi_join_batch(x, y, reference) if not DEBUG_DISABLE_JOIN else torch.zeros_like(x)
        
        return torch.cat([gp, join], dim=-1)
"""
# This is the bilinear layer.
class MVGeometricBilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, reference):
        
        gp = geometric_product_batch(x, y)           
        join = equi_join_batch(x, y, reference)      

        return torch.cat([gp, join], dim=-1)         
"""    
    
# Putting the layers together to form a block. 
class MVFeedforwardBlock(nn.Module):    
    def __init__(self):
        super().__init__()
        self.norm = MVLayerNorm()
        self.linear1 = MVLinear()
        self.gelu = MVGatedGelu()
        self.bilinear = MVGeometricBilinear()
        self.linear2_gp   = MVLinear()
        self.linear2_join = MVLinear()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, reference=None):
        
        x_proj  = self.linear1(x)
        x_gated = self.gelu(x_proj)

        
        ref = reference if reference is not None else x
        x_bilinear = self.bilinear(x, x_gated, ref)

        n = len(constants.components)
        gp_part   = x_bilinear[..., :n]
        join_part = x_bilinear[..., n:]

        x_out = self.linear2_gp(gp_part) + self.linear2_join(join_part)
        x_out = self.dropout(x_out)
        return x_out   

            