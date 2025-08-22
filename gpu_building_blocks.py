import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
from batch_operations import geometric_product_batch, equi_join_batch


# This is the linear layer.
class MVLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None):
        
        super().__init__()
        
        # Initializing w and v parameters with random numbers.
        self.w = nn.Parameter(torch.randn(5))
        self.v = nn.Parameter(torch.randn(4))
        # This is a scalar term to ensure equivariance. 
        self.scalar_bias = nn.Parameter(torch.zeros(1))  
        
        # Fixed parameters. Will not change during training. 
        self.register_buffer("grade_masks", constants.grade_masks)
        self.register_buffer("e0_geometric_products", constants.e0_geometric_products)
        
    def forward(self, x):
        """ 
        Args:
            x (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components))

        Returns:
            torch.tensor: Shape: (batch_size, num_tokens, len(constants.components))
        """           
        first_sum = sum(self.w[i] * (self.grade_masks[i] * x) for i in range(5))
        # In all the multivectors in all batches select the scalar component. 
        first_sum[:, :, 0] += self.scalar_bias
            
        second_sum = sum(self.v[j] * (self.grade_masks[j] * x @ self.e0_geometric_products.T)
                                        for j in range(4))      
            
        return first_sum + second_sum      



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
        # The shape is (batch_size, num_tokens, 1).
        inner_prod = ((x ** 2) * self.mask).sum(dim=-1, keepdim=True)
        # Now, the shape is (batch_size, 1, 1).
        norm = torch.sqrt(inner_prod.mean(dim=1, keepdim=True) + self.eps)
        
        return x / norm


class MVAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = MVLinear()
        self.key = MVLinear()
        self.value = MVLinear()
        
        # Fixed parameters. Will not change during training. 
        self.register_buffer("mask", constants.non_e0_mask)
        
        # For scaling. 
        self.scale = (len(constants.components)) ** -0.5

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).

        Returns:
            (torch.tensor): Shape: (batch_size, num_tokens, len(constants.components)).
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Masking out e0 components.
        q_masked = q * self.mask
        k_masked = k * self.mask        
        
        # Using bmm instead of einsum. bmm is faster.
        # https://docs.pytorch.org/docs/stable/generated/torch.bmm.html
        # Assuming the shape of q_masked and k_masked is 
        # (Batch_size, num_tokens, len(constants.components)), it is necessary to 
        # transpose k_masked to get shape (Batch_size, len(constants.components), num_tokens).
        # The shape of the product is (batch_size, num_tokens, num_tokens).
        attn_logits = torch.bmm(q_masked, k_masked.transpose(1, 2)) * self.scale
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
        
        # Concatenate outputs from all heads. Concat on feature dim.
        multihead_outputs = torch.cat([head(x) for head in self.heads], dim=-1)  
               
                 
        batch, tokens, _ = multihead_outputs.shape
        
        return self.output_proj(multihead_outputs.view(batch, tokens, -1))



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
            Shape: (batch_siza, num_tokens, 2*len(constants.components))
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
        