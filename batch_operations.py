# Author: Amjed Farooq Afzal. 
# This file contains the batch versions of geometric product,
# outer product, dual and join. If you need the simpler versions for 
# just one multivector, please see file multivector.py.

import torch
import constants
from multivector import Multivector
from constants import components, product_table  # blades + symbolic products  :contentReference[oaicite:0]{index=0}

# Build the 16×16×16 geometric product tensor from the symbolic table once.
def _build_geometric_product_tensor():
    n = len(components)  # 16
    idx = {b: i for i, b in enumerate(components)}
    G = torch.zeros(n, n, n, dtype=torch.float32)

    for left_blade, rights in product_table.items():
        i = idx[left_blade]
        for right_blade, result in rights.items():
            j = idx[right_blade]

            if result == '0':
                continue

            # sign and blade parsing: examples: '1', '-1', '1e2', '-1e0e1e3'
            sign = 1.0
            s = result
            if s.startswith('-'):
                sign = -1.0
                s = s[1:]  # strip '-'

            # strip leading '1' when present (e.g. '1e2' -> 'e2')
            if s.startswith('1') and len(s) > 1:
                s = s[1:]

            # Now s is one of the canonical blade labels in `components`, e.g. '1', 'e2', 'e0e1e3', etc.
            k = idx[s]
            G[i, j, k] = sign

    return G

# Base (CPU/float32) table; we’ll cast/copy per device/dtype as needed at runtime.
G_BASE = _build_geometric_product_tensor()

# Small cache so we don’t re-to() every batch
_GEOPROD_CACHE = {}
def _get_G_for(x_dtype, x_device):
    key = (x_dtype, x_device)
    if key not in _GEOPROD_CACHE:
        _GEOPROD_CACHE[key] = G_BASE.to(dtype=x_dtype, device=x_device)
    return _GEOPROD_CACHE[key]



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# <<<<<<  >>>>>>>
# This is just to “mirror” a constant to the inputs’ device (and cache it).
_cached = {}

def _on_device(t_cpu: torch.Tensor, dev):
    key = (id(t_cpu), dev)
    if key not in _cached:
        _cached[key] = t_cpu.to(dev)
    
    return _cached[key]
# <<<<<<  >>>>>>>


def geometric_product_batch(x, y):
    """
    Args:
        x, y: tensors of shape (batch_size, num_tokens, len(constants.components))
    Returns:
        geometric product: tensor of shape (batch_size, num_tokens, len(constants.components))
    """    
    # Shape: (batch_size, num_tokens, len(constants.components), 1)
    x_i = x.unsqueeze(-1)
    # Shape: (batch_size, num_tokens, 1, len(constants.components))  
    y_j = y.unsqueeze(-2)  
    # Shape: (batch_size, num_tokens, len(constants.components), len(constants.components))  
    xy_ij = x_i * y_j      

    G_cast = _get_G_for(xy_ij.dtype, xy_ij.device)
    
    return torch.einsum('btij,ijk->btk', xy_ij, G_cast)    
    


def build_dual_matrix():
    """Find the dual. To be used in dual method below. 

    Returns:
        x: tensor of shape (len(constants.components), len(constants.components))
    """
    n = len(constants.components)
    dual_mat = torch.zeros(n, n)
    for i, a in enumerate(constants.components):
        b = constants.dual_dict[a]
        if b.startswith("-"):
            sign = -1
            b = b[1:]
        else:
            sign = 1
        j = constants.components.index(b)
        dual_mat[i, j] = sign
    
    return dual_mat 


#DUAL_MATRIX = build_dual_matrix().to(device)
DUAL_MATRIX = build_dual_matrix()                # stay on CPU


def dual_batch(x):
    """
    Args:
        x: tensor of shape (batch, num_tokens, len(constants.components))
    Returns:
        dual(x): tensor of shape (batch, num_tokens, len(constants.components))
    """
    D = _on_device(DUAL_MATRIX, x.device)
    
    return torch.matmul(x, D)


def build_outer_mask():
    """To be used in outer product method below. 

    Returns:
        torch.tensor: Shape (len(constants.components), len(constants.components), 
        len(constants.components))
    """
    n = len(constants.components)
    outer_mask = torch.zeros(n, n, n)

    for i, a in enumerate(constants.components):
        for j, b in enumerate(constants.components):
            product = constants.product_table[a][b]

            if product == "0":
                continue            
            sign = 1
            if product == "-1":                
                k = constants.components.index("1")
                outer_mask[i, j, k] = -1
                continue
            if product == "1":                
                k = constants.components.index("1")
                outer_mask[i, j, k] = 1
                continue
            if product.startswith("-1") and len(product) > 2:
                sign = -1
                product = product[2:]
            elif product.startswith("1") and len(product) > 1:
                product = product[1:]
            

            if product in constants.components:
                ga = Multivector.blade_grade(a)
                gb = Multivector.blade_grade(b)
                gp = Multivector.blade_grade(product)
                if gp == ga + gb:
                    k = constants.components.index(product)
                    outer_mask[i, j, k] = sign

    return outer_mask


#OUTER_MASK = build_outer_mask().to(device)
OUTER_MASK = build_outer_mask()                 # stay on CPU


def outer_product_batch(x, y):
    """Calculates the outer product.

    Args:
        x, y: shape (batch_size, num_tokens, len(constants.components))        

    Returns:
        outer product: shape (batch_size, num_tokens, len(constants.components))
    """
    x_i = x.unsqueeze(-1)
    y_j = y.unsqueeze(-2)
    xy_ij = x_i * y_j
    M = _on_device(OUTER_MASK, x.device)
    
    return torch.einsum('btij,ijk->btk', xy_ij, M)


def join_batch(x, y):
    """
    Args:
        x, y: tensors of shape (batch_size, num_tokens, len(constants.components))
    Returns:
        Join(x, y): tensor of shape (batch_size, num_tokens, len(constants.components))
    """
    #x = x.to(DUAL_MATRIX.device)
    #y = y.to(DUAL_MATRIX.device)
    
    #x_dual = dual_batch(x)
    #y_dual = dual_batch(y)
    #outer = outer_product_batch(x_dual, y_dual)
        
    #return dual_batch(outer)
    return dual_batch(outer_product_batch(dual_batch(x), dual_batch(y)))



def equi_join_batch(x, y, reference):
    """
    Computes the E(3)-equivariant join of two multivectors in batch form.

    This is the full E(3)-equivariant version of the join operation, incorporating a 
    reference multivector (typically one of the inputs) to ensure correct transformation 
    behavior under reflections. Internally, the join is defined as the dual of the outer 
    product of the duals of x and y, scaled by the pseudoscalar component of the reference.

    Args:
        x (torch.Tensor): Tensor of shape (batch_size, num_tokens, 16), representing the first multivector.
        y (torch.Tensor): Tensor of shape (batch_size, num_tokens, 16), representing the second multivector.
        reference (torch.Tensor): Tensor of shape (batch_size, num_tokens, 16), representing the reference multivector.

    Returns:
        torch.Tensor: The E(3)-equivariant join of x and y, with shape (batch_size, 
        num_tokens, len(constants.components)).
    """

    join = join_batch(x, y)
    # Using orientation from the trivector e1e2e3:
    idx_triv = constants.components.index('e1e2e3')  # orientation channel
    triv = reference[..., idx_triv].unsqueeze(-1)
    triv_sign = torch.sign(triv)
    # Fallback to +1 when triv == 0 to avoid 0-sign.
    triv_sign = torch.where(triv_sign == 0, triv_sign.new_tensor(1.0), triv_sign)

    return join * triv_sign.to(join.device)