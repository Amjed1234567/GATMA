# This file contains the batch versions of geometric product,
# outer product, dual and join. If you need the simpler versions for 
# just one multivector, please see file multivector.py.

import torch
import constants
from multivector import Multivector


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_geometric_product_tensor():
    """Calculating the geometric product tensor to be used 
        in the below batch version of the geometric product.                 

        Returns:
             x (torch.tensor): Shape: (len(constants.components), 
             len(constants.components), len(constants.components))).
     """        
    n = len(constants.components)
    G = torch.zeros(n, n, n)
    for i, a in enumerate(constants.components):
        for j, b in enumerate(constants.components):
            product = constants.product_table[a][b]
            if product == "0":
                continue
            sign = 1
            if product == "1":
                sign = 1
            if product == "-1":
                sign = -1                
            if product.startswith("-") and len(product) > 2:
                sign = -1
                product = product[2:]
            if product.startswith("1") and len(product) > 1:
                product = product[1:]

            # Scalar result: "1" or "-1"
            if product == "-1" or product == "1":
                k = constants.components.index('1')
                G[i, j, k] = sign
            elif product in constants.components:
                k = constants.components.index(product)
                G[i, j, k] = sign
    return G

# Shape: (len(constants.components), len(constants.components), len(constants.components))).
GEOMETRIC_PRODUCT_TENSOR = build_geometric_product_tensor().to(device)


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

    result = torch.einsum('btij,ijk->btk', xy_ij, GEOMETRIC_PRODUCT_TENSOR)
    
    return result


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


DUAL_MATRIX = build_dual_matrix().to(device)


def dual_batch(x):
    """
    Args:
        x: tensor of shape (batch, num_tokens, len(constants.components))
    Returns:
        dual(x): tensor of shape (batch, num_tokens, len(constants.components))
    """
    return torch.matmul(x, DUAL_MATRIX)  


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


OUTER_MASK = build_outer_mask().to(device)


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
    result = torch.einsum('btij,ijk->btk', xy_ij, OUTER_MASK)
    
    return result


def join_batch(x, y):
    """
    Args:
        x, y: tensors of shape (batch_size, num_tokens, len(constants.components))
    Returns:
        Join(x, y): tensor of shape (batch_size, num_tokens, len(constants.components))
    """
    x = x.to(DUAL_MATRIX.device)
    y = y.to(DUAL_MATRIX.device)
    
    x_dual = dual_batch(x)
    y_dual = dual_batch(y)
    outer = outer_product_batch(x_dual, y_dual)
        
    return dual_batch(outer)



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
    # Shape: (batch_size, num_tokens, 1)
    pseudoscalar = reference[..., 15].unsqueeze(-1)  
    
    return join * pseudoscalar.to(join.device)
