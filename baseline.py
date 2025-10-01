from torch_geometric.datasets import QM9
import numpy as np
import torch

# <<<<<< This is the baseline model for the qm9 dataset. >>>>>>>>

dataset = QM9(root='data/QM9')

dipole_moment = []
for i in range(dataset.len()):
    dipole_moment.append(dataset[i].y[0][0].item())

# Find the mean dipole moment.
mean_ = np.array(dipole_moment).mean()

# Subtract mean from all the dipole moments.
new_values = [r - mean_ for r in dipole_moment]

# Subtract the two lists.
result = np.array(new_values) - np.array(dipole_moment)
# print("The mean absolute error is ...")
# print(np.absolute(result).mean())

# <<<<<<<<<<  End  >>>>>>>>>>>>

# T<<<<<<<< This is the baseline for the geometric data set. >>>>>>>>

def mean_dist(data:torch.FloatTensor):
    sum_dist = 0
    for i in range(data.shape[0]):
        sum_dist = sum_dist + data[i][7].item()

    return sum_dist / data.shape[0] # Mean of distances.


def calculate_mae(data:torch.FloatTensor):
    dist_list = []
    for i in range(data.shape[0]):
        dist_list.append(data[i][7].item())
    # Subtract mean from all the distances..
    new_values = [r - mean_dist(data) for r in dist_list]
    # Subtract the two lists.
    result = np.array(new_values) - np.array(dist_list)
    
    return np.absolute(result).mean()

