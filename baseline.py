# This is the baseline model for the qm9 dataset.

from torch_geometric.datasets import QM9
import numpy as np

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

