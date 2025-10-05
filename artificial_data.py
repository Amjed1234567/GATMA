import random as rnd
import math
import torch

def create_one_sample():
    # Plane parameters.
    a = rnd.choice([i for i in range(-10,10) if i not in [0]])
    b = rnd.choice([i for i in range(-10,10) if i not in [0]])
    c = rnd.choice([i for i in range(-10,10) if i not in [0]])
    d = rnd.choice([i for i in range(-10,10) if i not in [0]])
    # Point parameters.
    x = rnd.choice([i for i in range(-10,10) if i not in [0]])
    y = rnd.choice([i for i in range(-10,10) if i not in [0]])
    z = rnd.choice([i for i in range(-10,10) if i not in [0]])
    # Euclidian distance between point and plane.
    dist = abs(a*x + b*y + c*z + d) / math.sqrt(a**2 + b**2 + c**2)

    return [a, b, c, d, x, y, z, dist]


def create_data(data_size:int):
    data_list = []
    for i in range(data_size):
        sample = create_one_sample()
        data_list.append(sample)

    return torch.FloatTensor(data_list)