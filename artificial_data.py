import random as rnd
import math
import torch

def create_one_sample():
    # Plane parameters.
    a = rnd.randrange(1,10, 1)
    b = rnd.randrange(1,10, 1)
    c = rnd.randrange(1,10, 1)
    d = rnd.randrange(1,10, 1)
    # Point parameters.
    x = rnd.randrange(1,10, 1)
    y = rnd.randrange(1,10, 1)
    z = rnd.randrange(1,10, 1)
    # Euclidian distance between point and plane.
    dist = abs(a*x + b*y + c*z + d) / math.sqrt(a**2 + b**2 + c**2)

    return [a, b, c, d, x, y, z, dist]


def create_data(data_size:int):
    data_list = []
    for i in range(data_size):
        sample = create_one_sample()
        data_list.append(sample)

    return torch.FloatTensor(data_list)


