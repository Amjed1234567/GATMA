# Author: Amjed Farooq Afzal

import random
import math
import torch

def create_one_sample():
      
    total = 6
    start = -10 
    end = 10

    numbers = random.choices(range(start, end + 1), k=total)
    
    x1 = numbers[0]
    y1 = numbers[1]
    z1 = numbers[2]
    
    x2 = numbers[3]
    y2 = numbers[4]
    z2 = numbers[5]
    
    dist = math.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )
    
    return [x1, y1, z1, x2, y2, z2, dist]
    
        
def create_data(data_size:int):
    data_list = []
    for i in range(data_size):
        sample = create_one_sample()
        data_list.append(sample)

    return torch.FloatTensor(data_list)
