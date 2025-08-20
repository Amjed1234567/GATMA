# Author: Amjed Farooq Afzal. 

import precompute.constants_pre as constants_pre

from blade_processing import BladeProcessing

class BladeProducts:
    
    def __init__(self):
        self.components = constants_pre.components
    
    def blade_product_matrix(self):
        """Calculating all the blade products in the G301 algebra.

        Returns:
            2D list: A matrix of size 16 x 16 = 256. 
        """
        bp = BladeProcessing()
                
        matrix = [[col for col in range(len(constants_pre.components))] for row in range(len(constants_pre.components))]
        
        for i in range(len(self.components)):
            for j in range(len(self.components)):
                if ((i == 0) and (j == 0)):
                    matrix[i][j] = '1'
                elif (i == 0) and (j != 0):
                    matrix[i][j] =  '1' + self.components[j]               
                elif (j == 0) and (i != 0):
                    matrix[i][j] = '1' + self.components[i]
                else:                
                    sign, blade = bp.sort_blade(self.components[i] + self.components[j])
                    matrix[i][j] = bp.simplify_blade(sign, blade)
        
        return matrix                    
                

def main():
    
    bp = BladeProducts()    
    m = bp.blade_product_matrix()
    
    print(type(m))
    with open("pre.txt", "w") as f:
        for p in range(len(bp.components)):
            for q in range(len(bp.components)):
                f.write(bp.components[p] + " " + bp.components[q] + " " + m[p][q] + '\n')            


if __name__ == "__main__":
    main()    