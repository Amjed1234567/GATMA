# Author: Amjed Farooq Afzal. 

import precompute.constants_pre as constants_pre

class BladeProcessing:
        
    def __init__(self):       
        self.priority_dict = constants_pre.priority_dict
                 
    
    def blade_to_list(self, blade:str)->list:
        """A blade is converted into a list

        Args:
            blade (str): The blade is in str format.

        Returns:
            list: A list where each element is either e0, e1, e2 or e3.
        """
        new_list = []
        i = 0
        while (i < (len(blade)-1)):
            new_list.append(blade[i] + blade[i + 1])
            i += 2
        return new_list

    
    def sort_blade(self, blade: str)->str:
        """A blade in str format is sorted, meaning e0 appears before e1, and e1 appears before
            e2, etc.
        Args:
            blade (str): A blade in str format without number and '*'.
            Example: Not 3*e1e2e3, but only e1e2e3.
        Returns:
            tuple: (sign, list of basis vectors)
        """
            
        blade_list = self.blade_to_list(blade)
        # To keep track of the sign, everytime there is a swap of two basis vectors.
        sign = 1 
            
        # Iterating through the blade in list format (Outer loop). 
        for n in range(len(blade_list) - 1, 0, -1):
            swapped = False  # To keep track of if a swap occured. 
                
            for i in range(n):# Comparing elements next to each other (Inner loop). 
                if self.priority_dict[blade_list[i]] > self.priority_dict[blade_list[i + 1]]:
                    
                    # If higher value before lower value, swap. 
                    blade_list[i], blade_list[i + 1] = blade_list[i + 1], blade_list[i]
                            
                    swapped = True # There has been swap between two 
                    sign *= -1
                    
            if not swapped: # No swap. The list is already sorted. Stop. 
                    break
                
        return (sign, blade_list)
        
    
    def simplify_blade(self, sign:int, blade:list)->str:
        """A blade in the form of a list is simplified, i.e. e0e0 becomes '0', 
        e1e1 becomes '1', etc. 

        Args:
            sign (int): Either -1 or 1 depending on the previous sorting. 
            blade (list): A blade in the form of a list.

        Returns:
            str: The simplified blade. 
        """
        if (blade[0] == 'e0' and blade[1] == 'e0'):
            return '0'
        
        for i in range(len(blade)-1):
            if blade[i] == blade[i+1]:
                blade[i] = ""
                blade[i+1] = ""
        
        if sign == 1:
            sign = '1'
        else:
            sign = '-1'
            
        blade = ''.join(blade)      
                    
        return sign + blade            
