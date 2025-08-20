import torch
import math
import constants  

class Multivector:
    
    def __init__(self, coefficients: torch.Tensor):
        if not isinstance(coefficients, torch.Tensor):
            coefficients = torch.tensor(coefficients, dtype=torch.float32)
        self.coefficients = coefficients.to(coefficients.device)  

    
    def to_string(self):
        """Making a string representaion of the multivector.

        Returns:
            str: Multivector as a string. 
        """
        # Convert tensor to list.
        # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html#torch.Tensor.tolist 
        list_of_coefficients = self.coefficients.tolist()
        str_ = ""
        for i in range(len(constants.components)):
            str_ += str(list_of_coefficients[i]) + "*" + constants.components[i] + " + "
        return str_[:len(str_)-2]
    
    
    
    def geometric_product(self, M):
        """Calculating the geometric product between two multivectors.
        Using components_coefficients from constants.py.
        
        Args:
            M (Multivector): The method is used on a Multivector object.
            The argument is another multivector.

        Returns:
            Multivector: An instance of the Multivector class.
        """       
        x_coefficients = constants.components_coefficients.copy()
            
        for i in range(len(constants.components)):
            for j in range(len(constants.components)):
                product = constants.product_table[constants.components[i]][constants.components[j]]
                coef_product = self.coefficients[i] * M.coefficients[j]
                if (product == '-1'):
                    x_coefficients['1'] += (-1)*coef_product                
                elif (product == '1'):
                    x_coefficients['1'] += coef_product
                elif ((product[0] == '1') and (len(product) > 1)):
                    x_coefficients[product[1:]] += coef_product
                elif ((product[0] == '-') and (len(product) > 2)):
                    x_coefficients[product[2:]] += (-1)*coef_product   
        
        return Multivector(list(x_coefficients.values()))

    

    def add(self, M):
        """"Adding two multivectors.

        Args:
             M (Multivector): The method is used on a Multivector object.
            The argument is another multivector.

        Returns:
             Multivector: An instance of the Multivector class.
        """
        return Multivector(torch.add(self.coefficients, M.coefficients))
    
    
    def subtract(self, M):
        """Subtracting two multivectors.

        Args:
             M (Multivector): The method is used on a Multivector object.
            The argument is another multivector.

        Returns:
             Multivector: An instance of the Multivector class.
        """
        return Multivector(torch.subtract(self.coefficients, M.coefficients))
    
    
    @staticmethod
    def blade_grade(blade:str)->int:
        """Helper function used in outer_product function.
        Finds the grade of a blade. 

        Args:
            blade (str): A blade in the shape of a string.

        Returns:
            int: The grade of the blade. 
        """
        if (blade == "-1") or (blade == "0") or (blade == "1"):
            return 0
        elif (blade[0] == '-'):
            return int(len(blade[2:])/2)
        elif (blade[0] == '1'):
            return int(len(blade[1:])/2)
        else:
            return int(len(blade)/2)
       
            
    @staticmethod
    def blade_filter(blade1:str, blade2:str, blade3:str)->bool:
        """Helper function for outer_product.
        Determines if a blade should be kept or not.
        *) Only blades that satisfy grade(blade) = grade(M1_blade) + grade(M2_blade)
        should be kept. 

        Args:
            blade1 (str): A blade in the shape of a string.
            blade2 (str): A blade in the shape of a string.
            blade3 (str): A blade in the shape of a string.

        Returns:
            bool: True if *) is satisfied.   
        """
        return (Multivector.blade_grade(blade1) == Multivector.blade_grade(blade2) + 
                Multivector.blade_grade(blade3))

        
    def outer_product(self, M):
        """Finding the outer product. Using components_coefficients from 
        constants.py.

        Args:
            M (Multivector): The method is used on a Multivector object.
            The argument is another multivector.

        Returns:
            Multivector: An instance of the Multivector class.
        """       
        x_coefficients = constants.components_coefficients.copy()
            
        for i in range(len(constants.components)):
            for j in range(len(constants.components)):
                product = constants.product_table[constants.components[i]][constants.components[j]]
                coef_product = self.coefficients[i] * M.coefficients[j]
                filter = Multivector.blade_filter(product, constants.components[i], constants.components[j])
                    
                if ((product[0] == '1') and (len(product) > 1)) and filter:
                    x_coefficients[product[1:]] += coef_product
                elif ((product[0] == '-') and (len(product) > 2)) and filter:
                    x_coefficients[product[2:]] += (-1)*coef_product   
        
        return Multivector(torch.tensor(list(x_coefficients.values()), dtype=torch.float32))
    
    
    def multiply_with_scalar(self, k:float):
        """A multivector is multiplied with a scalar.

        Args:
            k (float): The scalar that is multiplied.

        Returns:
            Multivector: An instance of the Multivector class.
        """        
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, dtype=self.coefficients.dtype, device=self.coefficients.device)
        else:
            k = k.to(self.coefficients.device)

        # Just want to make sure it can be broadcasted. 
        if k.dim() == 0:
            k = k.view(1)  

        return Multivector(self.coefficients * k)

    
    
    
    def blade_projection(self, k):
        """Extract the grade k parts of a multivector.

        Args:
            k (int): The grade is between 0 and 4. 

        Returns:
            Multivector: An instance of the Multivector class.
        """
        projected_coefficients = []
        for i in range(len(constants.components)):
            if Multivector.blade_grade(constants.components[i]) == k:
                projected_coefficients.append(self.coefficients[i].item())
            else:
                projected_coefficients.append(0)
                
        return Multivector(torch.tensor(projected_coefficients))
    
    
    
    def dual(self):
        """Find the dual of a multivector. Using dual_dict from constants.py
        as a look-up table.

        Returns:
            Multivector: An instance of the Multivector class.
        """
        x_coefficients = constants.components_coefficients.copy()
                    
        for i in range(len(constants.components)):
            d = constants.dual_dict[constants.components[i]] # Finding the dual.    
            if d[0] == '-':
                x_coefficients[ d[1:] ] = -1 * self.coefficients[i].item()
            elif d == '1':
                x_coefficients[d] += self.coefficients[i].item()
            else:
                x_coefficients[d] += self.coefficients[i].item()
                
        return Multivector(torch.tensor(list(x_coefficients.values())))
    
    
    def join(self, M):
        """Computes the join of two multivectors.
        First, the dual of each multivector is found.
        Second, the outer product of the two duals is found.
        Finally, the dual of the outer product is found. 
        Please note that this version is correct for rotation-only equivariance.       

        Args:
            M (Multivector): The method is used on a Multivector object.

        Returns:
            Multivector: An instance of the Multivector class.
        """
        return self.dual().outer_product(M.dual()).dual()   
    
    
    def equi_join(self, M, reference):
        """Computes the join of two multivectors.
        This is the full E(3) equivariance version.
        Using a reference multivector e.g. one of the inputs. 

        Args:
            M (Multivector): The method is used on a Multivector object.
            reference (Multivector): Another Multivector object.

        Returns:
            Multivector: An instance of the Multivector class.
        """
        
        pseudoscalar = reference.coefficients[15]
        
        return self.join(M).multiply_with_scalar(pseudoscalar.to(self.coefficients.device))



     
    def reverse(self):
            """This will reverse a multivector consisting of only one blade,
            or a rotor, which typically consists of a scalar and a grade-2 blade.
            Please do not use it for general multivectors. 

            Returns:
                Multivector: An instance of the Multivector class.
            """
            x = [0]*len(constants.components)
            for i in range(len(constants.components)):
                if self.coefficients[i] != 0:
                    rev = constants.reverse_of_blade[constants.components[i]]
                    if rev[0] == '-':
                        x[i] = -1
                    else:
                        x[i] = 1
                        
            return Multivector(torch.tensor(x, dtype=torch.float32) * self.coefficients)



    def inner_product(self, M)->float:  
            """ Calculating inner product of two multivectors.

            Returns:
                float: The value (scalar) of the inner product.  
            """             
            # Only using the indices of components not containing e0 basis vector.
            indices = torch.tensor(constants.not_e0_indices)
            sum_of_products = (self.coefficients[indices] * M.coefficients[indices]).sum()
            
            return sum_of_products.item()
        
        
        
    def inverse(self):
        """Finding the inverse of a multivector. 
        Please note that this function only works on blades and rotors. 

        Raises:
            ValueError: The squared norm of the multivector should not be zero.

        Returns:
            Multivector: An instance of the Multivector class.
        """
        norm_squared = self.geometric_product(self.reverse()).coefficients[0].item()
        if norm_squared == 0:
            raise ValueError("Unable to invert, because norm^2 = 0")
        
        return self.reverse().multiply_with_scalar(1 / norm_squared) 
    
    
    def normalize(self):
        """Normalazing a multivector, meaning the norm will become one, but the 
        orientation will not change.
        Please note that this function only works on blades and rotors. 

        Raises:
            ValueError: The squared norm of the multivector should not be zero.

        Returns:
            Multivector: An instance of the Multivector class.
        """
        norm_squared = self.geometric_product(self.reverse()).coefficients[0].item()
        if norm_squared == 0:
            raise ValueError("Unable to normalize, because norm^2 = 0")
    
        norm = math.sqrt(norm_squared)
        
        return self.multiply_with_scalar(1.0 / norm)
    
    
    def rotate(self, R):
        """Rotating a multivector using sandwich product RMR*,
        where R is the rotor and R* is the inverse. M is the multivector.

        Args:
            R (Multivector): Must be a rotor. Usually consisting of a scalar 
            and a bivector. 

        Returns:
            Multivector: An instance of the Multivector class.
        """
        R = R.normalize() # Make sure the versor is normalized. 
        
        return R.geometric_product(self).geometric_product(R.inverse())
    
    
    
    def translate(self, deltax:float, deltay:float, deltaz:float):
        """Translates a multivector in the e0 direction using sandwich 
        product TMT*, where T is the translater and T* is the inverse. M is the multivector.   

        Args:
            deltax (float): Translation in the x-direction.
            deltay (float): Translation in the y-direction.
            deltaz (float): Translation in the z-direction.

        Returns:
            Multivector: An instance of the Multivector class.
        """
        new_tensor = torch.zeros(16) # Just to have an initial tensor. 
        
        # Building the translater = 1 + 0.5*(deltax*e1 + deltay*e2 + deltaz*e3)*e0.
        new_tensor[constants.components.index('1')] = 1.0
        new_tensor[constants.components.index('e0e1')] = 0.5 * deltax
        new_tensor[constants.components.index('e0e2')] = 0.5 * deltay
        new_tensor[constants.components.index('e0e3')] = 0.5 * deltaz
        
        translator = Multivector(new_tensor) 
        # This is the sandwich product.     
        return translator.geometric_product(self).geometric_product(translator.reverse())
    
        
        
    def reflect(self, mirror):
        """Reflects a multivector M across another multivector a.
        Using sandwich product -aMa*, where a is the mirror multivector and a* is the 
        inverse. 

        Args:
            mirror (Multivector): Should be invertible. 

        Returns:
            Multivector: An instance of the Multivector class.
        """
        mirror_inv = mirror.inverse()
        return mirror.geometric_product(self).geometric_product(mirror_inv).multiply_with_scalar(-1.0)
        