# Geometric Algebra Transformer for Molecular Analysis (GATMA).

## 1. Introduction

This repo contains the code of the Geometric Algebra Transformer for Molecular Analysis (GATMA), 
which has been trained on the [qm9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html) data set 
to calculate the dipole moment of a molecule.   GATMA is built using Projective Geometric Algebra (PGA). 
### Where to start
If you just want to understand the code, start by looking at the files multivector.py and constants.py. The remaining code is built on these two files. 
The next step is batch_operations.py that contains the batch versions of what you find in multivector.py.
Finally, you can look at gpu_building_blocks.py and gpu_transformer.py. 

## 2. A short description of the files

### artificial_data.py 
This file is used to create an artificial data set consisting of random pairs of 3D points and their Euclidean distance.

### baseline.py
The MAE of the baseline model is calculated here.

### batch_operations.py
This file contains the batch versions of geometric product, outer product, dual and join. If you need the simpler versions for 
just one multivector, please see file multivector.py.

### constants.py
All the constants used in this project.

### gpu_building_blocks
This file contains class MVLinear, class MVGatedGelu, class MVLayerNorm, class MVAttentionHead, class MVMultiHeadAttention, 
class MVGeometricBilinear, and class MVFeedforwardBlock. These are the building blocks of the Transformer. 

### gpu_transformer.py
This file contains the Transformer build using the building blocks from the file gpu_building_blocks.

### multivector.py
An instance of this class is a multivector. You will find methods that are used on multivector objects:\
to_string\
geometric_product\
add\
subtract\
outer_product\
multiply_with_scalar\
blade_projection\
dual\
join\
equi_join\
reverse\
inner_product\
inverse\
normalize\
rotate\
translate\
reflect

## 3. How to use this code

Just clone the repository. Make sure that you have a virtual environment installed.
The file train_0.py is used for training. To resume training with more epochs, you can use train_more.py. 
If you are using a university computer, you may need to create a .sh file. You will need a GPU to run this code. 

## 4. Citations
 
If you find this code useful, please cite:

@misc{gatma2025,
  author = {Afzal, Amjed Farooq},
  title = {GATMA},
  howpublished = {\url{https://github.com/ Amjed1234567/GATMA}},
  year = {2025},
}

## 5. License

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations 
in the disclaimer below) provided that the following conditions are met:

The name of Amjed Farooq Afzal may not be used to endorse or promote commercial products derived from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.  
THIS SOFTWARE IS PROVIDED BY AMJED FAROOQ AFZAL "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL AMJED FAROOQ AFZAL BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
