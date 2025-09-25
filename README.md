# Geometric Algebra Transformer for Molecular Analysis (GATMA).

## 1. Introduction

This repo contains the code of the Geometric Algebra Transformer for Molecular Analysis (GATMA), 
which has been trained on the [qm9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html) data set 
to calculate the dipole moment of a molecule.   GATMA is built using Projective Geometric Algebra (PGA). 

## 2. How to use this code

Just clone the repository. Make sure that you have a virtual environment installed.
The file train_0.py is used for training. To resume training with more epochs, you can use train_more.py. 
If you are using a university computer, you may need to create a .sh file. You will need a GPU to run this code. 

## Citations
 
If you find this code useful, please cite:

@misc{gatma2025,
  author = {Afzal, Amjed Farooq},
  title = {GATMA},
  howpublished = {\url{https://github.com/ Amjed1234567/GATMA}},
  year = {2025},
}

## 3. License

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
