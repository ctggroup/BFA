# BayesFactors   

## Description   
Bayesian probabilistic factorization using Gibbs sampling.

Implementation based on paper by Salakhutdinov and Mih (2008), https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf.


## Installation
To install simply type in the command line:    
`make`

and run with 

`./BayesFactors`

adding input options that are given below. The option specifying the input matrix file and the options specifying the dimensions of the matrix (no. of rows and columns) are required.


## Dependencies
To correctly compile the program you need to have installed the boost, Eigen and armadillo libraries.     

In ubuntu linux you type the following:   

for boost library:    
`sudo apt-get install libboost-all-dev`     

for eigen library:   
`sudo apt install libeigen3-dev`   

for armadillo please install the library from source as described here:   
http://arma.sourceforge.net/download.html   


## Input and options   

Possible options:   

### Required   
--M : No. of markers (required)    
--N : No. of individuals (required)   
--input : Input tag for genotype matrix file (required). The genotype matrix is assumed to have the suffix ".X" after the tag provided with input.   

### Optional   
--iter : No. of Gibbs sampling iterations (Default:100)   
--burnin: No. burnin iterations (Default: 10)   
--num_feat : No. of estimated factors (Default:3)   
--output : Results of the gibbs sampler are outputed using the tag provided here (Default:"BayesFactors_out"). The program will create a folder with this tag and then output the estimated factors and scores in that folder for every Gibbs iteration.   
--b0_m : Shrinkage constant for prior of column hyper-parameters (Default: 2)   
--b0_u : Shrinkage constant for prior of row hyper-parameters (Default: 2)   
--scale : Flag to substract mean and divide by standard deviation for each column of the input matrix before analysis.   
--miss : Flag to indicate that there is missing data. The data that should be masked as "missing" is specified with an Indicator matrix (NxM) that contains 1 or 0. The file must have the suffix ".Indicator" after the tag provided with input.   

## Examples   

### Example 1   
./BayesFactors --input inputfile --N 3000 --M 1269 --burnin 100 --iter 100 --num_feat 3 --scale   

The genotype matrix is assumed to have the suffix ".X". So in this example the program expects a file called inputfile.X to read in the genotype matrix.  

### Example 2   
./BayesFactors --input inputfile --N 3000 --M 1269 --burnin 100 --iter 100 --num_feat 5 --scale --miss   