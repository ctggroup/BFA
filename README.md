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

--M : No. of markers (required)    
--N : No. of individuals (required)   
--iter : No. of Gibbs sampling iterations (Default:100) 
--burnin: No. burnin iterations (Default: 10) 
--num_feat : No. of estimated factors (Default:3)   
--input : input tag for genotype matrix file (required). The genotype matrix is assumed to have the suffix ".X" after the tag provided with input.   
--output : Results of the gibbs sampler are outputed using the tag provided here (Default:"BayesFactors_out"). The program will create a folder with this tag and then output the estimated factors and scores in that folder for every Gibbs iteration.   

