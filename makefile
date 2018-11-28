CC = g++
CXXFLAGS = -g -O3 -lm -lgsl -lboost_program_options -larmadillo -llapack -lblas -std=c++11

BayesFactors: 
	$(CC) -o BayesFactors BayesFactors.cpp Sampling_functions.cpp $(CXXFLAGS)

clean: 
	rm BayesFactors