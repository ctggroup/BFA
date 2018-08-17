CC = g++
CXXFLAGS = -g -O3 -lm -lgsl -lboost_program_options -std=c++11

BayesC: 
	$(CC) -o BayesFactors BayesFactors.cpp $(CXXFLAGS)

clean: 
	rm BayesFactors