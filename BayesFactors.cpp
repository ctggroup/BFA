#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include <random>
#include <map>
#include <string>
#include <iomanip>

#include <unistd.h>
#include <string>
#include <algorithm>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/random.hpp>
#include <boost/program_options.hpp>
#include <iterator>
#include <armadillo>

#include "Sampling_functions.h"

using namespace std;
using namespace Eigen;
using namespace arma;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{

	po::options_description desc("Options");
	desc.add_options()
				("M", po::value<int>()->required(), "No. of markers")
				("N", po::value<int>()->required(), "No. of individuals")
				("num_feat", po::value<int>()->default_value(3), "No. of factors")
				("iter", po::value<int>()->default_value(100), "No. of Gibbs iterations")
				("input", po::value<std::string>()->required(),"Input filename")
				("out", po::value<std::string>()->default_value("BayesFactors_out"),"Output filename");

	srand(time(0));

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc),vm);
	po::notify(vm);

	int M=vm["M"].as<int>();
	int N=vm["N"].as<int>();
	int num_feat=vm["num_feat"].as<int>();
	int iter=vm["iter"].as<int>();
	string input=vm["input"].as<string>();
	string output=vm["out"].as<string>();

	MatrixXd X(N,M);

	int i,j,k,l,m=0;
	auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout<<"Started analysis!"<<endl;
	timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout << ctime(&timenow) << endl;

	ifstream f1(input+".X");
	if (f1){
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				f1 >> X(i,j);
				//cout<<X(i,j)<<endl;
			}
		}
		f1.close();
		cout<<"finished reading matrix X!"<<endl;
		timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
		cout << ctime(&timenow) << endl;
	}else{
		cout<<"the "+input+".X"+" file does not exist/cannot be opened!"<<endl;
		return 0;
	}

	//normalize matrix X (genotype matrix)
	RowVectorXd mean = X.colwise().mean();
	RowVectorXd sd = (X.rowwise() - mean).array().square().colwise().mean();
	X = (X.rowwise() - mean).array().rowwise() / sd.array();

	//Factor analysis
	//Initialization of latent variables. It can be done with ML estimates. Here its done just by sampling normal deviates.
	MatrixXd w1_M1_sample(M,num_feat);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < num_feat; j++)
		{
			w1_M1_sample(i,j)=rnorm(0,1);
		}

	}
	MatrixXd w1_P1_sample(N,num_feat);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < num_feat; j++)
		{
			w1_P1_sample(i,j)=rnorm(0,1);
		}

	}
	//Initialization of hyperparameters
	MatrixXd WI_u(num_feat,num_feat);
	VectorXd mu_u(num_feat);
	MatrixXd lambda_u(num_feat,num_feat);
	VectorXd mu0_u(num_feat);

	MatrixXd WI_m(num_feat,num_feat);
	VectorXd mu_m(num_feat);
	MatrixXd lambda_m(num_feat,num_feat);
	VectorXd mu0_m(num_feat);

	WI_m.setIdentity();
	mu_m.setZero();
	lambda_m.setIdentity();
	mu0_m.setZero();

	WI_u.setIdentity();
	mu_u.setZero();
	lambda_u.setIdentity();
	mu0_u.setZero();

	int b0_m=2;int df_m=num_feat;
	int b0_u=2;int df_u=num_feat;

	double alpha=1;

	//create folder to put output
	int systemRet = system("mkdir -p BayesFactors_out");
	if(systemRet == -1){
		cout<<"system command to create folder to place output FAILED!"<<endl;
	}

	//declare filestreams for output
	ofstream file_lambda_u;
	ofstream file_lambda_m;
	ofstream file_latentInd;
	ofstream file_latentSNPs;

	//clear files for hyper-parameters
	file_lambda_m.open ("BayesFactors_out/"+output+"_lambda_m.txt");
	file_lambda_u.open ("BayesFactors_out/"+output+"_lambda_u.txt");
	file_lambda_u.close();
	file_lambda_m.close();
	//appending hyper-parameters
	file_lambda_m.open ("BayesFactors_out/"+output+"_lambda_m.txt", std::ios_base::app);
	file_lambda_u.open ("BayesFactors_out/"+output+"_lambda_u.txt", std::ios_base::app);

	//begin Gibbs updates
	for (int i = 0; i < iter+1; i++)
	{

		//write out for each iteration the factors
		file_latentInd.open ("BayesFactors_out/"+output+".iter"+to_string(i)+".factors");
		file_latentInd << w1_P1_sample << ' ';

		file_latentInd << endl;
		file_latentInd.close();

		//write out for each iteration the scores
		file_latentSNPs.open ("BayesFactors_out/"+output+".iter"+to_string(i)+".scores");
		file_latentSNPs << w1_M1_sample << ' ';

		file_latentSNPs<< endl;
		file_latentSNPs.close();

		//write-out hyperparameters (covariance matrices)
		//file_lambda_m << i <<" ";
		for (int j = 0; j < lambda_m.rows(); j++){
		file_lambda_m << lambda_m.row(j) << " ";
		}
		file_lambda_m<<endl;
		//file_lambda_u << i<<" ";
		for (int j = 0; j <lambda_u.rows(); j++){
		file_lambda_u  << lambda_u.row(j) << " ";
		}
		file_lambda_u<<endl;

		//GIBBS UPDATES

		//update SNP hyperparameters
		sample_hyper(w1_M1_sample,WI_m,b0_m,mu0_m,df_m,mu_m,lambda_m);
		//update individual hyperparameters
		sample_hyper(w1_P1_sample,WI_u,b0_u,mu0_u,df_u,mu_u,lambda_u);
		//update individual parameters
		sample_ind (w1_M1_sample,w1_P1_sample,X,N,num_feat,lambda_u,mu_u,alpha);
		//update SNP parameters
		sample_SNP (w1_P1_sample,w1_M1_sample,X,M,num_feat,lambda_m,mu_m,alpha);

	}
	file_lambda_u.close();
	file_lambda_m.close();

	cout<<"Finished!"<<endl;
	timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout << ctime(&timenow) << endl;

	return 0;
}
