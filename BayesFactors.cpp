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
	("burnin", po::value<int>()->default_value(10), "No. burnin iterations")
	("b0_m", po::value<double>()->default_value(2), "b0_m")
	("b0_u", po::value<double>()->default_value(2), "b0_u")
	("input", po::value<std::string>()->required(),"Input filename")
	("out", po::value<std::string>()->default_value("BayesFactors_out"),"Output filename")
	("scale", "perform scaling")
	("missing", "missing data included?")
	;

	srand(time(0));

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc),vm);
	po::notify(vm);

	int M=vm["M"].as<int>();
	int N=vm["N"].as<int>();
	int num_feat=vm["num_feat"].as<int>();
	int iter=vm["iter"].as<int>();
	int burnin=vm["burnin"].as<int>();
	double b0_m=vm["b0_m"].as<double>();
	double b0_u=vm["b0_u"].as<double>();
	string input=vm["input"].as<string>();
	string output=vm["out"].as<string>();

	MatrixXd X(N,M);

	int i,j,k,l,m=0;
	auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout<<"Started analysis!"<<endl;
	timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout << ctime(&timenow) << endl;

//Read Genotype Matrix
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
//Read missing data indicator matrix
	MatrixXd Indicator(N,M);
	if (vm.count("missing")) {
	ifstream f2(input+".Indicator");
	if (f2){
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				f2 >> Indicator(i,j);
				//cout<<X(i,j)<<endl;
			}
		}
		f2.close();
		cout<<"finished reading matrix Indicator!"<<endl;
		timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
		cout << ctime(&timenow) << endl;
	}else{
		cout<<"the "+input+".Indicator"+" file does not exist/cannot be opened!"<<endl;
		return 0;
	}
	}

	//Normalize matrix X
	if (vm.count("scale")) {
		if (vm.count("missing")) {
			X=X.cwiseProduct(Indicator);
			RowVectorXd mean = X.colwise().sum().array()/Indicator.colwise().sum().array();
			RowVectorXd sqsum(M);
			RowVectorXd sd(M);
			sqsum.setZero();
			sd.setZero();
			//to calculate sd I add the squared deviations only for non-missing entries as specified in the Indicator matrix
			for (int j = 0; j < M; j++)
			{
				for (int i = 0; i < N; i++)
				{
					sqsum[j] +=pow((X(i,j)-mean[j]),2)*Indicator(i,j);
				}
				sd[j]=sqrt((sqsum[j]/(Indicator.col(j).sum() - 1)));
			}
			//Divide by sd but make sure that sd!=0
//Should be possible to integrate following code so that only one loop through M is done
			for (int j = 0; j < M; j++)
			{
				if (sd[j]==0){
					for (int i = 0; i < N; i++)
					{
						X(i,j) -= mean[j];
					}
				}else{
					for (int i = 0; i < N; i++)
					{
						X(i,j) -= mean[j];
						X(i,j) /= sd[j];
					}
				}
			}
		}else{

			RowVectorXd mean = X.colwise().mean();
			RowVectorXd sd = ((X.rowwise() - mean).array().square().colwise().sum() / (X.rows() - 1)).sqrt();
			X = (X.rowwise() - mean).array().rowwise() / sd.array();
		}
	}

	double Xglobalmean=X.mean();
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
	lambda_m.setIdentity();
	lambda_u.setIdentity();
	WI_u.setIdentity();

	mu_m.setZero();
	mu0_m.setZero();
	mu_u.setZero();
	mu0_u.setZero();

	int df_m=num_feat;
	int df_u=num_feat;

	//Initialization of residual variance
	MatrixXd epsilon_t(N,M);
	double Xm=X.mean();
	epsilon_t=X-w1_P1_sample*w1_M1_sample.transpose();
	//double sigma2_e=2;
	double sigma2_e=sample_residual_variance_gamma(epsilon_t);
	VectorXd sigma2_e_rowvec(N);
	sample_residual_row_variance_gamma(epsilon_t, sigma2_e_rowvec);

	//Intialize running means of latent variables
	MatrixXd Ew1_M1_sample(M,num_feat);
	MatrixXd Ew1_P1_sample(N,num_feat);

	Ew1_M1_sample=w1_M1_sample;
	Ew1_P1_sample=w1_P1_sample;


	//create folder to put output
	int systemRet = system("mkdir -p BayesFactors_out");
	if(systemRet == -1){
		cout<<"system command to create folder to place output FAILED!"<<endl;
	}

	//declare filestreams for output
	ofstream file_lambda_u;
	ofstream file_lambda_m;
	ofstream file_mu_u;
	ofstream file_mu_m;
	ofstream file_latentInd;
	ofstream file_latentSNPs;
	ofstream file_sigma2_e;
	ofstream file_sigma2_e_rowvec;

	ofstream file_ElatentInd;
	ofstream file_ElatentSNPs;

	//clear files for hyper-parameters
	file_lambda_m.open ("BayesFactors_out/"+output+"_lambda_m.txt");
	file_lambda_u.open ("BayesFactors_out/"+output+"_lambda_u.txt");
	file_mu_m.open ("BayesFactors_out/"+output+"_mu_m.txt");
	file_mu_u.open ("BayesFactors_out/"+output+"_mu_u.txt");
	file_sigma2_e.open ("BayesFactors_out/"+output+"_sigma2_e.txt");
	file_sigma2_e_rowvec.open ("BayesFactors_out/"+output+"_sigma2_e_rowvec.txt");

	file_lambda_u.close();
	file_lambda_m.close();
	file_mu_u.close();
	file_mu_m.close();
	file_sigma2_e.close();
	file_sigma2_e_rowvec.close();

	//appending sigma2_e
	file_sigma2_e.open ("BayesFactors_out/"+output+"_sigma2_e.txt", std::ios_base::app);
	file_sigma2_e_rowvec.open ("BayesFactors_out/"+output+"_sigma2_e_rowvec.txt", std::ios_base::app);
	//appending hyper-parameters
	file_lambda_m.open ("BayesFactors_out/"+output+"_lambda_m.txt", std::ios_base::app);
	file_lambda_u.open ("BayesFactors_out/"+output+"_lambda_u.txt", std::ios_base::app);
	file_mu_m.open ("BayesFactors_out/"+output+"_mu_m.txt", std::ios_base::app);
	file_mu_u.open ("BayesFactors_out/"+output+"_mu_u.txt", std::ios_base::app);


	//Burnin iterations
	cout<<"Starting burning "<<burnin<<" iterations"<<endl;
	for (int i = 0; i < burnin; i++)
	{
		//update residual variance
		epsilon_t=X-w1_P1_sample*w1_M1_sample.transpose();
		sigma2_e=sample_residual_variance_gamma(epsilon_t);
		sample_residual_row_variance_gamma(epsilon_t, sigma2_e_rowvec);

		//update SNP hyperparameters
		//sample_hyper(w1_M1_sample,WI_m,b0_m,mu0_m,df_m,mu_m,lambda_m);
		//update individual hyperparameters
		sample_hyper(w1_P1_sample,WI_u,b0_u,mu0_u,df_u,mu_u,lambda_u);
		//update individual parameters
		//sample_ind (Xglobalmean,w1_M1_sample,w1_P1_sample,X,N,num_feat,lambda_u,mu_u,sigma2_e_rowvec);
		sample_ind_missing (Xglobalmean,w1_M1_sample,w1_P1_sample,Indicator,X,N,M,num_feat,lambda_u,mu_u,sigma2_e_rowvec);
		sample_SNP_missing (Xglobalmean,w1_P1_sample,w1_M1_sample,Indicator,X,N,M,num_feat,lambda_m,mu_m,sigma2_e);
		//update SNP parameters
		//sample_SNP (Xglobalmean,w1_P1_sample,w1_M1_sample,X,M,num_feat,lambda_m,mu_m,sigma2_e);
		//update rows and columns together
		//sample_ind_SNP (w1_M1_sample,w1_P1_sample, X,N,M,num_feat,lambda_u,mu_u,lambda_m,mu_m,sigma2_e);


	}

	//Sampling iterations
	cout<<"Finished burnin, starting sampling "<<iter<<" iterations" <<endl;

	for (int i = 0; i < iter; i++)
	{

		//write out for each iteration the factors
		file_latentInd.open ("BayesFactors_out/"+output+".iter"+to_string(i+1)+".factors");
		file_latentInd << w1_P1_sample << ' ';

		file_latentInd << endl;
		file_latentInd.close();

		//write out for each iteration the scores
		file_latentSNPs.open ("BayesFactors_out/"+output+".iter"+to_string(i+1)+".scores");
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

		//write-out hyperparameters (means)
		for (int j = 0; j <num_feat; j++){
			file_mu_m<<mu_m[j]<<" ";
			file_mu_u<<mu_u[j]<<" ";
		}
		file_mu_m<<endl;
		file_mu_u<<endl;

		//write-out residual variance
		file_sigma2_e<<sigma2_e<<endl;

		for (int j = 0; j <sigma2_e_rowvec.rows(); j++){
			file_sigma2_e_rowvec  << sigma2_e_rowvec.row(j) << " ";
		}
		file_sigma2_e_rowvec<<endl;


		//GIBBS UPDATES
		//update residual variance
		epsilon_t=X-w1_P1_sample*w1_M1_sample.transpose();
		sigma2_e=sample_residual_variance_gamma(epsilon_t);
		sample_residual_row_variance_gamma(epsilon_t, sigma2_e_rowvec);

		//update SNP hyperparameters
		//sample_hyper(w1_M1_sample,WI_m,b0_m,mu0_m,df_m,mu_m,lambda_m);
		//update individual hyperparameters
		sample_hyper(w1_P1_sample,WI_u,b0_u,mu0_u,df_u,mu_u,lambda_u);

		//update individual parameters
		//sample_ind (Xglobalmean,w1_M1_sample,w1_P1_sample,X,N,num_feat,lambda_u,mu_u,sigma2_e_rowvec);
		sample_ind_missing (Xglobalmean,w1_M1_sample,w1_P1_sample,Indicator,X,N,M,num_feat,lambda_u,mu_u,sigma2_e_rowvec);
		sample_SNP_missing (Xglobalmean,w1_P1_sample,w1_M1_sample,Indicator,X,N,M,num_feat,lambda_m,mu_m,sigma2_e);
		//update SNP parameters
		//sample_SNP (Xglobalmean,w1_P1_sample,w1_M1_sample,X,M,num_feat,lambda_m,mu_m,sigma2_e);

		//running average has +1 value (1 from burnin)
		Ew1_M1_sample=Ew1_M1_sample+(w1_M1_sample-Ew1_M1_sample)/(i+2);
		Ew1_P1_sample=Ew1_P1_sample+(w1_P1_sample-Ew1_P1_sample)/(i+2);

	}

	//write-out means for latent variables
	file_ElatentInd.open ("BayesFactors_out/"+output+"_Efactors.txt");
	file_ElatentSNPs.open ("BayesFactors_out/"+output+"_Escores.txt");
	file_ElatentInd<<Ew1_P1_sample<<endl;
	file_ElatentSNPs<<Ew1_M1_sample<<endl;
	file_ElatentInd.close();
	file_ElatentSNPs.close();

	file_lambda_u.close();
	file_lambda_m.close();

	file_mu_u.close();
	file_mu_m.close();

	file_sigma2_e.close();
	file_sigma2_e_rowvec.close();


	cout<<"Finished!"<<endl;
	timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout << ctime(&timenow) << endl;

	return 0;
}
