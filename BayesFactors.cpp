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
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>
#include <boost/program_options.hpp>
#include <iterator>


using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Map;
using Eigen::Upper;
typedef Map<MatrixXd> MapMatd;

boost::random::mt19937 gen(time(0));

//distributions
double runif(double lower, double higher)
{
	boost::random::uniform_real_distribution<> dist(lower, higher);
	return dist(gen);
}

double rnorm(double mean, double sd)
{
	boost::random::normal_distribution<> dist(mean, sd);
	return dist(gen);
}


double rbeta(double alpha, double beta)
{

	boost::math::beta_distribution<> dist(alpha, beta);
	double q = quantile(dist, runif(0,1));

	return(q);
}

double rinvchisq(double df, double scale)
{

	boost::math::inverse_chi_squared_distribution<> dist(df, scale);
	double q = quantile(dist, runif(0,1));

	return(q);
}
int rbernoulli(double p)
{
	std::bernoulli_distribution dist(p);
	return dist(gen);
}

//sampling functions
double sample_mu(int N, double Esigma2,const VectorXd& Y,const MatrixXd& X,const VectorXd& beta)
{
	double mean=((Y-X*beta).sum())/N;
	double sd=sqrt(Esigma2/N);
	double mu=rnorm(mean,sd);
	return(mu);
}

//sample variance of beta
double sample_psi2_chisq(const VectorXd& beta,int NZ,double v0B,double s0B){
	double df=v0B+NZ;
	double scale=(beta.squaredNorm()*NZ+v0B*s0B)/(v0B+NZ);
	//cout<<NZ<<"\t"<<beta.squaredNorm()<<"\t"<<df<<"\t"<<scale<<"\t"<<endl;
	double psi2=rinvchisq(df, scale);
	return(psi2);
}

//sample error variance of Y
double sample_sigma_chisq(int N,const VectorXd& epsilon,double v0E,double s0E){
	double sigma2=rinvchisq(v0E+N, (epsilon.squaredNorm()+v0E*s0E)/(v0E+N));
	return(sigma2);
}

//sample mixture weight
double sample_w(int M,int NZ){
	double w=rbeta(1+NZ,1+(M-NZ));
	return(w);
}


void ReadFromFile(std::vector<double> &x, const std::string &file_name)
{
	std::ifstream read_file(file_name);
	assert(read_file.is_open());

	std::copy(std::istream_iterator<double>(read_file), std::istream_iterator<double>(),
			std::back_inserter(x));

	read_file.close();
}

int main(int argc, char *argv[])
{

	return 0;
}
