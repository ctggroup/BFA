#include <Eigen/Core>
#include <Eigen/Dense>
#include <armadillo>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>
#include "Sampling_functions.h"

using namespace std;
using namespace Eigen;
using namespace arma;

//boost::random::mt19937 gen(0);
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

double rgamma(double alpha, double beta)
{

	boost::math::gamma_distribution<> dist(alpha, beta);
	double q = quantile(dist, runif(0,1));

	return(q);
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

//switch between armadillo and eigen
Eigen::MatrixXd cast_eigen(arma::mat arma_A)
{
	Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
			arma_A.n_rows,
			arma_A.n_cols);

	return eigen_B;
}

arma::mat cast_arma(Eigen::MatrixXd& eigen_A)
{
	arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),false, false);
	return arma_B;
}


//sampling functions
double sample_hyper(const MatrixXd& w1_M1_sample, const MatrixXd& WI_m, double b0_m, const VectorXd& mu0_m, int df_m,VectorXd& mu_m,MatrixXd& lambda_m)
{
	//Sample from individual or SNP hyperparams (equation 14)
	int N = w1_M1_sample.rows();
	int num_feat=w1_M1_sample.cols();
	VectorXd u_bar(num_feat);
	MatrixXd S_bar(num_feat,num_feat);

	u_bar = w1_M1_sample.colwise().mean().transpose();
	//covariance to correlation
	S_bar = (w1_M1_sample.transpose()*w1_M1_sample)/N;

	//Gaussian-Wishard sampling
	//1.sample lambda from wishard distribution (lambda_m or lambda_u)
	//2.sample mu (mu_m or mu_u) from multivariate normal distribution with mean mu_0 (mu_temp) and variance (lambda*Lambda)^-1

	//Wishard-covariance matrix
	MatrixXd WI_post(num_feat,num_feat);
	WI_post = WI_m.inverse() + N*S_bar+((N*b0_m)/(b0_m+N))*((mu0_m - u_bar)*(mu0_m - u_bar).transpose());

	//WI_post = (WI_post + WI_post.transpose())/2;

	WI_post=WI_post.inverse();

	//Wishard-degrees of freedom
	int df_mpost = df_m+N;

	//Wishard draw, using the armadillo function. So I cast to an armadillo matrix, draw wishart then cast back to eigen matrix.
	//should just code the wishart function.
	mat arma_lambda =cast_arma(WI_post);
	lambda_m=cast_eigen(wishrnd(arma_lambda, df_mpost));

	//multivariate normal mean
	VectorXd mu_temp(num_feat);
	mu_temp = (b0_m*mu0_m + N*u_bar)/(b0_m+N);

	//Multivariate normal with mean mu_temp and cholesky decomposed covariance lam
	MatrixXd lam = (((b0_m+N)*lambda_m).inverse());

	lam=lam.llt().matrixU();
	lam.transposeInPlace();

	MatrixXd normV(num_feat,1);
	for (int i=0;i<num_feat;i++){
		normV(i,0)=rnorm(0,1);
	}

	mu_m = lam*normV+mu_temp;
/*
	if (N<3000){
		cout<<"u_bar"<<endl;
		cout<<u_bar<<endl;

		cout<<"S_bar"<<endl;
		cout<<S_bar<<endl;

		cout<<"WIPOST"<<endl;
		cout<<WI_post<<endl;

		cout<<"lam"<<endl;
		cout<<lam<<endl;

		cout<<"lambda_m"<<endl;
		cout<<lambda_m<<endl;

		cout<<"mu_m"<<endl;
		cout<<mu_m<<endl;


		cout<<"normV"<<endl;
		cout<<normV<<endl;
		cout<<"mu_temp"<<endl;
		cout<<mu_temp<<endl;

	}else{

		cout<<"S_bar_ind"<<endl;
		cout<<S_bar<<endl;
		cout<<"lam-ind"<<endl;
		cout<<lam<<endl;
		cout<<"lambda_u"<<endl;
		cout<<lambda_m<<endl;
		cout<<"mu_temp_ind"<<endl;
		cout<<mu_temp<<endl;
	}
*/

	return 0;
}

double sample_ind (const MatrixXd& w1_M1_sample, MatrixXd& w1_P1_sample, const MatrixXd& X,int num_p,int num_feat,const MatrixXd& lambda_u,const VectorXd& mu_u,const VectorXd& alpha)
{
	//random shuffling of individuals.
	std::vector<int> I;
	for (int i=0; i<num_p; ++i) {
		I.push_back(i);
	}

	std::random_shuffle(I.begin(), I.end());

	// Gibbs updates over individual latent vectors given hyperparams.
	// Infer posterior distribution over individual latent vectors (equation 11).

	for (int i=0;i<num_p;i++){
		//observed data row
		VectorXd rr = X.row(I[i]);
		//equation 12
		MatrixXd covar = ((alpha[i]*(w1_M1_sample.transpose()*w1_M1_sample)+lambda_u));
		covar=covar.inverse();
		//equation 13
		VectorXd mean_u = covar * (alpha[i]*(w1_M1_sample.transpose()*rr)+lambda_u*mu_u);
		//multivariate normal with mean mean_u and cholesky decomposed variance lam
		MatrixXd lam = covar.llt().matrixU();
		lam.transposeInPlace();

		MatrixXd normV(num_feat,1);
		for (int j=0;j<num_feat;j++){
			normV(j,0)=rnorm(0,1);
		}

		w1_P1_sample.row(I[i]) = (lam*normV+mean_u).transpose();

	}

	return 0;
}

double sample_SNP (const MatrixXd& w1_P1_sample, MatrixXd& w1_M1_sample, const MatrixXd& X,int num_p,int num_feat,const MatrixXd& lambda_m,const VectorXd& mu_m,double alpha)
{
	//random shuffling of markers.
	std::vector<int> I;
	for (int i=0; i<num_p; ++i) {
		I.push_back(i);
	}

	std::random_shuffle(I.begin(), I.end());

	// Gibbs updates over SNP latent vectors given hyperparams.
	// Infer posterior distribution over SNP latent vectors (equation 11).
	/*
	cout<<"lambda_m"<<endl;
	cout<<lambda_m<<endl;
	cout<<"mu_m"<<endl;
	cout<<lambda_m<<endl;
	*/
	for (int i=0;i<num_p;i++){
		//observed data column
		VectorXd rr = X.col(I[i]);
		//equation 12
		MatrixXd covar = ((alpha*(w1_P1_sample.transpose()*w1_P1_sample)+lambda_m));
		covar=covar.inverse();
		//equation 13
		VectorXd mean_m = covar * (alpha*(w1_P1_sample.transpose()*rr)+lambda_m*mu_m);
		//multivariate normal with mean mean_m and cholesky decomposed variance lam
		MatrixXd lam = covar.llt().matrixU();
		lam.transposeInPlace();

		MatrixXd normV(num_feat,1);
		for (int j=0;j<num_feat;j++){
			normV(j,0)=rnorm(0,1);
		}
		w1_M1_sample.row(I[i]) = (lam*normV+mean_m).transpose();
	}

	return 0;
}

double sample_ind_SNP (MatrixXd& w1_M1_sample, MatrixXd& w1_P1_sample, const MatrixXd& X,int num_p,int num_m,int num_feat,const MatrixXd& lambda_u,const VectorXd& mu_u,const MatrixXd& lambda_m,const VectorXd& mu_m,double alpha)
{
	//random shuffling of individuals.
	std::vector<int> I;
	for (int i=0; i<num_p; ++i) {
		I.push_back(i);
	}
	std::vector<int> J;
	for (int i=0; i<num_m; ++i) {
		J.push_back(i);
	}

	std::random_shuffle(I.begin(), I.end());
	std::random_shuffle(J.begin(), J.end());

	// Gibbs updates over individual latent vectors given hyperparams.
	// Infer posterior distribution over individual latent vectors (equation 11).
	int upd_row=0;
	int upd_col=0;
	int i=0;
	while (i<num_p+num_m){

		if(upd_row<num_p){
			//observed data row
			VectorXd rr = X.row(I[upd_row]);
			//equation 12
			MatrixXd covar = ((alpha*(w1_M1_sample.transpose()*w1_M1_sample)+lambda_u));
			covar=covar.inverse();
			//equation 13
			VectorXd mean_u = covar * (alpha*w1_M1_sample.transpose()*rr+lambda_u*mu_u);
			//multivariate normal with mean mean_u and cholesky decomposed variance lam
			MatrixXd lam = covar.llt().matrixU();
			lam.transposeInPlace();

			MatrixXd normV(num_feat,1);
			for (int j=0;j<num_feat;j++){
				normV(j,0)=rnorm(0,1);
			}

			w1_P1_sample.row(I[upd_row]) = (lam*normV+mean_u).transpose();
			upd_row=upd_row+1;
			i=i+1;
		}

		if(upd_col<num_m){
			//observed data column
			VectorXd rr = X.col(J[upd_col]);
			//equation 12
			MatrixXd covar = ((alpha*(w1_P1_sample.transpose()*w1_P1_sample)+lambda_m));
			covar=covar.inverse();
			//equation 13
			VectorXd mean_m = covar * (alpha*w1_P1_sample.transpose()*rr+lambda_m*mu_m);
			//multivariate normal with mean mean_m and cholesky decomposed variance lam
			MatrixXd lam = covar.llt().matrixU();
			lam.transposeInPlace();

			MatrixXd normV(num_feat,1);
			for (int j=0;j<num_feat;j++){
				normV(j,0)=rnorm(0,1);
			}
			w1_M1_sample.row(J[upd_col]) = (lam*normV+mean_m).transpose();
			upd_col=upd_col+1;
			i=i+1;
		}

	}

	return 0;
}


double sample_residual_variance_gamma(const MatrixXd& epsilon)
{
	int N=epsilon.rows()*epsilon.cols();
	double res=epsilon.squaredNorm();
	double sigma2=1/rgamma((N-1)/2,1/(0.5*res));
	//cout<<"residual info"<<endl;
	//cout<<N<<"\t"<<res<<"\t"<<sigma2<<endl;
	return sigma2;
}

double sample_residual_row_variance_gamma(const MatrixXd& epsilon, VectorXd& alpha)
{
	int N=epsilon.rows();
	int M=epsilon.cols();
	double res=0;
	for (int i=0;i<N;i++){
	res=epsilon.row(i).squaredNorm();
	alpha[i]=1/rgamma((M-1)/2,1/(0.5*res));
	}

	return 0;
}
