#include <Eigen/Core>
using namespace Eigen;

double runif(double lower, double higher);
double rnorm(double mean, double sd);
double rbeta(double alpha, double beta);
double rinvchisq(double df, double scale);
int rbernoulli(double p);

double sample_hyper(const MatrixXd& w1_M1_sample, const MatrixXd& WI_m, int b0_m, const VectorXd& mu0_m, int df_m,VectorXd& mu_m,MatrixXd& lambda_m);
double sample_ind (const MatrixXd& w1_M1_sample, MatrixXd& w1_P1_sample, const MatrixXd& X,int num_p,int num_feat,const MatrixXd& lambda_u,const VectorXd& mu_u,double alpha);
double sample_SNP (const MatrixXd& w1_P1_sample, MatrixXd& w1_M1_sample, const MatrixXd& X,int num_p,int num_feat,const MatrixXd& lambda_m,const VectorXd& mu_m,double alpha);
