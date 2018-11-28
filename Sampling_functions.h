#include <Eigen/Core>
using namespace Eigen;

double runif(double lower, double higher);
double rnorm(double mean, double sd);
double rbeta(double alpha, double beta);
double rinvchisq(double df, double scale);
int rbernoulli(double p);

double sample_hyper(const MatrixXd& w1_M1_sample, const MatrixXd& WI_m, double b0_m, const VectorXd& mu0_m, int df_m,VectorXd& mu_m,MatrixXd& lambda_m);
double sample_ind (double Xglobalmean,const MatrixXd& w1_M1_sample, MatrixXd& w1_P1_sample, const MatrixXd& X,int num_p,int num_feat,const MatrixXd& lambda_u,const VectorXd& mu_u,const VectorXd& alpha);
double sample_SNP (double Xglobalmean,const MatrixXd& w1_P1_sample, MatrixXd& w1_M1_sample, const MatrixXd& X,int num_p,int num_feat,const MatrixXd& lambda_m,const VectorXd& mu_m,double alpha);

double sample_ind_missing (double Xglobalmean,const MatrixXd& w1_M1_sample, MatrixXd& w1_P1_sample,const MatrixXd& Indicator, const MatrixXd& X,int num_p,int num_m,int num_feat,const MatrixXd& lambda_u,const VectorXd& mu_u,const VectorXd& alpha);
double sample_SNP_missing (double Xglobalmean,const MatrixXd& w1_P1_sample, MatrixXd& w1_M1_sample, const MatrixXd& Indicator,const MatrixXd& X,int num_p,int num_m,int num_feat,const MatrixXd& lambda_m,const VectorXd& mu_m,double alpha);

double sample_ind_SNP (MatrixXd& w1_M1_sample, MatrixXd& w1_P1_sample, const MatrixXd& X,int num_p,int num_m,int num_feat,const MatrixXd& lambda_u,const VectorXd& mu_u,const MatrixXd& lambda_m,const VectorXd& mu_m,double alpha);

double sample_residual_variance_gamma(const MatrixXd& epsilon);
double sample_residual_row_variance_gamma(const MatrixXd& epsilon, VectorXd& alpha);
