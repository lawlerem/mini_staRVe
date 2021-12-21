#include <TMB.hpp>
using namespace density;

template<class Type>
struct myTypeDefs{
  typedef Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SAES;
};

TMB_ATOMIC_VECTOR_FUNCTION(
  // ATOMIC_NAME
  spd_sqrt
  ,
  // OUTPUT_DIM
  tx.size()
  ,
  // ATOMIC_DOUBLE
  typedef myTypeDefs<double>::SAES SAES_t;
  int n=sqrt((double)tx.size());
  matrix<double> X=atomic::vec2mat(tx,n,n);
  SAES_t saes(X);
  matrix<double> sqrtX = saes.operatorSqrt();
  for(int i=0;i<n*n;i++)ty[i]=sqrtX(i);
  ,
  // ATOMIC_REVERSE ( vec(f'(X)) =  (f(X)^T %kronecker sum% f(X))^-1 * vec(X') )
  int n = sqrt((double)ty.size());
  matrix<Type> Y = atomic::vec2mat(ty, n, n); // f(X)
  matrix<Type> Yt = Y.transpose(); // f(x)^T
  matrix<Type> I(Y.rows(),Y.cols());
  I.setIdentity();
  matrix<Type> kronSum = kronecker(Yt,I)+kronecker(I,Y);
  matrix<Type> kronSumInv = atomic::matinv(kronSum);
  px = kronSumInv*vector<Type>(py);
)

template<class Type>
matrix<Type> spd_sqrt(matrix<Type> x){
  int n=x.rows();
  return atomic::vec2mat(spd_sqrt(atomic::mat2vec(x)),n,n);
}

template<class Type>
Type objective_function<Type>::operator() () {
  DATA_MATRIX(y);

  PARAMETER_MATRIX(working_response_pars);
  PARAMETER(logit_rho);

  int no = y.rows();
  int nv = y.cols();

  matrix<Type> y_std(no,nv);

  matrix<Type> response_pars = working_response_pars;
  response_pars.row(1) = exp(vector<Type>(working_response_pars.row(1))); // log_sd --> sd

  Type rho = 2*invlogit(logit_rho)-1;
  matrix<Type> R(nv,nv);
  R << 1, rho, rho, 1;
  Type logdetR = atomic::logdet(R);
  matrix<Type> Q = atomic::matinv(R);
  matrix<Type> I(nv,nv); I.setIdentity();
  MVNORM_t<Type> copula(R);

  Type nll = 0.0;

  for(int v=0; v<nv; v++) {
    for(int o=0; o<no; o++) {
      nll -= dnorm(y(o,v),response_pars(0,v),response_pars(1,v),true);
      y_std(o,v) = (y(o,v) - response_pars(0,v)) / response_pars(1,v);
    }
  }

  for(int o=0; o<no; o++) {
    vector<Type> yrow = vector<Type>(y_std.row(o));
    nll += Type(0.5)*logdetR + Type(0.5)*(yrow*vector<Type>(matrix<Type>(Q-I)*yrow)).sum();
  }

  SIMULATE{
    for(int o=0; o<no; o++) {
      y_std.row(o) = copula.simulate();
    }
    for(int v=0; v<nv; v++) {
      for(int o=0; o<no; o++) {
        y(o,v) = response_pars(1,v)*y_std(o,v)+response_pars(0,v);
      }
    }
    REPORT(y);
  }

  ADREPORT(response_pars);
  ADREPORT(R);

  REPORT(y_std);

  REPORT(response_pars);
  REPORT(R);

  return nll;
}
