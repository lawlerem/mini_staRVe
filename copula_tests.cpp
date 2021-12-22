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
  DATA_IVECTOR(yt);

  PARAMETER_MATRIX(w); // [t,var]
  PARAMETER_MATRIX(working_response_pars); // [par,var]
  PARAMETER_MATRIX(working_w_pars) // [par,var]
  PARAMETER(logit_rho);

  int nob = y.rows(); // # of observations
  int nv = y.cols(); // # of vars
  int nt = w.rows(); // # of time points

  matrix<Type> response_pars = working_response_pars;
  response_pars.row(1) = exp(vector<Type>(working_response_pars.row(1))); // log_sd --> sd

  matrix<Type> w_std(nt,nv);
  matrix<Type> w_pars = working_w_pars;
  w_pars.row(0) = exp(vector<Type>(working_w_pars.row(0))); // log_sd --> sd
  w_pars.row(1) = 2*invlogit(vector<Type>(working_w_pars.row(1)))-1; // logit_ar1 --> ar1

  Type rho = 2*invlogit(logit_rho)-1;
  matrix<Type> R(nv,nv);
  R << 1, rho, rho, 1;
  Type logdetR = atomic::logdet(R);
  matrix<Type> Q = atomic::matinv(R);
  matrix<Type> I(nv,nv); I.setIdentity();
  MVNORM_t<Type> copula(R);




  Type nll = 0.0;

  for(int v=0; v<nv; v++) {
    for(int ob=0; ob<nob; ob++) {
      nll -= dnorm(y(ob,v),response_pars(0,v)+w(yt(ob),v),response_pars(1,v),true);
    }
    for(int t=0; t<nt; t++) {
      Type mu, sd;
      if( t == 0 ) {
        mu = 0.0;
        sd = w_pars(0,v)/sqrt(1-pow(w_pars(1,v),2));
      } else {
        mu = w_pars(1,v)*w(t-1,v);
        sd = w_pars(0,v);
      }
      nll -= dnorm(w(t,v),mu,sd,true);
      w_std(t,v) = (w(t,v) - mu) / sd;
    }
  }

  for(int t=0; t<nt; t++) {
    vector<Type> wrow = vector<Type>(w_std.row(t));
    nll += Type(0.5)*logdetR + Type(0.5)*(wrow*vector<Type>(matrix<Type>(Q-I)*wrow)).sum();
  }

  SIMULATE{
    for(int t=0; t<nt; t++) {
      w_std.row(t) = copula.simulate();
    }
    for(int v=0; v<nv; v++) {
      for(int t=0; t<nt; t++) {
        Type mu, sd;
        if( t == 0 ) {
          mu = 0.0;
          sd = w_pars(0,v)/sqrt(1-pow(w_pars(1,v),2));
        } else {
          mu = w_pars(1,v)*w(t-1,v);
          sd = w_pars(0,v);
        }
        w(t,v) = sd*w_std(t,v) + mu;
      }
      for(int ob=0; ob<nob; ob++) {
        y(ob,v) = rnorm(response_pars(0,v)+w(yt(ob),v),response_pars(1,v));
      }
    }
    REPORT(y);
    REPORT(w);
  }

  ADREPORT(response_pars);
  ADREPORT(w_pars);
  ADREPORT(R);

  REPORT(response_pars);
  REPORT(R);

  return nll;
}
