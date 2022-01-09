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
matrix<Type> parallel_cov(Type sigma2, Type cor,int size) {
  matrix<Type> ans(size,size);
  for(int i=0; i<size; i++) {
    for(int j=0; j<size; j++) {
      // ans(i,j) = sigma2*pow(cor,abs(i-j));
      ans(i,j) = sigma2*pow(cor,i==j ? 0 : 1);
    }
  }
  return ans;
}

template<class Type>
Type objective_function<Type>::operator() () {
  DATA_MATRIX(y); // [i,var]
  DATA_IVECTOR(ys);
  DATA_IVECTOR(yt);

  PARAMETER_ARRAY(w); // [s,t,var]
  PARAMETER_MATRIX(wt); // [s,var]
  PARAMETER_MATRIX(working_response_pars); // [par,var]
  PARAMETER_MATRIX(working_w_pars) // [par,var] (par = sd, ar1, space_sd, space_cor)
  PARAMETER(logit_rho);

  int nob = y.rows(); // # of observations
  int ns = w.rows(); // # of space points
  int nt = w.col(0).cols(); // # of time points
  int nv = w.cols(); // # of vars
  array<Type> w_std(ns,nt,nv);
  array<Type> w_std_transpose(nv,nt,ns);

  matrix<Type> response_pars = working_response_pars;
  response_pars.row(1) = exp(vector<Type>(working_response_pars.row(1))); // log_sd --> sd

  matrix<Type> w_pars = working_w_pars;
  w_pars.row(0) = exp(vector<Type>(working_w_pars.row(0))); // log_sd --> sd
  w_pars.row(1) = 2*invlogit(vector<Type>(working_w_pars.row(1)))-1; // logit_ar1 --> ar1
  w_pars.row(2) = exp(vector<Type>(working_w_pars.row(2))); // log_sd --> sd (space)
  w_pars.row(3) = 2*invlogit(vector<Type>(working_w_pars.row(3)))-1; // logit_cor --> cor (space)

  vector<MVNORM_t<Type> > mvns(nv);
  for(int v=0; v<nv; v++) {
    MVNORM_t<Type> mvn(parallel_cov(w_pars(2,v),w_pars(3,v),ns));
    mvns(v) = mvn;
  }

  Type rho = 2*invlogit(logit_rho)-1;
  matrix<Type> R(nv,nv);
  R << 1, rho, rho, 1;
  Type logdetR = atomic::logdet(R);
  matrix<Type> Q = atomic::matinv(R);
  matrix<Type> I(nv,nv); I.setIdentity();
  MVNORM_t<Type> copula(R);

  Type nll = 0.0;

  for(int v=0; v<nv; v++) {
    for(int t=0; t<nt; t++) {
      Type mu, sd;
      vector<Type> muu(ns);
      if( t == 0 ) {
        mu = 0.0;
        sd = w_pars(0,v)/sqrt(1-pow(w_pars(1,v),2));
      } else {
        mu = w_pars(1,v)*wt(t-1,v);
        sd = w_pars(0,v);
      }
      nll -= dnorm(wt(t,v),mu,sd,true);

      if( t == 0 ) {
        muu.setZero();
        muu = muu + wt(t,v);
      } else {
        muu = w_pars(1,v)*vector<Type>(w.col(v).col(t-1)-wt(t-1,v)) + wt(t,v);
      }
      nll += mvns(v)(vector<Type>(w.col(v).col(t) - muu));
      w_std.col(v).col(t) = spd_sqrt(mvns(v).Q)*vector<Type>(w.col(v).col(t)-muu);
    }
    for(int ob=0; ob<nob; ob++) {
      nll -= dnorm(y(ob,v),response_pars(0,v)+w(ys(ob),yt(ob),v),response_pars(1,v),true);
    }
  }
  w_std_transpose = w_std.transpose();

  for(int s=0; s<ns; s++) {
    for(int t=0; t<nt; t++) {
      vector<Type> wrow = w_std_transpose.col(s).col(t);
      nll += Type(0.5)*logdetR + Type(0.5)*(wrow*vector<Type>(matrix<Type>(Q-I)*wrow)).sum();
    }
  }

  SIMULATE{
    for(int s=0; s<ns; s++) {
      for(int t=0; t<nt; t++) {
        w_std_transpose.col(s).col(t) = copula.simulate();
      }
    }
    w_std = w_std_transpose.transpose();
    for(int v=0; v<nv; v++) {
      for(int t=0; t<nt; t++) {
        Type mu, sd;
        vector<Type> muu(ns);
        if( t == 0 ) {
          mu = 0.0;
          sd = w_pars(0,v)/sqrt(1-pow(w_pars(1,v),2));
        } else {
          mu = w_pars(1,v)*wt(t-1,v);
          sd = w_pars(0,v);
        }
        wt(t,v) = rnorm(mu,sd);

        if( t == 0 ) {
          muu.setZero();
          muu = muu + wt(t,v);
        } else {
          muu = w_pars(1,v)*vector<Type>(w.col(v).col(t-1)-wt(t-1,v))+wt(t,v);
        }
        w.col(v).col(t) = spd_sqrt(mvns(v).Sigma)*vector<Type>(w_std.col(v).col(t)) +muu;
      }
      for(int ob=0; ob<nob; ob++) {
        y(ob,v) = rnorm(response_pars(0,v)+w(ys(ob),yt(ob),v),response_pars(1,v));
      }
    }
    REPORT(y);
    REPORT(w);
    REPORT(wt);
    // REPORT(w_std);
  }

  REPORT(y);
  REPORT(w);
  REPORT(wt);
  REPORT(w_std);
  REPORT(w_std_transpose);

  ADREPORT(response_pars);
  ADREPORT(w_pars);
  ADREPORT(R);

  REPORT(response_pars);
  REPORT(R);

  return nll;
}
