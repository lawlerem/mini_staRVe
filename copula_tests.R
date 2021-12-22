library(TMB)
compile("copula_tests.cpp","-O0 -g")
dyn.load(dynlib("copula_tests"))

nv<- 2
nl<- 5
nperl<- 10
nob<- nl*nperl

simdat<- list(
  y = matrix(0,nrow=nob,ncol=nv),
  y_level = rep(seq(nl)-1,each=nperl)
)
simpar<- list(
  w = matrix(0,nrow=nl,ncol=nv),
  working_response_pars = cbind(
    c(5,log(3)),
    c(5,log(2))
  ),
  working_w_pars = cbind(
    c(log(20)),
    c(log(20))
  ),
  logit_rho = qlogis(0.5*(c(-0.8)+1))
)

simobj<- MakeADFun(
  data = simdat,
  para = simpar,
  random = "w",
  map = list(
    working_response_pars=factor(c(1,2,1,4))
  ),
  DLL = "copula_tests",
  silent = TRUE
)

sim<- simobj$simulate()
plot(as.data.frame(sim$y))
cor(as.data.frame(sim$y))

fitobj<- MakeADFun(
  data = list(
    y = sim$y,
    y_level = simdat$y_level
  ),
  para = list(
    w = matrix(0,nrow=nl,ncol=nv),
    working_response_pars = cbind(
      c(0,0),
      c(0,0)
    ),
    working_w_pars = cbind(
      c(0),
      c(0)
    ),
    logit_rho = qlogis(0.5*(c(0)+1))
  ),
  map = list(
    working_response_pars=factor(c(1,2,1,4))
  ),
  random = "w",
  DLL = "copula_tests",
  silent = TRUE
)
fitopt<- nlminb(
  fitobj$par,
  fitobj$fn,
  fitobj$gr
)
fitsdr<- sdreport(fitobj)
(fitsdr<- list(
  Estimate = as.list(fitsdr,"Est",TRUE),
  StdError = as.list(fitsdr,"Std",TRUE)
))
(fitcheck<- checkConsistency(fitobj))
