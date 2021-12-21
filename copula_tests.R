library(TMB)
compile("copula_tests.cpp")
dyn.load(dynlib("copula_tests"))

nv<- 2
no<- 50

simdat<- list(
  y = matrix(0,nrow=no,ncol=nv)
)
simpar<- list(
  working_response_pars = cbind(
    c(5,log(3)),
    c(-10,log(2))
  ),
  logit_rho = qlogis(0.5*(c(0.9)+1))
)

simobj<- MakeADFun(
  data = simdat,
  para = simpar,
  DLL = "copula_tests",
  silent = TRUE
)

sim<- simobj$simulate()
plot(as.data.frame(sim$y))
cor(as.data.frame(sim$y))

fitobj<- MakeADFun(
  data = list(
    y = sim$y
  ),
  para = list(
    working_response_pars = cbind(
      c(0,0),
      c(0,0)
    ),
    logit_rho = qlogis(0.5*(c(0)+1))
  ),
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
