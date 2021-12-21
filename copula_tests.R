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
    c(-10,log(20))
  )
)

simobj<- MakeADFun(
  data = simdat,
  para = simpar,
  DLL = "copula_tests",
  silent = TRUE
)

sim<- simobj$simulate()


fitobj<- MakeADFun(
  data = list(
    y = sim$y
  ),
  para = list(
    working_response_pars = matrix(0,nrow=2,ncol=nv)
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
fitsdr<- list(
  Estimate = as.list(fitsdr,"Est",TRUE),
  StdError = as.list(fitsdr,"Std",TRUE)
)
fitcheck<- checkConsistency(fitobj)
