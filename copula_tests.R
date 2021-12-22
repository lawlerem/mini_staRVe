library(TMB)
compile("copula_tests.cpp","-O0 -g")
dyn.load(dynlib("copula_tests"))

nv<- 2
nt<- 15
npert<- 3
nob<- nt*npert

simdat<- list(
  y = matrix(0,nrow=nob,ncol=nv),
  yt = rep(seq(nt)-1,each=npert)
)
simpar<- list(
  w = matrix(0,nrow=nt,ncol=nv),
  working_response_pars = cbind(
    c(5,log(3)),
    c(5,log(2))
  ),
  working_w_pars = cbind(
    c(log(20),qlogis(0.5*c((0.7)+1))),
    c(log(10),qlogis(0.5*c((-0.7)+1)))
  ),
  logit_rho = qlogis(0.5*(c(-0.8)+1))
)
simmap<- list(
  working_response_pars=factor(c(1,2,1,4))
)

simobj<- MakeADFun(
  data = simdat,
  para = simpar,
  random = "w",
  map = simmap,
  DLL = "copula_tests",
  silent = TRUE
)

sim<- simobj$simulate()
par(mfrow=c(2,1))
  plot.ts(sim$y[,1])
  plot.ts(sim$y[,2])

fitobj<- MakeADFun(
  data = list(
    y = sim$y,
    yt = simdat$yt
  ),
  para = list(
    w = matrix(0,nrow=nt,ncol=nv),
    working_response_pars = cbind(
      c(0,0),
      c(0,0)
    ),
    working_w_pars = cbind(
      c(0,0),
      c(0,0)
    ),
    logit_rho = qlogis(0.5*(c(0)+1))
  ),
  map = simmap,
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
