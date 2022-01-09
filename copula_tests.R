plot_w<- function(w,wt,setpar=TRUE) {
  ns<- dim(w)[[1]]
  nt<- dim(w)[[2]]
  nv<- dim(w)[[3]]
  if( setpar ) {
    ogpar<- par(no.readonly=TRUE)
    on.exit(par(ogpar))
    par(mfrow=c(nv,1))
  } else {}
  cols<- RColorBrewer::brewer.pal(ns,"Set2")

  for( v in (seq(nv)) ) {
    plot(
      x = c(0,nt),
      y = c(floor(min(c(w[,,v],wt[,v]))),ceiling(max(c(w[,,v],wt[,v])))),
      type = "n",
      main = paste("Variable",v),
      xlab = "Time",
      ylab = "Value"
    )
    lines(
      x = seq(nt),
      y = wt[,v],
      col = "black"
    )
    for( s in (seq(ns)) ) {
      lines(
        x = seq(nt),
        y = w[s,,v],
        col = cols[s+1]
      )
    }
  }
}

library(TMB)
compile("copula_tests.cpp","-O0 -g")
dyn.load(dynlib("copula_tests"))

nv<- 2
ns<- 8
nt<- 15
npert<- 3
nob<- ns*nt*npert

simdat<- list(
  y = matrix(0,nrow=nob,ncol=nv),
  ys = rep(seq(ns)-1,nt*npert),
  yt = rep(seq(nt)-1,each=ns*npert)
)
simpar<- list(
  w = array(0,dim=c(ns,nt,nv)),
  wt = array(0,dim=c(nt,nv)),
  working_response_pars = cbind(
    c(0,log(3)),
    c(0,log(2))
  ),
  working_w_pars = cbind(
    c(log(20),qlogis(0.5*c((0.7)+1)),log(10),qlogis(0.5*c((0.2)+1))),
    c(log(10),qlogis(0.5*c((0.5)+1)),log(10),qlogis(0.5*c((0.2)+1)))
  ),
  logit_rho = qlogis(0.5*(c(0.7)+1))
)
simmap<- list(
  working_response_pars=factor(c(NA,2,NA,4)),
  working_w_pars = factor(c(1,2,3,NA,5,6,7,NA))
)

simobj<- MakeADFun(
  data = simdat,
  para = simpar,
  random = c("w","wt"),
  map = simmap,
  DLL = "copula_tests",
  silent = TRUE
)

sim<- simobj$simulate()
plot_w(sim$w,sim$wt)

# (simcheck<- checkConsistency(simobj))





fitobj<- MakeADFun(
  data = list(
    y = sim$y,
    ys = simdat$ys,
    yt = simdat$yt
  ),
  para = list(
    w = array(0,dim=c(ns,nt,nv)),
    wt = array(0,dim=c(nt,nv)),
    working_response_pars = cbind(
      c(0,0),
      c(0,0)
    ),
    working_w_pars = cbind(
      c(0,0,0,simpar$working_w_pars[4,1]),
      c(0,0,0,simpar$working_w_pars[4,2])
    ),
    logit_rho = qlogis(0.5*(c(0)+1))
  ),
  map = simmap,
  random = c("w","wt"),
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
# (fitcheck<- checkConsistency(fitobj))




par(mfcol=c(2,2))
plot_w(sim$w,sim$wt,FALSE)
plot_w(fitobj$report()$w,fitobj$report()$wt,FALSE)
