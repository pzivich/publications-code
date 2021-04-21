##############################################################
# Network-TMLE code to verify against
##############################################################

# Install corresponding R libraries if not available
# library(devtools)
# devtools::install_github('osofr/simcausal', build_vignettes = FALSE)
# devtools::install_github('osofr/tmlenet', build_vignettes = FALSE)
library(tmlenet)
options(tmlenet.verbose = TRUE, useglm=T)

Kmax <- 2 # max number of friends in the network
n <- 1000 # number of obs

# Loading Data
data(df_netKmax2)

# Defining summary measures
sW <- def_sW(netW1 = W1[[1:Kmax]]) + 
  def_sW(sum.netW1 = sum(W1[[1:Kmax]]))
sA <- def_sA(A = A) +
  def_sA(netA = A[[1:Kmax]], replaceNAw0=TRUE) +
  def_sA(sum.netA = sum(A[[1:Kmax]]), replaceNAw0=TRUE)
summaries = eval.summaries(data = df_netKmax2, Kmax = Kmax,
               sW = sW, sA = sA, IDnode = "IDs", NETIDnode = "Net_str")

# Estimating network-TMLE
res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.35)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str",
                     optPars=list(n_MCsims=1000))
res_K2_1a$EY_gstar1$estimates
res_K2_1a$EY_gstar1$condW.IC.vars
res_K2_1a$EY_gstar1$condW.CIs

res_K2_1a <- tmlenet(data = df_netKmax2, Kmax = Kmax,
                     intervene1.sA = def_new_sA(A=rbinom(nrow(df_netKmax2), 1, 0.65)),
                     sW = sW, sA = sA,
                     Qform = "Y ~ A + sum.netA + W1 + sum.netW1",
                     hform.g0 = "A + netA ~ W1 + sum.netW1",
                     IDnode = "IDs", NETIDnode = "Net_str",
                     optPars=list(n_MCsims=1000))
res_K2_1a$EY_gstar1$estimates
res_K2_1a$EY_gstar1$condW.IC.vars
res_K2_1a$EY_gstar1$condW.CIs
