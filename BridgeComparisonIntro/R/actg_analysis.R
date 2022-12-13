################################################################################
# ACTG 320 - ACTG 175 Fusion: main analysis
#       This script runs the procedure for the estimation of the risk difference
#
# Paul Zivich (2022/6/11)
################################################################################

library(dplyr)

# Loading the estimator function
setwd("publications-code/BridgeComparisonIntro")
source("R/Chimera.R")

# Loading data set
d = read.csv("data/actg_data_formatted.csv")
dr = d %>% filter(50 <= d[, "cd4"] & d[, "cd4"] <= 300)

### Estimation ###
ans = survival.fusion.ipw(dr, 
                          treatment='art', 
                          sample='study',
                          outcome='delta', 
                          censor='censor',
                          time='t', 
                          sample_model=study ~ male + black + idu + age + 
                              age_rs0 +  age_rs1 + age_rs2 + 
                              as.factor(karnof_cat),
                          treatment_model=art ~ 1,
                          censor_model=Surv(t, censor) ~ male + black + idu + 
                              age + age_rs0 + age_rs1 + age_rs2 + study +  
                              as.factor(karnof_cat) + strata(art),
                          diagnostic=F, 
                          permutation=F, 
                          verbose=T,
                          bootstrap_n=1000)

# Risk difference at t=365 (or last jump)
message("Risk Difference")
tail(ans, n=1)

# Twister plot from Figure 2
twister_plot(ans,
             xvar = rd,
             lcl = rd_lcl,
             ucl = rd_ucl,
             yvar = t,
             xlab = "Risk Difference",
             ylab = "Days",
             reference_line = 0.0)

# Console output
#
# =================================================================
# Sampling Model
# 
# Call:
# glm(formula = model, family = binomial(), data = data)
# 
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.9939  -1.2723   0.7367   0.8882   1.6470  

# Coefficients:
#     Estimate Std. Error z value Pr(>|z|)    
# (Intercept)            -3.177379   1.498744  -2.120 0.034004 *  
# male                   -0.222021   0.188718  -1.176 0.239407    
# black                   0.063093   0.161236   0.391 0.695569    
# idu                     0.168026   0.204250   0.823 0.410709    
# age                     0.117050   0.055780   2.098 0.035869 *  
# age_rs0                -0.002567   0.005106  -0.503 0.615123    
# age_rs1                -0.003170   0.010660  -0.297 0.766199    
# age_rs2                 0.006907   0.008017   0.862 0.388937    
# as.factor(karnof_cat)1  0.506244   0.145548   3.478 0.000505 ***
# as.factor(karnof_cat)2  0.586946   0.242453   2.421 0.015483 *  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1301.0  on 1033  degrees of freedom
# Residual deviance: 1241.7  on 1024  degrees of freedom
# AIC: 1261.7
# 
# Number of Fisher Scoring iterations: 4
# 
# =================================================================
# =================================================================
# Treatment Model : S=0
# 
# Call:
# glm(formula = model, family = binomial(), data = ds0)
# 
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.4904  -1.4904   0.8939   0.8939   0.8939  
# 
# Coefficients:
#    Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   0.7112     0.1164   6.108 1.01e-09 ***
#  ---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 423.32  on 333  degrees of freedom
# Residual deviance: 423.32  on 333  degrees of freedom
# AIC: 425.32
# 
# Number of Fisher Scoring iterations: 4
# 
# =================================================================
# =================================================================
# Treatment Model : S=1
# 
# Call:
# glm(formula = model, family = binomial(), data = ds1)

# Deviance Residuals: 
#    Min      1Q  Median      3Q     Max  
# -1.192  -1.192   1.163   1.163   1.163  
# 
# Coefficients:
#     Estimate Std. Error z value Pr(>|z|)
# (Intercept)  0.03429    0.07560   0.454     0.65
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 970.2  on 699  degrees of freedom
# Residual deviance: 970.2  on 699  degrees of freedom
# AIC: 972.2
# 
# Number of Fisher Scoring iterations: 3
#
# =================================================================
# =================================================================
# Censoring Model
# Call:
# coxph(formula = model, data = data, method = "breslow")
# 
# n= 1034, number of events= 663 
# 
# coef  exp(coef)   se(coef)      z Pr(>|z|)    
# male                    3.374e-03  1.003e+00  1.035e-01  0.033   0.9740    
# black                   2.357e-01  1.266e+00  9.192e-02  2.564   0.0104 *  
# idu                    -3.328e-02  9.673e-01  1.076e-01 -0.309   0.7571    
# age                    -9.390e-03  9.907e-01  3.876e-02 -0.242   0.8086    
# age_rs0                 8.595e-05  1.000e+00  3.344e-03  0.026   0.9795    
# age_rs1                 3.168e-04  1.000e+00  6.502e-03  0.049   0.9611    
# age_rs2                -1.571e-03  9.984e-01  4.500e-03 -0.349   0.7271    
# study                   4.105e+00  6.066e+01  3.023e-01 13.582   <2e-16 ***
# as.factor(karnof_cat)1 -1.717e-01  8.422e-01  8.424e-02 -2.039   0.0415 *  
# as.factor(karnof_cat)2 -9.466e-02  9.097e-01  1.312e-01 -0.722   0.4705    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Concordance= 0.719  (se = 0.012 )
# Likelihood ratio test= 550.9  on 10 df,   p=<2e-16
# Wald test            = 202  on 10 df,   p=<2e-16
# Score (logrank) test = 478.7  on 10 df,   p=<2e-16
# 
# =================================================================
# 
#      t         rd      rd.se      rd_lcl      rd_ucl
# 73 365 -0.2048207 0.06532376  -0.3328553 -0.07678617
# 