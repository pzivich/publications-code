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
                              age + age_rs0 + age_rs1 + age_rs2 + 
                              as.factor(karnof_cat) + strata(art),
                          diagnostic=F, 
                          permutation=F, 
                          verbose=T)

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


# =================================================================
# Location Model
# Call:
#     glm(formula = model, family = binomial(), data = data)
# 
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.9939  -1.2723   0.7367   0.8882   1.6470  
# 
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
#  as.factor(karnof_cat)2  0.586946   0.242453   2.421 0.015483 *  
# ---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
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
# Treatment Model : S=0
#
# Call:
#     glm(formula = model, family = binomial(), data = ds0)
# 
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.4904  -1.4904   0.8939   0.8939   0.8939  
# 
# Coefficients:
#     Estimate Std. Error z value Pr(>|z|)    
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
# Treatment Model : S=1
#
# Call:
#     glm(formula = model, family = binomial(), data = ds1)
# 
# Deviance Residuals: 
#     Min      1Q  Median      3Q     Max  
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
# Censoring Model
# Call:
#     coxph(formula = model, data = data, method = "breslow")
# 
# n= 334, number of events= 19 
# 
# coef exp(coef)  se(coef)      z Pr(>|z|)
# male                   -0.389471  0.677415  0.575189 -0.677    0.498
# black                   0.707193  2.028290  0.498644  1.418    0.156
# idu                    -0.085390  0.918154  0.776747 -0.110    0.912
# age                    -0.049910  0.951315  0.102764 -0.486    0.627
 #age_rs0                -0.002819  0.997185  0.012498 -0.226    0.822
 #age_rs1                 0.012453  1.012531  0.034081  0.365    0.715
# age_rs2                -0.009568  0.990478  0.032196 -0.297    0.766
# as.factor(karnof_cat)1 -0.103880  0.901333  0.506149 -0.205    0.837
# as.factor(karnof_cat)2 -0.370323  0.690511  1.050516 -0.353    0.724
# 
# Concordance= 0.74  (se = 0.039 )
# Likelihood ratio test= 7.33  on 9 df,   p=0.6
# Wald test            = 7.66  on 9 df,   p=0.6
# Score (logrank) test = 8.36  on 9 df,   p=0.5
# 
# =================================================================
# Censoring Model
# Call:
#    coxph(formula = model, data = data, method = "breslow")
# 
# n= 700, number of events= 644 
# 
# coef  exp(coef)   se(coef)      z Pr(>|z|)  
# male                    0.0144245  1.0145291  0.1054695  0.137   0.8912  
# black                   0.2068861  1.2298425  0.0940144  2.201   0.0278 *
# idu                    -0.0308143  0.9696557  0.1087482 -0.283   0.7769  
# age                     0.0043851  1.0043947  0.0425831  0.103   0.9180  
# age_rs0                -0.0006076  0.9993926  0.0035843 -0.170   0.8654  
# age_rs1                 0.0009483  1.0009488  0.0067985  0.139   0.8891  
# age_rs2                -0.0016150  0.9983863  0.0045931 -0.352   0.7251  
# as.factor(karnof_cat)1 -0.1710468  0.8427821  0.0855892 -1.998   0.0457 *
# as.factor(karnof_cat)2 -0.0953946  0.9090142  0.1324783 -0.720   0.4715  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Concordance= 0.568  (se = 0.013 )
# Likelihood ratio test= 23.54  on 9 df,   p=0.005
# Wald test            = 21.68  on 9 df,   p=0.01
# Score (logrank) test = 22.07  on 9 df,   p=0.009
#
# Risk Difference
#      t         rd      rd_se     rd_lcl      rd_ucl
# 72 358 -0.2030732 0.06812496 -0.3365981 -0.06954831
