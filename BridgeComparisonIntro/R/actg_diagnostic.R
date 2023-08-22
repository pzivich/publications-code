################################################################################
# ACTG 320 - ACTG 175 Fusion: diagnostic results
#       This script runs the proposed diagnostics using the ACTG example. 
#       Graphical and diagnostic tests are both demonstrated.
#
# Paul Zivich (2023/8/22)
################################################################################

library(dplyr)
set.seed(20211011)

# Setting working drive
setwd("publications-code/BridgeComparisonIntro")
source("R/Chimera.R")

# Loading data set
d = read.csv("data/actg_data_formatted.csv")
dr = d %>% filter(50 <= d[, "cd4"] & d[, "cd4"] <= 300)

### Diagnostics ###

# Unadjusted model
ans = survival.fusion.ipw(d, 
                          treatment='art', 
                          sample='study',
                          outcome='delta', 
                          censor='censor',
                          time='t', 
                          sample_model=study ~ 1,
                          treatment_model=art ~ 1,
                          censor_model=Surv(t, censor) ~ male + black + idu + 
                              age + age_rs0 + age_rs1 + age_rs2 + study + 
                              as.factor(karnof_cat) + strata(art),
                          diagnostic_plot=T, 
                          diagnostic_test=T, 
                          bootstrap_n=1000,
                          verbose=F)

# Adjusted model
ans = survival.fusion.ipw(d, 
                          treatment='art', 
                          sample='study',
                          outcome='delta', 
                          censor='censor',
                          time='t', 
                          sample_model=study ~ male + black + idu + age + 
                              age_rs0 + age_rs1 + age_rs2 + 
                              as.factor(karnof_cat),
                          treatment_model=art ~ 1,
                          censor_model=Surv(t, censor) ~ male + black + idu + 
                              age + age_rs0 + age_rs1 + age_rs2 + study + 
                              as.factor(karnof_cat) + strata(art),
                          diagnostic_plot=T, 
                          diagnostic_test=T, 
                          bootstrap_n=1000,
                          verbose=F)

# Sampling model including CD4
ans = survival.fusion.ipw(d, 
                          treatment='art', 
                          sample='study',
                          outcome='delta', 
                          censor='censor',
                          time='t', 
                          sample_model=study ~ male + black + idu + age + 
                              age_rs0 + age_rs1 + age_rs2 + 
                              as.factor(karnof_cat) + cd4 + cd4_rs0 + cd4_rs1,
                          treatment_model=art ~ 1,
                          censor_model=Surv(t, censor) ~ male + black + idu + 
                              age + age_rs0 + age_rs1 + age_rs2 + study + 
                              as.factor(karnof_cat) + strata(art) + cd4 + 
                              cd4_rs0 + cd4_rs1,
                          diagnostic_plot=T, 
                          diagnostic_test=T, 
                          bootstrap_n=1000,
                          verbose=F)

# Restricted by CD4
ans = survival.fusion.ipw(dr, 
                          treatment='art', 
                          sample='study',
                          outcome='delta', 
                          censor='censor',
                          time='t', 
                          sample_model=study ~ male + black + idu + age + 
                              age_rs0 + age_rs1 + age_rs2 + 
                              as.factor(karnof_cat),
                          treatment_model=art ~ 1,
                          censor_model=Surv(t, censor) ~ male + black + idu + 
                              age + age_rs0 + age_rs1 + age_rs2 + study + 
                              as.factor(karnof_cat) + strata(art),
                          diagnostic_plot=T, 
                          diagnostic_test=T, 
                          bootstrap_n=1000,
                          verbose=F)

# Console Output
#
# ====================================================
# Diagnostic Test
# ====================================================
# No. Bootstraps:  1000
# ----------------------------------------------------
# Area:     32.496
# 95% CI:   24.174 40.818
# P-value:  0
# ====================================================
#
# ====================================================
# Diagnostic Test
# ====================================================
# No. Bootstraps:  1000
# ----------------------------------------------------
# Area:     30.051
# 95% CI:   20.572 39.529
# P-value:  0
# ====================================================
# 
# ====================================================
# Diagnostic Test
# ====================================================
# No. Bootstraps:  1000
# ----------------------------------------------------
# Area:     36.367
# 95% CI:   28.002 44.732
# P-value:  0
# ====================================================
# 
# ====================================================
# Diagnostic Test
# ====================================================
# No. Bootstraps:  1000
# ----------------------------------------------------
# Area:     7.944
# 95% CI:   -2.606 18.493
# P-value:  0.14
# ====================================================
