################################################################################
# ACTG 320 - ACTG 175 Fusion: diagnostic results
#       This script runs the proposed diagnostics using the ACTG example. 
#       Graphical and permutation tests are both demonstrated.
#
# Paul Zivich (2022/6/11)
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
                          diagnostic=T, 
                          permutation=T, 
                          permutation_n=10000, 
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
                          diagnostic=T, 
                          permutation=T,
                          permutation_n=10000,
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
                          diagnostic=T,
                          permutation=T,
                          permutation_n=10000,
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
                          diagnostic=T, 
                          permutation=T,
                          permutation_n=10000, 
                          bootstrap_n=1000,
                          verbose=F)

# Console Output
#
# ====================================================
# Permutation Test
# ====================================================
# Observed area:  32.50
# No. Permutations:  10000
# ----------------------------------------------------
# P-value:  0
# ====================================================
#
# ====================================================
# Permutation Test
# ====================================================
# Observed area:  30.05
# No. Permutations:  10000
# ----------------------------------------------------
# P-value:  0
# ====================================================
# 
# ====================================================
# Permutation Test
# ====================================================
# Observed area:  36.3671117881398
# No. Permutations:  10000
# ----------------------------------------------------
# P-value:  0
# ====================================================
# 
# ====================================================
# Permutation Test
# ====================================================
# Observed area:  9.51598461324015
# No. Permutations:  10000
# ----------------------------------------------------
# P-value:  0.0723
# ====================================================
