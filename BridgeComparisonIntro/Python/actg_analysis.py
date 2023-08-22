####################################################################################################################
# ACTG 320 - ACTG 175 Fusion: main analysis
#       This script runs the procedure for the estimation of the risk difference
#
# Paul Zivich (2023/08/22)
####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chimera import SurvivalFusionIPW


if __name__ == "__main__":
    ###############################
    # Loading Data and Setup
    d = pd.read_csv("../data/actg_data_formatted.csv", sep=",")

    # Restricting by CD4
    dr = d.loc[(d['cd4'] >= 50) & (d['cd4'] <= 300)].copy()

    ###############################
    # Estimation
    afipw = SurvivalFusionIPW(df=dr, treatment='art', outcome='delta', time='t',
                              sample='study', censor='censor', verbose=True)
    afipw.sampling_model("male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat)",
                         bound=None)
    afipw.treatment_model(model="1", bound=None)
    afipw.censoring_model("male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat) + study",
                          censor_shift=1e-4, bound=None, stratify_by_sample=False, strata='art')
    r = afipw.estimate(variance="bootstrap", bs_iterations=1000,
                       n_cpus=28, seed=20230705)

    print("\nRisk difference at t=365")
    print(np.round(r.iloc[-1, 1:5], 3), '\n')

    ###############################
    # Twister plot of main result
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))          # Generating axes
    ax.vlines(0, 0, 370, colors='gray', linestyles='--', label=None)  # Reference at RD=0
    ax.step(r["RD"],                                                  # Risk Difference column
            r["t"].shift(-1).ffill(),                                 # time column (shift to align)
            color='k', where='post', label="Unadjusted")
    ax.fill_betweenx(r["t"],                                          # time column (no shift needed here)
                     r["RD_UCL"],                                     # upper confidence limit
                     r["RD_LCL"],                                     # lower confidence limit
                     label=None, color='gray', alpha=0.5, step='post')
    # Setting up axes labels
    ax.set_xlabel(r"Risk Difference")
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylabel("Days")
    ax.set_ylim([0, 366])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 365])
    # Pop-out label for easy interpretations on the Twister plot
    ax2 = ax.twiny()                                                  # Duplicate the x-axis to create a separate label
    ax2.set_xlabel("Favors triple therapy                Favors monotherapy",
                   fontdict={"size": 10})
    ax2.set_xticks([])                                                # Removes top x-axes tick marks
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 36))
    # Getting rid of extra labels
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig("figure2.png", format="png", dpi=300)
    plt.close()

# ==============================================================================
# Sampling Model
#                  Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:                  study   No. Observations:                 1034
# Model:                            GLM   Df Residuals:                     1024
# Model Family:                Binomial   Df Model:                            9
# Link Function:                  logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -620.85
# Date:                Wed, 05 Jul 2023   Deviance:                       1241.7
# Time:                        09:55:17   Pearson chi2:                 1.04e+03
# No. Iterations:                     4
# Covariance Type:            nonrobust
# ========================================================================================
#                            coef    std err          z      P>|z|      [0.025      0.975]
# ----------------------------------------------------------------------------------------
# Intercept               -3.1774      1.499     -2.120      0.034      -6.115      -0.240
# C(karnof_cat)[T.1.0]     0.5062      0.146      3.478      0.001       0.221       0.792
# C(karnof_cat)[T.2.0]     0.5869      0.242      2.421      0.015       0.112       1.062
# male                    -0.2220      0.189     -1.176      0.239      -0.592       0.148
# black                    0.0631      0.161      0.391      0.696      -0.253       0.379
# idu                      0.1680      0.204      0.823      0.411      -0.232       0.568
# age                      0.1170      0.056      2.098      0.036       0.008       0.226
# age_rs0                 -0.0026      0.005     -0.503      0.615      -0.013       0.007
# age_rs1                 -0.0032      0.011     -0.297      0.766      -0.024       0.018
# age_rs2                  0.0069      0.008      0.862      0.389      -0.009       0.023
# ========================================================================================
# ==============================================================================
# ==============================================================================
# Treatment Model, study=0
#                  Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:                    art   No. Observations:                  334
# Model:                            GLM   Df Residuals:                      333
# Model Family:                Binomial   Df Model:                            0
# Link Function:                  logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -211.66
# Date:                Wed, 05 Jul 2023   Deviance:                       423.32
# Time:                        09:55:17   Pearson chi2:                     334.
# No. Iterations:                     4
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.7112      0.116      6.108      0.000       0.483       0.939
# ==============================================================================
# ==============================================================================
# ==============================================================================
# Treatment Model, study=1
#                  Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:                    art   No. Observations:                  700
# Model:                            GLM   Df Residuals:                      699
# Model Family:                Binomial   Df Model:                            0
# Link Function:                  logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -485.10
# Date:                Wed, 05 Jul 2023   Deviance:                       970.20
# Time:                        09:55:17   Pearson chi2:                     700.
# No. Iterations:                     3
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.0343      0.076      0.454      0.650      -0.114       0.182
# ==============================================================================
# ==============================================================================
# ==============================================================================
# Censoring Models
#                                 Results: PHReg
# ==============================================================================
# Model:                       PH Reg           Num strata:                3
# Dependent variable:          t                Min stratum size:          110
# Ties:                        Breslow          Max stratum size:          568
# Sample size:                 1031             Avg stratum size:          344.7
# Num. events:                 663
# ------------------------------------------------------------------------------
#                       log HR log HR SE    HR      t    P>|t|   [0.025  0.975]
# ------------------------------------------------------------------------------
# C(karnof_cat)[T.1.0] -0.1717    0.0842  0.8422 -2.0386 0.0415  0.7140   0.9934
# C(karnof_cat)[T.2.0] -0.0947    0.1312  0.9097 -0.7216 0.4705  0.7035   1.1764
# male                  0.0034    0.1035  1.0034  0.0326 0.9740  0.8192   1.2290
# black                 0.2357    0.0919  1.2657  2.5638 0.0104  1.0571   1.5156
# idu                  -0.0333    0.1076  0.9673 -0.3092 0.7571  0.7833   1.1944
# age                  -0.0094    0.0388  0.9907 -0.2423 0.8086  0.9182   1.0688
# age_rs0               0.0001    0.0033  1.0001  0.0257 0.9795  0.9936   1.0067
# age_rs1               0.0003    0.0065  1.0003  0.0487 0.9611  0.9876   1.0131
# age_rs2              -0.0016    0.0045  0.9984 -0.3490 0.7271  0.9897   1.0073
# study                 4.1053    0.3023 60.6592 13.5818 0.0000 33.5437 109.6940
# ==============================================================================
# Confidence intervals are for the hazard ratios
# ==============================================================================
#
# Risk difference at t=365
# RD       -0.205
# RD_SE     0.063
# RD_LCL   -0.328
# RD_UCL   -0.082
