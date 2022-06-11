####################################################################################################################
# ACTG 320 - ACTG 175 Fusion: main analysis
#       This script runs the procedure for the estimation of the risk difference
#
# Paul Zivich (2022/6/11)
####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chimera import SurvivalFusionIPW


# __main__ call is not needed here (since no Pool) but putting regardless
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
    afipw.censoring_model("male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat)",
                          censor_shift=1e-4, bound=None, stratify_by_sample=True, strata='art')
    r = afipw.estimate()
    print(r[['RD', 'R2_S1', 'R1_S1', 'R1_S0', 'R0_S0']])
    print((r['R2_S1']-r['R1_S1']) + (r['R1_S0']-r['R0_S0']))

    print("\nRisk difference at t=365")
    print(np.round(r.iloc[-1, 1:5], 2), '\n')

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
    # plt.savefig("results/figure2_draft.png", format="png", dpi=300)
    plt.show()

# ==============================================================================
# Sampling Model
#                  Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:                  study   No. Observations:                 1034
# Model:                            GLM   Df Residuals:                     1024
# Model Family:                Binomial   Df Model:                            9
# Link Function:                  logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -620.85
# Date:                Sat, 11 Jun 2022   Deviance:                       1241.7
# Time:                        09:10:06   Pearson chi2:                 1.04e+03
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
# Date:                Sat, 11 Jun 2022   Deviance:                       423.32
# Time:                        09:10:06   Pearson chi2:                     334.
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
# Date:                Sat, 11 Jun 2022   Deviance:                       970.20
# Time:                        09:10:06   Pearson chi2:                     700.
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
# ------------------------------------------------------------------------------
# study=0
# ------------------------------------------------------------------------------
#                               Results: PHReg
# ==========================================================================
# Model:                       PH Reg         Num strata:              2
# Dependent variable:          t              Min stratum size:        110
# Ties:                        Breslow        Max stratum size:        224
# Sample size:                 329            Avg stratum size:        167.0
# Num. events:                 19
# --------------------------------------------------------------------------
#                       log HR log HR SE   HR      t    P>|t|  [0.025 0.975]
# --------------------------------------------------------------------------
# C(karnof_cat)[T.1.0] -0.1039    0.5061 0.9013 -0.2052 0.8374 0.3342 2.4306
# C(karnof_cat)[T.2.0] -0.3703    1.0505 0.6905 -0.3525 0.7245 0.0881 5.4122
# male                 -0.3895    0.5752 0.6774 -0.6771 0.4983 0.2194 2.0915
# black                 0.7072    0.4986 2.0283  1.4182 0.1561 0.7633 5.3899
# idu                  -0.0854    0.7767 0.9182 -0.1099 0.9125 0.2003 4.2081
# age                  -0.0499    0.1028 0.9513 -0.4857 0.6272 0.7778 1.1636
# age_rs0              -0.0028    0.0125 0.9972 -0.2256 0.8215 0.9731 1.0219
# age_rs1               0.0125    0.0341 1.0125  0.3654 0.7148 0.9471 1.0825
# age_rs2              -0.0096    0.0322 0.9905 -0.2972 0.7663 0.9299 1.0550
# ==========================================================================
# Confidence intervals are for the hazard ratios
# ------------------------------------------------------------------------------
# study=1
# ------------------------------------------------------------------------------
#                               Results: PHReg
# ==========================================================================
# Model:                       PH Reg         Num strata:              2
# Dependent variable:          t              Min stratum size:        344
# Ties:                        Breslow        Max stratum size:        356
# Sample size:                 700            Avg stratum size:        350.0
# Num. events:                 644
# --------------------------------------------------------------------------
#                       log HR log HR SE   HR      t    P>|t|  [0.025 0.975]
# --------------------------------------------------------------------------
# C(karnof_cat)[T.1.0] -0.1710    0.0856 0.8428 -1.9985 0.0457 0.7126 0.9967
# C(karnof_cat)[T.2.0] -0.0954    0.1325 0.9090 -0.7201 0.4715 0.7011 1.1785
# male                  0.0144    0.1055 1.0145  0.1368 0.8912 0.8251 1.2475
# black                 0.2069    0.0940 1.2298  2.2006 0.0278 1.0229 1.4787
# idu                  -0.0308    0.1087 0.9697 -0.2834 0.7769 0.7835 1.2000
# age                   0.0044    0.0426 1.0044  0.1030 0.9180 0.9240 1.0918
# age_rs0              -0.0006    0.0036 0.9994 -0.1695 0.8654 0.9924 1.0064
# age_rs1               0.0009    0.0068 1.0009  0.1395 0.8891 0.9877 1.0144
# age_rs2              -0.0016    0.0046 0.9984 -0.3516 0.7251 0.9894 1.0074
# ==========================================================================
# Confidence intervals are for the hazard ratios
# ==============================================================================
#
# Risk difference at t=365
# RD       -0.20
# RD_SE     0.07
# RD_LCL   -0.34
# RD_UCL   -0.07
