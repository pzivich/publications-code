#######################################################################################################################
# Practical Implementation of g-computation for marginal structural models
#
# Paul Zivich (2023/11/29)
#######################################################################################################################

# Importing libraries
import numpy as np
import pandas as pd
from formulaic import model_matrix
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

# Reading in data set
d = pd.read_csv("actg.csv")

# Model specifications
g_model = ("male + idu + white + C(karnof) "
           "+ agec + age_rs1 + age_rs2 + age_rs3 "
           "+ cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3")
m_model = ("treat + male + treat:male + idu + white + C(karnof) "
           "+ agec + age_rs1 + age_rs2 + age_rs3 "
           "+ cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3 "
           "+ male:(cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3)")
msm_model = "treat + treat:male + male"

##########################################################################
# G-computation as an M-estimator

X = model_matrix(m_model, d)         # Outcome model design matrix
y = np.asarray(d['cd4_20wk'])        # Outcome variable

da = d.copy()                        # Copy original data set
da['treat'] = 1                      # Set A to 1 in the copy
X1 = model_matrix(m_model, da)       # Get outcome model design matrix with A=1
M1 = model_matrix(msm_model, da)     # Get MSM design matrix with A=1
da['treat'] = 0                      # Set A to 0 in the copy
X0 = model_matrix(m_model, da)       # Get outcome model design matrix with A=0
M0 = model_matrix(msm_model, da)     # Get MSM design matrix with A=0
idx_msm = M1.shape[1]                # Number of columns in MSM design matrix
idx_m = X.shape[1]                   # Number of columns in outcome model design matrix


def psi(theta):
    # Defining the estimating function for delicatessen
    beta = theta[:idx_msm]           # Parameters of interest
    alpha = theta[idx_msm:]          # Nuisance parameters

    # Outcome nuisance model
    ee_reg = ee_regression(alpha, X=X, y=y, model='linear')
    y1hat = np.dot(X1, alpha)        # Pseudo-outcome under A=1
    y0hat = np.dot(X0, alpha)        # Pseudo-outcome under A=0

    # Marginal structural model
    ee_msm1 = ee_regression(beta, X=M1, y=y1hat, model='linear')
    ee_msm0 = ee_regression(beta, X=M0, y=y0hat, model='linear')
    ee_msm = ee_msm1 + ee_msm0       # Combining score functions are described in paper

    # Returning stacked estimating functions
    return np.vstack([ee_msm, ee_reg])


# Applying M-estimator via delicatessen
init_vals = [0., ]*idx_msm + [0., ]*idx_m
estr = MEstimator(psi, init=init_vals)
estr.estimate()

# Storing results in a table to output
table = pd.DataFrame()
table['Parameter'] = M1.columns
table['Estimate-gcomp'] = estr.theta[:idx_msm]
table['SE-gcomp'] = np.sqrt(np.diag(estr.variance)[:idx_msm])
table['LCL-gcomp'] = estr.confidence_intervals()[:idx_msm, 0]
table['UCL-gcomp'] = estr.confidence_intervals()[:idx_msm, 1]
table['CLD-gcomp'] = table['UCL-gcomp'] - table['LCL-gcomp']

##########################################################################
# Inverse Probability Weighting

W = model_matrix(g_model, d)         # Propensity score model design matrix
M = model_matrix(msm_model, d)       # MSM design matrix with observed data
a = np.asarray(d['treat'])           # Treatment variable
y = np.asarray(d['cd4_20wk'])        # Outcome variable
idx_msm = M.shape[1]                 # Number of columns in MSM design matrix
idx_g = W.shape[1]                   # Number of columns in propensity score design matrix


def psi(theta):
    # Defining the estimating function for delicatessen
    beta = theta[:idx_msm]           # Parameters of interest
    gamma = theta[idx_msm:]          # Nuisance parameters

    # Propensity score nuisance model
    ee_reg = ee_regression(gamma, X=W, y=a, model='logistic')
    pr_a = inverse_logit(np.dot(W, gamma))          # Propensity score
    pr_a = np.clip(pr_a, a_min=0.01, a_max=0.99)    # Truncating PS to prevent extreme weights
    ipw = a / pr_a + (1-a) / (1-pr_a)               # Transforming into IPTW

    # Marginal structural model
    ee_msm = ee_regression(beta, X=M, y=y,
                           model='linear',
                           weights=ipw)

    # Returning stacked estimating functions
    return np.vstack([ee_msm, ee_reg])


# Applying M-estimator via delicatessen
init_vals = [350., -20., 40., 0., ] + [0., ]*idx_g
estr = MEstimator(psi, init=init_vals)
estr.estimate()

# Storing results in a table to output
table['Estimate-ipw'] = estr.theta[:idx_msm]
table['SE-ipw'] = np.sqrt(np.diag(estr.variance)[:idx_msm])
table['LCL-ipw'] = estr.confidence_intervals()[:idx_msm, 0]
table['UCL-ipw'] = estr.confidence_intervals()[:idx_msm, 1]
table['CLD-ipw'] = table['UCL-ipw'] - table['LCL-ipw']

##########################################################################
# Results

# table.round(1).to_csv("msm_results.csv", index=False)  # Output as CSV to drop into paper
print(table.round(1))

# END
