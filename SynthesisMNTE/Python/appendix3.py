#####################################################################################################################
# Using a Synthesis of Statistical and Mathematical Models to Account for Missing Data in Public Health Research
#   This file runs the AIPW model analysis reported in Appendix 3.
#
# Paul Zivich (2025/10/9)
#####################################################################################################################

###############################################
# Loading packages

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit
from formulaic import model_matrix

from mathmodel import MathModel

np.random.seed(2905141)

###############################################
# Setting up data

d = pd.read_csv("../data/nhanes.csv")
d = d.dropna(subset=['height', 'weight'])
d['height'] = d['height'] / 2.54
d['intercept'] = 1
d['r'] = np.where(d['sbp'].isna(), 0, 1)

y_model = "female*(age + age_sp1 + age_sp2 + height + h_sp1 + h_sp2 + weight + w_sp1 + w_sp2)"
m_model = "female + age + age_sp1 + age_sp2 + height + h_sp1 + h_sp2 + weight + w_sp1 + w_sp2"


###############################################
# Extrapolation

X = model_matrix(y_model, d)
W = model_matrix(m_model, d)
r = np.where(d['sbp'].isna(), 0, 1)            # Indicator if the outcome was observed
y_no_miss = np.where(r == 1, d['sbp'], -9999)  # Replacing NaN with -9999 to avoid NaN-related deli errors


def psi_missing(theta):
    ee_ipw = ee_regression(theta,                           # Estimating functions for a regression model of
                           X=W,                             # ... parametric model for age, gender, weight, height
                           y=r,                             # ... on missingness
                           model='logistic',                # ... using logistic regression
                           weights=d['sample_weight'])      # ... using sample weights
    return ee_ipw


estr = MEstimator(psi_missing, init=[0., ]*W.shape[1])
estr.estimate()
init_ipw = list(estr.theta)


def psi(theta):
    # Estimating function for the extrapolation mean
    mu = theta[0]                                           # Parameter of interest
    dimW = W.shape[1]                                       # Number of parameters in IPW model
    alpha = theta[1: 1+dimW]                                # Nuisance parameters for IPW
    beta = theta[1+dimW:]                                   # Nuisance parameters for g-computation
    ee_ipw = ee_regression(alpha,                           # Estimating functions for a regression model of
                           X=W,                             # ... parametric model for age, gender, weight, height
                           y=r,                             # ... on missingness
                           model='logistic',                # ... using logistic regression
                           weights=d['sample_weight'])      # ... using sample weights
    pr_obs = inverse_logit(np.dot(W, alpha))                # Probability of being observed
    ipmw = 1 / pr_obs                                       # Inverse Probability of Missing Weight
    overall_weight = np.asarray(d['sample_weight']) * ipmw  # Overall weight to use for outcome model
    ee_out = ee_regression(beta,                            # Estimating functions for a regression model of
                           X=X,                             # ... parametric model for age, gender, weight, height
                           y=y_no_miss,                     # ... on outcomes where NaN was replaced
                           model='linear',                  # ... using linear regression
                           weights=overall_weight) * r      # ... using sample weights and among complete-cases only
    yhat = np.dot(X, beta)                                  # Compute the predicted Y values for all obs
    ee_mean = d['sample_weight']*(yhat - mu)                # Compute the weighted mean of the predicted Y values
    return np.vstack([ee_mean, ee_ipw, ee_out])             # Returning stacked estimating equations


# Computing the M-estimator
estr = MEstimator(psi, init=[100., ] + init_ipw + [100., ] + [0., ]*(X.shape[1] - 1))
estr.estimate()

print("Extrapolation")
print(estr.theta[0])
print(estr.confidence_intervals()[0, :])
print("")


###############################################
# Synthesis approach

# Storage for parameter of interest estimates
mu_hats = []

# Monte-Carlo procedure for point and confidence interval estimation
for i in range(10000):
    ds = d.sample(n=d.shape[0], replace=True)             # Resample observed data with replacement

    # Subsetting data into positive and non-positive regions
    dx0 = ds.loc[ds['age'] < 8].copy()                    # Subset resampled data to non-positive region
    dx1 = ds.loc[ds['age'] >= 8].copy()                   # Subset resampled data to positive region

    # Statistical model
    fam1 = sm.families.Binomial()                                      # GLM specification
    fm1 = smf.glm("r ~ " + m_model, data=dx1, family=fam1).fit()       # Fitting GLM
    pr_obs = fm1.predict(dx1)                                          # Predicted probability of being observed
    ipw = np.where(dx1['r'] == 1, 1/pr_obs, 0) * dx1['sample_weight']  # Inverse probability weight
    fam2 = sm.families.Gaussian()                                      # GLM specification
    fm2 = smf.glm("sbp ~ " + y_model, data=dx1, family=fam2,           # Fitting GLM
                  freq_weights=ipw).fit()                              # ... with weights
    dx1['sbp-hat'] = fm2.predict(dx1)                                  # Predictions for positive observations

    # Mathematical model
    math_model = MathModel()                                           # Initialize the mathematical model class
    bp = math_model.simulate_blood_pressure(female=dx0['female'],      # Simulate a single SBP given gender,
                                            age=dx0['age'],            # ... age,
                                            height=dx0['height'])      # ... and height
    dx0['sbp-hat'] = bp                                                # Add simulated SBP to non-positive data

    # Stack predictions together (including all variables here is not necessary)
    dx = pd.concat([dx0[['id', 'age', 'female', 'sbp', 'sbp-hat', 'sample_weight']],
                    dx1[['id', 'age', 'female', 'sbp', 'sbp-hat', 'sample_weight']]])

    # Compute the parameter of interest by hand
    mu_i = np.sum(dx['sbp-hat'] * dx['sample_weight']) / np.sum(dx['sample_weight'])
    mu_hats.append(mu_i)  # Add parameter of interest from this Monte-Carlo procedure to on-going list


# Evaluating Monte-Carlo procedure results
mu_hat = np.median(mu_hats)                           # Median corresponds to the point estimate
mu_hat_ci = np.percentile(mu_hats, q=[2.5, 97.5])     # Confidence intervals computed via the percentiles

print("Synthesis")
print(mu_hat)
print(mu_hat_ci)
print("")

# Output: 2025/10/10
# --------------------------------------------------------------------------------------------------------------
# Extrapolation
# 101.09450853874321
# [ 98.3754103  103.81360678]
#
# Synthesis
# 100.47208680802606
# [ 99.91983401 101.00784009]
