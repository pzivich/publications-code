#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file runs the analysis reported in the main paper.
#
# Paul Zivich (2024/10/15)
#####################################################################################################################

###############################################
# Loading packages

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import regression_predictions

from mathmodel import MathModel

np.random.seed(905141)

###############################################
# Setting up data

d = pd.read_csv("../data/nhanes.csv")
d = d.dropna(subset=['height', ])
d['height'] = d['height'] / 2.54
d['intercept'] = 1


###############################################
# Complete-case analysis

dr = d.loc[d['sbp'].notnull()].copy()    # Restricting data to complete-cases
y = np.asarray(dr['sbp'])                # Converting outcome column in NumPy array


def psi(theta):
    # Estimating function for the complete-case mean
    mu = theta[0]
    ee_mean = dr['sample_weight']*(y - mu)
    return ee_mean


# Computing the M-estimator
estr = MEstimator(psi, init=[100., ])
estr.estimate()

print("Complete-case")
print(estr.theta[0])
print(estr.confidence_intervals()[0, :])
print("")


###############################################
# Extrapolation

def psi(theta):
    # Estimating function for the extrapolation mean
    mu = theta[0]                                           # Parameter of interest
    beta = theta[1:]                                        # Nuisance parameters
    r = np.where(d['sbp'].isna(), 0, 1)                     # Indicator if the outcome was observed
    y_no_miss = np.where(r == 1, d['sbp'], -9999)           # Replacing NaN with -9999 to avoid NaN-related deli errors
    ee_reg = ee_regression(beta,                            # Estimating functions for a regression model of
                           X=d[['intercept', 'age']],       # ... linear model for age
                           y=y_no_miss,                     # ... on outcomes where NaN was replaced
                           model='linear',                  # ... using linear regression
                           weights=d['sample_weight']) * r  # ... using sample weights and among complete-cases only
    yhat = np.dot(d[['intercept', 'age']], beta)            # Compute the predicted Y values for all obs
    ee_mean = d['sample_weight']*(yhat - mu)                # Compute the weighted mean of the predicted Y values
    return np.vstack([ee_mean, ee_reg])                     # Returning stacked estimating equations


# Computing the M-estimator
estr = MEstimator(psi, init=[100., 100., 0.])
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
    reg_cols = []                                         # Storage for new regression column names
    for j in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:      # For each unique age in the positive region
        age_label = 'age' + str(j)                        # ... create a new column label
        reg_cols.append(age_label)                        # ... add column label to the running list
        dx1[age_label] = np.where(dx1['age'] == j, 1, 0)  # ... create indicator variable for that specific age

    def psi(theta):
        # Estimating function for statistical nuisance model
        r = np.where(dx1['sbp'].isna(), 0, 1)                   # Indicator if the outcome was observed
        y_no_miss = np.where(r == 1, dx1['sbp'], -9999)         # Replacing NaN with -9999 to avoid NaN-related errors
        return ee_regression(theta,                             # Estimating functions for a regression model of
                             X=dx1[reg_cols],                   # ... indicator terms for age
                             y=y_no_miss,                       # ... on outcomes where NaN was replaced
                             model='linear',                    # ... using linear regression
                             weights=dx1['sample_weight']) * r  # ... using sample weights and among complete-cases only

    # Computing M-estimator
    estr = MEstimator(psi, init=[100., ]*len(reg_cols))
    estr.estimate()

    # Short-cut deli function to get predicted SBP values from the model estimates
    preds = regression_predictions(X=dx1[reg_cols], theta=estr.theta, covariance=estr.variance)
    dx1['sbp-hat'] = preds[:, 0]      # Predicted Y for the positive region

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
# Complete-case
# 104.72203263097508
# [104.11666427 105.32740099]
#
# Extrapolation
# 101.57597868021931
# [100.78515856 102.3667988 ]
#
# Synthesis
# 100.48765980889932
# [ 99.94526258 101.02284028]
