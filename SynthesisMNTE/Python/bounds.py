#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file runs the bounds and sensitivity analysis reported in the main paper.
#
# Paul Zivich (2025/10/03)
#####################################################################################################################

###############################################
# Loading packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression

np.random.seed(905141)

###############################################
# Setting up data

d = pd.read_csv("../data/nhanes.csv")
d = d.dropna(subset=['height', ])
d['height'] = d['height'] / 2.54
d['intercept'] = 1

###############################################
# Synthesis bounds

reg_cols = []  # Storage for new regression column names
for j in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:  # For each unique age in the positive region
    age_label = 'age' + str(j)  # ... create a new column label
    reg_cols.append(age_label)  # ... add column label to the running list
    d[age_label] = np.where(d['age'] == j, 1, 0)  # ... create indicator variable for that specific age


def psi(theta):
    mu = theta[0]
    beta = theta[1:]

    # Data preparation for missing
    r = np.where(d['sbp'].isna(), 0, 1)                     # Indicator if the outcome was observed
    y_no_miss = np.where(r == 1, d['sbp'], -9999)           # Replacing NaN with -9999 to avoid NaN-related errors
    x_star = np.where(d['age'] < 8, 0, 1)

    # Estimating function for statistical nuisance model
    ee_reg = ee_regression(beta,                            # Estimating functions for a regression model of
                           X=d[reg_cols],                   # ... indicator terms for age
                           y=y_no_miss,                     # ... on outcomes where NaN was replaced
                           model='linear',                  # ... using linear regression
                           weights=d['sample_weight']) * r  # ... using sample weights and among complete-cases only

    # Estimating function for bound
    yhat = np.dot(d[reg_cols], beta)                        # Predicting SBP for all observations
    yhat = np.where(x_star == 1, yhat, mu_nonpos)           # Setting SBP for those in nonpositive region
    ee_mean = d['sample_weight']*(yhat - mu)                # Compute the estimating function for the weighted mean

    return np.vstack([ee_mean, ee_reg])                     # Returning stacked estimating equations


# Computing lower bound with M-estimator
mu_nonpos = 70
estr = MEstimator(psi, init=[100., ] + [100., ] * len(reg_cols))
estr.estimate()
lbound = estr.theta[0]
lbound_ci = estr.confidence_intervals()[0, 0]

# Computing upper bound with M-estimator
mu_nonpos = 120
estr = MEstimator(psi, init=[100., ] + [100., ] * len(reg_cols))
estr.estimate()
ubound = estr.theta[0]
ubound_ci = estr.confidence_intervals()[0, 1]

# Results
print("Synthesis Bounds")
print("Bounds:", [lbound, ubound])
print("95% CI:", [lbound_ci, ubound_ci])

###############################################
# Synthesis sensitivity analysis

# stat-model parameters
beta_hat = estr.theta[1:]

p, l, u = [], [], []
values = np.linspace(70, 120, 40)

for v in values:
    # Fitting estimator for given point
    mu_nonpos = v
    estr = MEstimator(psi, init=[100., ] + list(beta_hat))
    estr.estimate()
    ci = estr.confidence_intervals()[0, :]

    # Storing output results
    p.append(estr.theta[0])
    l.append(ci[0])
    u.append(ci[1])


# Generating figure
plt.axhline(104.7, color='#298c8c', label='Complete-case', zorder=1)
plt.fill_between([70, 120], [104.1, 104.1], [105.3, 105.3], color='#298c8c', alpha=0.2, zorder=2)
plt.axhline(101.6, color='#800074', label='Extrapolation', zorder=3)
plt.fill_between([70, 120], [100.8, 100.8], [102.4, 102.4], color='#800074', alpha=0.2, zorder=4)
plt.plot(values, p, '-', color='k', label='Sensitivity Analysis', zorder=6)
plt.fill_between(values, l, u, color='k', alpha=0.2, zorder=5)
plt.xlabel(r"Input Values for $E[Y \mid X^* = 0]$")
plt.ylabel("Mean Systolic Blood Pressure")
plt.ylim([90, 115])
plt.xlim([70, 120])
plt.legend()
plt.tight_layout()
plt.show()
