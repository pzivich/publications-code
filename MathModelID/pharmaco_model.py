#######################################################################################################################
# Towards a Unified Framework for Statistical and Mathematical Modeling
#
# Paul Zivich (2026/03/23)
#######################################################################################################################

import numpy as np
import pandas as pd
from itertools import product

########################################################
# Setup

# Loading NHANES data
d = pd.read_csv("data/nhanes.csv")
weight = np.asarray(d['sample_weight'])


def f():
    # Baseline SBP density model (based on NHANES observations)
    b = np.asarray(d['sbp'])
    return b


def g(a, theta):
    # Effective-dose model
    return a * theta[0] * theta[1]


def ybar(b, a, m, lambda_):
    # Pharmaco-dynamic model for dose-response relationship
    change = lambda_[0] + a*lambda_[1]*(m / (lambda_[2] + m)) + lambda_[3]*a
    return b - change


def h(b, a, m, lambda_):
    # Categorization for hypertension status at follow-up
    y = ybar(b=b, a=a, m=m, lambda_=lambda_)
    return np.where(y < 140, 1, 0)


# Mathematical model parameter specifications
theta_1_range = [0.25, 0.40]
lambda_1_range = [16.3, 36.3]
lambda_2_range = [0.1, 13.0]

# Creating all unique combinations of mathematical model paramters
all_combos = list(product(theta_1_range, lambda_1_range, lambda_2_range))

########################################################
# Mathematical Model for Placebo

a = 0
output0 = []
for values in all_combos:
    theta_1, lambda_1, lambda_2 = values
    b = f()
    m = g(a=a, theta=[10, theta_1])
    yhat = h(b=b, a=a, m=m, lambda_=[0, lambda_1, lambda_2, 0])
    mu_bar = np.average(yhat, weights=weight)
    output0.append(mu_bar)


########################################################
# Mathematical Model for Active Drug

a = 1
output1 = []
for values in all_combos:
    theta_1, lambda_1, lambda_2 = values
    b = f()
    m = g(a=a, theta=[10, theta_1])
    yhat = h(b=b, a=a, m=m, lambda_=[0, lambda_1, lambda_2, 0])
    mu_bar = np.average(yhat, weights=weight)
    output1.append(mu_bar)


########################################################
# Computing bounds for the parameter of interest

risk_pairs = product(output1, output0)
risk_differences = []
for rp in risk_pairs:
    rd = rp[0] - rp[1]
    risk_differences.append(rd)

print("Bounds")
print(np.asarray([np.min(risk_differences), np.max(risk_differences)]).round(2))

# OUTPUT
# Bounds
# [0.23 0.91]
