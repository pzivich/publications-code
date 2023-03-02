#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Solution 2: Restrict the covariate set -- GetTested illustrative example
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

from helper import load_data

#####################################
# Loading trial data
d = load_data(full=False)


#####################################
# Solution 2: G-computation

# Data preparation
da = d.copy()
da['group'] = 1
Xa1 = np.asarray(da[['intercept', 'group', 'age', 'age_sq']])
da['group'] = 0
Xa0 = np.asarray(da[['intercept', 'group', 'age', 'age_sq']])
s = np.asarray(d['clinic'])
y = np.asarray(d['anytest'])


def psi_gcomp(theta):
    psi = theta[0]
    r1, r0 = theta[1], theta[2]
    beta = theta[3:]

    # Estimating the nuisance outcome model
    ee_out = ee_regression(theta=beta,
                           X=d[['intercept', 'group', 'age', 'age_sq']],
                           y=d['anytest'],
                           model='logistic')
    ee_out = np.nan_to_num(ee_out, copy=True, nan=0.)

    # Estimating equations for parameters of interest
    ee_r1 = s*(inverse_logit(np.dot(Xa1, beta)) - r1)
    ee_r0 = s*(inverse_logit(np.dot(Xa0, beta)) - r0)
    ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

    return np.vstack([ee_rd,
                     ee_r1,
                     ee_r0,
                     ee_out])


estr = MEstimator(psi_gcomp, init=[0, 0.5, 0.5, 0, 0, 0, 0])
estr.estimate(solver='lm')

est_g = estr.theta[0]
ci_g = estr.confidence_intervals()[0, :]

print(est_g)
print(ci_g)

#####################################
# Solution 2: Inverse Prob Weight

a = np.asarray(d['group'])


def psi_ipw(theta):
    psi = theta[0]
    r1, r0 = theta[1], theta[2]
    alpha = [theta[3], ]
    beta = theta[4:]

    # Estimating the nuisance action model
    ee_act = ee_regression(theta=alpha,
                           X=d[['intercept']],
                           y=d['group'],
                           model='logistic')
    ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
    pi_a = inverse_logit(np.dot(np.asarray(d[['intercept']]), alpha))

    # Estimating the nuisance transport model
    ee_trp = ee_regression(theta=beta,
                           X=d[['intercept', 'age', 'age_sq']],
                           y=d['clinic'],
                           model='logistic')
    pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'age', 'age_sq']]), beta))

    # Creating IPW
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))

    # Estimating equations for parameters of interest
    ee_r1 = np.where(s == 1, 0, y - r1) * iosw * (a == 1) * iptw
    ee_r0 = np.where(s == 1, 0, y - r0) * iosw * (a == 0) * iptw
    ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

    return np.vstack([ee_rd,
                     ee_r1,
                     ee_r0,
                     ee_trp,
                     ee_act])


estr = MEstimator(psi_ipw, init=[0, 0.5, 0.5, 0, 0, 0, 0])
estr.estimate(solver='lm')

est_w = estr.theta[0]
ci_w = estr.confidence_intervals()[0, :]

print(est_w)
print(ci_w)
