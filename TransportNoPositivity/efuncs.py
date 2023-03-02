#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Stacked estimating functions for restriction approaches
#
# Paul Zivich
#######################################################################################################################

import numpy as np
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


def ee_restrict_gcomp(theta, y, s, X, Xa1, Xa0):
    psi = theta[0]
    r1, r0 = theta[1], theta[2]
    beta = theta[3:]

    # Estimating the nuisance outcome model
    ee_out = ee_regression(theta=beta,
                           X=X,
                           y=y,
                           model='logistic')
    ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

    # Estimating equations for parameters of interest
    ee_r1 = s*(inverse_logit(np.dot(Xa1, beta)) - r1)
    ee_r0 = s*(inverse_logit(np.dot(Xa0, beta)) - r0)
    ee_rd = np.ones(X.shape[0]) * (r1 - r0) - psi

    return np.vstack([ee_rd,
                     ee_r1,
                     ee_r0,
                     ee_out])


def ee_restrict_ipw(theta, y, s, a, W, null):
    psi = theta[0]
    r1, r0 = theta[1], theta[2]
    alpha = [theta[3], ]
    beta = theta[4:]

    # Estimating the nuisance action model
    ee_act = ee_regression(theta=alpha,
                           X=null,
                           y=a,
                           model='logistic')
    ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
    pi_a = inverse_logit(np.dot(null, alpha))

    # Estimating the nuisance transport model
    ee_trp = ee_regression(theta=beta,
                           X=W,
                           y=s,
                           model='logistic')
    pi_s = inverse_logit(np.dot(W, beta))

    # Creating IPW
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))

    # Estimating equations for parameters of interest
    ee_r1 = np.where(s == 1, 0, y - r1) * iosw * (a == 1) * iptw
    ee_r0 = np.where(s == 1, 0, y - r0) * iosw * (a == 0) * iptw
    ee_rd = np.ones(W.shape[0]) * (r1 - r0) - psi

    return np.vstack([ee_rd,
                     ee_r1,
                     ee_r0,
                     ee_trp,
                     ee_act])
