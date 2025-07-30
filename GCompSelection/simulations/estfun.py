#####################################################################################################################
# Estimating functions for the described g-computation estimators
#####################################################################################################################

import numpy as np
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


def psi_standard_gcomp(theta, y, s, X, X1, X0):
    X = np.asarray(X)
    X1 = np.asarray(X1)
    X0 = np.asarray(X0)
    y = np.asarray(y)
    s = np.asarray(s)

    rd, r1, r0 = theta[0:3]
    beta = theta[3:]

    # Outcome nuisance model
    ee_out = ee_regression(beta, X=X, y=y, model='logistic')
    ee_out = ee_out * s
    y1hat = inverse_logit(np.dot(X1, beta))
    y0hat = inverse_logit(np.dot(X0, beta))

    # Risk functions
    ee_r1 = y1hat - r1
    ee_r0 = y0hat - r0
    ee_rd = np.ones(y.shape) * (r1 - r0 - rd)

    return np.vstack([ee_rd, ee_r1, ee_r0, ee_out])


def psi_gcomp_conditional(theta, y, a, s, X, X1, X0):
    X = np.asarray(X)
    X1 = np.asarray(X1)
    X0 = np.asarray(X0)
    y = np.asarray(y)
    a = np.asarray(a)
    s = np.asarray(s)

    rd, r1, r0 = theta[0:3]
    beta = theta[3:]

    # Outcome nuisance model
    ee_out = ee_regression(beta, X=X, y=y, model='logistic')
    ee_out = ee_out * s
    y1hat = inverse_logit(np.dot(X1, beta))
    y0hat = inverse_logit(np.dot(X0, beta))

    # Risk functions
    ee_r1 = a*(y1hat - r1)
    ee_r0 = (1-a)*(y0hat - r0)
    ee_rd = np.ones(y.shape) * (r1 - r0 - rd)

    return np.vstack([ee_rd, ee_r1, ee_r0, ee_out])


def psi_gcomp_nested(theta, y, s, X, X1, X0, W, W1, W0):
    X = np.asarray(X)
    X1 = np.asarray(X1)
    X0 = np.asarray(X0)
    W = np.asarray(W)
    W1 = np.asarray(W1)
    W0 = np.asarray(W0)
    y = np.asarray(y)
    s = np.asarray(s)

    idX = 3 + X.shape[1]
    idW = idX + W.shape[1]

    rd, r1, r0 = theta[0:3]
    beta = theta[3:idX]
    gamma_1 = theta[idX: idW]
    gamma_0 = theta[idW:]

    # Inner outcome nuisance model
    ee_inner = ee_regression(beta, X=X, y=y, model='logistic')
    ee_inner = ee_inner * s
    y1hat = inverse_logit(np.dot(X1, beta))
    y0hat = inverse_logit(np.dot(X0, beta))

    # Outer outcome nuisance model
    ee_outer1 = ee_regression(gamma_1, X=W, y=y1hat, model='logistic')
    ee_outer0 = ee_regression(gamma_0, X=W, y=y0hat, model='logistic')

    y1hat_outer = inverse_logit(np.dot(W1, gamma_1))
    y0hat_outer = inverse_logit(np.dot(W0, gamma_0))

    # Risk functions
    ee_r1 = y1hat_outer - r1
    ee_r0 = y0hat_outer - r0
    ee_rd = np.ones(y.shape) * (r1 - r0 - rd)

    return np.vstack([ee_rd, ee_r1, ee_r0, ee_inner, ee_outer1, ee_outer0])


def psi_ipw_case1(theta, y, a, s, W):
    W = np.asarray(W)
    y = np.asarray(y)
    s = np.asarray(s)

    rd, r1, r0 = theta[0:3]
    alpha = theta[3:]

    # Nuisance model for weights
    ee_log = ee_regression(alpha, y=s, X=W, model='logistic')

    # Computing the weights
    pi_s = inverse_logit(np.dot(W, alpha))
    ipw = s / pi_s

    # Weighted means for risk functions
    ee_r1 = a * ipw * (y - r1)
    ee_r0 = (1-a) * ipw * (y - r0)
    ee_rd = np.ones(y.shape) * (r1 - r0 - rd)

    return np.vstack([ee_rd, ee_r1, ee_r0, ee_log])


def psi_ipw_case2(theta, y, a, s, W, Z):
    W = np.asarray(W)
    Z = np.asarray(Z)
    y = np.asarray(y)
    s = np.asarray(s)

    idW = 3 + W.shape[1]

    rd, r1, r0 = theta[0:3]
    alpha = theta[3:idW]
    gamma = theta[idW:]

    # Nuisance model for selection
    ee_slc = ee_regression(alpha, y=s, X=W, model='logistic')

    # Nuisance model for confounding
    ee_act = ee_regression(gamma, y=a, X=Z, model='logistic')

    # Computing the weights
    pi_s = inverse_logit(np.dot(W, alpha))
    pi_a = inverse_logit(np.dot(Z, gamma))
    ipmw = s / pi_s
    iptw = a / pi_a + (1-a)/(1-pi_a)
    ipw = ipmw * iptw

    # Weighted means for risk functions
    ee_r1 = a * ipw * (y - r1)
    ee_r0 = (1-a) * ipw * (y - r0)
    ee_rd = np.ones(y.shape) * (r1 - r0 - rd)

    return np.vstack([ee_rd, ee_r1, ee_r0, ee_slc, ee_act])

