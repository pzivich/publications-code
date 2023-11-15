#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Defining the estimating functions for each approach implemented
#
# Paul Zivich
#######################################################################################################################

import numpy as np
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


def ee_stat_aipw(theta, y, a, s, Z, W, X, Xa1, Xa0, outcome_model='linear'):
    """Statistical AIPW model based on the weighted regression implementation. Used by the restricted target population,
    restricted covariate set, and extrapolation approaches.

    Parameters
    ----------
    theta :
        Parameter vector
    y :
        Outcome array
    a :
        Action array
    s :
        Sample indicator array
    Z :
        Design matrix for propensity score model
    W :
        Design matrix for the sampling model
    X :
        Design matrix for the outcome model
    Xa1 :
        Design matrix for the outcome model with A set to 1
    Xa0 :
        Design matrix for the outcome model with A set to 0
    outcome_model :
        Distribution of Y for the outcome model
    """
    idxZ = Z.shape[1]            # IPTW model parameter number
    idxW = W.shape[1] + idxZ     # IOSW model parameter number
    psi, r1, r0 = theta[0:3]     # Parameters of interest
    eta1 = theta[3:3+idxZ]       # Nuisance parameters: Pr(A|R=0)
    eta2 = theta[3+idxZ:3+idxW]  # Nuisance parameters: Pr(R|V,W)
    eta3 = theta[3+idxW:]        # Nuisance parameters: Pr(Y|A,V,W,R=0)

    # Estimating nuisance model: action process
    ee_act = ee_regression(theta=eta1,
                           X=Z,
                           y=a,
                           model='logistic')
    ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
    pi_a = inverse_logit(np.dot(Z, eta1))

    # Estimating nuisance model: sample process
    ee_sam = ee_regression(theta=eta2,
                           X=W,
                           y=s,
                           model='logistic')
    pi_s = inverse_logit(np.dot(W, eta2))

    # Computing the overall weights
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))
    ipw = iptw*iosw

    # Estimating nuisance model: outcome process
    ee_out = ee_regression(theta=eta3,
                           X=X,
                           y=y,
                           model=outcome_model,
                           weights=ipw)
    ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)
    if outcome_model == 'logistic':
        y1hat = inverse_logit(np.dot(Xa1, eta3))
        y0hat = inverse_logit(np.dot(Xa0, eta3))
    else:
        y1hat = np.dot(Xa1, eta3)
        y0hat = np.dot(Xa0, eta3)

    # Estimating functions for parameters of interest
    ee_r1 = s*(y1hat - r1)
    ee_r0 = s*(y0hat - r0)
    ee_rd = np.ones(y.shape[0]) * (r1 - r0) - psi

    # Returning stacked estimating functions
    return np.vstack([ee_rd, ee_r1, ee_r0,
                     ee_act, ee_sam, ee_out])


def ee_stat_msm(theta, y, MSM, model='linear'):
    """Statistical marginal structural model for mathematical model information.

    Parameters
    ----------
    theta :
        Parameter vector
    y :
        Outcome array
    MSM :
        Design matrix for marginal structural model
    model :
        Distribution of Y for the outcome model
    """
    return ee_regression(theta=theta,
                         y=y,
                         X=MSM,
                         model=model)


def ee_stat_cace(theta, y, X, Xa1, Xa0, CACE, model='linear'):
    """Statistical conditional average causal effect model for mathematical model information.

    Parameters
    ----------
    theta :
        Parameter vector
    y :
        Outcome array
    X :
        Design matrix for the outcome model
    Xa1 :
        Design matrix for the outcome model with A set to 1
    Xa0 :
        Design matrix for the outcome model with A set to 0
    CACE :
        Design matrix for conditional average causal effect model
    model :
        Distribution of Y for the outcome model
    """
    idxC = CACE.shape[1]           # CACE parameter number
    gamma = theta[0:idxC]          # Nuisance Paramters: CACE
    eta = theta[idxC:]             # Nuisance parameters: Pr(A|R=0)

    # Estimating nuisance model: outcome process
    ee_out = ee_regression(theta=eta,
                           X=X,
                           y=y,
                           model=model)
    if model == 'logistic':
        y1hat = inverse_logit(np.dot(Xa1, eta))
        y0hat = inverse_logit(np.dot(Xa0, eta))
    else:
        y1hat = np.dot(Xa1, eta)
        y0hat = np.dot(Xa0, eta)

    # Estimating conditional average causal effect model
    ee_cace = ee_regression(theta=gamma,
                            X=CACE,
                            y=y1hat - y0hat,
                            model='linear')

    return np.vstack([ee_cace, ee_out])
