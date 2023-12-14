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


def ee_synth_aipw_msm(theta, y, a, s, r, c, Z, W, X, MSM, math_contribution, model='linear'):
    """Synthesis AIPW model estimating functions based on marginal structural models.

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
    r :
        Indicator of observations in the positive region
    c :
        Indicator for the original (non-duplicated) observations
    Z :
        Design matrix for propensity score model
    W :
        Design matrix for the sampling model
    X :
        Design matrix for the outcome model
    MSM :
        Design matrix for the marginal structural model
    math_contribution :
        Contribution of the mathematical model to each observation
    model :
        Distribution of Y for the outcome model
    """
    idxC = MSM.shape[1]           # MSM parameter number
    idxZ = Z.shape[1] + idxC       # IPTW model parameter number
    idxW = W.shape[1] + idxZ       # IOSW model parameter number
    psi, r1, r0 = theta[0:3]       # Parameters of interest
    alpha = theta[3:3+idxC]        # Nuisance Paramters: CACE
    eta1 = theta[3+idxC:3+idxZ]    # Nuisance parameters: Pr(A|R=0)
    eta2 = theta[3+idxZ: 3+idxW]   # Nuisance parameters: Pr(R|V,W)
    eta3 = theta[3+idxW:]          # Nuisance parameters: Pr(Y|A,V,W,R=0)

    # Estimating nuisance model: action process
    ee_act = ee_regression(theta=eta1,
                           X=Z,
                           y=a,
                           model='logistic')
    ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
    ee_act = ee_act * (1-s) * (1-c)
    pi_a = inverse_logit(np.dot(Z, eta1))

    # Estimating nuisance model: sample process
    ee_sam = ee_regression(theta=eta2,
                           X=W,
                           y=s,
                           model='logistic')
    ee_sam = ee_sam * r * (1-c)
    pi_s = inverse_logit(np.dot(W, eta2))

    # Computing the overall weights
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))
    ipw = iptw*iosw

    # Estimating nuisance model: outcome process
    ee_out = ee_regression(theta=eta3,
                           X=X,
                           y=y,
                           model=model,
                           weights=ipw)
    ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)
    ee_out = ee_out * (1-s)
    if model == 'logistic':
        yhat = inverse_logit(np.dot(X, eta3))
    else:
        yhat = np.dot(X, eta3)

    # Marginal structural model with AIPW pseudo-outcomes
    ee_msm = ee_regression(theta=alpha,
                           y=yhat,
                           X=MSM,
                           model=model)
    ee_msm = np.nan_to_num(ee_msm, copy=False, nan=0.)
    ee_msm = ee_msm * c * r

    # Estimating equations for parameters of interest
    stat_contribution = np.dot(MSM, alpha)
    if model == 'logistic':
        pred_y = inverse_logit(stat_contribution + math_contribution)
    else:
        pred_y = stat_contribution + math_contribution
    ee_r1 = np.where(a == 1, pred_y - r1, 0) * s
    ee_r0 = np.where(a == 0, pred_y - r0, 0) * s
    ee_ace = np.ones(y.shape[0])*(r1 - r0 - psi)

    return np.vstack([ee_ace, ee_r1, ee_r0, ee_msm,
                      ee_act, ee_sam, ee_out])


def ee_synth_msm_only(theta, y, a, s, r, Z, W, X, X1, X0, M1, M0, model='linear'):
    """Statistical MSM for the synthesis AIPW model based on the MSM. This estimating equation estimates the MSM for
    the positive region(s).

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
    r :
        Indicator of observations in the positive region
    Z :
        Design matrix for propensity score model
    W :
        Design matrix for the sampling model
    X :
        Design matrix for the outcome model
    MSM :
        Design matrix for the marginal structural model
    model :
        Distribution of Y for the outcome model
    """
    idxC = M1.shape[1]            # MSM parameter number
    idxZ = Z.shape[1] + idxC       # IPTW model parameter number
    idxW = W.shape[1] + idxZ       # IOSW model parameter number
    alpha = theta[0:idxC]          # Nuisance Paramters: CACE
    eta1 = theta[idxC:idxZ]        # Nuisance parameters: Pr(A|R=0)
    eta2 = theta[idxZ: idxW]       # Nuisance parameters: Pr(R|V,W)
    eta3 = theta[idxW:]            # Nuisance parameters: Pr(Y|A,V,W,R=0)

    # Estimating nuisance model: action process
    ee_act = ee_regression(theta=eta1,
                           X=Z,
                           y=a,
                           model='logistic')
    ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
    ee_act = ee_act * (1-s)
    pi_a = inverse_logit(np.dot(Z, eta1))

    # Estimating nuisance model: sample process
    ee_sam = ee_regression(theta=eta2,
                           X=W,
                           y=s,
                           model='logistic')
    ee_sam = ee_sam * r
    pi_s = inverse_logit(np.dot(W, eta2))

    # Computing the overall weights
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))
    ipw = iptw*iosw

    # Estimating nuisance model: outcome process
    ee_out = ee_regression(theta=eta3,
                           X=X,
                           y=y,
                           model=model,
                           weights=ipw)
    ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)
    ee_out = ee_out * (1-s)
    if model == 'logistic':
        y1hat = inverse_logit(np.dot(X1, eta3))
        y0hat = inverse_logit(np.dot(X0, eta3))
    else:
        y1hat = np.dot(X1, eta3)
        y0hat = np.dot(X0, eta3)

    # Marginal structural model with AIPW pseudo-outcomes
    ee_msm1 = ee_regression(theta=alpha, y=y1hat, X=M1, model=model)
    ee_msm0 = ee_regression(theta=alpha, y=y0hat, X=M0, model=model)
    ee_msm = ee_msm1 + ee_msm0
    ee_msm = np.nan_to_num(ee_msm, copy=False, nan=0.)
    ee_msm = ee_msm * r * s

    # Returning the stacked estimating equations
    return np.vstack([ee_msm, ee_act, ee_sam, ee_out])


def ee_synth_aipw_cace(theta, y, a, s, r, Z, W, X, Xa1, Xa0, CACE, math_contribution, model='linear'):
    """Synthesis AIPW model estimating functions based on conditional average causal effect models.

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
    r :
        Indicator of observations in the positive region
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
    CACE :
        Design matrix for the conditional average causal effect model
    math_contribution :
        Contribution of the mathematical model to each observation
    model :
        Distribution of Y for the outcome model
    """
    idxC = CACE.shape[1]           # MSM parameter number
    idxZ = Z.shape[1] + idxC       # IPTW model parameter number
    idxW = W.shape[1] + idxZ       # IOSW model parameter number
    psi = theta[0]                 # Parameters of interest
    gamma = theta[1:1+idxC]        # Nuisance Paramters: CACE
    eta1 = theta[1+idxC:1+idxZ]    # Nuisance parameters: Pr(A|R=0)
    eta2 = theta[1+idxZ: 1+idxW]   # Nuisance parameters: Pr(R|V,W)
    eta3 = theta[1+idxW:]          # Nuisance parameters: Pr(Y|A,V,W,R=0)

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
                           model='logistic') * r
    pi_s = inverse_logit(np.dot(W, eta2))

    # Computing the overall weights
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))
    ipw = iptw*iosw

    # Estimating nuisance model: outcome process
    ee_out = ee_regression(theta=eta3,
                           X=X,
                           y=y,
                           model=model,
                           weights=ipw)
    ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

    # Generating pseudo-outcomes
    if model == 'logistic':
        y1hat = inverse_logit(np.dot(Xa1, eta3))
        y0hat = inverse_logit(np.dot(Xa0, eta3))
    else:
        y1hat = np.dot(Xa1, eta3)
        y0hat = np.dot(Xa0, eta3)

    # Estimating nuisance model: conditional average causal effect
    ee_cace = ee_regression(theta=gamma,
                            X=CACE,
                            y=y1hat - y0hat,
                            model='linear')
    ee_cace = np.nan_to_num(ee_cace, copy=False, nan=0.)
    ee_cace = ee_cace * r * s

    # Estimating equations for parameters of interest
    ee_ace = s * (np.dot(CACE, gamma) + math_contribution - psi)

    return np.vstack([ee_ace, ee_cace,
                     ee_act, ee_sam, ee_out])


def ee_synth_cace_only(theta, y, a, s, r, Z, W, X, Xa1, Xa0, CACE, model='linear'):
    """Synthesis AIPW model estimating functions based on conditional average causal effect models.

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
    r :
        Indicator of observations in the positive region
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
    CACE :
        Design matrix for the conditional average causal effect model
    math_contribution :
        Contribution of the mathematical model to each observation
    model :
        Distribution of Y for the outcome model
    """
    idxC = CACE.shape[1]           # MSM parameter number
    idxZ = Z.shape[1] + idxC       # IPTW model parameter number
    idxW = W.shape[1] + idxZ       # IOSW model parameter number
    gamma = theta[0:idxC]          # Nuisance Paramters: CACE
    eta1 = theta[idxC: idxZ]       # Nuisance parameters: Pr(A|R=0)
    eta2 = theta[idxZ: idxW]       # Nuisance parameters: Pr(R|V,W)
    eta3 = theta[idxW:]            # Nuisance parameters: Pr(Y|A,V,W,R=0)

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
                           model='logistic') * r
    pi_s = inverse_logit(np.dot(W, eta2))

    # Computing the overall weights
    iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
    iosw = (pi_s / (1 - pi_s))
    ipw = iptw*iosw

    # Estimating nuisance model: outcome process
    ee_out = ee_regression(theta=eta3,
                           X=X,
                           y=y,
                           model=model,
                           weights=ipw)
    ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

    # Generating pseudo-outcomes
    if model == 'logistic':
        y1hat = inverse_logit(np.dot(Xa1, eta3))
        y0hat = inverse_logit(np.dot(Xa0, eta3))
    else:
        y1hat = np.dot(Xa1, eta3)
        y0hat = np.dot(Xa0, eta3)

    # Estimating nuisance model: conditional average causal effect
    ee_cace = ee_regression(theta=gamma,
                            X=CACE,
                            y=y1hat - y0hat,
                            model='linear')
    ee_cace = np.nan_to_num(ee_cace, copy=False, nan=0.)
    ee_cace = ee_cace * r * s

    return np.vstack([ee_cace, ee_act, ee_sam, ee_out])
