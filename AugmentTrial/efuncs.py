########################################################################################################################
# Combining Randomized and Observational Data: The Case of Postnatal Prophylaxis Against HIV
#       - Estimating functions to call in the simulation study
#
# Paul Zivich (2026/10/26)
########################################################################################################################

# Setup the environment
import numpy as np
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


def ee_naive(theta, y, a, s):
    """Trial-only risk ratio estimator"""
    rd, lnrr, r1, r0 = theta     # Parameters

    # Risks
    ee_r1 = (y - r1) * a                                             # Trial-only risk A=1
    ee_r0 = (y - r0) * s * (1-a)                                     # Trial-only risk A=0
    ee_rd = np.ones(y.shape[0])*(r1 - r0) - rd                       # Trial-only RD
    ee_rr = np.ones(y.shape[0])*(np.log(r1) - np.log(r0)) - lnrr     # Trial-only log-RR

    # Returning stacked functions
    return np.vstack([ee_rd, ee_rr, ee_r1, ee_r0])


def ee_naive_pool(theta, y, a):
    """Naive-pooling risk ratio estimator"""
    rd, lnrr, r1, r0 = theta

    # Risks
    ee_r1 = (y - r1) * a                                             # Trial-only risk A=1
    ee_r0 = (y - r0) * (1-a)                                         # Trial-only risk A=0
    ee_rd = np.ones(y.shape[0])*(r1 - r0) - rd                       # Trial-only RD
    ee_rr = np.ones(y.shape[0])*(np.log(r1) - np.log(r0)) - lnrr     # Trial-only log-RR

    # Returning stacked functions
    return np.vstack([ee_rd, ee_rr, ee_r1, ee_r0])


def ee_aipw_wr(theta, y, a, s, W, X):
    """Augmented inverse probability weighting estimator for pooled control arm. This version uses the
    weighted-regression implementation.
    """
    # Pre-processing the inputs
    idW = W.shape[1]                 # Design matrix dimensions
    rd, lnrr, r1, r0 = theta[:4]     # Parameters of interest
    alpha = theta[4:4+idW]           # Nuisance parameters for the sampling weights
    beta = theta[4+idW:]             # Nuisance parameters for the outcome model

    # Sample nuisance model
    ee_smp = ee_regression(theta=alpha, X=W, y=s, model='logistic')  # Logistic model for sampling score
    pi_s = inverse_logit(np.dot(W, alpha))                           # Predicted probability of sampling (score)
    iosw = s + ((1-s) * pi_s / (1-pi_s))                             # Constructing the inverse odds weights

    # Outcome nuisance model
    ee_out = ee_regression(theta=beta, X=X, y=y, model='logistic', weights=iosw) * (1-a)  # Outcome model fit among A=0
    y0hat = inverse_logit(np.dot(X, beta))                                                # Predictions from model

    # Risks
    ee_r1 = (y - r1) * s * a                                       # Trial-only A=1 risk estimating function
    ee_r0 = (y0hat - r0) * s                                       # Pooled risk under A=0 estimating function
    ee_rd = np.ones(y.shape[0])*(r1 - r0) - rd                     # RD estimating function
    ee_rr = np.ones(y.shape[0])*(np.log(r1) - np.log(r0)) - lnrr   # Log-RR estimating function

    # Returning stacked functions
    return np.vstack([ee_rd,  ee_rr,
                      ee_r1, ee_r0, ee_smp, ee_out])


def fit_mestimator(psi, starting_values, scenario=1):
    """Utility function for fitting the M-estimators and processing their output results for simulation studies. This
    handles cases where root-finding doesn't converge nicely.
    """
    # Pulling the true values for each scenario
    if scenario == 1:
        true_rd, true_rr = -0.01, 0.5
    elif scenario == 2:
        true_rd, true_rr = -0.05, 0.5
    elif scenario == 3:
        true_rd, true_rr = 0., 1.
    else:
        raise ValueError("No scenario specifications available beyond 1,2,3.")

    # Fitting the corresponding M-estimator and calculating simulation stats
    try:
        estr = MEstimator(psi, init=starting_values)   # Fitting M-estimator
        estr.estimate(maxiter=50000)                   # ... lots of possible iterations as max
        ests = estr.theta                              # ... extract the relevant coefficients
        stderr = np.sqrt(np.diag(estr.variance))       # ... get the standard errors
        confint = estr.confidence_intervals()          # ... the confidence intervals
        pvals = estr.p_values()                        # ... and the P-values

        # Organizing the output results for the iteration
        output = [ests[0] - true_rd, stderr[0], confint[0, 0], confint[0, 1],
                  confint[0, 0] <= true_rd <= confint[0, 1], pvals[0],
                  ests[1] - np.log(true_rr), stderr[1], confint[1, 0], confint[1, 1],
                  confint[1, 0] <= np.log(true_rr) <= confint[1, 1], pvals[1]]

    # When something fails, return NaN nicely organized for the outputs
    except RuntimeError:
        output = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    # Return the result-formatted row of the simulation iteration
    return output
