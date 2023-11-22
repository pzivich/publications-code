#####################################################################################################################
# Bespoke Instrumental Variable via two-stage regression as an M-estimator
#       Estimating function for Python simulations and example
#
# Paul Zivich (2023/11/22)
####################################################################################################################

import numpy as np
from delicatessen.estimating_equations import ee_regression


def ee_2sr_bsiv(theta, y, r, a, L):
    # Parameters
    idxL = L.shape[1] + 2
    beta = theta[0:2]        # Parameters of interest
    alpha = theta[2:idxL]    # Nuisance parameters for E[Y | L]
    gamma = theta[idxL:]     # Nuisance parameters for E[A | L]

    # Control arm model (E[Y | L])
    ee_control = ee_regression(alpha, L, y,        # Regression for Y given L
                               model='linear')     # ... via linear model
    ee_control = ee_control * (1 - r)              # Restricting to control arm for fit
    yhat = np.dot(L, alpha)                        # Predicted values for Y

    # Received treatment model (E[A | L, R=1])
    ee_receipt = ee_regression(gamma, L, a,        # Regression for A given L
                               model='linear')     # ... via linear model
    ee_receipt = ee_receipt * r                    # Restricting to non-control arm for fit
    ahat = np.dot(L, gamma)                        # Predicted values for A

    # Controlled direct effect
    X = np.asarray([np.ones(ahat.shape), ahat]).T  # Stacking a new design matrix together
    ee_cde = ee_regression(beta, X=X, y=y,         # Regression for Y given A-hat
                           model='linear',         # ... via linear model
                           offset=yhat)            # ... with Y-hat offset term
    ee_cde = ee_cde * r                            # Restricting to non-control arm for fit

    # Returning stacked estimating functions
    return np.vstack([ee_cde, ee_control, ee_receipt])
