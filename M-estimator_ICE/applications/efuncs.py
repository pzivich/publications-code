####################################################################################################################
# Empirical sandwich variance estimator for iterated conditional expectation g-computation
#   Defining estimating functions
#
# Paul Zivich
####################################################################################################################

import numpy as np
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


def ee_ice_gformula(theta, y, X_array, Xa_array, stratified=None):
    """Iterated Conditional Expectation (ICE) g-computation for longitudinal (repeated measures) data of a binary
    outcome.

    Note
    ----
    This estimating equation only supports binary repeated measure data (i.e., it does not support survival data or
    other repeated measure outcome types).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 1+b values if ``X0`` is ``None``, and 3+b values if ``X0`` is not ``None``.
    y : ndarray, list, vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1.
    X_array : ndarray, list, vector
        A 1D vector of 2-dimensional design matrices of n observed values for b variables.
    Xa_array : ndarray, list, vector
        A 1D vector of 2-dimensional design matrices of n observed values for b variables under the plan of interest.
    stratified : None, ndarray, list, vector, optional
        Optional argument to implement the stratified ICE g-computation estimator. Default is ``None`` which implements
        the unstratified estimator. For the stratified estimator, provide 1D vector of 1-dimensional vectors of n values
        indicating if a unit followed the plan (1) or 0 otherwise at each time. Nuisance models are then fit to the
        corresponding subset.

    Examples
    --------

    >>> from delicatessen import MEstimator

    >>> def psi_ice(theta):
    >>>     return ee_ice_gformula(theta=theta, y=y2,
    >>>                            X_array=[X1, X0], Xa_array=[Xa1, Xa0])

    >>> estr_ice = MEstimator(psi_ice, init=[0.25,
    >>>                                      0., 0., 0., 0., 0.,
    >>>                                      0., 0., 0., ])
    >>> estr_ice.estimate(solver='lm')

    References
    ----------

    """
    # Setup for iterative model estimation
    mu = theta[0]              # First parameter (0) is for the causal risk
    theta_first = 1            # Second index (1) is start of the logistic model parameters
    y_star = y                 # First iteration uses the observed outcomes at final time
    ee_evald = []              # Storage for evaluated estimating equations

    # Checking dimensions of input design matrices
    t = len(X_array)                             # Number of distinct time points determined by X_array shape
    if t != len(Xa_array):                       # Check that design matrices match their shapes...
        raise ValueError("The input number of design "
                         "matrices (number of time points) does not "
                         "match between X_array "
                         "and Xa_array.")
    if stratified is not None:                   # If providing a stratification
        if t != len(stratified):                 # ... also check that those lengths match
            raise ValueError("The input number of design matrices "
                             "(number of time points) does not match "
                             "between X_array and stratified.")

    # Iterative Evaluation of the Estimating Equations
    #  ... by working through each time point
    #  ... provided input is expected to be in backwards order
    for i in range(t):
        # Pre-processing design matrices and parameters
        X = X_array[i]                           # Dimensions of the corresponding observed design matrix
        Xa = Xa_array[i]                         # Dimensions of the corresponding intervention design matrix
        if X.shape[1] != Xa.shape[1]:            # Checking dimensions match for more informative error from me
            raise ValueError("The dimensions of the design "
                             "matrices at index " +
                             str(i) + " do not align.")
        if stratified is not None:               # When stratified is provided
            under_plan = stratified[i]           # ... indicator if followed policy is used for restricting the model
        else:                                    # Otherwise
            under_plan = 1                       # ... use all observations to fit the outcome model
        theta_last = theta_first + X.shape[1]    # Find the last expected parameter for current model
        beta = theta[theta_first: theta_last]    # Extract the corresponding parameters from input theta

        # Estimating nested fractional logistic model
        ee_log = ee_regression(beta,                        # Regression model parameters
                               X=X,                         # ... design matrix to use in the model
                               y=y_star,                    # ... outcome (observed at Tau, or predicted at all others)
                               model='logistic')            # ... logistic model for BINARY endpoints
        ee_log = ee_log * under_plan                        # Whether the plan was followed up to current time
        ee_log = np.nan_to_num(ee_log, copy=False, nan=0.)  # Filling in any NaN (missing Y,A,W) with 0
        ee_evald.append(ee_log)                             # Adding to list of evaluated models

        # Generating predicted values and post-processing
        y_star = inverse_logit(np.dot(Xa, beta))            # Predicted values of Y with intervention design matrix
        theta_first = theta_last                            # Update what is considered the first parameter to extract

    # Causal mean evaluation
    ee_mean = y_star - mu             # Estimating equation for causal mean (simple mean of final predictions)
    ee_evald.insert(0, ee_mean)       # Adding causal mean estimating function at first index

    # Returning the stacked estimating equations
    return np.vstack(ee_evald)
