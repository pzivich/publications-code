######################################################################################################################
# Implementation of the estimating equations pooled logistic model compatible with delicatessen
#
# Paul Zivich (Last update: 2025/4/17)
######################################################################################################################

import numpy as np
from delicatessen.utilities import inverse_logit
from delicatessen.estimating_equations.processing import generate_weights


def survival_to_measure(survival, measure):
    r"""Function to convert from survival to other measures. Options include survival, risk (cumulative incidence), or
    cumulative hazard.

    Parameters
    ----------
    survival : float, int, ndarray
        Survival values to convert to other measure.
    measure : str
        Measure to convert input to. Options include: ``'survival'``, ``'risk'``, ``'chazard'``.

    Returns
    -------
    Transformed values.
    """
    measure = measure.lower()
    if measure == 'survival':
        return survival
    elif measure == 'risk':
        return 1 - survival
    elif measure == 'chazard':
        return -1 * np.log(survival)
    else:
        raise ValueError("The measure `" + str(measure) + "` is not supported")


def ee_pooled_logit(theta, t, delta, X, S=None, unique_times=None, weights=None):
    r"""Estimating equations for a pooled logistic regression model for time-to-event data.

    Parameters
    ----------
    theta : ndarray, list, vector
        Parameter vector for the pooled logistic model. Composed of the parameters for the baseline covariates and the
        time coefficients.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times.
    delta : ndarray, list, vector
        1-dimensional vector of `n` event indicators, where 1 indicates an event and 0 indicates right censoring.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    S : ndarray, None, optional
        Optional argument for parametric function form specifications for time. Default is ``None``, which uses disjoint
        indicators to model time. Expected to have ``np.max(t)`` rows.
    unique_times : None, ndarray, list, vector, optional
        Optional argument to compute the disjoint indicators for only a subset of terms. This argument is intended for
        use with disjoint indicators for time that are stratified by some external variable. Otherwise, this argument
        should not be used. This argument is also ignored when ``S`` is not ``None``.
    weights : None, ndarray, list, vector, optional
        Optional argument to implement a weighted pooled logistic regression model. Input weights can be either
        1-dimensional (time-fixed weights) or 2-dimensional (time-varying weights). Note that if a 2-dimensional matrix
        is supplied, it must be of dimension `n` by the unique time intervals (differs by whether ``S`` is specified).

    Returns
    -------
    array :
        Returns a (`b`+`k`)-by-`n` NumPy array evaluated for the input ``theta``.
    """
    # Pre-processing input data
    t = np.asarray(t)                 # Convert to NumPy array
    delta = np.asarray(delta)         # Convert to NumPy array
    X = np.asarray(X)                 # Convert to NumPy array
    xp = X.shape[1]                   # Get shape of X array to divide parameter vector
    beta_x = theta[:xp]               # Beta parameters for X design matrix
    beta_s = np.asarray(theta[xp:])   # Beta parameters for S design matrix

    if S is None:
        if unique_times is None:
            event_times = t[delta == 1]
            unique_times = np.unique(event_times)
        else:
            unique_times = np.asarray(unique_times)
        n_time_steps = unique_times.shape[0]

        # Creating design matrix for time
        time_design_matrix = np.identity(n=len(unique_times))
        time_design_matrix[:, 0] = 1
    else:
        time_design_matrix = np.asarray(S)
        unique_times = np.asarray(range(1, int(np.max(t))+1, 1))
        n_time_steps = len(unique_times)       #
        if n_time_steps != time_design_matrix.shape[0]:
            raise ValueError("A total of " + str(unique_times) + " unit-time intervals were created based on the "
                             "input times, but the specific time design matrix has "
                             + str(time_design_matrix.shape[0]) + " rows. These values are expected to match")

    # Log-odds contributions for covariate and time
    log_odds_w = np.dot(X, beta_x)
    log_odds_t = np.dot(time_design_matrix, beta_s)

    # Computing residuals
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))       # Stacked copies of X contributions for intervals
    y_obs = delta * (t == unique_times[:, None]).astype(int)         # Event indicator at time intervals matrix
    y_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None])  # Predicted event at time intervals matrix
    in_risk_set = (t >= unique_times[:, None]).astype(int)           # Indicator if individual is in the risk set at k
    residual_matrix = (y_obs - y_pred) * in_risk_set                 # Computing residuals at time intervals matrix

    # Incorporating specified weights
    weights = generate_weights(weights, n_obs=t.shape[0])            # Pre-processing weight argument
    if weights.ndim == 2:
        if weights.shape[1] != n_time_steps or weights.shape[0] != t.shape[0]:
            raise ValueError("If a 2D weight matrix is provided, it must (1) have the same number of rows as "
                             "observations, and (2) match the number of time points. A total of "
                             + str(t.shape[0]) + " observations were provided with "
                             + str(n_time_steps) + " time intervals, but the weight "
                             "matrix was " + str(weights.shape))
        weights = weights.T
    residual_matrix = residual_matrix * weights                      # Multiplying residuals by weights prior to sum

    # Getting score matrix for X
    n_ones = np.ones(shape=(1, n_time_steps))       # Vector of ones to ease cumulative sum across time intervals
    y_resid = np.dot(n_ones, residual_matrix)[0]    # Adding together residual contributions across all time intervals
    x_score = y_resid[:, None] * X                  # Compute the score for the current interval for X

    # Getting score matrix for S
    if S is None:                                  # If using disjoint indicators for time
        t_score = residual_matrix                  # ... simply return the residual_matrix
    else:                                          # Otherwise
        t_score = np.dot(S.T, residual_matrix)     # ... matrix multiplication of time design matrix with residuals

    # Overall score matrix stacked together
    score_plogit = np.vstack([x_score.T, t_score])

    # Returning the score function for the stacked sums
    return score_plogit


def pooled_logit_prediction(theta, t, delta, X, S=None, times_to_predict=None, measure='survival', unique_times=None):
    r"""

    Parameters
    ----------
    theta : ndarray, list, vector
        Estimated parameter vector for the pooled logistic model. Composed of the parameters for the baseline
        covariates and the time coefficients. These should be the values optimized by ``ee_pooled_logit``.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times.
    delta : ndarray, list, vector
        1-dimensional vector of `n` event indicators, where 1 indicates an event and 0 indicates right censoring.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    S : ndarray, None, optional
        Optional argument for parametric function form specifications for time. Default is ``None``, which uses disjoint
        indicators to model time. Expected to have ``np.max(t)`` rows.
    times_to_predict : int, float, ndarray, list, vector, None, optional
        Time(s) to generate predicted values for. Specified times must be :math:`[0, \tau]`. Default is ``None``, which
        generates predicted values at each unique event time (if ``S=None``) or at each unit-time interval (``S!=None``)
    measure : str, optional
        Measure to generate predicted values of. Options are ``'survival'``, ``'risk'``, ``'chazard'``.
    unique_times : None, ndarray, list, vector, optional
        Optional argument to compute the disjoint indicators for only a subset of terms. This argument is intended for
        use with disjoint indicators for time that are stratified by some external variable. Otherwise, this argument
        should not be used. This argument is also ignored when ``S`` is not ``None``.

    Returns
    -------
    array :
        Returns a `t`-by-`n` NumPy array evaluated for the input ``theta``.
    """
    # Pre-processing input data
    t = np.asarray(t)                 # Convert to NumPy array
    delta = np.asarray(delta)         # Convert to NumPy array
    X = np.asarray(X)                 # Convert to NumPy array
    xp = X.shape[1]                   # Get shape of X array to divide parameter vector
    beta_x = theta[:xp]               # Beta parameters for X design matrix
    beta_s = np.asarray(theta[xp:])   # Beta parameters for S design matrix

    if S is None:
        if unique_times is None:
            event_times = t[delta == 1]
            unique_times = np.unique(event_times)
        else:
            unique_times = np.asarray(unique_times)
        n_time_steps = unique_times.shape[0]

        # Creating design matrix for time
        time_design_matrix = np.identity(n=len(unique_times))
        time_design_matrix[:, 0] = 1
    else:
        time_design_matrix = np.asarray(S)
        unique_times = np.asarray(range(1, int(np.max(t))+1, 1))
        n_time_steps = len(unique_times)       #

    # Log-odds contributions for covariate and time
    log_odds_w = np.dot(X, beta_x)
    log_odds_t = np.dot(time_design_matrix, beta_s)

    # # Computing full matrix of predicted values for each time
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))       # Stacked copies of X contributions for intervals
    y_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None])  # Predicted event at time intervals matrix
    survival_prediction = np.cumprod(1 - y_pred, axis=0)
    prediction_matrix = survival_to_measure(survival_prediction, measure=measure)
    prediction_t0 = survival_to_measure(1, measure=measure)

    #
    if times_to_predict is None:
        return prediction_matrix
    else:
        predictions = []
        for time in times_to_predict:
            if time == 0 or time < unique_times[0]:
                prediction = np.ones(t.shape[0]) * prediction_t0
            elif time > np.max(t):
                raise ValueError("Cannot predict beyond the maximum observed time")
            else:
                if unique_times[-1] <= time:
                    pred_matrix_index = -1
                else:
                    further_times = unique_times[time < unique_times]
                    if len(further_times) < 1:
                        nearest = unique_times[time >= unique_times][-1]  # looks at jump point before
                    else:
                        nearest = unique_times[time < unique_times][0]  # Looks at jump point after
                    pred_matrix_index = np.where(unique_times == nearest)[0][0] - 1
                prediction = prediction_matrix[pred_matrix_index, :]
            predictions.append(prediction)
        return np.asarray(predictions)
