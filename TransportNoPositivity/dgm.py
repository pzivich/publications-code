#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Data generating mechanisms
#
# Paul Zivich
#######################################################################################################################


import numpy as np
import pandas as pd
from scipy.stats import logistic
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression

from helper import trapezoid


def calculate_truth(n):
    """Function to calculate the true average causal effect by simulating the clinic population and potential outcomes
    under treatment and no treatment.

    Parameters
    ----------
    n : int
        Number of observations

    Returns
    -------
    float
    """
    d = generate_clinic(n=n)
    ya1 = generate_potential_outcomes(data=d, a=1)
    ya0 = generate_potential_outcomes(data=d, a=0)
    return np.mean(ya1) - np.mean(ya0)


def generate_data(n1, n0):
    """Generate observations of clinic and trial data for simulations.

    Parameters
    ----------
    n1 : int
        Observations in the clinic data.
    n0 : int
        Observations in the trial data

    Returns
    -------
    pandas.DataFrame
    """
    d1 = generate_clinic(n=n1)
    d0 = generate_trial(n=n0)
    d = pd.concat([d1, d0], ignore_index=True)     # Stacking data together
    d['V_i25'] = np.where(d['V'] > 25, d['V'], 0)  # Creating an indicator for age above 25
    d['intercept'] = 1                             # Creating an intercept term
    return d


def generate_background_info(marginal, n_secret, third_population=False):
    """Function to conduct the secret trial and return the estimated parameters.

    Parameters
    ----------
    marginal : bool
        Whether to return delta (marginal parameters, True) or beta (conditional parameters, False)
    n_secret : int
        Number of observations in the secret trial.
    third_population : bool, optional
        Whether the trial comes from a third population (different age distribution from both the trial and clinic).

    Returns
    -------
    list
    """
    # Generating secret study data
    if third_population:                                      # Population that the secret trial was conducted in
        d = generate_third(n=n_secret)                        # ... generating baseline covariate for secret trial
    else:
        d = generate_clinic(n=n_secret)                       # ... generating baseline covariate for secret trial
    d['A'] = np.random.binomial(n=1, p=0.5, size=n_secret)    # Randomly assigning treatment in the secret trial
    d['AW'] = d['A']*d['W']                                   # Creating A-W interaction term
    ya1 = generate_potential_outcomes(data=d, a=1)            # Generating potential outcomes under a=1
    ya0 = generate_potential_outcomes(data=d, a=0)            # Generating potential outcomes under a=0
    d['Y'] = np.where(d['A'] == 1, ya1, ya0)                  # Causal consistency
    d['intercept'] = 1                                        # Adding intercept term

    # Logic for estimating functions to use
    if marginal:
        # Parameters for IPW model synthesis
        def psi(theta):
            return ee_regression(theta=theta,
                                 X=d[['W', 'AW', 'intercept', 'A']],
                                 y=d['Y'],
                                 model='logistic')

        init_vals = [0, 0, 0, 0]
    else:
        # Parameters for g-computation model synthesis
        def psi(theta):
            return ee_regression(theta=theta,
                                 X=d[['W', 'AW', 'intercept', 'A', 'V']],
                                 y=d['Y'],
                                 model='logistic')

        init_vals = [0, 0, 0, 0, 0]

    # M-estimator for the secret trial and the corresponding models
    estr = MEstimator(psi, init=init_vals)
    estr.estimate(solver='lm', maxiter=2000)

    # Return the W, AW parameters only and the covariance matrix
    return estr.theta[0:2], estr.variance[0:2, 0:2]


def generate_clinic(n):
    """Generates the baseline covariate distribution for the clinic

    Parameters
    ----------
    n : int
        Number of observations to generate

    Returns
    -------
    pandas.DataFrame
    """
    d = pd.DataFrame()
    d['V'] = trapezoid(mini=18, mode1=18, mode2=25, maxi=30, size=n)
    d['V'] = np.round(d['V'])
    d['W'] = np.random.binomial(n=1, p=0.667, size=n)
    d['S'] = 1
    return d


def generate_trial(n):
    """Generates the baseline covariate distribution for the trial

    Parameters
    ----------
    n : int
        Number of observations to generate

    Returns
    -------
    pandas.DataFrame
    """
    d = pd.DataFrame()
    d['V'] = trapezoid(mini=18, mode1=25, mode2=30, maxi=30, size=n)
    d['V'] = np.round(d['V'])
    d['W'] = 0
    d['A'] = np.random.binomial(n=1, p=0.5, size=n)
    ya1 = generate_potential_outcomes(data=d, a=1)
    ya0 = generate_potential_outcomes(data=d, a=0)
    d['Y'] = np.where(d['A'] == 1, ya1, ya0)
    d['S'] = 0
    return d


def generate_third(n):
    """Generates the baseline covariate distribution for a third population (used only for the secret trial for a
    different age distribution population).

    Parameters
    ----------
    n : int
        Number of observations to generate

    Returns
    -------
    pandas.DataFrame
    """
    d = pd.DataFrame()
    d['V'] = trapezoid(mini=18, mode1=29, mode2=30, maxi=30, size=n)
    d['V'] = np.round(d['V'])
    d['W'] = np.random.binomial(n=1, p=0.667, size=n)
    d['S'] = 1
    return d


def generate_potential_outcomes(data, a):
    """Generates the potential outcomes for data under A=a

    Parameters
    ----------
    data : pandas.DataFrame
        Data set of observations to generate potential outcomes for
    a : int
        Action or treatment to generate potential outcomes under

    Returns
    -------
    np.array
    """
    logit = -3.25 + 1.50*a - 0.65*a*data['W'] - 0.02*data['W'] + 0.08*data['V']
    ya = np.random.binomial(n=1,
                            p=logistic.cdf(logit),
                            size=data.shape[0])
    return ya
