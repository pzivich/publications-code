####################################################################################################################
# Empirical sandwich variance estimator for iterated conditional expectation g-computation
#   Defining utility functions for simulations and applied example
#
# Paul Zivich
####################################################################################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import logistic


def generate_potential(n):
    """Function to generate the potential outcome data for all observations
    """
    d = pd.DataFrame()
    d['W0'] = np.random.binomial(n=1, p=0.5, size=n)

    d['Y1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.5 + 0.5 - 2*d['W0']), size=n)
    d['Y1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.5 + 0 - 2*d['W0']), size=n)
    d['W1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1 + d['W0'] - 1), size=n)
    d['W1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1 + d['W0'] - 0), size=n)

    d['Y2a1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.5 + 0.1 + 1.2 - 0.5*d['W0'] - 2*d['W1a1']), size=n)
    d['Y2a0a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.5 + 0.0 + 1.2 - 0.5*d['W0'] - 2*d['W1a0']), size=n)
    d['Y2a1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.5 + 0.1 + 0.0 - 0.5*d['W0'] - 2*d['W1a1']), size=n)
    d['Y2a0a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.5 + 0.0 + 0.0 - 0.5*d['W0'] - 2*d['W1a0']), size=n)

    d['W2a1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1 + d['W1a1'] + 0.5*d['W0'] - 0.2 - 1), size=n)
    d['W2a1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1 + d['W1a1'] + 0.5*d['W0'] - 0.2 - 0), size=n)
    d['W2a0a1'] = np.random.binomial(n=1, p=logistic.cdf(-1 + d['W1a0'] + 0.5*d['W0'] - 0.0 - 1), size=n)
    d['W2a0a0'] = np.random.binomial(n=1, p=logistic.cdf(-1 + d['W1a0'] + 0.5*d['W0'] - 0.0 - 0), size=n)

    d['Y3a1a1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 1.2 - 0.5*d['W1a1'] - 2*d['W2a1a1']),
                                       size=n)
    d['Y3a0a1a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 1.2 - 0.5*d['W1a0'] - 2*d['W2a0a1']),
                                       size=n)
    d['Y3a1a0a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 1.2 - 0.5*d['W1a1'] - 2*d['W2a1a0']),
                                       size=n)
    d['Y3a0a0a1'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 1.2 - 0.5*d['W1a0'] - 2*d['W2a0a0']),
                                       size=n)
    d['Y3a1a1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 0.0 - 0.5*d['W1a1'] - 2*d['W2a1a1']),
                                       size=n)
    d['Y3a0a1a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.1 + 0.0 - 0.5*d['W1a0'] - 2*d['W2a0a1']),
                                       size=n)
    d['Y3a1a0a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 0.0 - 0.5*d['W1a1'] - 2*d['W2a1a0']),
                                       size=n)
    d['Y3a0a0a0'] = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 0.0 + 0.0 + 0.0 - 0.5*d['W1a0'] - 2*d['W2a0a0']),
                                       size=n)
    return d


def generate_observed(data):
    """Function to transform the potential outcome data for all observations into the observed data.
    """
    d = data.copy()
    n = d.shape[0]
    d['A0'] = np.random.binomial(n=1, p=logistic.cdf(1 - 2.0*d['W0']), size=n)
    d['W1'] = np.where(d['A0'] == 1, d['W1a1'], d['W1a0'])
    d['A1'] = np.random.binomial(n=1, p=logistic.cdf(-1 - 0.2*d['W0'] - d['W1'] + 1.75*d['A0']), size=n)
    d['W2'] = np.where((d['A0'] == 1) & (d['A1'] == 1), d['W2a1a1'], np.nan)
    d['W2'] = np.where((d['A0'] == 1) & (d['A1'] == 0), d['W2a1a0'], d['W2'])
    d['W2'] = np.where((d['A0'] == 0) & (d['A1'] == 1), d['W2a0a1'], d['W2'])
    d['W2'] = np.where((d['A0'] == 0) & (d['A1'] == 0), d['W2a0a0'], d['W2'])
    d['A2'] = np.random.binomial(n=1, p=logistic.cdf(-1 - 0.2*d['W1'] - d['W2'] + 1.75*d['A1']), size=n)

    # Generating observed outcomes
    d['Y1'] = np.where(d['A0'] == 1, d['Y1a1'], np.nan)
    d['Y1'] = np.where(d['A0'] == 0, d['Y1a0'], d['Y1'])
    d['Y2'] = np.where((d['A0'] == 1) & (d['A1'] == 1),
                       d['Y2a1a1'], np.nan)
    d['Y2'] = np.where((d['A0'] == 0) & (d['A1'] == 1),
                       d['Y2a0a1'], d['Y2'])
    d['Y2'] = np.where((d['A0'] == 1) & (d['A1'] == 0),
                       d['Y2a1a0'], d['Y2'])
    d['Y2'] = np.where((d['A0'] == 0) & (d['A1'] == 0),
                       d['Y2a0a0'], d['Y2'])
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 1) & (d['A2'] == 1),
                       d['Y3a1a1a1'], np.nan)
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 1) & (d['A2'] == 0),
                       d['Y3a1a1a0'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 0) & (d['A2'] == 1),
                       d['Y3a1a0a1'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 1) & (d['A2'] == 1),
                       d['Y3a0a1a1'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 1) & (d['A1'] == 0) & (d['A2'] == 0),
                       d['Y3a1a0a0'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 1) & (d['A2'] == 0),
                       d['Y3a0a1a0'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 0) & (d['A2'] == 1),
                       d['Y3a0a0a1'], d['Y3'])
    d['Y3'] = np.where((d['A0'] == 0) & (d['A1'] == 0) & (d['A2'] == 0),
                       d['Y3a0a0a0'], d['Y3'])

    # Generating censoring
    d['C1'] = np.random.binomial(n=1, p=logistic.cdf(-2.5 - 0.5*d['A0']), size=n)
    d['Y1'] = np.where(d['C1'] == 1, np.nan, d['Y1'])

    d['C2'] = np.random.binomial(n=1, p=logistic.cdf(-2.5 - 0.5*d['A1']), size=n)
    d['W1'] = np.where((d['C2'] == 1) | (d['C1'] == 1), np.nan, d['W1'])
    d['A1'] = np.where((d['C2'] == 1) | (d['C1'] == 1), np.nan, d['A1'])
    d['Y2'] = np.where((d['C2'] == 1) | (d['C1'] == 1), np.nan, d['Y2'])

    d['C3'] = np.random.binomial(n=1, p=logistic.cdf(-2.5 - 0.5*d['A2']), size=n)
    d['W2'] = np.where((d['C1'] == 1) | (d['C2'] == 1) | (d['C3'] == 1), np.nan, d['W2'])
    d['A2'] = np.where((d['C1'] == 1) | (d['C2'] == 1) | (d['C3'] == 1), np.nan, d['A2'])
    d['Y3'] = np.where((d['C1'] == 1) | (d['C2'] == 1) | (d['C3'] == 1), np.nan, d['Y3'])

    return d[['W0', 'A0', 'Y1', 'W1', 'A1', 'Y2', 'W2', 'A2', 'Y3']].copy()


def rescale(variable):
    """Helper function to rescale a variable to a standard normal.
    """
    return (variable - np.mean(variable)) / np.std(variable)


def indicator_terms(data, variable, values):
    """Helper function to generate indicator term columns.
    """
    for val in values:
        is_na = np.where(data[variable].isna(), np.nan, 1)
        in_cat = (data[variable] == val)
        new_label = variable + "_" + str(val)
        data[new_label] = in_cat * is_na


def ice_point(y_t2, X_t1, X_t0, Xa_t1, Xa_t0):
    """Helper function to run point-estimation for ICE g-computation with statsmodels. This is used for the bootstrap
    implementation.
    """
    # Wave III
    logm = sm.GLM(endog=y_t2, exog=X_t1, family=sm.families.Binomial(),
                  missing='drop').fit()
    ystar1 = logm.predict(Xa_t1)

    # Wave I
    logm = sm.GLM(endog=ystar1, exog=X_t0, family=sm.families.Binomial(),
                  missing='drop').fit()
    ystar0 = logm.predict(Xa_t0)

    return np.mean(ystar0)


def ice_prevent_bootstrap(params):
    """Helper function to run variance-estimation for always-prevent ICE g-computation with statsmodels. This is used
    for the bootstrap implementation.
    """
    ids, d, da, cols1, cols0 = params
    df = d.iloc[ids].copy()
    dfa = da.iloc[ids].copy()
    y = np.asarray(df['htn_w4'])
    X1 = np.asarray(df[cols1])
    X0 = np.asarray(df[cols0])
    Xa1 = np.asarray(dfa[cols1])
    Xa0 = np.asarray(dfa[cols0])
    return ice_point(y_t2=y, X_t1=X1, X_t0=X0, Xa_t1=Xa1, Xa_t0=Xa0)


def ice_ban_bootstrap(params):
    """Helper function to run variance-estimation for ban contrast ICE g-computation with statsmodels. This is used
    for the bootstrap implementation.
    """
    ids, d, da, cols1, cols0 = params
    df = d.iloc[ids].copy()
    dfa = da.iloc[ids].copy()
    y = np.asarray(df['htn_w4'])
    X1 = np.asarray(df[cols1])
    X0 = np.asarray(df[cols0])
    Xa1 = np.asarray(dfa[cols1])
    Xa0 = np.asarray(dfa[cols0])
    natural = ice_point(y_t2=y, X_t1=X1, X_t0=X0, Xa_t1=X1, Xa_t0=X0)
    ban = ice_point(y_t2=y, X_t1=X1, X_t0=X0, Xa_t1=Xa1, Xa_t0=Xa0)
    return natural - ban

