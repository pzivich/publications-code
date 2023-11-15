#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Data generating mechanisms for the simulations
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd
from scipy.stats import logistic


def calculate_truth(n, scenario):
    # Calculate the truth for the target population
    d = generate_target_population(n=n)
    ya1 = generate_potential_outcomes(data=d, a=1, scenario=scenario)
    ya0 = generate_potential_outcomes(data=d, a=0, scenario=scenario)
    return np.mean(ya1) - np.mean(ya0)


def generate_data(n1, n0, scenario):
    # Generate the observed data for specified sample size and scenario
    d1 = generate_target_population(n=n1)
    d0 = generate_secondary_population(n=n0, scenario=scenario)
    d = pd.concat([d1, d0], ignore_index=True)
    d['V_s300'] = np.where(d['V'] >= 300, d['V'] - 300, 0)
    d['V_s800'] = np.where(d['V'] >= 800, d['V'] - 300, 0)
    d['V_star'] = np.where(d['V'] <= 300, 1, 0)
    d['intercept'] = 1
    return d


def generate_target_trial(n, scenario):
    # Generate observed data for the trial in the target population for the mathematical models
    d = generate_target_population(n=n)
    d['A'] = np.random.binomial(n=1, p=0.5, size=n)
    ya1 = generate_potential_outcomes(data=d, a=1, scenario=scenario)
    ya0 = generate_potential_outcomes(data=d, a=0, scenario=scenario)
    d['Y'] = np.where(d['A'] == 1, ya1, ya0)
    d['S'] = 1
    d['V_s300'] = np.where(d['V'] >= 300, d['V'] - 300, 0)
    d['V_s800'] = np.where(d['V'] >= 800, d['V'] - 300, 0)
    d['V_star'] = np.where(d['V'] <= 300, 1, 0)
    d['intercept'] = 1
    return d


def generate_target_population(n):
    # Generate the target population baseline variables
    d = pd.DataFrame()
    d['V'] = np.random.weibull(a=1.5, size=n)*375.
    d['V'] = np.round(d['V'])
    d['W'] = np.random.binomial(n=1, p=0.20, size=n)
    d['S'] = 1
    return d


def generate_secondary_population(n, scenario):
    # Generate the secondary population data by sampling from the target
    n_larger = n*3
    d = pd.DataFrame()
    d['V'] = np.random.weibull(a=1.5, size=n_larger)*370.
    d['V'] = np.round(d['V'])
    d['W'] = np.random.binomial(n=1, p=0.20, size=n_larger)

    # Sampling model
    log_odds = -0.02*d['V'] + 2*d['W']
    pr_s = np.where(d['V'] > 300, 0, logistic.cdf(log_odds))
    idx = np.random.choice(d.index, size=n, replace=False, p=pr_s / np.sum(pr_s))

    ds = d.iloc[idx].copy()
    ds['A'] = np.random.binomial(n=1, p=0.5, size=n)
    ya1 = generate_potential_outcomes(data=ds, a=1, scenario=scenario)
    ya0 = generate_potential_outcomes(data=ds, a=0, scenario=scenario)
    ds['Y'] = np.where(ds['A'] == 1, ya1, ya0)
    ds['S'] = 0
    return ds


def generate_potential_outcomes(data, a, scenario):
    # Calculate the potential outcomes for a data set with given A and scenario
    n = data.shape[0]
    if scenario == 2:
        lin_model = (-20 + a*70 + data['V']
                     + 0.2*a*data['V']
                     - 0.2*a*(data['V']-300)*(data['V'] >= 300)
                     - 0.3*a*(data['V']-800)*(data['V'] >= 800)
                     - 2*data['W'] + 5*a*data['W'])
    else:
        lin_model = (-20 + a*70 + data['V']
                     + 0.12*a*data['V']
                     - 2*data['W'] + 5*a*data['W'])
    ya = lin_model + np.random.normal(loc=1, scale=25, size=n)
    ya = np.where(ya < 0, 1, ya)
    ya = np.round(ya)
    return ya
