#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Solution 3: Model Synthesis G-computation -- GetTested illustrative example
#
# Paul Zivich
#######################################################################################################################

import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit
from multiprocessing import Pool

from helper import load_data, trapezoid


iters = 10000
n_cpus = 7
rng = default_rng(SeedSequence(77777777))


def g_mechanistic_computation(params):
    clinic_data, stat_param, mech_param = params

    # Generate predicted probabilities
    dca = clinic_data.copy()
    dca['group'] = 1
    dca['a_w'] = dca['gender'] * dca['group']
    X1 = np.asarray(dca[['intercept', 'group', 'age', 'age_sq']])
    W1 = np.asarray(dca[['gender', 'a_w']])
    ya1 = inverse_logit(np.dot(X1, stat_param) + np.dot(W1, mech_param))

    dca['group'] = 0
    dca['a_w'] = dca['gender'] * dca['group']
    X0 = np.asarray(dca[['intercept', 'group', 'age', 'age_sq']])
    W0 = np.asarray(dca[['gender', 'a_w']])
    ya0 = inverse_logit(np.dot(X0, stat_param) + np.dot(W0, mech_param))

    return np.mean(ya1) - np.mean(ya0)


if __name__ == '__main__':
    for scenario in [1, 2, 3, 4]:
        #####################################
        # Loading trial data
        d = load_data(full=False)

        #####################################
        # Data preparation

        d0 = d.loc[d['clinic'] == 0].copy()
        d1 = d.loc[d['clinic'] == 1].copy()

        #####################################
        # Solution 3: G-computation

        def psi_outcome_model(theta):
            return ee_regression(theta=theta,
                                 X=d0[['intercept', 'group', 'age', 'age_sq']],
                                 y=d0['anytest'],
                                 model='logistic')

        # Estimating the statistical model parameters
        estr = MEstimator(psi_outcome_model, init=[-5, 1.5, 0, 0])
        estr.estimate(solver='lm')
        stat_means = estr.theta
        stat_covar = estr.variance

        # Monte-Carlo Procedure
        stat_params = rng.multivariate_normal(stat_means, cov=stat_covar, size=iters)
        if scenario == 1:
            mech_params = [[0, 0], ] * iters
        elif scenario == 2:
            mech_params = []
            for i in range(iters):
                mech_params.append([trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0]])
        elif scenario == 3:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(-0.0160, scale=0.1761, size=1)[0],
                                    rng.normal(-0.6270, scale=0.2227, size=1)[0]])
        elif scenario == 4:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(0.0160, scale=0.1761, size=1)[0],
                                    rng.normal(0.6270, scale=0.2227, size=1)[0]])
        else:
            raise ValueError("Invalid scenario")

        # Packaging data to give to each Pool process
        params = [[d1.sample(n=d1.shape[0], replace=True),
                   stat_params[j],
                   mech_params[j],
                   ] for j in range(iters)]

        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(g_mechanistic_computation,  # Call outside function
                                      params))                    # provide packed input

        print("================================")
        if scenario == 1:
            print("Scenario: Strict Null")
        if scenario == 2:
            print("Scenario: Uncertain Null")
        if scenario == 3:
            print("Scenario: Accurate")
        if scenario == 4:
            print("Scenario: Inaccurate")
        print(np.round(np.median(estimates), 5))
        print(np.round(np.percentile(estimates, q=[2.5, 97.5]), 5))
        print("================================")
