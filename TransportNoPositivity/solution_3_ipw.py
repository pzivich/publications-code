#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Solution 3: Model Synthesis IPW -- GetTested illustrative example
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


def mechanistic_ipw(params):
    clinic_data, stat_param, mech_param = params

    # Generate predicted probabilities
    dca = clinic_data.copy()
    dca['group'] = 1
    dca['a_w'] = dca['gender'] * dca['group']
    X1 = np.asarray(dca[['intercept', 'group']])
    W1 = np.asarray(dca[['gender', 'a_w']])
    ya1 = inverse_logit(np.dot(X1, stat_param) + np.dot(W1, mech_param))

    dca['group'] = 0
    dca['a_w'] = dca['gender'] * dca['group']
    X0 = np.asarray(dca[['intercept', 'group']])
    W0 = np.asarray(dca[['gender', 'a_w']])
    ya0 = inverse_logit(np.dot(X0, stat_param) + np.dot(W0, mech_param))

    return np.mean(ya1) - np.mean(ya0)


if __name__ == '__main__':
    #####################################
    # Loading trial data
    d = load_data(full=False)

    #####################################
    # Data preparation

    d0 = d.loc[d['clinic'] == 0].copy()
    d1 = d.loc[d['clinic'] == 1].copy()
    s = np.asarray(d['clinic'])
    a = np.asarray(d['group'])

    #####################################
    # Solution 3: IPW

    def psi_msm_model(theta):
        alpha, beta, gamma = theta[0], theta[1:4], theta[4:]

        # Estimating the nuisance action model
        ee_act = np.nan_to_num((1 - s) * (a - alpha), copy=True, nan=0.)
        pi_a = (a == 1) * alpha + (a == 0) * (1 - alpha) + s
        iptw = 1 / pi_a

        # Estimating the nuisance sampling model
        ee_trp = ee_regression(theta=beta,
                               X=d[['intercept', 'age', 'age_sq']],
                               y=d['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'age', 'age_sq']]), beta))
        odds = pi_s / (1 - pi_s)
        iosw = s * 1 + (1 - s) * odds

        # Estimating the marginal structural model
        ee_msm = ee_regression(theta=gamma,
                               X=d[['intercept', 'group']],
                               y=d['anytest'],
                               weights=iptw * iosw,
                               model='logistic') * (1 - s)
        ee_msm = np.nan_to_num(ee_msm, copy=True, nan=0.)

        return np.vstack([ee_act, ee_trp, ee_msm])

    estr = MEstimator(psi_msm_model, init=[0.5, -1.2, -0.1, 0., -1.3, 1.5])
    estr.estimate(solver='lm', maxiter=2000)
    stat_means = estr.theta[4:]
    stat_covar = estr.variance[4:, 4:]

    for scenario in [1, 2, 3, 4]:
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
                mech_params.append([rng.normal(0.1380, scale=0.1931, size=1)[0],
                                    rng.normal(-0.6914, scale=0.2460, size=1)[0]])
        elif scenario == 4:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(-0.1380, scale=0.1931, size=1)[0],
                                    rng.normal(0.6914, scale=0.2460, size=1)[0]])
        else:
            raise ValueError("Invalid scenario")

        # Packaging data to give to each Pool process
        params = [[d1.sample(n=d1.shape[0], replace=True),
                   stat_params[j],
                   mech_params[j],
                   ] for j in range(iters)]

        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(mechanistic_ipw,  # Call outside function
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
