#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Appendix C: Non-positivity by a continuous covariate
#
# Paul Zivich
#######################################################################################################################

import numpy as np
from numpy.random import SeedSequence, default_rng
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit
from multiprocessing import Pool

from helper import load_data, trapezoid

###########################################
# Setup
iters = 10000                               # Number of iterations for Monte Carlo procedure
n_cpus = 7                                  # Number of CPUs to use for pooling
cutoff = 22                                 # Age cut-off to use for positivity
np.random.seed(77777777)                    # Setting global seed
rng = default_rng(SeedSequence(77777777))   # Setting seed generator specific to parameter draws


def g_iteration(params):
    # Revised helper function to run synthesis g-computation in continuous setting with Pool given parameters
    clinic_data, stat_param, mech_param = params

    # Generate predicted probabilities
    dca = clinic_data.copy()
    dca['group'] = 1
    dca['group_age'] = dca['age'] * dca['group']
    dca['group_age'] = dca['group'] * dca['age']
    dca['group_int'] = dca['group'] * dca['age_int']
    dca['group_ageg'] = dca['group'] * dca['age_g']
    X1 = np.asarray(dca[['intercept', 'group', 'gender', 'age', 'group_age']])
    W1 = np.asarray(dca[['age_int', 'group_int', 'age_g', 'group_ageg']])
    ya1 = inverse_logit(np.dot(X1, stat_param) + np.dot(W1, mech_param))

    dca['group'] = 0
    dca['group_age'] = dca['age'] * dca['group']
    dca['group_age'] = dca['group'] * dca['age']
    dca['group_int'] = dca['group'] * dca['age_int']
    dca['group_ageg'] = dca['group'] * dca['age_g']
    X0 = np.asarray(dca[['intercept', 'group', 'gender', 'age', 'group_age']])
    W0 = np.asarray(dca[['age_int', 'group_int', 'age_g', 'group_ageg']])
    ya0 = inverse_logit(np.dot(X0, stat_param) + np.dot(W0, mech_param))
    # Return point estimate
    return np.mean(ya1) - np.mean(ya0)


def ipw_iteration(params):
    # Revised helper function to run synthesis g-computation in continuous setting with Pool given parameters
    clinic_data, stat_param, mech_param = params

    # Generate predicted probabilities
    dca = clinic_data.copy()
    dca['group'] = 1
    dca['group_age'] = dca['age'] * dca['group']
    dca['group_age'] = dca['group'] * dca['age']
    dca['group_int'] = dca['group'] * dca['age_int']
    dca['group_ageg'] = dca['group'] * dca['age_g']
    X1 = np.asarray(dca[['intercept', 'group']])
    W1 = np.asarray(dca[['age_int', 'group_int', 'age_g', 'group_ageg']])
    ya1 = inverse_logit(np.dot(X1, stat_param) + np.dot(W1, mech_param))

    dca['group'] = 0
    dca['group_age'] = dca['age'] * dca['group']
    dca['group_age'] = dca['group'] * dca['age']
    dca['group_int'] = dca['group'] * dca['age_int']
    dca['group_ageg'] = dca['group'] * dca['age_g']
    X0 = np.asarray(dca[['intercept', 'group']])
    W0 = np.asarray(dca[['age_int', 'group_int', 'age_g', 'group_ageg']])
    ya0 = inverse_logit(np.dot(X0, stat_param) + np.dot(W0, mech_param))
    # Return point estimate
    return np.mean(ya1) - np.mean(ya0)


if __name__ == '__main__':
    ##########################################################################################
    # 0: Benchmark
    print("Benchmark")
    print("================================")

    # Loading and some data prep for continuous case
    d = load_data(full=True)
    d['age'] = d['age'] - cutoff                         # Center age at 22
    d['age_int'] = np.where(d['age'] > 0, 1, 0)          # Indicator if age > 22
    d['age_g'] = np.where(d['age'] > 0, d['age'], 0)     # Age if age > 22
    d['group_age'] = d['group'] * d['age']               # Interaction term for A & V
    d['group_ageg'] = d['group'] * d['age_g']            # Interaction term for AV and V>22
    d['group_int'] = d['age_int'] * d['group']           # Interaction term for A and V>22
    s = np.asarray(d['clinic'])
    a = np.asarray(d['group'])
    y = np.asarray(d['anytest'])

    #################
    # G-computation
    da = d.copy()
    da['group'] = 1
    da['group_int'] = da['age_int'] * da['group']
    da['group_age'] = da['group'] * da['age']
    da['group_ageg'] = da['group'] * da['age_g']
    Xa1 = np.asarray(da[['intercept', 'group', 'gender', 'age', 'group_age',
                         'age_int', 'group_int', 'age_g', 'group_ageg']])
    da['group'] = 0
    da['group_int'] = da['age_int'] * da['group']
    da['group_age'] = da['group'] * da['age']
    da['group_ageg'] = da['group'] * da['age_g']
    Xa0 = np.asarray(da[['intercept', 'group', 'gender', 'age', 'group_age',
                         'age_int', 'group_int', 'age_g', 'group_ageg']])

    def psi_gcomp(theta):
        psi, r1, r0, beta = theta[0], theta[1], theta[2], theta[3:]

        # Estimating the nuisance outcome model
        ee_out = ee_regression(theta=beta,
                               X=d[['intercept', 'group', 'gender', 'age', 'group_age',
                                    'age_int', 'group_int', 'age_g', 'group_ageg']],
                               y=d['anytest'],
                               model='logistic')
        ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

        # Estimating equations for parameters of interest
        ee_r1 = s*(inverse_logit(np.dot(Xa1, beta)) - r1)
        ee_r0 = s*(inverse_logit(np.dot(Xa0, beta)) - r0)
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                         ee_r1,
                         ee_r0,
                         ee_out])

    # M-estimator for benchmark
    estr = MEstimator(psi_gcomp, init=[0, 0.5, 0.5,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0])
    estr.estimate(solver='lm')
    est_g = estr.theta[0]
    ci_g = estr.confidence_intervals()[0, :]

    print("g-computation")
    print(est_g)
    print(ci_g)

    #################
    # IPW (saturated MSM)

    def psi_ipw(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        alpha = [theta[3], ]
        beta = theta[4:]

        # Estimating the nuisance action model
        ee_act = ee_regression(theta=alpha,
                               X=d[['intercept']],
                               y=d['group'],
                               model='logistic')
        ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
        pi_a = inverse_logit(np.dot(np.asarray(d[['intercept']]), alpha))

        # Estimating the nuisance transport model
        ee_trp = ee_regression(theta=beta,
                               X=d[['intercept', 'gender', 'age']],
                               y=d['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'gender', 'age']]), beta))

        # Creating IPW
        iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
        iosw = (pi_s / (1 - pi_s))

        # Estimating equations for parameters of interest
        ee_r1 = np.where(s == 1, 0, y - r1) * iosw * (a == 1) * iptw
        ee_r0 = np.where(s == 1, 0, y - r0) * iosw * (a == 0) * iptw
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                          ee_r1,
                          ee_r0,
                          ee_trp,
                          ee_act])

    # M-estimator for benchmark
    estr = MEstimator(psi_ipw, init=[0, 0.5, 0.5,
                                     0, 0, 0, 0])
    estr.estimate(solver='lm')
    est_w = estr.theta[0]
    ci_w = estr.confidence_intervals()[0, :]

    print("IPW - MSM")
    print(est_w)
    print(ci_w)

    #################
    # IPW (faux MSM)
    da = d.copy()
    da['group'] = 1
    da['group_int'] = da['group'] * da['age_int']
    da['group_ageg'] = da['group'] * da['age_g']
    Xa1 = np.asarray(da[['intercept', 'group', 'age_int', 'group_int', 'age_g', 'group_ageg']])
    da['group'] = 0
    da['group_int'] = da['group'] * da['age_int']
    da['group_ageg'] = da['group'] * da['age_g']
    Xa0 = np.asarray(da[['intercept', 'group', 'age_int', 'group_int', 'age_g', 'group_ageg']])

    def psi_ipw(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        alpha = [theta[3], ]
        beta = theta[4:7]
        gamma = theta[7:]

        # Estimating the nuisance action model
        ee_act = ee_regression(theta=alpha,
                               X=d[['intercept']],
                               y=d['group'],
                               model='logistic')
        ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
        pi_a = inverse_logit(np.dot(np.asarray(d[['intercept']]), alpha))

        # Estimating the nuisance transport model
        ee_trp = ee_regression(theta=beta,
                               X=d[['intercept', 'gender', 'age']],
                               y=d['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'gender', 'age']]), beta))

        # Creating IPW
        iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
        iosw = (pi_s / (1 - pi_s))

        # Estimating the faux / parametric marginal structural model
        ee_msm = ee_regression(theta=gamma,
                               X=d[['intercept', 'group',
                                    'age_int', 'group_int', 'age_g', 'group_ageg']],
                               y=d['anytest'],
                               weights=iptw*iosw,
                               model='logistic') * (1-s)
        ee_msm = np.nan_to_num(ee_msm, copy=True, nan=0.)
        ya1 = inverse_logit(np.dot(Xa1, gamma))               # Predictions under a=1 from the MSM
        ya0 = inverse_logit(np.dot(Xa0, gamma))               # Predictions under a=0 from the MSM

        # Estimating equations for parameters of interest
        ee_r1 = s*(ya1 - r1)
        ee_r0 = s*(ya0 - r0)
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                          ee_r1,
                          ee_r0,
                          ee_trp,
                          ee_act,
                          ee_msm])

    # M-estimator for benchmark
    estr = MEstimator(psi_ipw, init=[0, 0.5, 0.5,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0])
    estr.estimate(solver='lm')
    est_w = estr.theta[0]
    ci_w = estr.confidence_intervals()[0, :]

    print("IPW - faux MSM")
    print(est_w)
    print(ci_w)

    ##########################################################################################
    # Comparing solutions
    ##########################################################################################

    # Data preparation
    d = load_data(full=True)
    d['age'] = d['age'] - cutoff                            # Shifting age by subtracting 22
    d['group_age'] = d['group'] * d['age']                  # Interaction term for A & V
    d['age_g'] = np.where(d['age'] > 0, d['age'], 0)        # Age if V > 22
    d['age_int'] = np.where(d['age'] > 0, 1, 0)             # Indicator if age > 22
    d = d.loc[(d['clinic'] == 1) | (d['age'] <= 0)].copy()  # Subset trial to V <= 22 only

    ##########################################################################################
    # Solution 1: Restrict target population

    print()
    print("Restrict Target Population")
    print("================================")

    ds = d.loc[d['age'] <= 0].copy()                       # Restricting target population by age
    s = np.asarray(ds['clinic'])
    a = np.asarray(ds['group'])
    y = np.asarray(ds['anytest'])

    #################
    # G-computation
    da = ds.copy()
    da['group'] = 1
    da['group_age'] = da['group'] * da['age']
    Xa1 = np.asarray(da[['intercept', 'group', 'gender', 'age', 'group_age']])
    da['group'] = 0
    da['group_age'] = da['group'] * da['age']
    Xa0 = np.asarray(da[['intercept', 'group', 'gender', 'age', 'group_age']])

    def psi_gcomp(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        beta = theta[3:]

        # Estimating the nuisance outcome model
        ee_out = ee_regression(theta=beta,
                               X=ds[['intercept', 'group', 'gender', 'age', 'group_age']],
                               y=ds['anytest'],
                               model='logistic')
        ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

        # Estimating equations for parameters of interest
        ee_r1 = s*(inverse_logit(np.dot(Xa1, beta)) - r1)
        ee_r0 = s*(inverse_logit(np.dot(Xa0, beta)) - r0)
        ee_rd = np.ones(ds.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                         ee_r1,
                         ee_r0,
                         ee_out])

    # M-estimator for restricted population
    estr = MEstimator(psi_gcomp, init=[0, 0.5, 0.5,
                                       0, 0, 0, 0, 0])
    estr.estimate(solver='lm')
    est_g = estr.theta[0]
    ci_g = estr.confidence_intervals()[0, :]

    print("g-computation")
    print(est_g)
    print(ci_g)

    #################
    # IPW

    def psi_ipw(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        alpha = [theta[3], ]
        beta = theta[4:]

        # Estimating the nuisance action model
        ee_act = ee_regression(theta=alpha,
                               X=ds[['intercept']],
                               y=ds['group'],
                               model='logistic')
        ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
        pi_a = inverse_logit(np.dot(np.asarray(ds[['intercept']]), alpha))

        # Estimating the nuisance transport model
        ee_trp = ee_regression(theta=beta,
                               X=ds[['intercept', 'gender', 'age']],
                               y=ds['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(ds[['intercept', 'gender', 'age']]), beta))

        # Creating IPW
        iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
        iosw = (pi_s / (1 - pi_s))

        # Estimating equations for parameters of interest
        ee_r1 = np.where(s == 1, 0, y - r1) * iosw * (a == 1) * iptw
        ee_r0 = np.where(s == 1, 0, y - r0) * iosw * (a == 0) * iptw
        ee_rd = np.ones(ds.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                         ee_r1,
                         ee_r0,
                         ee_trp,
                         ee_act])


    # M-estimator for restricted population
    estr = MEstimator(psi_ipw, init=[0, 0.5, 0.5,
                                     0,
                                     0, 0, 0])
    estr.estimate(solver='lm')
    est_w = estr.theta[0]
    ci_w = estr.confidence_intervals()[0, :]

    print("IPW")
    print(est_w)
    print(ci_w)

    ##########################################################################################
    # Solution 2: Restrict covariate set

    print()
    print("Restrict Covariate Set")
    print("================================")

    s = np.asarray(d['clinic'])
    a = np.asarray(d['group'])
    y = np.asarray(d['anytest'])

    #################
    # G-computation
    da = d.copy()
    da['group'] = 1
    da['group_age'] = da['group'] * da['age']
    Xa1 = np.asarray(da[['intercept', 'group', 'gender']])
    da['group'] = 0
    da['group_age'] = da['group'] * da['age']
    Xa0 = np.asarray(da[['intercept', 'group', 'gender']])

    def psi_gcomp(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        beta = theta[3:]

        # Estimating the nuisance outcome model
        ee_out = ee_regression(theta=beta,
                               X=d[['intercept', 'group', 'gender']],
                               y=d['anytest'],
                               model='logistic')
        ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

        # Estimating equations for parameters of interest
        ee_r1 = s*(inverse_logit(np.dot(Xa1, beta)) - r1)
        ee_r0 = s*(inverse_logit(np.dot(Xa0, beta)) - r0)
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                         ee_r1,
                         ee_r0,
                         ee_out])

    # M-estimator for restricted population
    estr = MEstimator(psi_gcomp, init=[0, 0.5, 0.5,
                                       0, 0, 0])
    estr.estimate(solver='lm')
    est_g = estr.theta[0]
    ci_g = estr.confidence_intervals()[0, :]

    print("g-computation")
    print(est_g)
    print(ci_g)

    #################
    # IPW

    def psi_ipw(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        alpha = [theta[3], ]
        beta = theta[4:]

        # Estimating the nuisance action model
        ee_act = ee_regression(theta=alpha,
                               X=d[['intercept']],
                               y=d['group'],
                               model='logistic')
        ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
        pi_a = inverse_logit(np.dot(np.asarray(d[['intercept']]), alpha))

        # Estimating the nuisance transport model
        ee_trp = ee_regression(theta=beta,
                               X=d[['intercept', 'gender']],
                               y=d['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'gender']]), beta))

        # Creating IPW
        iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
        iosw = (pi_s / (1 - pi_s))

        # Estimating equations for parameters of interest
        ee_r1 = np.where(s == 1, 0, y - r1) * iosw * (a == 1) * iptw
        ee_r0 = np.where(s == 1, 0, y - r0) * iosw * (a == 0) * iptw
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                         ee_r1,
                         ee_r0,
                         ee_trp,
                         ee_act])

    # M-estimator for restricted population
    estr = MEstimator(psi_ipw, init=[0, 0.5, 0.5,
                                     0,
                                     0, 0])
    estr.estimate(solver='lm')
    est_w = estr.theta[0]
    ci_w = estr.confidence_intervals()[0, :]

    print("IPW")
    print(est_w)
    print(ci_w)

    ##########################################################################################
    # Solution 3: Extrapolate

    print()
    print("Extrapolate")
    print("================================")

    s = np.asarray(d['clinic'])
    a = np.asarray(d['group'])
    y = np.asarray(d['anytest'])

    #################
    # G-computation
    da = d.copy()
    da['group'] = 1
    da['group_age'] = da['group'] * da['age']
    Xa1 = np.asarray(da[['intercept', 'group', 'gender', 'age', 'group_age']])
    da['group'] = 0
    da['group_age'] = da['group'] * da['age']
    Xa0 = np.asarray(da[['intercept', 'group', 'gender', 'age', 'group_age']])

    def psi_gcomp(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        beta = theta[3:]

        # Estimating the nuisance outcome model
        ee_out = ee_regression(theta=beta,
                               X=d[['intercept', 'group', 'gender', 'age', 'group_age']],
                               y=d['anytest'],
                               model='logistic')
        ee_out = np.nan_to_num(ee_out, copy=False, nan=0.)

        # Estimating equations for parameters of interest
        ee_r1 = s*(inverse_logit(np.dot(Xa1, beta)) - r1)
        ee_r0 = s*(inverse_logit(np.dot(Xa0, beta)) - r0)
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                          ee_r1,
                          ee_r0,
                          ee_out])

    # M-estimator for restricted population
    estr = MEstimator(psi_gcomp, init=[0, 0.5, 0.5,
                                       0, 0, 0, 0, 0])
    estr.estimate(solver='lm')
    est_g = estr.theta[0]
    ci_g = estr.confidence_intervals()[0, :]

    print("g-computation")
    print(est_g)
    print(ci_g)

    #################
    # IPW

    def psi_ipw(theta):
        psi, r1, r0 = theta[0], theta[1], theta[2]
        alpha = [theta[3], ]
        beta = theta[4:]

        # Estimating the nuisance action model
        ee_act = ee_regression(theta=alpha,
                               X=d[['intercept']],
                               y=d['group'],
                               model='logistic')
        ee_act = np.nan_to_num(ee_act, copy=True, nan=0.)
        pi_a = inverse_logit(np.dot(np.asarray(d[['intercept']]), alpha))

        # Estimating the nuisance transport model
        ee_trp = ee_regression(theta=beta,
                               X=d[['intercept', 'gender', 'age']],
                               y=d['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'gender', 'age']]), beta))

        # Creating IPW
        iptw = 1 / np.where(a == 1, pi_a, 1 - pi_a)
        iosw = (pi_s / (1 - pi_s))

        # Estimating equations for parameters of interest
        ee_r1 = np.where(s == 1, 0, y - r1) * iosw * (a == 1) * iptw
        ee_r0 = np.where(s == 1, 0, y - r0) * iosw * (a == 0) * iptw
        ee_rd = np.ones(d.shape[0]) * (r1 - r0) - psi

        return np.vstack([ee_rd,
                         ee_r1,
                         ee_r0,
                         ee_trp,
                         ee_act])

    # M-estimator for restricted population
    estr = MEstimator(psi_ipw, init=[0, 0.5, 0.5, 0, 0, 0, 0])
    estr.estimate(solver='lm')
    est_w = estr.theta[0]
    ci_w = estr.confidence_intervals()[0, :]

    print("IPW")
    print(est_w)
    print(ci_w)

    ##########################################################################################
    # Solution 4: Synthesis of statistical and simulation models

    print()
    print("Model Synthesis")
    print("================================")

    d0 = d.loc[d['clinic'] == 0].copy()
    d1 = d.loc[d['clinic'] == 1].copy()

    s = np.asarray(d['clinic'])
    a = np.asarray(d['group'])
    y = np.asarray(d['anytest'])

    #################
    # G-computation
    print("g-computation")

    def psi_outcome_model(theta):
        return ee_regression(theta=theta,
                             X=d0[['intercept', 'group', 'gender', 'age', 'group_age']],
                             y=d0['anytest'],
                             model='logistic')

    # Estimating the statistical model parameters
    estr = MEstimator(psi_outcome_model, init=[0., 0., 0., 0., 0.])
    estr.estimate(solver='lm')
    stat_means = estr.theta
    stat_covar = estr.variance

    # Monte-Carlo Procedure for each combination of statistical model parameters
    for scenario in [1, 2, 3, 4, 5]:
        stat_params = rng.multivariate_normal(stat_means, cov=stat_covar, size=iters)
        if scenario == 1:
            mech_params = [[0, 0, 0, 0], ] * iters
        elif scenario == 2:
            mech_params = []
            for i in range(iters):
                mech_params.append([trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0]])
        elif scenario == 3:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(0.4443, scale=0.3229, size=1)[0],
                                    rng.normal(0.1415, scale=0.4184, size=1)[0],
                                    rng.normal(-0.1330, scale=0.0963, size=1)[0],
                                    rng.normal(0.0966, scale=0.1263, size=1)[0]])
        elif scenario == 4:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(-0.4443, scale=0.3229, size=1)[0],
                                    rng.normal(-0.1415, scale=0.4184, size=1)[0],
                                    rng.normal(0.1330, scale=0.0963, size=1)[0],
                                    rng.normal(-0.0966, scale=0.1263, size=1)[0]])
        elif scenario == 5:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(0.4558, scale=0.2894, size=1)[0],
                                    rng.normal(0.1261, scale=0.3734, size=1)[0],
                                    rng.normal(-0.0293, scale=0.0562, size=1)[0],
                                    rng.normal(0.0658, scale=0.0743, size=1)[0]])
        else:
            raise ValueError("Invalid scenario")

        # Packaging data to give to each Pool process
        params = [[d1.sample(n=d1.shape[0], replace=True),
                   stat_params[j],
                   mech_params[j],
                   ] for j in range(iters)]
        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(g_iteration,                # Call outside function
                                      params))                    # ... provide packed input

        if scenario == 1:
            print("Scenario: Strict Null")
        if scenario == 2:
            print("-----")
            print("Scenario: Uncertain Null")
        if scenario == 3:
            print("-----")
            print("Scenario: Accurate")
        if scenario == 4:
            print("-----")
            print("Scenario: Inaccurate")
        if scenario == 5:
            print("-----")
            print("Scenario: Reversed")
        print(np.round(np.median(estimates), 5))
        print(np.round(np.percentile(estimates, q=[2.5, 97.5]), 5))

    #################
    # IPW
    print()
    print("IPW")

    def psi_msm_model(theta):
        alpha, beta, gamma = theta[0], theta[1:4], theta[4:]

        # Estimating the nuisance action model
        ee_act = np.nan_to_num((1 - s) * (a - alpha), copy=True, nan=0.)
        pi_a = (a == 1) * alpha + (a == 0) * (1 - alpha) + s
        iptw = 1 / pi_a

        # Estimating the nuisance sampling model
        ee_trp = ee_regression(theta=beta,
                               X=d[['intercept', 'gender', 'age']],
                               y=d['clinic'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(d[['intercept', 'gender', 'age']]), beta))
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

    # Estimating the statistical model parameters
    estr = MEstimator(psi_msm_model, init=[0.5, 0., 0., 0., 0., 0.])
    estr.estimate(solver='lm', maxiter=2000)
    stat_means = estr.theta[4:]
    stat_covar = estr.variance[4:, 4:]

    # Monte-Carlo Procedure for each combination of statistical model parameters
    for scenario in [1, 2, 3, 4, 5]:
        stat_params = rng.multivariate_normal(stat_means, cov=stat_covar, size=iters)
        if scenario == 1:
            mech_params = [[0, 0, 0, 0], ] * iters
        elif scenario == 2:
            mech_params = []
            for i in range(iters):
                mech_params.append([trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0],
                                    trapezoid(-2, -1, 1, 2, size=1)[0]])
        elif scenario == 3:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(0.4558, scale=0.2894, size=1)[0],
                                    rng.normal(0.1261, scale=0.3734, size=1)[0],
                                    rng.normal(-0.0293, scale=0.0562, size=1)[0],
                                    rng.normal(0.0658, scale=0.0743, size=1)[0]])
        elif scenario == 4:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(-0.4558, scale=0.2894, size=1)[0],
                                    rng.normal(-0.1261, scale=0.3734, size=1)[0],
                                    rng.normal(0.0293, scale=0.0562, size=1)[0],
                                    rng.normal(-0.0658, scale=0.0743, size=1)[0]])
        elif scenario == 5:
            mech_params = []
            for i in range(iters):
                mech_params.append([rng.normal(0.4443, scale=0.3229, size=1)[0],
                                    rng.normal(0.1415, scale=0.4184, size=1)[0],
                                    rng.normal(-0.1330, scale=0.0963, size=1)[0],
                                    rng.normal(0.0966, scale=0.1263, size=1)[0]])
        else:
            raise ValueError("Invalid scenario")

        # Packaging data to give to each Pool process
        params = [[d1.sample(n=d1.shape[0], replace=True),
                   stat_params[j],
                   mech_params[j],
                   ] for j in range(iters)]
        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(ipw_iteration,     # Call outside function
                                      params))           # ... provide packed input

        if scenario == 1:
            print("Scenario: Strict Null")
        if scenario == 2:
            print("-----")
            print("Scenario: Uncertain Null")
        if scenario == 3:
            print("-----")
            print("Scenario: Accurate")
        if scenario == 4:
            print("-----")
            print("Scenario: Inaccurate")
        if scenario == 5:
            print("-----")
            print("Scenario: Reversed")
        print(np.round(np.median(estimates), 5))
        print(np.round(np.percentile(estimates, q=[2.5, 97.5]), 5))
