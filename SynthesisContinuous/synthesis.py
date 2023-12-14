#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Synthesis estimators
#
# Paul Zivich
#######################################################################################################################

import patsy
import numpy as np
import pandas as pd
from delicatessen import MEstimator
from multiprocessing import Pool

from efuncs import ee_synth_aipw_msm, ee_synth_msm_only, ee_synth_aipw_cace, ee_synth_cace_only


class SynthesisMSM:
    """Synthesis estimator based on a marginal structural model. The statistical portion of the synthesis estimator
    is estimated using weighted regression augmented inverse probability weighting.

    Parameters
    ----------
    data :
        Data set containing variables of interest
    outcome : str
        Outcome column label
    action : str
        Action column label
    sample : str
        Sample indicator column label
    positive_region :
        Indicator of whether a unit is in the positive region(s) or not
    """
    def __init__(self, data, outcome, action, sample, positive_region):
        self.data = data
        self.outcome = outcome
        self.action = action
        self.sample = sample
        self.positive = positive_region

        # Generating stacked data for the Snowden et al. trick
        d = self.data.copy()
        self.data_a1 = d.copy()
        self.data_a1[self.action] = 1
        self.data_a1[self.outcome] = np.nan
        self.data_a0 = d.copy()
        self.data_a0[self.action] = 0
        self.data_a0[self.outcome] = np.nan

        # Initialize storage for results
        self.estimates = None
        self.ace = None
        self.ace_var = None
        self.ace_ci = None
        self.bounds = None
        self.bounds_ci = None

        self._PS_ = None
        self._action_model_ = None
        self._SW_ = None
        self._sample_model_ = None
        self._X_ = None
        self._Xa1_ = None
        self._Xa0_ = None
        self._outcome_model_ = None
        self._MSM1_ = None
        self._MSM0_ = None
        self._msm_model_ = None
        self._math_model_ = None
        self._math_param_ = None

    def action_model(self, model):
        self._PS_ = patsy.dmatrix(model, self.data,
                                  return_type='dataframe',
                                  NA_action=patsy.NAAction(NA_types=[]))
        self._action_model_ = model

    def sample_model(self, model):
        self._SW_ = patsy.dmatrix(model, self.data,
                                  return_type='dataframe',
                                  NA_action=patsy.NAAction(NA_types=[]))
        self._sample_model_ = model

    def outcome_model(self, model):
        self._X_ = patsy.dmatrix(model, self.data,
                                 return_type='dataframe',
                                 NA_action=patsy.NAAction(NA_types=[]))
        self._Xa1_ = patsy.dmatrix(model, self.data_a1,
                                   return_type='dataframe',
                                   NA_action=patsy.NAAction(NA_types=[]))
        self._Xa0_ = patsy.dmatrix(model, self.data_a0,
                                   return_type='dataframe',
                                   NA_action=patsy.NAAction(NA_types=[]))
        self._outcome_model_ = model

    def marginal_structural_model(self, model):
        self._MSM1_ = patsy.dmatrix(model, self.data_a1,
                                    return_type='dataframe',
                                    NA_action=patsy.NAAction(NA_types=[]))
        self._MSM0_ = patsy.dmatrix(model, self.data_a0,
                                    return_type='dataframe',
                                    NA_action=patsy.NAAction(NA_types=[]))
        self._msm_model_ = model

    def math_model(self, model, parameters):
        self._math_model_ = model
        self._math_param_ = parameters

    def estimate(self, mc_iterations, n_cpus=1):
        # Estimating statistical model parameters
        msm_np = self._MSM1_.shape[1]
        stat = self.estimate_stat_msm(init=None, solver='lm')
        alpha_msm = stat.theta[0:msm_np]
        alpha_cov_msm = stat.variance[0:msm_np, 0:msm_np]

        # Drawing statistical model parameters
        stat_params = np.random.multivariate_normal(mean=alpha_msm, cov=alpha_cov_msm, size=mc_iterations)

        # Preparing data for estimation procedure via Pool
        d1 = self.data.loc[self.data[self.sample] == 1].copy()
        n1 = d1.shape[0]
        params = [[d1.sample(n=n1, replace=True),
                   stat_params[j], self._math_param_[j],
                   self._msm_model_, self._math_model_, self.action,
                   ] for j in range(mc_iterations)]

        # Running point estimation with multiple CPUs
        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(_msm_semiparametric_bootstrap_,  # Call outside function
                                      params))                         # ... and provide packed input

        # Processing results
        self.estimates = estimates
        self.ace = np.nanmedian(estimates)
        self.ace_var = np.nanvar(estimates)
        self.ace_ci = np.nanpercentile(estimates, q=[2.5, 97.5])

    def estimate_stat_msm(self, init=None, solver='lm'):
        y = np.asarray(self.data[self.outcome])
        a = np.asarray(self.data[self.action])
        s = np.asarray(self.data[self.sample])
        r = np.asarray(self.data[self.positive])
        Z = np.asarray(self._PS_)
        W = np.asarray(self._SW_)
        X = np.asarray(self._X_)
        X1 = np.asarray(self._Xa1_)
        X0 = np.asarray(self._Xa0_)
        M1 = np.asarray(self._MSM1_)
        M0 = np.asarray(self._MSM0_)

        def psi_stat_msm(theta):
            return ee_synth_msm_only(theta, y, a, s, r,
                                     Z, W,
                                     X, X1, X0,
                                     M1, M0,
                                     model='linear')

        if init is None:
            init = [-15., 67., 0., 0.] + [0., ]*Z.shape[1] + [0., ]*W.shape[1] + [0., ]*X.shape[1]

        estr = MEstimator(psi_stat_msm, init=init)
        estr.estimate(solver=solver, maxiter=100000)
        return estr


class SynthesisCACE:
    """Synthesis estimator based on a conditional average causal effect model. The statistical portion of the synthesis
    estimator is estimated using weighted regression augmented inverse probability weighting.

    Parameters
    ----------
    data :
        Data set containing variables of interest
    outcome : str
        Outcome column label
    action : str
        Action column label
    sample : str
        Sample indicator column label
    positive_region :
        Indicator of whether a unit is in the positive region(s) or not
    """
    def __init__(self, data, outcome, action, sample, positive_region):
        self.data = data
        self.outcome = outcome
        self.action = action
        self.sample = sample
        self.positive = positive_region

        # Generating data under the all policies
        self.data_a1 = self.data.copy()
        self.data_a1[self.action] = 1
        self.data_a0 = self.data.copy()
        self.data_a0[self.action] = 0

        # Initialize storage for results
        self.estimates = None
        self.ace = None
        self.ace_var = None
        self.ace_ci = None
        self.bounds = None
        self.bounds_ci = None

        self._PS_ = None
        self._action_model_ = None
        self._SW_ = None
        self._sample_model_ = None
        self._X_ = None
        self._Xa1_ = None
        self._Xa0_ = None
        self._outcome_model_ = None
        self._CACE_ = None
        self._cace_model_ = None
        self._MATH_ = None
        self._math_model_ = None
        self._math_param_ = None

    def action_model(self, model):
        self._PS_ = patsy.dmatrix(model, self.data,
                                  return_type='dataframe',
                                  NA_action=patsy.NAAction(NA_types=[]))
        self._action_model_ = model

    def sample_model(self, model):
        self._SW_ = patsy.dmatrix(model, self.data,
                                  return_type='dataframe')
        self._sample_model_ = model

    def outcome_model(self, model):
        self._X_ = patsy.dmatrix(model, self.data,
                                 return_type='dataframe',
                                 NA_action=patsy.NAAction(NA_types=[]))
        self._Xa1_ = patsy.dmatrix(model, self.data_a1,
                                   return_type='dataframe')
        self._Xa0_ = patsy.dmatrix(model, self.data_a0,
                                   return_type='dataframe')
        self._outcome_model_ = model

    def cace_model(self, model):
        self._CACE_ = patsy.dmatrix(model, self.data,
                                    return_type='dataframe')
        self._cace_model_ = model

    def math_model(self, model, parameters):
        self._MATH_ = patsy.dmatrix(model, self.data,
                                    return_type='dataframe')
        self._math_model_ = model
        self._math_param_ = parameters

    def estimate(self, mc_iterations, n_cpus=1):
        # Estimating statistical model parameters
        cace_np = self._CACE_.shape[1]
        stat = self.estimate_stat_cace(init=None, solver='lm')
        gamma_msm = stat.theta[0:cace_np]
        gamma_cov_msm = stat.variance[0:cace_np, 0:cace_np]

        # Drawing statistical model parameters
        stat_params = np.random.multivariate_normal(mean=gamma_msm, cov=gamma_cov_msm, size=mc_iterations)

        # Preparing data for estimation procedure via Pool
        d1 = self.data.loc[self.data[self.sample] == 1].copy()
        n1 = d1.shape[0]
        params = [[d1.sample(n=n1, replace=True),
                   stat_params[j], self._math_param_[j],
                   self._cace_model_, self._math_model_,
                   ] for j in range(mc_iterations)]

        # Running point estimation with multiple CPUs
        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(_cace_semiparametric_bootstrap_,  # Call outside function
                                      params))                          # ... and provide packed input

        # Processing results
        self.estimates = estimates
        self.ace = np.nanmedian(estimates)
        self.ace_var = np.nanvar(estimates)
        self.ace_ci = np.nanpercentile(estimates, q=[2.5, 97.5])

    def estimate_stat_cace(self, init=None, solver='lm'):
        y = np.asarray(self.data[self.outcome])
        a = np.asarray(self.data[self.action])
        s = np.asarray(self.data[self.sample])
        r = np.asarray(self.data[self.positive])
        Z = np.asarray(self._PS_)
        W = np.asarray(self._SW_)
        X = np.asarray(self._X_)
        Xa1 = np.asarray(self._Xa1_)
        Xa0 = np.asarray(self._Xa0_)
        CACE = np.asarray(self._CACE_)

        def psi_stat_cace(theta):
            return ee_synth_cace_only(theta, y, a, s, r,
                                      Z, W,
                                      X, Xa1, Xa0, CACE,
                                      model='linear')

        if init is None:
            init = [0.]*CACE.shape[1] + [0., ]*Z.shape[1] + [0., ]*W.shape[1] + [0., ]*X.shape[1]

        estr = MEstimator(psi_stat_cace, init=init)
        estr.estimate(solver=solver, maxiter=100000)
        return estr

    def estimate_bounds(self, lower, upper, init=None, solver='lm'):
        def psi_synth_cace(theta):
            return ee_synth_aipw_cace(theta=theta, y=y, a=a, s=s, r=r,
                                      Z=Z, W=W, X=X, Xa1=Xa1, Xa0=Xa0, CACE=CACE,
                                      math_contribution=math_offset)

        y = np.asarray(self.data[self.outcome])
        a = np.asarray(self.data[self.action])
        s = np.asarray(self.data[self.sample])
        r = np.asarray(self.data[self.positive])
        Z = np.asarray(self._PS_)
        W = np.asarray(self._SW_)
        X = np.asarray(self._X_)
        Xa1 = np.asarray(self._Xa1_)
        Xa0 = np.asarray(self._Xa0_)
        CACE = np.asarray(self._CACE_)

        # Calculating lower bound
        math_offset = np.dot(self._MATH_, lower).flatten()
        if init is None:
            init = [100., ] + [0., ]*CACE.shape[1] + [0., ]*Z.shape[1] + [0., ]*W.shape[1] + [0., ]*X.shape[1]

        estr = MEstimator(psi_synth_cace, init=init)
        estr.estimate(solver=solver, maxiter=100000)
        lbound = estr.theta[0]
        lbound_ci = estr.confidence_intervals()[0, 0]

        # Calculating upper bound
        math_offset = np.dot(self._MATH_, upper).flatten()
        estr = MEstimator(psi_synth_cace, init=init)
        estr.estimate(solver=solver, maxiter=100000)
        ubound = estr.theta[0]
        ubound_ci = estr.confidence_intervals()[0, 1]

        # Storing results
        self.bounds = lbound, ubound
        self.bounds_ci = lbound_ci, ubound_ci


def _msm_semiparametric_bootstrap_(params):
    d, sp, mp, msm_model, math_model, action = params

    # Setting actions in copy of data
    d1 = d.copy()
    d1[action] = 1
    W1 = patsy.dmatrix(msm_model, d1, return_type='dataframe')
    W1_star = patsy.dmatrix(math_model, d1, return_type='dataframe')
    ya1 = np.dot(W1, sp) + np.dot(W1_star, mp)

    d0 = d.copy()
    d0[action] = 0
    W0 = patsy.dmatrix(msm_model, d0, return_type='dataframe')
    W0_star = patsy.dmatrix(math_model, d0, return_type='dataframe')
    ya0 = np.dot(W0, sp) + np.dot(W0_star, mp)

    return np.mean(ya1 - ya0)


def _cace_semiparametric_bootstrap_(params):
    d, sp, mp, cace_model, math_model = params

    # Setting actions in copy of data
    V1 = patsy.dmatrix(cace_model, d, return_type='dataframe')
    V1_star = patsy.dmatrix(math_model, d, return_type='dataframe')
    cace = np.dot(V1, sp) + np.dot(V1_star, mp)
    return np.mean(cace)
