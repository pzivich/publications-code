#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Synthesis estimators
#
# Paul Zivich
#######################################################################################################################

import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from delicatessen import MEstimator
from multiprocessing import Pool

from efuncs import ee_synth_aipw_msm, ee_synth_aipw_cace


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
        self.copy_ind = "__msm_copy__"
        self.data[self.copy_ind] = 0
        d = self.data.loc[self.data[self.sample] == 1].copy()
        self.data_a1 = d.copy()
        self.data_a1[self.action] = 1
        self.data_a1[self.outcome] = np.nan
        self.data_a1[self.copy_ind] = 1
        self.data_a0 = d.copy()
        self.data_a0[self.action] = 0
        self.data_a0[self.outcome] = np.nan
        self.data_a0[self.copy_ind] = 1
        self.stack_data = pd.concat([self.data, self.data_a1, self.data_a0],
                                    ignore_index=True)

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
        self._MSM_ = None
        self._msm_model_ = None
        self._MATH_ = None
        self._math_model_ = None
        self._math_param_ = None

    def action_model(self, model):
        self._PS_ = patsy.dmatrix(model, self.stack_data,
                                  return_type='dataframe',
                                  NA_action=patsy.NAAction(NA_types=[]))
        self._action_model_ = model

    def sample_model(self, model):
        self._SW_ = patsy.dmatrix(model, self.stack_data,
                                  return_type='dataframe',
                                  NA_action=patsy.NAAction(NA_types=[]))
        self._sample_model_ = model

    def outcome_model(self, model):
        self._X_ = patsy.dmatrix(model, self.stack_data,
                                 return_type='dataframe',
                                 NA_action=patsy.NAAction(NA_types=[]))
        self._outcome_model_ = model

    def marginal_structural_model(self, model):
        self._MSM_ = patsy.dmatrix(model, self.stack_data,
                                   return_type='dataframe',
                                   NA_action=patsy.NAAction(NA_types=[]))
        self._msm_model_ = model

    def math_model(self, model, parameters):
        self._MATH_ = patsy.dmatrix(model, self.stack_data,
                                    return_type='dataframe',
                                    NA_action=patsy.NAAction(NA_types=[]))
        self._math_model_ = model
        self._math_param_ = parameters

    def estimate(self, mc_iterations, n_cpus=1):
        # Preparing data for estimation procedure
        params = []
        d1 = self.data.loc[self.data[self.sample] == 1].copy()
        n1 = d1.shape[0]
        d0 = self.data.loc[self.data[self.sample] == 0].copy()
        n0 = d0.shape[0]
        meta_data = [self.outcome, self.action, self.sample, self.positive,
                     self._action_model_, self._sample_model_, self._outcome_model_,
                     self._msm_model_, self._math_model_]

        # Packaging data set here so no seed issues later
        for i in range(mc_iterations):
            d1s = d1.sample(n=n1, replace=True)
            d0s = d0.sample(n=n0, replace=True)
            ds = pd.concat([d1s, d0s])
            params.append([ds, self._math_param_[i], meta_data])

        # Running point estimation with multiple CPUs
        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(_msm_resample_,  # Call outside function
                                      params))         # provide packed input

        # Processing results
        self.estimates = estimates
        self.ace = np.nanmedian(estimates)
        self.ace_var = np.nanvar(estimates)
        self.ace_ci = np.nanpercentile(estimates, q=[2.5, 97.5])

    def point_estimate(self, math_parameters):
        # Computing the point estimate for a single random draw
        a = np.asarray(self.stack_data[self.action])
        s = np.asarray(self.stack_data[self.sample])
        r = np.asarray(self.stack_data[self.positive])
        math_offset = np.dot(self._MATH_, math_parameters).flatten()

        # Fitting action process model
        f = sm.families.Binomial()
        act = smf.glm(self.action+" ~ "+self._action_model_,
                      self.stack_data.loc[self.stack_data[self.copy_ind] == 0],
                      family=f).fit()
        pr_a1 = act.predict(self.stack_data)
        iptw = 1 / np.where(a == 1, pr_a1, 1 - pr_a1)

        # Fitting sample process model
        f = sm.families.Binomial()
        smp = smf.glm(self.sample+" ~ "+self._sample_model_,
                      self.stack_data.loc[(self.stack_data[self.copy_ind] == 0) & r],
                      family=f).fit()
        pr_s1 = smp.predict(self.stack_data)
        iosw = s + (1-s)*(pr_s1 / (1 - pr_s1))

        # Fitting the outcome process model
        ipw = iptw*iosw
        f = sm.families.Gaussian()
        out = smf.glm(self.outcome + " ~ "+self._outcome_model_,
                      self.stack_data,
                      freq_weights=ipw, family=f).fit()
        self.stack_data['ydiffhat'] = out.predict(self.stack_data)

        # Fitting the MSM
        msm = smf.glm("ydiffhat ~ " + self._msm_model_,
                      self.stack_data.loc[(self.stack_data[self.copy_ind] == 1) & r],
                      family=f).fit()
        ypred = msm.predict(self.stack_data) + math_offset
        ya1 = ypred[(self.stack_data[self.action] == 1) & (self.stack_data[self.sample] == 1)]
        ya0 = ypred[(self.stack_data[self.action] == 0) & (self.stack_data[self.sample] == 1)]
        return np.mean(ya1) - np.mean(ya0)


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
        # Preparing data for estimation procedure
        params = []
        d1 = self.data.loc[self.data[self.sample] == 1].copy()
        n1 = d1.shape[0]
        d0 = self.data.loc[self.data[self.sample] == 0].copy()
        n0 = d0.shape[0]
        meta_data = [self.outcome, self.action, self.sample, self.positive,
                     self._action_model_, self._sample_model_, self._outcome_model_,
                     self._cace_model_, self._math_model_]

        # Packaging data set here so no seed issues later
        for i in range(mc_iterations):
            d1s = d1.sample(n=n1, replace=True)
            d0s = d0.sample(n=n0, replace=True)
            ds = pd.concat([d1s, d0s])
            params.append([ds, self._math_param_[i], meta_data])

        # Running point estimation with multiple CPUs
        with Pool(processes=n_cpus) as pool:
            estimates = list(pool.map(_cace_resample_,  # Call outside function
                                      params))          # provide packed input

        # Processing results
        self.estimates = estimates
        self.ace = np.nanmedian(estimates)
        self.ace_var = np.nanvar(estimates)
        self.ace_ci = np.nanpercentile(estimates, q=[2.5, 97.5])

    def point_estimate(self, math_parameters):
        # Computing the point estimate for a single random draw
        a = np.asarray(self.data[self.action])
        s = np.asarray(self.data[self.sample])
        r = np.asarray(self.data[self.positive])
        math_offset = np.dot(self._MATH_, math_parameters).flatten()

        # Fitting action process model
        f = sm.families.Binomial()
        act = smf.glm(self.action+" ~ "+self._action_model_,
                      self.data,
                      family=f).fit()
        pr_a1 = act.predict(self.data)
        iptw = 1 / np.where(a == 1, pr_a1, 1 - pr_a1)

        # Fitting sample process model
        f = sm.families.Binomial()
        smp = smf.glm(self.sample+" ~ "+self._sample_model_,
                      self.data.loc[r == 1],
                      family=f).fit()
        pr_s1 = smp.predict(self.data)
        iosw = s + (1-s)*(pr_s1 / (1 - pr_s1))

        # Fitting the outcome process model
        ipw = iptw*iosw
        f = sm.families.Gaussian()
        out = smf.glm(self.outcome+" ~ "+self._outcome_model_,
                      self.data,
                      freq_weights=ipw, family=f).fit()
        self.data['ydiffhat'] = out.predict(self.data_a1) - out.predict(self.data_a0)

        # Fitting the CACE model
        f = sm.families.Gaussian()
        cac = smf.glm('ydiffhat ~ '+self._cace_model_, self.data.loc[(s == 1) & (r == 1)], family=f).fit()
        ydiff = cac.predict(self.data) + math_offset

        ydiff_s = ydiff[self.data[self.sample] == 1]
        return np.mean(ydiff_s)

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
        estr.estimate(solver=solver, maxiter=10000)
        lbound = estr.theta[0]
        lbound_ci = estr.confidence_intervals()[0, 0]

        # Calculating upper bound
        math_offset = np.dot(self._MATH_, upper).flatten()
        estr = MEstimator(psi_synth_cace, init=init)
        estr.estimate(solver=solver, maxiter=10000)
        ubound = estr.theta[0]
        ubound_ci = estr.confidence_intervals()[0, 1]

        # Storing results
        self.bounds = lbound, ubound
        self.bounds_ci = lbound_ci, ubound_ci


def _msm_resample_(params):
    # Internal function to call the synthesis point procedure for multi-CPU
    data, math, meta = params

    saipw = SynthesisMSM(data=data, outcome=meta[0], action=meta[1],
                         sample=meta[2], positive_region=meta[3])
    saipw.action_model(meta[4])
    saipw.sample_model(meta[5])
    saipw.outcome_model(meta[6])
    saipw.marginal_structural_model(meta[7])
    saipw.math_model(meta[8], parameters=None)
    try:
        est = saipw.point_estimate(math_parameters=math)
    except:
        est = np.nan
    return est


def _cace_resample_(params):
    # Internal function to call the synthesis point procedure for multi-CPU
    data, math, meta = params

    saipw = SynthesisCACE(data=data, outcome=meta[0], action=meta[1],
                          sample=meta[2], positive_region=meta[3])
    saipw.action_model(meta[4])
    saipw.sample_model(meta[5])
    saipw.outcome_model(meta[6])
    saipw.cace_model(meta[7])
    saipw.math_model(meta[8], parameters=None)
    try:
        est = saipw.point_estimate(math_parameters=math)
    except:
        est = np.nan
    return est
