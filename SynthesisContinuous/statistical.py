#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Statistical estimators
#
# Paul Zivich
#######################################################################################################################

import patsy
import numpy as np
from delicatessen import MEstimator

from efuncs import ee_stat_aipw, ee_stat_msm, ee_stat_cace


class StatAIPW:
    """Weighted regression statistical augmented inverse probability weighting (AIPW) estimator.

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
    """
    def __init__(self, data, outcome, action, sample):
        self.data = data
        self.outcome = outcome
        self.action = action
        self.sample = sample

        # Generating data under the all policies
        self.data_a1 = self.data.copy()
        self.data_a1[self.action] = 1
        self.data_a0 = self.data.copy()
        self.data_a0[self.action] = 0

        # Initialize storage for results
        self.ace = None
        self.ace_var = None
        self.ace_ci = None

        self._PS_ = None
        self._action_model_ = None
        self._SW_ = None
        self._sample_model_ = None
        self._X_ = None
        self._Xa1_ = None
        self._Xa0_ = None
        self._outcome_model_ = None

    def action_model(self, model):
        # Processing the model specification into the design matrix
        self._PS_ = patsy.dmatrix(model, self.data,
                                  return_type='dataframe',
                                  NA_action=patsy.NAAction(NA_types=[]))
        self._action_model_ = model

    def sample_model(self, model):
        # Processing the model specification into the design matrix
        self._SW_ = patsy.dmatrix(model, self.data,
                                  return_type='dataframe')
        self._sample_model_ = model

    def outcome_model(self, model):
        # Processing the model specification into the design matrices
        self._X_ = patsy.dmatrix(model, self.data,
                                 return_type='dataframe',
                                 NA_action=patsy.NAAction(NA_types=[]))
        self._Xa1_ = patsy.dmatrix(model, self.data_a1,
                                   return_type='dataframe')
        self._Xa0_ = patsy.dmatrix(model, self.data_a0,
                                   return_type='dataframe')
        self._outcome_model_ = model

    def estimate(self, init=None, solver='lm'):
        # Preparing data for the design matrix
        y = np.asarray(self.data[self.outcome])
        a = np.asarray(self.data[self.action])
        s = np.asarray(self.data[self.sample])
        Z = np.asarray(self._PS_)
        W = np.asarray(self._SW_)
        X = np.asarray(self._X_)
        Xa1 = np.asarray(self._Xa1_)
        Xa0 = np.asarray(self._Xa0_)

        # Creating initial root-finding values if none given
        if init is None:
            init = [100., 200., 100., ] + [0., ]*Z.shape[1] + [0., ]*W.shape[1] + [0., ]*X.shape[1]

        # Estimating equation solving procedure
        def psi_aipw(theta):
            return ee_stat_aipw(theta=theta, y=y, a=a, s=s,
                                Z=Z, W=W, X=X, Xa1=Xa1, Xa0=Xa0)

        estr = MEstimator(psi_aipw, init=init)
        estr.estimate(solver=solver, maxiter=100000)
        ci = estr.confidence_intervals(alpha=0.05)

        # Storing results from the procedure
        self.ace = estr.theta[0]
        self.ace_var = estr.variance[0, 0]
        self.ace_ci = ci[0, :]


class StatMSM:
    """Estimate the parameters of a marginal structural model using a randomized trial in the target population.

    Parameters
    ----------
    data :
        Data set containing variables of interest
    outcome : str
        Outcome column label
    """

    def __init__(self, data, outcome):
        self.data = data
        self.outcome = outcome

        # Initialize storage for results
        self.params = None
        self.params_var = None

        self._MSM_ = None

    def marginal_structural_model(self, model):
        # Processing the model specification into the design matrix
        self._MSM_ = patsy.dmatrix(model, self.data,
                                   return_type='dataframe')

    def estimate(self, init=None, solver='lm'):
        # Preparing data for the design matrix
        y = np.asarray(self.data[self.outcome])
        MSM = np.asarray(self._MSM_)

        # Creating initial root-finding values if none given
        if init is None:
            init = [0., ]*MSM.shape[1]

        # Estimating equation solving procedure
        def psi_aipw(theta):
            return ee_stat_msm(theta=theta, y=y, MSM=MSM)

        estr = MEstimator(psi_aipw, init=init)
        estr.estimate(solver=solver, maxiter=100000)

        # Storing results from the procedure
        self.params = estr.theta
        self.params_var = np.diag(estr.variance)


class StatCACE:
    """Estimate the parameters of a conditional average causal effect model using a randomized trial in the target
    population. Here, an outcome model is used to generate predictions under each a.

    Parameters
    ----------
    data :
        Data set containing variables of interest
    outcome : str
        Outcome column label
    action : str
        Action column label
    """

    def __init__(self, data, outcome, action):
        self.data = data
        self.outcome = outcome
        self.action = action

        # Generating data under the all-policy
        self.data_a1 = self.data.copy()
        self.data_a1[self.action] = 1
        self.data_a0 = self.data.copy()
        self.data_a0[self.action] = 0

        # Initialize storage for results
        self.params = None
        self.params_var = None

        self._X_ = None
        self._Xa1_ = None
        self._Xa0_ = None
        self._outcome_model_ = None
        self._CACE_ = None

    def outcome_model(self, model):
        # Processing the model specification into the design matrix
        self._X_ = patsy.dmatrix(model, self.data,
                                 return_type='dataframe')
        self._Xa1_ = patsy.dmatrix(model, self.data_a1,
                                   return_type='dataframe')
        self._Xa0_ = patsy.dmatrix(model, self.data_a0,
                                   return_type='dataframe')
        self._outcome_model_ = model

    def cace_model(self, model):
        # Processing the model specification into the design matrix
        self._CACE_ = patsy.dmatrix(model, self.data,
                                    return_type='dataframe')

    def estimate(self, init=None, solver='lm'):
        # Preparing data for the design matrix
        y = np.asarray(self.data[self.outcome])
        X = np.asarray(self._X_)
        Xa1 = np.asarray(self._Xa1_)
        Xa0 = np.asarray(self._Xa0_)
        CACE = np.asarray(self._CACE_)

        # Creating initial root-finding values if none given
        if init is None:
            init = [0., ]*CACE.shape[1] + [0., ]*X.shape[1]

        # Estimating equation solving procedure
        def psi_aipw(theta):
            return ee_stat_cace(theta=theta, y=y, X=X, Xa1=Xa1, Xa0=Xa0, CACE=CACE)

        estr = MEstimator(psi_aipw, init=init)
        estr.estimate(solver=solver, maxiter=100000)

        # Storing results from the procedure
        self.params = estr.theta
        self.params_var = np.diag(estr.variance)
