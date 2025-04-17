######################################################################################################################
# Code to simplify the estimator calls for the simulation experiments
#
# Paul Zivich (Last update: 2025/4/17)
######################################################################################################################

import numpy as np
from delicatessen import MEstimator
from delicatessen.utilities import spline

from efuncs import ee_pooled_logit, pooled_logit_prediction


class PooledLogitEE:
    def __init__(self, data, time, delta, action):
        self.data = data.copy()
        self.time = time
        self.delta = delta
        self.action = action

        # Storage for all parameters
        self.point = []
        self.variance = []
        self.lower_ci = []
        self.upper_ci = []

        # Nuisance parameter stuff
        self.x_matrix = None
        self.pregen_matrix = None
        self.t_steps = list(range(1, 31))
        self.unique_event_times = list(np.unique(self.data.loc[self.data[self.delta] == 1, self.time]))
        self.unique_event_times_a1 = list(np.unique(self.data.loc[(self.data[self.delta] == 1)
                                                                  & (self.data[self.action] == 1), self.time]))
        self.unique_event_times_a0 = list(np.unique(self.data.loc[(self.data[self.delta] == 1)
                                                                  & (self.data[self.action] == 0), self.time]))
        self.design_parser = None
        self.time1_inits, self.time0_inits = None, None
        self.x_inits = None

    def nuisance_model(self, covariates, time):
        # Setting up time design matrices
        if time == 'constant':
            self.pregen_matrix = self.time_fixed_matrix(interval=self.t_steps)
            self.time1_inits = [-5., ]
            self.time0_inits = [-5., ]
        elif time == 'linear':
            self.pregen_matrix = self.time_linear_matrix(interval=self.t_steps)
            self.time1_inits = [-4., 0., ]
            self.time0_inits = [-4., 0., ]
        elif time == 'log':
            self.pregen_matrix = self.time_log_matrix(interval=self.t_steps)
            self.time1_inits = [-4., 0., ]
            self.time0_inits = [-4., 0., ]
        elif time == 'spline':
            self.pregen_matrix = self.time_spline_matrix(interval=self.t_steps)
            self.time1_inits = [-4., 0., 0., 0., 0., 0., ]
            self.time0_inits = [-4., 0., 0., 0., 0., 0., ]
        elif time == 'disjoint':
            self.design_parser = None
            self.time1_inits = [-2., ] + [0., ]*(len(self.unique_event_times_a1) - 1)
            self.time0_inits = [-2., ] + [0., ]*(len(self.unique_event_times_a0) - 1)
        else:
            raise ValueError("Invalid option for time design matrix")

        # Covariate design matrix
        self.x_matrix = np.asarray(self.data[covariates])
        self.x_inits = [0., ]*self.x_matrix.shape[1]

    def estimate(self):
        if self.x_matrix is None:
            raise ValueError("nuisance_model() must be called before estimate()")

        def psi_plogit_a1(theta):
            ee_plog = ee_pooled_logit(theta=theta,
                                      t=self.data[self.time], delta=self.data[self.delta],
                                      X=self.x_matrix, S=self.pregen_matrix,
                                      unique_times=self.unique_event_times_a1)
            ee_plog = ee_plog * np.asarray(self.data[self.action] == 1)[None, :]
            return ee_plog

        def psi_plogit_a0(theta):
            ee_plog = ee_pooled_logit(theta=theta,
                                      t=self.data[self.time], delta=self.data[self.delta],
                                      X=self.x_matrix, S=self.pregen_matrix,
                                      unique_times=self.unique_event_times_a0)
            ee_plog = ee_plog * np.asarray(self.data[self.action] == 0)[None, :]
            return ee_plog

        def psi_rd(theta):
            # Extracting parameters
            rds = theta[:3]
            idM1 = 3 + len(self.x_inits) + len(self.time1_inits)
            beta1 = theta[3: idM1]
            beta0 = theta[idM1:]

            # Nuisance models
            ee_plog1 = psi_plogit_a1(theta=beta1)
            ee_plog0 = psi_plogit_a0(theta=beta0)

            # Predictions to get risk differences
            risk1 = pooled_logit_prediction(theta=beta1, t=self.data[self.time], delta=self.data[self.delta],
                                            X=self.x_matrix, S=self.pregen_matrix,
                                            times_to_predict=[10, 20, 30], measure='risk',
                                            unique_times=self.unique_event_times_a1)
            risk0 = pooled_logit_prediction(theta=beta0, t=self.data[self.time], delta=self.data[self.delta],
                                            X=self.x_matrix, S=self.pregen_matrix,
                                            times_to_predict=[10, 20, 30], measure='risk',
                                            unique_times=self.unique_event_times_a0)
            ee_rd = (risk1 - risk0) - np.asarray(rds)[:, None]

            # Returning stacked estimating equations
            return np.vstack([ee_rd, ee_plog1, ee_plog0])

        init_vals = self.x_inits + self.time1_inits
        estr_nuisance1 = MEstimator(psi_plogit_a1, init=init_vals)
        estr_nuisance1.estimate(solver='lm', maxiter=50000)
        init_nuisance_a1 = list(estr_nuisance1.theta)

        init_vals = self.x_inits + self.time0_inits
        estr_nuisance0 = MEstimator(psi_plogit_a0, init=init_vals)
        estr_nuisance0.estimate(solver='lm', maxiter=50000)
        init_nuisance_a0 = list(estr_nuisance0.theta)

        init_vals = [0., 0., 0., ] + init_nuisance_a1 + init_nuisance_a0
        estr = MEstimator(psi_rd, init=init_vals)
        estr.estimate(solver='lm', maxiter=50000)

        self.point = estr.theta[:3]
        self.variance = np.diag(estr.variance)[:3]
        self.lower_ci = estr.confidence_intervals()[:3, 0]
        self.upper_ci = estr.confidence_intervals()[:3, 1]

    @staticmethod
    def time_fixed_matrix(interval):
        ones = np.ones(len(interval))
        return np.asarray([ones, ]).T

    @staticmethod
    def time_linear_matrix(interval):
        ones = np.ones(len(interval))
        tlin = ones * interval
        matrix = [ones, tlin, ]
        return np.asarray(matrix).T

    @staticmethod
    def time_log_matrix(interval):
        ones = np.ones(len(interval))
        tlin = ones * interval
        matrix = [ones, np.log(tlin), ]
        return np.asarray(matrix).T

    @staticmethod
    def time_spline_matrix(interval):
        ones = np.ones(len(interval))[:, None]
        tlin = np.asarray(interval)
        tmat = spline(tlin, knots=[5, 10, 15, 20, 25],
                      power=2, restricted=True, normalized=False)
        matrix = np.concatenate([ones, tlin[:, None], tmat], axis=1)
        return matrix
