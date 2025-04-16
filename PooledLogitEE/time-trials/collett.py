import warnings
import numpy as np
import pandas as pd
from time import time
from delicatessen import MEstimator
from delicatessen.utilities import spline

from standard import PooledLogitGComputation
from ..efuncs import ee_pooled_logit, pooled_logit_prediction

warnings.filterwarnings("ignore")

# DISJOINT INDICATOR
# EE implementation
# RUNTIME: 0.3128323554992676
#
# Standard -- 1 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
#      -0.189233  0.025134 -0.499957  0.121492
#
# RUNTIME: 495.21120142936707
# 489.70824694633484 523.9646537303925
#
# Standard -- 7 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
#      -0.189233  0.025134 -0.499957  0.121492
# RUNTIME: 171.3387439250946
#
# SPLINES
# EE implementation
# RUNTIME: 1.199704647064209
#
# Standard -- 1 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
#      -0.177775  0.020096 -0.455622  0.100072
# RUNTIME: 53.583834409713745
#
# Standard -- 7 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
#      -0.177775  0.020096 -0.455622  0.100072
# RUNTIME: 15.764265298843384


if __name__ == "__main__":
    #########################################################################
    # Setup data
    d = pd.read_csv("../data/collett.dat", sep='\s+',
                    names=['patient', 'time', 'delta', 'treat', 'init', 'size'])
    d['novel'] = d['treat'] - 1
    d['intercept'] = 1

    d1 = d.copy()
    d1['novel'] = 1
    d0 = d.copy()
    d0['novel'] = 0

    a = np.asarray(d['novel'])
    t = np.asarray(d['time'])
    y = np.asarray(d['delta'])
    W = np.asarray(d[['init', 'size', ]])

    ######################################
    # Example 1a: Disjoint Indicator
    print("DISJOINT INDICATOR")

    event_times = [0, ] + list(np.unique(d.loc[d['delta'] == 1, 'time'])) + [59, ]
    event_times_a1 = list(np.unique(d.loc[(d['delta'] == 1) & (d['novel'] == 1), 'time']))
    event_times_p1 = [0, ] + event_times_a1 + [59, ]
    event_times_a0 = list(np.unique(d.loc[(d['delta'] == 1) & (d['novel'] == 0), 'time']))
    event_times_p0 = [0, ] + event_times_a0 + [59, ]
    params_rd = len(event_times)
    params_r1 = len(event_times_p1)
    params_r0 = len(event_times_p0)
    params_plr_a1 = len(event_times_a1)
    params_plr_a0 = len(event_times_a0)

    def psi_plogit_a1(theta):
        ee_plog = ee_pooled_logit(theta, t=t, delta=y, X=W, unique_times=event_times_a1)
        ee_plog = ee_plog * (a == 1)[None, :]
        return ee_plog

    def psi_plogit_a0(theta):
        ee_plog = ee_pooled_logit(theta, t=t, delta=y, X=W, unique_times=event_times_a0)
        ee_plog = ee_plog * (a == 0)[None, :]
        return ee_plog

    def psi_rd(theta):
        # Extracting parameters
        rds = theta[:params_rd]
        idPLR = params_rd + W.shape[1] + params_plr_a1
        beta1 = theta[params_rd: idPLR]
        beta0 = theta[idPLR:]

        # Nuisance models
        ee_plog1 = psi_plogit_a1(theta=beta1)
        ee_plog0 = psi_plogit_a0(theta=beta0)

        # Predictions to get risk differences
        risk1 = pooled_logit_prediction(theta=beta1, delta=y, t=t, X=W,
                                        times_to_predict=event_times, measure='risk', unique_times=event_times_a1)
        risk0 = pooled_logit_prediction(theta=beta0, delta=y, t=t, X=W,
                                        times_to_predict=event_times, measure='risk', unique_times=event_times_a0)
        ee_rd = (risk1 - risk0) - np.asarray(rds)[:, None]

        # Returning stacked estimating equations
        return np.vstack([ee_rd, ee_plog1, ee_plog0])

    print("EE implementation")
    run_times = []
    for i in range(5):
        start = time()
        inits = ([0., ]*params_rd
                 + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a1 - 1)
                 + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a0 - 1))
        estr = MEstimator(psi_rd, init=inits)
        estr.estimate()
        run_times.append(time() - start)

    print("RUNTIME:", np.median(run_times))
    print(run_times)

    print("Standard -- 1 CPU")
    runtimes = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.outcome_model(model='novel*(init + size + C(time))')
        results = plgc.estimate(n_cpus=1, bs_iterations=1000, seed=80921)
        runtimes.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(runtimes))
    print(np.min(runtimes), np.max(runtimes))

    print("Standard -- 7 CPU")
    runtimes = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.outcome_model(model='novel*(init + size + C(time))')
        results = plgc.estimate(n_cpus=7, bs_iterations=1000, seed=80921)
        runtimes.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(runtimes))
    print(np.min(runtimes), np.max(runtimes))

    ######################################
    # Example 1b: Splines
    print("SPLINES")

    t_steps = np.asarray(range(1, 60))
    tp_intervals = [0, ] + list(range(1, 59, 1)) + [59, ]
    params_risk = len(tp_intervals)

    intercept = np.ones(t_steps.shape)[:, None]
    time_splines = spline(t_steps, knots=[10, 20, 30, 40],
                          power=2, restricted=True, normalized=False)
    s_matrix = np.concatenate([intercept, t_steps[:, None], time_splines], axis=1)

    def psi_plogit_spline_a1(theta):
        ee_plog = ee_pooled_logit(theta=theta, t=t, delta=y, X=W, S=s_matrix)
        ee_plog = ee_plog * (a == 1)[None, :]
        return ee_plog

    def psi_plogit_spline_a1w(theta):
        ee_plog = ee_pooled_logit(theta=theta, t=t, delta=y, X=W, S=s_matrix)
        ee_plog = ee_plog * (a == 1)[None, :]
        return ee_plog

    def psi_plogit_spline_a0(theta):
        ee_plog = ee_pooled_logit(theta=theta, t=t, delta=y, X=W, S=s_matrix)
        ee_plog = ee_plog * (a == 0)[None, :]
        return ee_plog

    def psi_rd(theta):
        # Extracting parameters
        risks = theta[:params_risk]
        idPLRM = params_risk + 7
        beta1 = theta[params_risk:idPLRM]
        beta0 = theta[idPLRM:]

        # Nuisance models
        ee_plog1 = psi_plogit_spline_a1(theta=beta1)
        ee_plog0 = psi_plogit_spline_a0(theta=beta0)

        # Predictions to get risk differences
        risk1 = pooled_logit_prediction(theta=beta1, t=t, delta=y, X=W, S=s_matrix,
                                        times_to_predict=tp_intervals, measure='risk')
        risk0 = pooled_logit_prediction(theta=beta0, t=t, delta=y, X=W, S=s_matrix,
                                        times_to_predict=tp_intervals, measure='risk')
        ee_rd = (risk1 - risk0) - np.asarray(risks)[:, None]

        # Returning stacked estimating equations
        return np.vstack([ee_rd, ee_plog1, ee_plog0])

    print("EE implementation")
    run_times = []
    for i in range(5):
        start = time()
        inits = [0., ] * params_risk + [0., 0., -4., ] + [0., ]*4 + [0., 0., -4., ] + [0., ]*4
        estr = MEstimator(psi_rd, init=inits)
        estr.estimate()
        run_times.append(time() - start)

    print("RUNTIME:", np.mean(run_times))
    print(run_times)

    print("Standard -- 1 CPU")
    runtimes = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.create_time_splines(term=2, knots=[10, 20, 30, 40])
        plgc.outcome_model(model='novel*(init + size + time + time_spline1 + time_spline2 + time_spline3)')
        results = plgc.estimate(n_cpus=1, bs_iterations=1000, seed=80921)
        runtimes.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(runtimes))
    print(np.min(runtimes), np.max(runtimes))

    print("Standard -- 7 CPU")
    runtimes = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.create_time_splines(term=2, knots=[10, 20, 30, 40])
        plgc.outcome_model(model='novel*(init + size + time + time_spline1 + time_spline2 + time_spline3)')
        results = plgc.estimate(n_cpus=7, bs_iterations=1000, seed=80921)
        runtimes.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(runtimes))
    print(np.min(runtimes), np.max(runtimes))
