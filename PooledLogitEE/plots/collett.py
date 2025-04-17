######################################################################################################################
# Code to generate the risk function plots for Collett
#
# Paul Zivich (Last update: 2025/4/16)
######################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.utilities import spline

from plotting import twister_plot
from efuncs import ee_pooled_logit, pooled_logit_prediction


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


def psi_r1(theta):
    # Extracting parameters
    risks = theta[:params_r1]
    beta = theta[params_r1:]

    # Nuisance models
    ee_plog = psi_plogit_a1(theta=beta)

    # Predictions to get risk differences
    risk1 = pooled_logit_prediction(theta=beta, delta=y, t=t, X=W,
                                    times_to_predict=event_times_p1, measure='risk', unique_times=event_times_a1)
    ee_rd = risk1 - np.asarray(risks)[:, None]

    # Returning stacked estimating equations
    return np.vstack([ee_rd, ee_plog])


def psi_r0(theta):
    # Extracting parameters
    risks = theta[:params_r0]
    beta = theta[params_r0:]

    # Nuisance models
    ee_plog = psi_plogit_a0(theta=beta)

    # Predictions to get risk differences
    risk1 = pooled_logit_prediction(theta=beta, delta=y, t=t, X=W,
                                    times_to_predict=event_times_p0, measure='risk', unique_times=event_times_a0)
    ee_rd = risk1 - np.asarray(risks)[:, None]

    # Returning stacked estimating equations
    return np.vstack([ee_rd, ee_plog])


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


inits = [0., ]*params_r1 + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a1 - 1)
estr = MEstimator(psi_r1, init=inits)
estr.estimate()

r1_results = pd.DataFrame()
r1_results['time'] = event_times_p1
r1_results['risk'] = estr.theta[:params_r1]
r1_ci = estr.confidence_intervals()[:params_r1, :]
r1_results['lcl'] = r1_ci[:, 0]
r1_results['ucl'] = r1_ci[:, 1]

inits = [0., ]*params_r0 + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a0 - 1)
estr = MEstimator(psi_r0, init=inits)
estr.estimate()

r0_results = pd.DataFrame()
r0_results['time'] = event_times_p0
r0_results['risk'] = estr.theta[:params_r0]
r0_ci = estr.confidence_intervals()[:params_r0, :]
r0_results['lcl'] = r0_ci[:, 0]
r0_results['ucl'] = r0_ci[:, 1]

inits = ([0., ]*params_rd
         + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a1 - 1)
         + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a0 - 1))
estr = MEstimator(psi_rd, init=inits)
estr.estimate()

rd = estr.theta[:params_rd]
rd_ci = estr.confidence_intervals()[:params_rd, :]
rd_results = pd.DataFrame()
rd_results['time'] = event_times
rd_results['rd'] = rd
rd_results['lcl'] = rd_ci[:, 0]
rd_results['ucl'] = rd_ci[:, 1]
print("Results -- Disjoint")
print(rd_results.tail(1))

# Generating Figure
fig, axes = plt.subplots(2, 2, width_ratios=[3, 1], figsize=[7.2, 3.6*2])
ax0 = axes[0, 0]
ax0.fill_between(r1_results['time'], r1_results['lcl'], r1_results['ucl'], color='blue', alpha=0.1, step='post')
ax0.fill_between(r0_results['time'], r0_results['lcl'], r0_results['ucl'], color='red', alpha=0.1, step='post')
ax0.step(r1_results['time'], r1_results['risk'], color='blue', where='post', label='Novel')
ax0.step(r0_results['time'], r0_results['risk'], color='red', where='post', label='Standard')
ax0.set_xlim([0, 60])
ax0.set_ylim([0, 1])
ax0.set_xlabel("Time (months)")
ax0.set_ylabel("Risk")
ax0.legend(loc='upper left')
ax0.text(-9, 0.97, "A", fontsize=14)

ax1 = axes[0, 1]
twister_plot(rd_results, point='rd', lcl='lcl', ucl='ucl', time='time', color='k', ax=ax1, favors=False)
ax1.set_ylim([0, 60])
ax1.set_ylabel("Time (months)")
ax1.set_xlabel("Risk Difference")

######################################
# Example 1b: Splines

t_steps = np.asarray(range(1, 60))
tp_intervals = list(range(0, 60))
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


def psi_r1(theta):
    # Extracting parameters
    risks = theta[:params_risk]
    beta = theta[params_risk:]

    # Nuisance models
    ee_plog = psi_plogit_spline_a1(theta=beta)

    # Predictions to get risk differences
    risk1 = pooled_logit_prediction(theta=beta, t=t, delta=y, X=W, S=s_matrix,
                                    times_to_predict=tp_intervals, measure='risk')
    ee_rd = risk1 - np.asarray(risks)[:, None]

    # Returning stacked estimating equations
    return np.vstack([ee_rd, ee_plog])


def psi_r0(theta):
    # Extracting parameters
    risks = theta[:params_risk]
    beta = theta[params_risk:]

    # Nuisance models
    ee_plog = psi_plogit_spline_a0(theta=beta)

    # Predictions to get risk differences
    risk0 = pooled_logit_prediction(theta=beta, t=t, delta=y, X=W, S=s_matrix,
                                    times_to_predict=tp_intervals, measure='risk')

    ee_rd = risk0 - np.asarray(risks)[:, None]

    # Returning stacked estimating equations
    return np.vstack([ee_rd, ee_plog])


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


inits = list(np.linspace(0, 0.5, params_risk)) + [0, 0., -4., ] + [0., ]*4
estr = MEstimator(psi_r1, init=inits)
estr.estimate()

r1_results = pd.DataFrame()
r1_results['time'] = tp_intervals
r1_results['risk'] = estr.theta[:params_risk]
r1_ci = estr.confidence_intervals()[:params_risk, :]
r1_results['lcl'] = r1_ci[:, 0]
r1_results['ucl'] = r1_ci[:, 1]

inits = list(np.linspace(0., 0.5, params_risk)) + [0., 0., -4., ] + [0., ]*4
estr = MEstimator(psi_r0, init=inits)
estr.estimate()

r0_results = pd.DataFrame()
r0_results['time'] = tp_intervals
r0_results['risk'] = estr.theta[:params_risk]
r0_ci = estr.confidence_intervals()[:params_risk, :]
r0_results['lcl'] = r0_ci[:, 0]
r0_results['ucl'] = r0_ci[:, 1]

inits = [0., ]*params_risk + [0., 0., -4., ] + [0., ]*4 + [0., 0., -4., ] + [0., ]*4
estr = MEstimator(psi_rd, init=inits)
estr.estimate()

rd_results = pd.DataFrame()
rd_results['time'] = tp_intervals
rd_results['rd'] = estr.theta[:params_risk]
rd_ci = estr.confidence_intervals()[:params_risk, :]
rd_results['lcl'] = rd_ci[:, 0]
rd_results['ucl'] = rd_ci[:, 1]
print("Results -- Spline")
print(rd_results.tail(1))

# fig, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[3, 1], figsize=[7.2, 3.6])

ax2 = axes[1, 0]
ax2.fill_between(r1_results['time'], r1_results['lcl'], r1_results['ucl'], color='blue', alpha=0.1)
ax2.fill_between(r0_results['time'], r0_results['lcl'], r0_results['ucl'], color='red', alpha=0.1)
ax2.plot(r1_results['time'], r1_results['risk'], color='blue', label='Novel')
ax2.plot(r0_results['time'], r0_results['risk'], color='red', label='Standard')
ax2.set_xlim([0, 60])
ax2.set_ylim([0, 1])
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Risk")
ax2.legend(loc='upper left')
ax2.text(-9, 0.97, "B", fontsize=14)

ax3 = axes[1, 1]
twister_plot(rd_results, point='rd', lcl='lcl', ucl='ucl', time='time', color='k', ax=ax3, favors=False, step=False)
ax3.set_xticks([-0.5, 0., 0.5])
ax3.set_ylim([0, 60])
ax3.set_ylabel("Time (days)")
ax3.set_xlabel("Risk Difference")
plt.tight_layout()
plt.savefig("figure_collett.png", format='png', dpi=300)
plt.show()
