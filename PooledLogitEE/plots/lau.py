import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.utilities import spline

from efuncs import ee_pooled_logit, pooled_logit_prediction
from plotting import twister_plot

#########################################################################
# Setup data
d = pd.read_csv("../data/lau.csv")

# Generating splines for continuous variables
d[['cd4_sp1', 'cd4_sp2']] = spline(d['cd4nadir'], knots=[2.1, 3.5, 5.2], power=2, restricted=True, normalized=False)
d[['age_sp1', 'age_sp2']] = spline(d['ageatfda'], knots=[25, 35, 50], power=2, restricted=True, normalized=False)

# Maxing maximum follow-up 10 years
d['event'] = np.where(d['eventtype'] == 2, 1, 0)
d['event'] = np.where(d['t'] > 10, 0, d['event'])
d['t'] = np.where(d['t'] > 10, 10, d['t'])

# Transforming time into days
d['days'] = np.ceil(d['t'] * 365.25)
d['days'].astype(int)
d['months'] = np.ceil(d['days'] / 30.437)
d['months'].astype(int)

time_var = 'days'
delta_var = 'event'
act_var = 'BASEIDU'
a = np.asarray(d[act_var])
t = np.asarray(d[time_var])
y = np.asarray(d[delta_var])
W = np.asarray(d[['black', 'cd4nadir', 'cd4_sp1', 'cd4_sp2', 'ageatfda', 'age_sp1', 'age_sp2']])


######################################
# Example 2a: Disjoint Indicator

event_times = [0, ] + list(np.unique(d.loc[d[delta_var] == 1, time_var])) + [np.max(t), ]
event_times_a1 = list(np.unique(d.loc[(d[delta_var] == 1) & (d[act_var] == 1), time_var]))
event_times_p1 = [0, ] + event_times_a1 + [np.max(t), ]
event_times_a0 = list(np.unique(d.loc[(d[delta_var] == 1) & (d[act_var] == 0), time_var]))
event_times_p0 = [0, ] + event_times_a0 + [np.max(t), ]
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
    risk0 = pooled_logit_prediction(theta=beta, delta=y, t=t, X=W,
                                    times_to_predict=event_times_p0, measure='risk', unique_times=event_times_a0)
    ee_rd = risk0 - np.asarray(risks)[:, None]

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


inits = [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a1 - 1)
estr = MEstimator(psi_plogit_a1, init=inits)
estr.estimate()
inits = [0., ]*params_r1 + list(estr.theta)
estr = MEstimator(psi_r1, init=inits)
estr.estimate()

r1_results = pd.DataFrame()
r1_results['time'] = event_times_p1
r1_results['risk'] = estr.theta[:params_r1]
r1_ci = estr.confidence_intervals()[:params_r1, :]
r1_results['lcl'] = r1_ci[:, 0]
r1_results['ucl'] = r1_ci[:, 1]

inits = [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a0 - 1)
estr = MEstimator(psi_plogit_a0, init=inits)
estr.estimate()
inits = [0., ]*params_r0 + list(estr.theta)
estr = MEstimator(psi_r0, init=inits)
estr.estimate()

r0_results = pd.DataFrame()
r0_results['time'] = event_times_p0
r0_results['risk'] = estr.theta[:params_r0]
r0_ci = estr.confidence_intervals()[:params_r0, :]
r0_results['lcl'] = r0_ci[:, 0]
r0_results['ucl'] = r0_ci[:, 1]

inits = [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a1 - 1)
estr = MEstimator(psi_plogit_a1, init=inits)
estr.estimate()
starting_a1 = list(estr.theta)
inits = [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a0 - 1)
estr = MEstimator(psi_plogit_a0, init=inits)
estr.estimate()
starting_a0 = list(estr.theta)
inits = [0., ]*params_rd + starting_a1 + starting_a0
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

fig, axes = plt.subplots(2, 2, width_ratios=[3, 1], figsize=[7.2, 3.6*2])
ax0 = axes[0, 0]
ax0.fill_between(r1_results['time'], r1_results['lcl'], r1_results['ucl'], color='blue', alpha=0.1, step='post')
ax0.fill_between(r0_results['time'], r0_results['lcl'], r0_results['ucl'], color='red', alpha=0.1, step='post')
ax0.step(r1_results['time'], r1_results['risk'], color='blue', where='post', label='IDU')
ax0.step(r0_results['time'], r0_results['risk'], color='red', where='post', label='No IDU')
ax0.set_xlim([-5, np.max(t)+5])
ax0.set_ylim([0, 1])
ax0.set_xlabel("Time (days)")
ax0.set_ylabel("Risk")
ax0.legend(loc='upper left')
ax0.text(-540, 0.97, "A", fontsize=14)

ax1 = axes[0, 1]
twister_plot(rd_results, point='rd', lcl='lcl', ucl='ucl', time='time', color='k', ax=ax1, favors=False)
ax1.set_ylim([-5, np.max(t)+5])
ax1.set_ylabel("Time (days)")
ax1.set_xlabel("Risk Difference")

######################################
# Example 2b: Splines

max_time = int(np.max(t))
t_steps = np.asarray(range(1, max_time+1))
tp_intervals = [0, ] + list(range(1, max_time+1, 50)) + [np.max(t), ]
params_risk = len(tp_intervals)

intercept = np.ones(t_steps.shape)[:, None]
time_splines = spline(t_steps, knots=[500, 1000, 2000, 3000, 3500],
                      power=2, restricted=True, normalized=False)
s_matrix = np.concatenate([intercept, t_steps[:, None], time_splines], axis=1)


def psi_plogit_spline_a1(theta):
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
    idPLRM = params_risk + W.shape[1] + s_matrix.shape[1]
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


inits = [0., ]*W.shape[1] + [-8., ] + [0., ]*5
estr = MEstimator(psi_plogit_spline_a1, init=inits)
estr.estimate()
inits = list(np.linspace(0, 0.5, params_risk)) + list(estr.theta)
estr = MEstimator(psi_r1, init=inits)
estr.estimate()

r1_results = pd.DataFrame()
r1_results['time'] = tp_intervals
r1_results['risk'] = estr.theta[:params_risk]
r1_ci = estr.confidence_intervals()[:params_risk, :]
r1_results['lcl'] = r1_ci[:, 0]
r1_results['ucl'] = r1_ci[:, 1]

inits = [0., ]*W.shape[1] + [-8., ] + [0., ]*5
estr = MEstimator(psi_plogit_spline_a0, init=inits)
estr.estimate()
inits = list(np.linspace(0, 0.5, params_risk)) + list(estr.theta)
estr = MEstimator(psi_r0, init=inits)
estr.estimate()

r0_results = pd.DataFrame()
r0_results['time'] = tp_intervals
r0_results['risk'] = estr.theta[:params_risk]
r0_ci = estr.confidence_intervals()[:params_risk, :]
r0_results['lcl'] = r0_ci[:, 0]
r0_results['ucl'] = r0_ci[:, 1]

inits = [0., ] * W.shape[1] + [-8., ] + [0., ] * 5
estr = MEstimator(psi_plogit_spline_a1, init=inits)
estr.estimate()
starting_a1 = list(estr.theta)
inits = [0., ] * W.shape[1] + [-8., ] + [0., ] * 5
estr = MEstimator(psi_plogit_spline_a0, init=inits)
estr.estimate()
starting_a0 = list(estr.theta)
inits = [0., ] * params_risk + starting_a1 + starting_a0
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

ax2 = axes[1, 0]
ax2.fill_between(r1_results['time'], r1_results['lcl'], r1_results['ucl'], color='blue', alpha=0.1)
ax2.fill_between(r0_results['time'], r0_results['lcl'], r0_results['ucl'], color='red', alpha=0.1)
ax2.plot(r1_results['time'], r1_results['risk'], color='blue', label='IDU')
ax2.plot(r0_results['time'], r0_results['risk'], color='red', label='No IDU')
ax2.set_xlim([-5, np.max(t)+5])
ax2.set_ylim([0, 1])
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Risk")
ax2.legend(loc='upper left')
ax2.text(-540, 0.97, "B", fontsize=14)

ax3 = axes[1, 1]
twister_plot(rd_results, point='rd', lcl='lcl', ucl='ucl', time='time', color='k', ax=ax3, favors=False, step=False)
ax3.set_ylim([-5, np.max(t)+5])
ax3.set_ylabel("Time (days)")
ax3.set_xlabel("Risk Difference")
plt.tight_layout()
plt.savefig("figure_lau.png", format='png', dpi=300)
plt.show()
