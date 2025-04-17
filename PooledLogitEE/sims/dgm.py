######################################################################################################################
# Code for the data generating mechanism for the simulation experiments
#
# Paul Zivich (Last update: 2025/4/17)
######################################################################################################################

import numpy as np
import pandas as pd
from delicatessen.utilities import logit, inverse_logit, spline


def censor(data, time, event_time, censor_time, event_indicator):
    data[time] = np.min(data[[event_time, censor_time]], axis=1)
    data[event_indicator] = np.where(data[time] == data[event_time], 1, 0)


def dgm(n, truth=False):
    data = pd.DataFrame()

    # Generating baseline covariates
    data['W'] = np.random.uniform(-1, 1, size=n)
    data[['W_sp1', 'W_sp2']] = spline(data['W'], knots=[-0.8, 0., 0.8], restricted=True)
    data['A'] = np.random.binomial(n=1, p=inverse_logit(-1.5*data['W']), size=n)
    data['tau'] = 30

    # Generating event times
    data['T1'] = np.ceil((50. + 5.*data['W'] + 15*1) * np.random.weibull(a=0.75, size=n))
    censor(data=data, time='T1_star', event_time='T1', censor_time='tau', event_indicator='delta1')
    data['T0'] = np.ceil((50. + 5.*data['W'] + 15*0) * np.random.weibull(a=1.5, size=n))
    censor(data=data, time='T0_star', event_time='T0', censor_time='tau', event_indicator='delta0')
    data['T'] = np.where(data['A'] == 1, data['T1_star'], data['T0_star'])
    data['deltaA'] = np.where(data['A'] == 1, data['delta1'], data['delta0'])
    data['I'] = 1

    # Applying censoring
    data['C'] = np.ceil(np.random.exponential(38., size=n))
    censor(data=data, time='T_star', event_time='T', censor_time='C', event_indicator='deltaC')
    data['delta'] = np.where((data['deltaA'] == 1) & (data['deltaC'] == 1), 1, 0)

    if truth:
        return data[['T1_star', 'delta1', 'T0_star', 'delta0']]
    else:
        return data.drop(columns=['T1_star', 'delta1', 'T0_star', 'delta0'])
