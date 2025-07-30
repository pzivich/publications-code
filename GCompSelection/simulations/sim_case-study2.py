#####################################################################################################################
# Case Study 2: Simulation Experiment
#####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator

from dgm import dgm_example2
from estfun import psi_standard_gcomp, psi_gcomp_nested, psi_ipw_case2
from postprocess import sim_results_table

############################################################
# Hyperparameters for simulation

n_runs = 5000
n_sample = 1000
np.random.seed(7777777)

############################################################
# Truth calculation

dt = dgm_example2(n=1000000, truth=True)
a = np.asarray(dt['A'])
y = np.asarray(dt['Y'])


def psi_cc(theta):
    ee_rd = (theta[1] - theta[2]) - theta[0] * np.ones(y.shape[0])
    ee_y1 = a*(y - theta[1])
    ee_y0 = (1-a)*(y - theta[2])
    return np.vstack([ee_rd, ee_y1, ee_y0])


estr = MEstimator(psi_cc, init=[0., 0.5, 0.5])
estr.estimate()
truth = estr.theta[0]
print("Truth:", truth)

############################################################
# Simulations

results = pd.DataFrame(columns=['bias_cc', 'se_cc', 'cover_cc',
                                'bias_gz', 'se_gz', 'cover_gz',
                                'bias_gx', 'se_gx', 'cover_gx',
                                'bias_gs', 'se_gs', 'cover_gs',
                                'bias_ng', 'se_ng', 'cover_ng',
                                'bias_w', 'se_w', 'cover_w',
                                ])

for i in range(n_runs):
    # Generating sample data
    d = dgm_example2(n=n_sample, truth=False)
    d['AZ'] = d['A'] * d['Z']
    d['AX'] = d['A'] * d['X']
    d['AXZ'] = d['A'] * d['X'] * d['Z']
    ds = d.loc[d['S'] == 1].copy()
    da1 = d.copy()
    da1['A'] = 1
    da1['AZ'] = da1['Z']
    da0 = d.copy()
    da0['A'] = 0
    da0['AZ'] = 0
    row = []

    # Complete-case estimator
    a, y = ds['A'], ds['Y']
    estr = MEstimator(psi_cc, init=[0., 0.5, 0.5])
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 0]:
        row.append(1)
    else:
        row.append(0)

    # G-computation Z-only
    def psi_sg(theta):
        return psi_standard_gcomp(theta=theta, y=d['Y'], s=d['S'],
                                  X=d[['I', 'A', 'Z']],
                                  X1=da1[['I', 'A', 'Z']],
                                  X0=da0[['I', 'A', 'Z']])
    inits = [0., 0.5, 0.5, 0., 0., 0.]
    estr = MEstimator(psi_sg, init=inits)
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 1]:
        row.append(1)
    else:
        row.append(0)

    # G-computation X-only
    def psi_sg(theta):
        return psi_standard_gcomp(theta=theta, y=d['Y'], s=d['S'],
                                  X=d[['I', 'A', 'X']],
                                  X1=da1[['I', 'A', 'X']],
                                  X0=da0[['I', 'A', 'X']])
    inits = [0., 0.5, 0.5, 0., 0., 0.]
    estr = MEstimator(psi_sg, init=inits)
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 1]:
        row.append(1)
    else:
        row.append(0)

    # G-computation X,Z
    def psi_sg(theta):
        return psi_standard_gcomp(theta=theta, y=d['Y'], s=d['S'],
                                  X=d[['I', 'A', 'Z', 'X']],
                                  X1=da1[['I', 'A', 'Z', 'X']],
                                  X0=da0[['I', 'A', 'Z', 'X']])
    inits = [0., 0.5, 0.5, 0., 0., 0., 0.]
    estr = MEstimator(psi_sg, init=inits)
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 1]:
        row.append(1)
    else:
        row.append(0)

    # Nested g-computation
    def psi_ng(theta):
        return psi_gcomp_nested(theta=theta, y=d['Y'], s=d['S'],
                                X=d[['I', 'A', 'Z', 'X']],
                                X1=da1[['I', 'A', 'Z', 'X']],
                                X0=da0[['I', 'A', 'Z', 'X']],
                                W=d[['I', 'A', 'Z']],
                                W1=da1[['I', 'A', 'Z']],
                                W0=da0[['I', 'A', 'Z']])
    inits = [0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    estr = MEstimator(psi_ng, init=inits)
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 1]:
        row.append(1)
    else:
        row.append(0)

    # IPW
    def psi_w(theta):
        return psi_ipw_case2(theta=theta, y=d['Y'], a=d['A'], s=d['S'],
                             W=d[['I', 'X']], Z=d[['I', 'Z']])

    inits = [0., 0.5, 0.5, 0., 0., 0., 0.]
    estr = MEstimator(psi_w, init=inits)
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 1]:
        row.append(1)
    else:
        row.append(0)

    # Stacking Results to Output
    results.loc[len(results.index)] = row


############################################################
# Results

table = sim_results_table(results, estimators=['Complete-case', 'Z-only', 'X-only', 'X,Z', 'Nested', 'IPW'],
                          bias=['bias_cc', 'bias_gz', 'bias_gx', 'bias_gs', 'bias_ng', 'bias_w'],
                          se=['se_cc', 'se_gz', 'se_gx', 'se_gs', 'se_ng', 'se_w'],
                          coverage=['cover_cc', 'cover_gz', 'cover_gx', 'cover_gs', 'cover_ng', 'cover_w'])
print(table.round(3))

# 29/07/2025
#
# Truth: -0.2048799822473868
#
#                 bias    ese   rmse    ser  coverage
# estimator
# Complete-case  0.037  0.034  0.050  1.008     0.000
# Z-only         0.025  0.033  0.041  1.005     0.896
# X-only         0.042  0.039  0.058  1.002     0.833
# X,Z            0.029  0.038  0.047  1.000     0.897
# Nested         0.004  0.036  0.036  1.004     0.950
# IPW           -0.000  0.038  0.038  1.003     0.949
