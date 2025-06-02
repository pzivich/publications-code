#####################################################################################################################
# Case Study 1: Simulation Experiment
#####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_glm

from dgm import dgm_example1
from estfun import psi_standard_gcomp, psi_gcomp_conditional
from postprocess import sim_results_table

############################################################
# Hyperparameters for simulation

n_runs = 5000
n_sample = 1000
np.random.seed(7777777)

############################################################
# Truth calculation

dt = dgm_example1(n=10000000, truth=True)
X = np.asarray(dt[['I', 'A']])
y = np.asarray(dt['Y'])


def psi_cc(theta):
    return ee_glm(theta, X=X, y=y,
                  distribution='binomial',
                  link='identity')


estr = MEstimator(psi_cc, init=[0.5, -0.2])
estr.estimate()
truth = estr.theta[1]
print(truth)

############################################################
# Simulations

results = pd.DataFrame(columns=['bias_cc', 'se_cc', 'cover_cc',
                                'bias_sg', 'se_sg', 'cover_sg',
                                'bias_pg', 'se_pg', 'cover_pg',
                                ])

for i in range(n_runs):
    # Generating sample data
    d = dgm_example1(n=n_sample, truth=False)
    d['AW'] = d['A'] * d['W']
    ds = d.loc[d['S'] == 1].copy()
    da1 = d.copy()
    da1['A'] = 1
    da1['AW'] = da1['W']
    da0 = d.copy()
    da0['A'] = 0
    da0['AW'] = 0
    row = []

    # Complete-case estimator
    X, y = ds[['I', 'A']], ds['Y']
    estr = MEstimator(psi_cc, init=[0.5, -0.2])
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[1] - truth)
    row.append(estr.variance[1, 1]**0.5)
    if ci[1, 0] < truth < ci[1, 1]:
        row.append(1)
    else:
        row.append(0)

    # Standard g-computation
    def psi_sg(theta):
        return psi_standard_gcomp(theta=theta, y=d['Y'], s=d['S'],
                                  X=d[['I', 'A', 'W', 'AW']],
                                  X1=da1[['I', 'A', 'W', 'AW']],
                                  X0=da0[['I', 'A', 'W', 'AW']])

    inits = [0., 0.5, 0.5, 0., 0., 0., 0., ]
    estr = MEstimator(psi_sg, init=inits)
    estr.estimate()
    ci = estr.confidence_intervals()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0]**0.5)
    if ci[0, 0] < truth < ci[0, 1]:
        row.append(1)
    else:
        row.append(0)

    # Proposed g-computation
    def psi_pg(theta):
        return psi_gcomp_conditional(theta=theta, y=d['Y'], a=d['A'], s=d['S'],
                                     X=d[['I', 'A', 'W', 'AW']],
                                     X1=da1[['I', 'A', 'W', 'AW']],
                                     X0=da0[['I', 'A', 'W', 'AW']])

    inits = [0., 0.5, 0.5, 0., 0., 0., 0., ]
    estr = MEstimator(psi_pg, init=inits)
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

table = sim_results_table(results, estimators=['Complete-case', 'Standard', 'Proposed'],
                          bias=['bias_cc', 'bias_sg', 'bias_pg'],
                          se=['se_cc', 'se_sg', 'se_pg'],
                          coverage=['cover_cc', 'cover_sg', 'cover_pg'])
print(table.round(3))
