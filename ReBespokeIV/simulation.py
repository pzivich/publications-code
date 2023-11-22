#####################################################################################################################
# Bespoke Instrumental Variable via two-stage regression as an M-estimator
#       Replication of the simulation study but with the sandwich variance estimator
#
# Paul Zivich (2023/11/22)
####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator

from estfunc import ee_2sr_bsiv


# Setup
sims = 1000
exclude_restrict = False
n = 5000


def generate_data(n, exr):
    d = pd.DataFrame()
    d['L1'] = np.random.uniform(-1, 1, size=n)
    d['L2'] = np.random.binomial(n=1, p=0.5, size=n)
    d['R'] = np.random.binomial(n=1, p=0.5, size=n)
    pr_a = 0.25 + 0.25*d['L1'] + 0.25*d['L2']
    a_flip = np.random.binomial(n=1, p=pr_a, size=n)
    d['A'] = np.where(d['R'] == 1, a_flip, 0)
    d['C'] = 1

    # Outcome generation for scenarios
    if exr:
        y_det = 1 + 1*d['L1'] + 1*d['L2'] + 1*d['A']
    else:
        y_det = 1 + 1*d['L1'] + 1*d['L2'] + 1*d['A'] + 1*d['R']
    d['Y'] = y_det + np.random.normal(size=n)

    return d


def psi(theta):
    return ee_2sr_bsiv(theta=theta, y=y, a=a, r=r, L=L)


# Creating result storage
results = pd.DataFrame(columns=['beta0', 'beta0_var', 'beta0_ci',
                                'beta1', 'beta1_var', 'beta1_ci'])

if exclude_restrict:
    beta0_truth = 0
    np.random.seed(7777777)
else:
    beta0_truth = 1
    np.random.seed(96369)
beta1_truth = 1

for i in range(sims):
    row = []

    # Sample data and process for estimating equations
    d = generate_data(n=n, exr=exclude_restrict)
    y = np.asarray(d['Y'])
    a = np.asarray(d['A'])
    r = np.asarray(d['R'])
    L = np.asarray(d[['C', 'L1']])

    # Apply M-estimator
    estr = MEstimator(psi, init=[0., 0.,
                                 0., 0., 0., 0., ])
    estr.estimate(deriv_method='exact')
    beta0ci = estr.confidence_intervals()[0, :]
    beta1ci = estr.confidence_intervals()[1, :]

    # Processing output to save
    beta0, beta1 = estr.theta[0] - beta0_truth, estr.theta[1] - beta1_truth
    beta0v, beta1v = np.diag(estr.variance)[0:2]
    if beta0ci[0] < beta0_truth < beta0ci[1]:
        beta0c = 1
    else:
        beta0c = 0
    if beta1ci[0] < beta1_truth < beta1ci[1]:
        beta1c = 1
    else:
        beta1c = 0

    row = row + [beta0, beta0v, beta0c]
    row = row + [beta1, beta1v, beta1c]
    results.loc[len(results.index)] = row


def calculate_metrics(data, estimator):
    bias = np.mean(data[estimator])
    ase = np.mean(np.sqrt(data[estimator + '_var']))
    ese = np.std(data[estimator])
    ser = ase / ese
    cover = np.mean(data[estimator + '_ci'])
    return bias, ser, cover


def create_table(data):
    """Function to create a table of the simulation results

    Parameters
    ----------
    data : pandas.DataFrame
        Simulation output from the `run_estimators.py` file

    Returns
    -------
    pandas.DataFrame
    """
    # Creating blank DataFrame
    table = pd.DataFrame(columns=['Estimator', 'Bias', 'SER', 'Coverage'])

    # String indicators for all scenarios explored
    estrs = ['beta0', 'beta1']

    # Calculating metrics for each estimator, then adding as a row in the table
    for estr in estrs:
        bias, cld, cover = calculate_metrics(data=data, estimator=estr)
        table.loc[len(table.index)] = [estr, bias, cld, cover]

    # Return processed simulation results
    return table


table = create_table(data=results)
print(table)

# OUTPUT
# -------------------------------------------------
# With exclusion restriction
#   Estimator      Bias       SER  Coverage
# 0     beta0 -0.000898  1.026384     0.948
# 1     beta1  0.002159  1.031067     0.954
#
# Without exclusion restriction
#   Estimator      Bias       SER  Coverage
# 0     beta0  0.000286  1.000151     0.953
# 1     beta1  0.002314  0.975548     0.942
