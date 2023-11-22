#####################################################################################################################
# Bespoke Instrumental Variable via two-stage regression as an M-estimator
#
# Paul Zivich (2023/11/22)
####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator

from estfunc import ee_2sr_bsiv

# Reading in data
d = pd.read_csv("processed.csv")
d['intercept'] = 1

# Processing data into array for delicatessen
r = np.asarray(d['R'])
a = np.asarray(d['A'])
L = np.asarray(d[['intercept', 'L1', 'L2']])
y = np.asarray(d['Y'])

# Running the M-estimator procedure


def psi(theta):
    return ee_2sr_bsiv(theta=theta, y=y, a=a, r=r, L=L)


estr = MEstimator(psi, init=[0., ]*8)
estr.estimate()
estimates = np.asarray(estr.theta)
ci = estr.confidence_intervals()

# Results
print("Beta_0", np.round(estimates[0], 2), np.round(ci[0, :], 6))
print("Beta_1", np.round(estimates[1], 2), np.round(ci[1, :], 6))


# Repeating but with a bootstrap
bs_iters = 5000
np.random.seed(20231122)
beta0_rep, beta1_rep = [], []

for i in range(bs_iters):
    ds = d.sample(frac=1, replace=True)
    r = np.asarray(ds['R'])
    a = np.asarray(ds['A'])
    L = np.asarray(ds[['intercept', 'L1', 'L2']])
    y = np.asarray(ds['Y'])

    # Applying M-estimator to resampled data
    estr = MEstimator(psi, init=[0., ] * 8)
    estr.estimate()
    beta0_rep.append(estr.theta[0])
    beta1_rep.append(estr.theta[1])

beta0_se = np.std(beta0_rep, ddof=0)
beta1_se = np.std(beta1_rep, ddof=0)
beta0_ci = [estimates[0] - 1.96*beta0_se,
            estimates[0] + 1.96*beta0_se]
beta1_ci = [estimates[1] - 1.96*beta1_se,
            estimates[1] + 1.96*beta1_se]

print("Beta_0", np.round(estimates[0], 2), np.round(beta0_ci, 6))
print("Beta_1", np.round(estimates[1], 2), np.round(beta1_ci, 6))

# OUTPUT
# -------------------------------------------------
# Sandwich
# Beta_0 -0.33 [-0.428549 -0.233846]
# Beta_1 0.18 [0.004587 0.358593]
# Bootstrap
# Beta_0 -0.33 [-0.431346 -0.23105 ]
# Beta_1 0.18 [-0.00159   0.364769]
