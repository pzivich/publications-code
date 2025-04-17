######################################################################################################################
# Code to run the simulation experiments
#   Results are shown as comments at the end
#
# Paul Zivich (Last update: 2025/4/17)
######################################################################################################################

import numpy as np
import pandas as pd

from dgm import dgm
from estimators import PooledLogitEE

np.random.seed(338092421)

runs = 5000
n_obs = 250

truth = pd.read_csv("truth.csv")
truth = np.asarray(truth['KM_estimate'])
true_10, true_20, true_30 = truth

results_10, results_20, results_30 = [], [], []
for i in range(runs):
    print("...starting", i+1)
    result10_row, result20_row, result30_row = [], [], []
    d = dgm(n=n_obs, truth=False)

    # Pooled Logit Exponential
    try:
        plee = PooledLogitEE(data=d, time='T_star', delta='delta', action='A')
        plee.nuisance_model(covariates=['W', 'W_sp1', 'W_sp2'], time='constant')
        plee.estimate()
        for j in [plee.point, plee.variance, plee.lower_ci, plee.upper_ci]:
            result10_row.append(j[0])
            result20_row.append(j[1])
            result30_row.append(j[2])
    except:
        for j in [1, 2, 3, 4]:
            result10_row.append(np.nan)
            result20_row.append(np.nan)
            result30_row.append(np.nan)

    # Pooled Logit Gompertz
    try:
        plee = PooledLogitEE(data=d, time='T_star', delta='delta', action='A')
        plee.nuisance_model(covariates=['W', 'W_sp1', 'W_sp2'], time='linear')
        plee.estimate()
        for j in [plee.point, plee.variance, plee.lower_ci, plee.upper_ci]:
            result10_row.append(j[0])
            result20_row.append(j[1])
            result30_row.append(j[2])
    except:
        for j in [1, 2, 3, 4]:
            result10_row.append(np.nan)
            result20_row.append(np.nan)
            result30_row.append(np.nan)

    # Pooled Logit Weibull
    try:
        plee = PooledLogitEE(data=d, time='T_star', delta='delta', action='A')
        plee.nuisance_model(covariates=['W', 'W_sp1', 'W_sp2'], time='log')
        plee.estimate()
        for j in [plee.point, plee.variance, plee.lower_ci, plee.upper_ci]:
            result10_row.append(j[0])
            result20_row.append(j[1])
            result30_row.append(j[2])
    except:
        for j in [1, 2, 3, 4]:
            result10_row.append(np.nan)
            result20_row.append(np.nan)
            result30_row.append(np.nan)

    # Pooled Logit Splines
    try:
        plee = PooledLogitEE(data=d, time='T_star', delta='delta', action='A')
        plee.nuisance_model(covariates=['W', 'W_sp1', 'W_sp2'], time='spline')
        plee.estimate()
        for j in [plee.point, plee.variance, plee.lower_ci, plee.upper_ci]:
            result10_row.append(j[0])
            result20_row.append(j[1])
            result30_row.append(j[2])
    except:
        for j in [1, 2, 3, 4]:
            result10_row.append(np.nan)
            result20_row.append(np.nan)
            result30_row.append(np.nan)

    # Pooled Logit Disjoint
    try:
        plee = PooledLogitEE(data=d, time='T_star', delta='delta', action='A')
        plee.nuisance_model(covariates=['W', 'W_sp1', 'W_sp2'], time='disjoint')
        plee.estimate()
        for j in [plee.point, plee.variance, plee.lower_ci, plee.upper_ci]:
            result10_row.append(j[0])
            result20_row.append(j[1])
            result30_row.append(j[2])
    except:
        for j in [1, 2, 3, 4]:
            result10_row.append(np.nan)
            result20_row.append(np.nan)
            result30_row.append(np.nan)

    # Adding the new rows
    results_10.append(result10_row)
    results_20.append(result20_row)
    results_30.append(result30_row)


metric_cols = ['p', 'v', 'l', 'u']
estr_cols = ['ple', 'plg', 'plw', 'pls', 'pld']
columns = []
for ec in estr_cols:
    columns = columns + [ec + "_" + c for c in metric_cols]

for end_time, results_t, truth in zip([10, 20, 30], [results_10, results_20, results_30], [true_10, true_20, true_30]):
    results = pd.DataFrame(results_t, columns=columns)
    for estimator in estr_cols:
        results[estimator+'_b'] = results[estimator+'_p'] - truth
        results[estimator+'_s'] = results[estimator+'_v'] ** 0.5
        results[estimator+'_c'] = np.where((results[estimator+"_l"] <= truth) & (truth <= results[estimator+'_u']),
                                           1, 0)

    # Saving simulations output
    results.to_csv("sim_t"+str(end_time)+"_n"+str(n_obs)+".csv")


# N = 250
#              Bias     ESE       SER    RMSE Coverage MISS-Bias
# Estimator
# ple        -0.070   0.040     0.983   0.080    0.552     0.000
# plg         0.002   0.047     0.982   0.047    0.943     0.000
# plw         0.005   0.046     0.971   0.046    0.939     0.000
# pls        -0.002   0.050     0.983   0.050    0.938    15.000
# pld         0.000   0.051     0.987   0.051    0.946     0.000
#
#              Bias     ESE       SER    RMSE Coverage MISS-Bias
# Estimator
# ple        -0.013   0.064     0.982   0.065    0.935     0.000
# plg         0.021   0.064     0.977   0.067    0.930     0.000
# plw         0.003   0.063     0.976   0.063    0.940     0.000
# pls        -0.001   0.070     0.988   0.070    0.944    15.000
# pld        -0.001   0.071     0.986   0.071    0.946     0.000
#
#              Bias     ESE       SER    RMSE Coverage MISS-Bias
# Estimator
# ple         0.069   0.079     0.979   0.105    0.843     0.000
# plg        -0.007   0.085     0.984   0.085    0.942     0.000
# plw        -0.005   0.084     0.985   0.084    0.941     0.000
# pls        -0.001   0.085     0.982   0.085    0.938    15.000
# pld        -0.001   0.085     0.981   0.085    0.941     0.000

# N = 500
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.071  0.027  1.000  0.076    0.257     0.000
# plg         0.002  0.033  0.996  0.033    0.947     0.000
# plw         0.005  0.031  0.995  0.032    0.947     0.000
# pls        -0.002  0.035  0.993  0.035    0.942     0.000
# pld         0.000  0.036  0.995  0.036    0.948     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.013  0.045  0.995  0.047    0.933     0.000
# plg         0.022  0.045  0.995  0.050    0.921     0.000
# plw         0.003  0.044  0.995  0.044    0.946     0.000
# pls        -0.000  0.049  0.999  0.049    0.946     0.000
# pld         0.000  0.049  1.000  0.049    0.947     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple         0.070  0.056  0.993  0.089    0.750     0.000
# plg        -0.006  0.060  0.997  0.060    0.946     0.000
# plw        -0.004  0.059  0.996  0.059    0.946     0.000
# pls        -0.001  0.060  0.997  0.060    0.949     0.000
# pld        -0.001  0.060  0.997  0.060    0.949     0.000

# N = 1000
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.072  0.019  0.994  0.075    0.039     0.000
# plg         0.002  0.023  1.003  0.023    0.951     0.000
# plw         0.005  0.022  1.002  0.022    0.944     0.000
# pls        -0.002  0.024  1.014  0.024    0.951     0.000
# pld         0.000  0.025  1.012  0.025    0.954     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.014  0.032  0.991  0.035    0.917     0.000
# plg         0.022  0.031  0.997  0.038    0.894     0.000
# plw         0.003  0.031  0.995  0.031    0.943     0.000
# pls        -0.001  0.035  0.982  0.035    0.945     0.000
# pld        -0.000  0.036  0.981  0.036    0.943     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple         0.069  0.040  0.989  0.080    0.577     0.000
# plg        -0.006  0.043  0.989  0.043    0.942     0.000
# plw        -0.005  0.042  0.988  0.042    0.943     0.000
# pls        -0.001  0.043  0.989  0.043    0.945     0.000
# pld        -0.001  0.043  0.989  0.043    0.945     0.000

# N = 2000
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.073  0.013  0.992  0.074    0.000     0.000
# plg         0.002  0.016  0.991  0.017    0.946     0.000
# plw         0.004  0.016  0.990  0.016    0.941     0.000
# pls        -0.002  0.017  0.990  0.018    0.943     0.000
# pld        -0.000  0.018  0.995  0.018    0.946     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.015  0.022  0.990  0.027    0.888     0.000
# plg         0.022  0.022  0.986  0.031    0.823     0.000
# plw         0.002  0.022  0.986  0.022    0.948     0.000
# pls        -0.001  0.025  0.983  0.025    0.946     0.000
# pld        -0.000  0.025  0.984  0.025    0.948     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple         0.069  0.028  0.988  0.075    0.304     0.000
# plg        -0.006  0.030  0.988  0.031    0.943     0.000
# plw        -0.004  0.030  0.989  0.030    0.946     0.000
# pls        -0.001  0.030  0.988  0.030    0.949     0.000
# pld        -0.001  0.030  0.988  0.030    0.949     0.000
