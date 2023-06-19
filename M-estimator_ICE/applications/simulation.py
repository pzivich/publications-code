####################################################################################################################
# Iterated Conditional Expectation (ICE) G-computation as an M-estimator
#   Running simulation experiment to check sandwich variance
#
# Paul Zivich
####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator

from funcs import generate_potential, generate_observed
from efuncs import ee_ice_gformula

#####################################
# Setup for simulations
n = 5000                           # Number of observations in a sample
sims = 5000                        # Number of simulation iterations
np.random.seed(8094141 + n*100)    # Seed (based somewhat on n so differs across)

#####################################
# Truth computation
# d = generate_potential(n=10000000)
# truth3_a1 = np.mean(d['Y3a1a1a1'])
# truth3_a0 = np.mean(d['Y3a0a0a0'])
# print(truth3_a1, truth3_a0)
truth3_a1, truth3_a0 = 0.4427662, 0.1356909   # Truth was pre-computed using above code

#####################################
# Setup for storage of results

pice_bias_a1, pice_se_a1, pice_lcl_a1, pice_ucl_a1 = [], [], [], []
pice_bias_a0, pice_se_a0, pice_lcl_a0, pice_ucl_a0 = [], [], [], []
sice_bias_a1, sice_se_a1, sice_lcl_a1, sice_ucl_a1 = [], [], [], []
sice_bias_a0, sice_se_a0, sice_lcl_a0, sice_ucl_a0 = [], [], [], []

#####################################
# Estimating Equations


def psi_ice_3a1(theta):
    """non-stratified ICE g-computation for always act"""
    return ee_ice_gformula(theta=theta, y=y3,
                           X_array=[X2, X1, X0],
                           Xa_array=[Xa2, Xa1, Xa0])


def psi_ice_3a0(theta):
    """non-stratified ICE g-computation for never act"""
    return ee_ice_gformula(theta=theta, y=y3,
                           X_array=[X2, X1, X0],
                           Xa_array=[Xb2, Xb1, Xb0])


def psi_ice_strat_3a1(theta):
    """stratified ICE g-computation for always act"""
    return ee_ice_gformula(theta=theta, y=y3,
                           X_array=[X2s, X1s, X0s],
                           Xa_array=[X2s, X1s, X0s],
                           stratified=[p2_a1, p1_a1, p0_a1])


def psi_ice_strat_3a0(theta):
    """stratified ICE g-computation for never act"""
    return ee_ice_gformula(theta=theta, y=y3,
                           X_array=[X2s, X1s, X0s],
                           Xa_array=[X2s, X1s, X0s],
                           stratified=[p2_a0, p1_a0, p0_a0])


#####################################
# Simulations

for i in range(sims):
    # Generating a random sample data set
    d = generate_potential(n=n)    # Generating data with potential outcomes
    d = generate_observed(data=d)  # Transforming potential data into observed data
    d['intercept'] = 1             # Adding intercept

    # Setting up design matrices for M-estimators
    y2 = np.asarray(d['Y2'])
    y3 = np.asarray(d['Y3'])
    X2 = np.asarray(d[['intercept', 'A1', 'A2', 'W1', 'W2']])
    X1 = np.asarray(d[['intercept', 'A0', 'A1', 'W0', 'W1']])
    X0 = np.asarray(d[['intercept', 'A0', 'W0']])
    X2s = np.asarray(d[['intercept', 'W1', 'W2']])
    X1s = np.asarray(d[['intercept', 'W0', 'W1']])
    X0s = np.asarray(d[['intercept', 'W0']])

    # Determining whether observation followed policy of interest
    p2_a1 = np.asarray(d['A0'] * d['A1'] * d['A2'])
    p2_a0 = np.asarray((1-d['A0']) * (1-d['A1']) * (1-d['A2']))
    p1_a1 = np.asarray(d['A0'] * d['A1'])
    p1_a0 = np.asarray((1-d['A0']) * (1-d['A1']))
    p0_a1 = np.asarray(d['A0'])
    p0_a0 = np.asarray(1-d['A0'])

    # Creating data that followed the policy
    da = d.copy()
    for a in ['A0', 'A1', 'A2']:
        da[a] = 1
    Xa2 = np.asarray(da[['intercept', 'A1', 'A2', 'W1', 'W2']])
    Xa1 = np.asarray(da[['intercept', 'A0', 'A1', 'W0', 'W1']])
    Xa0 = np.asarray(da[['intercept', 'A0', 'W0']])
    for a in ['A0', 'A1', 'A2']:
        da[a] = 0
    Xb2 = np.asarray(da[['intercept', 'A1', 'A2', 'W1', 'W2']])
    Xb1 = np.asarray(da[['intercept', 'A0', 'A1', 'W0', 'W1']])
    Xb0 = np.asarray(da[['intercept', 'A0', 'W0']])

    #####################################
    # ICE-g-computation implementations

    # Alway-act non-stratified ICE
    estr_ice = MEstimator(psi_ice_3a1, init=[0.5,
                                             0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0.,
                                             0., 0., 0., ])
    estr_ice.estimate(solver='lm', maxiter=10000)
    ice_ci = estr_ice.confidence_intervals()[0, :]
    pice_bias_a1.append(estr_ice.theta[0] - truth3_a1)
    pice_se_a1.append(np.sqrt(estr_ice.variance[0, 0]))
    pice_lcl_a1.append(ice_ci[0])
    pice_ucl_a1.append(ice_ci[1])

    # Never-act non-stratified ICE
    estr_ice = MEstimator(psi_ice_3a0, init=[0.5,
                                             0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0.,
                                             0., 0., 0., ])
    estr_ice.estimate(solver='lm', maxiter=10000)
    ice_ci = estr_ice.confidence_intervals()[0, :]
    pice_bias_a0.append(estr_ice.theta[0] - truth3_a0)
    pice_se_a0.append(np.sqrt(estr_ice.variance[0, 0]))
    pice_lcl_a0.append(ice_ci[0])
    pice_ucl_a0.append(ice_ci[1])

    # Alway-act stratified ICE
    estr_ice = MEstimator(psi_ice_strat_3a1, init=[0.5,
                                                   0., 0., 0.,
                                                   0., 0., 0.,
                                                   0., 0., ])
    try:                                                        # Catches failures to converge and marks as NaN
        estr_ice.estimate(solver='lm', maxiter=10000)
        ice_ci = estr_ice.confidence_intervals()[0, :]
        sice_bias_a1.append(estr_ice.theta[0] - truth3_a1)
        sice_se_a1.append(np.sqrt(estr_ice.variance[0, 0]))
        sice_lcl_a1.append(ice_ci[0])
        sice_ucl_a1.append(ice_ci[1])
    except RuntimeError:
        sice_bias_a1.append(np.nan)
        sice_se_a1.append(np.nan)
        sice_lcl_a1.append(np.nan)
        sice_ucl_a1.append(np.nan)

    # Never-act stratified ICE
    estr_ice = MEstimator(psi_ice_strat_3a0, init=[0.5,
                                                   0., 0., 0.,
                                                   0., 0., 0.,
                                                   0., 0., ])
    try:                                                        # Catches failures to converge and marks as NaN
        estr_ice.estimate(solver='lm', maxiter=10000)
        ice_ci = estr_ice.confidence_intervals()[0, :]
        sice_bias_a0.append(estr_ice.theta[0] - truth3_a0)
        sice_se_a0.append(np.sqrt(estr_ice.variance[0, 0]))
        sice_lcl_a0.append(ice_ci[0])
        sice_ucl_a0.append(ice_ci[1])
    except RuntimeError:
        sice_bias_a0.append(np.nan)
        sice_se_a0.append(np.nan)
        sice_lcl_a0.append(np.nan)
        sice_ucl_a0.append(np.nan)

    # Check current progress because I am impatient
    if (i+1) % 100 == 0:
        print(i+1)


#####################################
# Format and Output Results

results = pd.DataFrame()

results['bias_g_pa1'] = pice_bias_a1
results['se_g_pa1'] = pice_se_a1
results['lcl_g_pa1'] = pice_lcl_a1
results['ucl_g_pa1'] = pice_ucl_a1
results['cov_g_pa1'] = np.where((results['ucl_g_pa1'] > truth3_a1) & (results['lcl_g_pa1'] < truth3_a1), 1, 0)
results['cov_g_pa1'] = np.where(results['ucl_g_pa1'].isna(), np.nan, results['cov_g_pa1'])
results['bias_g_pa0'] = pice_bias_a0
results['se_g_pa0'] = pice_se_a0
results['lcl_g_pa0'] = pice_lcl_a0
results['ucl_g_pa0'] = pice_ucl_a0
results['cov_g_pa0'] = np.where((results['ucl_g_pa0'] > truth3_a0) & (results['lcl_g_pa0'] < truth3_a0), 1, 0)
results['cov_g_pa0'] = np.where(results['ucl_g_pa0'].isna(), np.nan, results['cov_g_pa0'])

results['bias_g_sa1'] = sice_bias_a1
results['se_g_sa1'] = sice_se_a1
results['lcl_g_sa1'] = sice_lcl_a1
results['ucl_g_sa1'] = sice_ucl_a1
results['cov_g_sa1'] = np.where((results['ucl_g_sa1'] > truth3_a1) & (results['lcl_g_sa1'] < truth3_a1), 1, 0)
results['cov_g_sa1'] = np.where(results['ucl_g_sa1'].isna(), np.nan, results['cov_g_sa1'])
results['bias_g_sa0'] = sice_bias_a0
results['se_g_sa0'] = sice_se_a0
results['lcl_g_sa0'] = sice_lcl_a0
results['ucl_g_sa0'] = sice_ucl_a0
results['cov_g_sa0'] = np.where((results['ucl_g_sa0'] > truth3_a0) & (results['lcl_g_sa0'] < truth3_a0), 1, 0)
results['cov_g_sa0'] = np.where(results['ucl_g_sa0'].isna(), np.nan, results['cov_g_sa0'])

print("ICE-G-COMPUTATION")
print(results[['bias_g_pa1', 'se_g_pa1', 'cov_g_pa1']].describe())
print(results[['bias_g_sa1', 'se_g_sa1', 'cov_g_sa1']].describe())
print(results[['bias_g_pa0', 'se_g_pa0', 'cov_g_pa0']].describe())
print(results[['bias_g_sa0', 'se_g_sa0', 'cov_g_sa0']].describe())

results.to_csv("results/sim1_n"+str(n)+".csv", index=False)
