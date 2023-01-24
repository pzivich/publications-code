#####################################################################################################################
# Introducing Proximal Causal Inference for Epidemiologists
#   PN Zivich, SR Cole, JK Edwards, GE Mulholland, BE Shook-Sa, E Tchetgen Tchetgen
#
#   Python code the the described simulations
#
# Paul Zivich (2022/11/13)
#####################################################################################################################

import numpy as np
import pandas as pd
from scipy.stats import logistic
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression

n_sims = 4000                 # Setting number of iterations for each scenario
n_sample_size = 500            # Setting sample size for scenarios
truth = -1                     # Setting the true population ACE
np.random.seed(7777777)        # Setting random seed for consistent results


def data_generate(n, scenario):
    # Setting up scenario coefficients
    if scenario == 1:            # all g-computations should work
        beta_u, beta_z = 0, 0
    elif scenario == 2:          # only proxy g-computation should work
        beta_u, beta_z = 1, 0
    elif scenario == 3:          # no g-computations should work
        beta_u, beta_z = 1, 1
    else:                        # catching bad scenario specifications
        raise ValueError("Invalid scenario")

    # Generating data
    d = pd.DataFrame()
    d['X'] = np.random.normal(size=n)            # Generate age
    d['U'] = d['X'] + np.random.normal(size=n)   # Generate immune function
    d['Z'] = d['U'] + np.random.normal(size=n)   # Generate treatment proxy
    d['W'] = d['U'] + np.random.normal(size=n)   # Generating outcome proxy

    # Generate treatment
    pr_a = logistic.cdf(d['Z'] + beta_u*d['U'] + d['X'])
    d['A'] = np.random.binomial(n=1, p=pr_a, size=n)

    # Generate potential outcomes
    ya0 = d['W'] + beta_u*d['U'] + d['X'] + beta_z*d['Z'] + np.random.normal(scale=1, size=n)
    ya1 = ya0 + truth

    # Generate observed outcome
    d['Y'] = np.where(d['A'] == 1, ya1, ya0)

    # Adding in an intercept term to the data set
    d['C'] = 1

    # Returning the generated data
    return d


def psi_standardgcomp(theta):
    # Linear regression model from delicatessen
    y_model = ee_regression(theta=theta,             # Regression model from delicatessen
                            X=y_vars,                # ... with vars to predict
                            y=y,                     # ... the outcome
                            model='linear')          # ... using a linear regression model
    return y_model                                   # Return stacked estimating equations for delicatessen


def psi_proxygcomp(theta):
    # Dividing theta into the parameters for each model
    alpha = theta[4:]
    beta = theta[0:4]

    # Outcome proxy nuisance model
    w_model = ee_regression(theta=alpha,             # Regression model from delicatessen
                            X=w_vars,                # ... with vars to predict
                            y=w,                     # ... the outcome proxy
                            model='linear')          # ... using a linear regression model
    w_hat = np.dot(w_vars, alpha)                    # Generating predicted value of W (outcome proxy)

    # Actual outcome nuisance model
    y_model = ee_regression(theta=beta,              # Linear regression model from delicatessen
                            X=np.c_[y_vars, w_hat],  # Stack vars to predict Y with W-hat
                            y=y,                     # Y: outcome
                            model='linear')

    return np.vstack([y_model,                       # Return stacked estimating equations for delicatessen
                      w_model])


########################################
# Scenarios

# Simulating all scenarios within a for loop (running in parallel not a concern here)
for scenario in [1, 2, 3]:
    # Storage in lists for results from the g-computations
    bias_msg, est_sd_msg, cover_msg = [], [], []           # Storage for minimal g-computation
    bias_sg, est_sd_sg, cover_sg = [], [], []              # Storage for g-computation
    bias_pg, est_sd_pg, cover_pg = [], [], []              # Storage for proximal g-computation

    # For loop for the number of iterations to run each simulation scenario
    for i in range(n_sims):                                # Loop over the number of sims
        # Generating the data set
        d = data_generate(n=n_sample_size,                 # ... generate data with N observations
                          scenario=scenario)               # ... for current scenario from outer loop
        y = np.asarray(d['Y'])                             # ... then extract Y from data set as a numpy array

        # Minimal standard g-computation
        y_vars = np.asarray(d[['A', 'C', 'W', 'X']])       # ... extract vars to predict Y as a numpy array
        mest = MEstimator(psi_standardgcomp,               # ... specify the M-estimator
                          init=[0, 0, 0, 0])               # ... with generic init vals
        mest.estimate(solver='lm')                         # ... compute for the parameter estimates
        est = mest.theta[0]                                # ... extract point estimate
        var = mest.variance[0, 0]                          # ... extract variance estimate
        ci = mest.confidence_intervals()[0]                # ... extract confidence intervals
        bias_msg.append(est - truth)                       # ... calculate and store bias
        est_sd_msg.append(var**0.5)                        # ... store estimated standard error
        if ci[0] < truth < ci[1]:                          # ... confidence interval coverage logic
            cover_msg.append(1)
        else:
            cover_msg.append(0)

        # Standard g-computation
        y_vars = np.asarray(d[['A', 'C', 'Z', 'W', 'X']])  # ... extract vars to predict Y as a numpy array
        mest = MEstimator(psi_standardgcomp,               # ... specify the M-estimator
                          init=[0, 0, 0, 0, 0])            # ... with generic init vals
        mest.estimate(solver='lm')                         # ... compute for the parameter estimates
        est = mest.theta[0]                                # ... extract point estimate
        var = mest.variance[0, 0]                          # ... extract variance estimate
        ci = mest.confidence_intervals()[0]                # ... extract confidence intervals
        bias_sg.append(est - truth)                        # ... calculate and store bias
        est_sd_sg.append(var**0.5)                         # ... store estimated standard error
        if ci[0] < truth < ci[1]:                          # ... confidence interval coverage logic
            cover_sg.append(1)
        else:
            cover_sg.append(0)

        # Proximal g-computation
        w = np.asarray(d['W'])                             # ... proximal outcome as a numpy array
        w_vars = np.asarray(d[['C', 'A', 'Z', 'X']])       # ... extract vars to predict W as a numpy array
        y_vars = np.asarray(d[['A', 'C', 'X']])            # ... extract vars to predict Y (_excluding_ W)
        mest = MEstimator(psi_proxygcomp,                  # ... specify the M-estimator
                          init=[0, 0, 0, 0,                # ... with generic init vals
                                0, 0, 0, 0])
        mest.estimate(solver='lm')                         # ... compute for the parameter estimates
        est = mest.theta[0]                                # ... extract point estimate
        var = mest.variance[0, 0]                          # ... extract variance estimate
        ci = mest.confidence_intervals()[0]                # ... extract confidence intervals
        bias_pg.append(est - truth)                        # ... calculate and store bias
        est_sd_pg.append(var**0.5)                         # ... store estimated standard error
        if ci[0] < truth < ci[1]:                          # ... confidence interval coverage logic
            cover_pg.append(1)
        else:
            cover_pg.append(0)

    # Print the simulation results to the console for each scenario
    print("====================================")
    print("Scenario:", scenario)
    print("====================================")
    print("Minimal standard g-computation")
    print("------------------------------------")
    print("Bias:", np.round(np.mean(bias_msg), 3))
    print("ESE: ", np.round(np.std(bias_msg), 3))
    print("ASE: ", np.round(np.mean(est_sd_msg), 3))
    print("SER: ", np.round(np.mean(est_sd_msg) / np.std(bias_msg), 3))
    print("RMSE:", np.round(np.sqrt(np.mean(bias_msg)**2 + np.var(bias_msg)), 3))
    print("Cvr :", np.round(np.mean(cover_msg), 3))
    print("------------------------------------")
    print("Standard g-computation")
    print("------------------------------------")
    print("Bias:", np.round(np.mean(bias_sg), 3))
    print("ESE: ", np.round(np.std(bias_sg), 3))
    print("ASE: ", np.round(np.mean(est_sd_sg), 3))
    print("SER: ", np.round(np.mean(est_sd_sg) / np.std(bias_sg), 3))
    print("RMSE:", np.round(np.sqrt(np.mean(bias_sg)**2 + np.var(bias_sg)), 3))
    print("Cvr :", np.round(np.mean(cover_sg), 3))
    print("------------------------------------")
    print("Proxy g-computation")
    print("------------------------------------")
    print("Bias:", np.round(np.mean(bias_pg), 3))
    print("ESE: ", np.round(np.std(bias_pg), 3))
    print("ASE: ", np.round(np.mean(est_sd_pg), 3))
    print("SER: ", np.round(np.mean(est_sd_pg) / np.std(bias_pg), 3))
    print("RMSE:", np.round(np.sqrt(np.mean(bias_pg)**2 + np.var(bias_pg)), 3))
    print("Cvr :", np.round(np.mean(cover_pg), 3))
    print("====================================")
    print("")

# Results: 2022/11/12
#
# ====================================
# Scenario: 1
# ====================================
# Minimal standard g-computation
# ------------------------------------
# Bias: -0.001
# ESE:  0.109
# ASE:  0.108
# SER:  0.992
# RMSE: 0.109
# Cvr : 0.946
# ------------------------------------
# Standard g-computation
# ------------------------------------
# Bias: 0.0
# ESE:  0.119
# ASE:  0.118
# SER:  0.988
# RMSE: 0.119
# Cvr : 0.95
# ------------------------------------
# Proxy g-computation
# ------------------------------------
# Bias: 0.0
# ESE:  0.119
# ASE:  0.119
# SER:  0.994
# RMSE: 0.119
# Cvr : 0.95
# ====================================
#
# ====================================
# Scenario: 2
# ====================================
# Minimal standard g-computation
# ------------------------------------
# Bias: 0.655
# ESE:  0.141
# ASE:  0.138
# SER:  0.982
# RMSE: 0.67
# Cvr : 0.002
# ------------------------------------
# Standard g-computation
# ------------------------------------
# Bias: 0.265
# ESE:  0.149
# ASE:  0.146
# SER:  0.98
# RMSE: 0.304
# Cvr : 0.562
# ------------------------------------
# Proxy g-computation
# ------------------------------------
# Bias: -0.003
# ESE:  0.21
# ASE:  0.208
# SER:  0.993
# RMSE: 0.21
# Cvr : 0.946
# ====================================
#
# ====================================
# Scenario: 3
# ====================================
# Minimal standard g-computation
# ------------------------------------
# Bias: 1.957
# ESE:  0.209
# ASE:  0.207
# SER:  0.994
# RMSE: 1.968
# Cvr : 0.0
# ------------------------------------
# Standard g-computation
# ------------------------------------
# Bias: 0.262
# ESE:  0.147
# ASE:  0.146
# SER:  0.996
# RMSE: 0.3
# Cvr : 0.565
# ------------------------------------
# Proxy g-computation
# ------------------------------------
# Bias: -0.9
# ESE:  0.555
# ASE:  0.553
# SER:  0.996
# RMSE: 1.057
# Cvr : 0.678
# ====================================

# END OF SCRIPT
