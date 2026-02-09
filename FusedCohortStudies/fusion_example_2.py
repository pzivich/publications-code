#######################################################################################################################
# Fused Cohort Studies: Replication of Appendix Section 8
#
# Paul Zivich (2026/02/09)
#######################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


######################################################################################################################
# Setup

# Reading in data
d = pd.read_csv("Satoex20jan26.dat", header=0, names=['A', 'W', 'R', 'Y', 'V'], sep=r'\s+')
d['W'] = pd.to_numeric(d['W'], errors='coerce')
d['V'] = pd.to_numeric(d['V'], errors='coerce')
d['AW'] = d['A']*d['W']
d['C'] = 1

######################################################################################################################
# Estimation Approaches

# Storage for results
rows = []

############################################
# Study 1 Only

############################################
# Estimator 1: IPW with R=1

def psi_1(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    alpha = theta[3:]
    dr = d.loc[d['R'] == 1].copy()
    y = np.asarray(dr['Y'])
    a = np.asarray(dr['A'])
    W = np.asarray(dr[['C', 'W']])

    # Estimate propensity scores
    ee_ps = ee_regression(theta=alpha, X=W, y=a, model='logistic')
    ps = inverse_logit(np.dot(W, alpha))
    ipw = a / ps + (1-a) / (1-ps)

    # Estimate interest parameters with Hajek
    ee_mu1 = ipw * a * (y - mu1)
    ee_mu0 = ipw * (1-a) * (y - mu0)
    ee_mud = np.ones(y.shape) * np.log(mu1) - np.log(mu0) - mud

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0, ee_ps])


estr1 = MEstimator(psi_1, init=[0, 0.5, 0.5, 0, 0])
estr1.estimate()
ci = estr1.confidence_intervals()
rows.append(["Study-1", np.exp(estr1.theta[0]), estr1.variance[0, 0]**0.5, np.exp(ci[0, 0]), np.exp(ci[0, 1])])


############################################
# Study 2 Only

def psi_2(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    dr = d.loc[d['R'] == 0].copy()
    y = np.asarray(dr['V'])
    a = np.asarray(dr['A'])

    # Estimate interest parameters with Hajek
    ee_mu1 = a * (y - mu1)
    ee_mu0 = (1-a) * (y - mu0)
    ee_mud = np.ones(y.shape) * np.log(mu1) - np.log(mu0) - mud

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0])


estr2 = MEstimator(psi_2, init=[0, 0.5, 0.5])
estr2.estimate()
ci = estr2.confidence_intervals()
rows.append(["Study-2", np.exp(estr2.theta[0]), estr2.variance[0, 0]**0.5, np.exp(ci[0, 0]), np.exp(ci[0, 1])])


############################################
# IPW fusion estimator

def psi_w(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    delta1, delta0 = theta[3], theta[4]
    alpha = theta[5]
    beta = theta[6]
    gamma = theta[7:]
    r = np.asarray(d['R'])
    y_star = np.asarray(d['Y'])
    y = np.asarray(d['V'].fillna(-9))
    a = np.asarray(d['A'])
    W = np.asarray(d[['C', 'W']].fillna(-9))

    # Estimating measurement error parameters
    ee_sens = (1-r) * y * (y_star - alpha)
    ee_spec = (1-r) * (1-y) * ((1-y_star) - beta)

    # Estimate propensity scores
    ee_ps = ee_regression(theta=gamma, X=W, y=a, model='logistic') * r
    ps = inverse_logit(np.dot(W, gamma))
    ipw = a / ps + (1-a) / (1-ps)

    # Estimate mismeasured interest parameters with Hajek
    ee_delta1 = r * ipw * a * (y_star - delta1)
    ee_delta0 = r * ipw * (1-a) * (y_star - delta0)

    # Estimate mismeasured interest parameters with Hajek
    ee_mud = np.ones(y.shape) * np.log(mu1) - np.log(mu0) - mud
    ee_mu1 = np.ones(y.shape) * mu1 * (alpha + beta - 1) - (delta1 + beta - 1)
    ee_mu0 = np.ones(y.shape) * mu0 * (alpha + beta - 1) - (delta0 + beta - 1)

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0,
                      ee_delta1, ee_delta0,
                      ee_sens, ee_spec,
                      ee_ps])


estrw = MEstimator(psi_w, init=[0, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0, 0])
estrw.estimate()
ci = estrw.confidence_intervals()
rows.append(["Fusion IPW", np.exp(estrw.theta[0]), estrw.variance[0, 0]**0.5, np.exp(ci[0, 0]), np.exp(ci[0, 1])])


############################################
# G-computation fusion estimator

def psi_g(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    delta1, delta0 = theta[3], theta[4]
    sens = theta[5]
    spec = theta[6]
    beta = theta[7:]

    r = np.asarray(d['R'])
    y_star = np.asarray(d['Y'])
    y = np.asarray(d['V'].fillna(-9))
    X = np.asarray(d[['C', 'A', 'W', 'AW']].fillna(-9))
    da = d.copy()
    da['A'] = 1
    da['AW'] = da['A'] * da['W']
    X1 = np.asarray(da[['C', 'A', 'W', 'AW']].fillna(-9))
    da['A'] = 0
    da['AW'] = da['A'] * da['W']
    X0 = np.asarray(da[['C', 'A', 'W', 'AW']].fillna(-9))

    # Estimating measurement error parameters
    ee_sens = (1-r) * y * (y_star - sens)
    ee_spec = (1-r) * (1-y) * ((1-y_star) - spec)

    # Estimate nuisance models
    ee_om = ee_regression(theta=beta, X=X, y=y_star, model='logistic') * r
    y1hat = inverse_logit(np.dot(X1, beta))
    y0hat = inverse_logit(np.dot(X0, beta))

    # Estimate mismeasured interest parameters
    ee_delta1 = r * (y1hat - delta1)
    ee_delta0 = r * (y0hat - delta0)

    # Estimate interest parameters correcting for measurement error
    ee_r1 = np.ones(y.shape) * mu1 * (sens + spec - 1) - (delta1 + spec - 1)
    ee_r0 = np.ones(y.shape) * mu0 * (sens + spec - 1) - (delta0 + spec - 1)
    ee_lrr = np.ones(y.shape) * np.log(mu1) - np.log(mu0) - mud

    # Return stacked estimating equations
    return np.vstack([ee_lrr, ee_r1, ee_r0,
                      ee_delta1, ee_delta0,
                      ee_sens, ee_spec, ee_om])


estrg = MEstimator(psi_g, init=[0., 0.5, 0.5,
                                0.5, 0.5,
                                0.75, 0.75,
                                0, 0, 0, 0])
estrg.estimate()
ci = estrg.confidence_intervals()
rows.append(["Fusion G-comp", np.exp(estrg.theta[0]), estrg.variance[0, 0]**0.5, np.exp(ci[0, 0]), np.exp(ci[0, 1])])


############################################
# AIPW fusion estimator

def psi_a(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    delta1, delta0 = theta[3], theta[4]
    sens = theta[5]
    spec = theta[6]
    gamma = theta[7:9]
    beta = theta[9:]

    r = np.asarray(d['R'])
    y_star = np.asarray(d['Y'])
    y = np.asarray(d['V'].fillna(-9))
    a = np.asarray(d['A'])
    W = np.asarray(d[['C', 'W']].fillna(-9))
    X = np.asarray(d[['C', 'A', 'W', 'AW']].fillna(-9))
    da = d.copy()
    da['A'] = 1
    da['AW'] = da['A'] * da['W']
    X1 = np.asarray(da[['C', 'A', 'W', 'AW']].fillna(-9))
    da['A'] = 0
    da['AW'] = da['A'] * da['W']
    X0 = np.asarray(da[['C', 'A', 'W', 'AW']].fillna(-9))

    # Estimating measurement error parameters
    ee_sens = (1-r) * y * (y_star - sens)
    ee_spec = (1-r) * (1-y) * ((1-y_star) - spec)

    # Estimate propensity scores
    ee_ps = ee_regression(theta=gamma, X=W, y=a, model='logistic') * r
    ps = inverse_logit(np.dot(W, gamma))
    ipw = a / ps + (1-a) / (1-ps)

    # Estimate nuisance models
    ee_om = ee_regression(theta=beta, X=X, y=y_star, model='logistic', weights=ipw) * r
    y1hat = inverse_logit(np.dot(X1, beta))
    y0hat = inverse_logit(np.dot(X0, beta))

    # Estimate mismeasured interest parameters
    ee_delta1 = r * (y1hat - delta1)
    ee_delta0 = r * (y0hat - delta0)

    # Estimate interest parameters correcting for measurement error
    ee_r1 = np.ones(y.shape) * mu1 * (sens + spec - 1) - (delta1 + spec - 1)
    ee_r0 = np.ones(y.shape) * mu0 * (sens + spec - 1) - (delta0 + spec - 1)
    ee_lrr = np.ones(y.shape) * np.log(mu1) - np.log(mu0) - mud

    # Return stacked estimating equations
    return np.vstack([ee_lrr, ee_r1, ee_r0,
                      ee_delta1, ee_delta0,
                      ee_sens, ee_spec, ee_ps, ee_om])


estrg = MEstimator(psi_a, init=[0., 0.5, 0.5,
                                0.5, 0.5,
                                0.75, 0.75,
                                0, 0, 0, 0, 0, 0])
estrg.estimate()
ci = estrg.confidence_intervals()
rows.append(["Fusion AIPW", np.exp(estrg.theta[0]), estrg.variance[0, 0]**0.5, np.exp(ci[0, 0]), np.exp(ci[0, 1])])

############################################
# Displaying Results

results = pd.DataFrame(rows, columns=['Estimator', 'Estimate', 'StdErr', 'LCL', 'UCL'])
results = results.set_index('Estimator')
print(results.round(2))

# OUTPUT
#
#                Estimate  StdErr   LCL   UCL
# Estimator
# Study-1            0.96    0.05  0.87  1.05
# Study-2            1.02    0.08  0.87  1.21
# Fusion IPW         0.81    0.24  0.51  1.29
# Fusion G-comp      0.81    0.24  0.51  1.29
# Fusion AIPW        0.81    0.24  0.51  1.29
