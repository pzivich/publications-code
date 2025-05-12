#######################################################################################################################
# Fused Cohort Studies: Replication of Results from Tables 1 and 2
#
# Paul Zivich (2025/01/16)
#######################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

######################################################################################################################
# Setup

# Loading in the tabled data from the paper
d = pd.DataFrame()
d['W'] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
d['A'] = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
d['V'] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
d['Y'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
d['n'] = [209, 70, 7, 63, 102, 34, 2, 14, 104, 35, 6, 55, 179, 60, 6, 54]
d = pd.DataFrame(np.repeat(d.values, d['n'], axis=0), columns=d.columns)
d['C'] = 1
d['AW'] = d['A'] * d['W']

# Oracle copy of the data
d_oracle = d.copy()

# Copy the data and set as missing
d1 = d.copy()
d1['R'] = 1
d1['V'] = np.nan
d0 = d.copy()
d0['R'] = 0
d0['W'] = np.nan
d0['AW'] = np.nan
d = pd.concat([d1, d0])
# d.info()

######################################################################################################################
# Estimation Approaches

# Storage for results
rows = []


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
    ee_mud = np.ones(y.shape) * (mu1 - mu0) - mud
    ee_mu1 = ipw * a * (y - mu1)
    ee_mu0 = ipw * (1-a) * (y - mu0)

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0, ee_ps])


estr1 = MEstimator(psi_1, init=[0, 0.5, 0.5, 0, 0])
estr1.estimate()
ci = estr1.confidence_intervals()
rows.append(["Study-1", estr1.theta[0], estr1.variance[0, 0]**0.5, ci[0, 0], ci[0, 1]])


############################################
# Estimator 2: Naive with R=0

def psi_2(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    dr = d.loc[d['R'] == 0].copy()
    y = np.asarray(dr['V'])
    a = np.asarray(dr['A'])

    # Estimate interest parameters with Hajek
    ee_mud = np.ones(y.shape) * (mu1 - mu0) - mud
    ee_mu1 = a * (y - mu1)
    ee_mu0 = (1-a) * (y - mu0)

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0])


estr2 = MEstimator(psi_2, init=[0, 0.5, 0.5])
estr2.estimate()
ci = estr2.confidence_intervals()
rows.append(["Study-2", estr2.theta[0], estr2.variance[0, 0]**0.5, ci[0, 0], ci[0, 1]])

############################################
# Estimator 3: Meta-analysis

p1, p2 = estr1.theta[0], estr2.theta[0]
v1, v2 = estr1.variance[0, 0], estr2.variance[0, 0]
inv_var_w = 1 / (1/v1 + 1/v2)

est_meta = inv_var_w * (p1/v1 + p2/v2)
lcl = est_meta - 1.96 * np.sqrt(inv_var_w)
ucl = est_meta + 1.96 * np.sqrt(inv_var_w)
rows.append(["Meta", est_meta, inv_var_w, lcl, ucl])


############################################
# Estimator 4: Pooled-analysis

def psi_4(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    y = np.asarray(d['Y'])
    a = np.asarray(d['A'])

    # Estimate interest parameters with Hajek
    ee_mud = np.ones(y.shape) * (mu1 - mu0) - mud
    ee_mu1 = a * (y - mu1)
    ee_mu0 = (1-a) * (y - mu0)

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0])


estr4 = MEstimator(psi_4, init=[0, 0.5, 0.5])
estr4.estimate()
ci = estr4.confidence_intervals()
rows.append(["Pool", estr4.theta[0], estr4.variance[0, 0]**0.5, ci[0, 0], ci[0, 1]])


############################################
# Estimator 5: IPW fusion estimator

def psi_5(theta):
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
    ee_mud = np.ones(y.shape) * (mu1 - mu0) - mud
    ee_mu1 = np.ones(y.shape) * mu1 * (alpha + beta - 1) - (delta1 + beta - 1)
    ee_mu0 = np.ones(y.shape) * mu0 * (alpha + beta - 1) - (delta0 + beta - 1)

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0,
                      ee_delta1, ee_delta0,
                      ee_sens, ee_spec,
                      ee_ps])


estr5 = MEstimator(psi_5, init=[0, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0, 0])
estr5.estimate()
ci = estr5.confidence_intervals()
rows.append(["Fusion", estr5.theta[0], estr5.variance[0, 0]**0.5, ci[0, 0], ci[0, 1]])


############################################
# Estimator 6: Oracle estimator

def psi_6(theta):
    # Setup data
    mud, mu1, mu0 = theta[:3]
    alpha = theta[3:]
    y = np.asarray(d_oracle['V'])
    a = np.asarray(d_oracle['A'])
    W = np.asarray(d_oracle[['C', 'W']])

    # Estimate propensity scores
    ee_ps = ee_regression(theta=alpha, X=W, y=a, model='logistic')
    ps = inverse_logit(np.dot(W, alpha))
    ipw = a / ps + (1-a) / (1-ps)

    # Estimate interest parameters with Hajek
    ee_mud = np.ones(y.shape) * (mu1 - mu0) - mud
    ee_mu1 = ipw * a * (y - mu1)
    ee_mu0 = ipw * (1-a) * (y - mu0)

    # Return stacked estimating equations
    return np.vstack([ee_mud, ee_mu1, ee_mu0, ee_ps])


estr6 = MEstimator(psi_6, init=[0, 0.5, 0.5, 0, 0])
estr6.estimate()
ci = estr6.confidence_intervals()
rows.append(["Oracle", estr6.theta[0], estr6.variance[0, 0]**0.5, ci[0, 0], ci[0, 1]])

######################################################################################################################
# Printing results for TABLE 2

results = pd.DataFrame(rows, columns=['Estimator', 'Estimate', 'StdErr', 'LCL', 'UCL'])
results = results.set_index('Estimator')
print((results * 100).round(1))


######################################################################################################################
# G-computation and IPW Fusion Estimators Stacked Together for Illustration

def psi_f(theta):
    # Setup data
    sens, spec = theta[0], theta[1]
    alpha = theta[2:4]
    delta1_w, delta0_w = theta[4], theta[5]
    rd_w, r1_w, r0_w = theta[6], theta[7], theta[8]
    beta = theta[9:13]
    delta1_g, delta0_g = theta[13], theta[14]
    rd_g, r1_g, r0_g = theta[15], theta[16], theta[17]

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

    # Estimate nuisance models
    ee_ps = ee_regression(theta=alpha, X=W, y=a, model='logistic') * r
    ps = inverse_logit(np.dot(W, alpha))
    ipw = a / ps + (1-a) / (1-ps)
    ee_om = ee_regression(theta=beta, X=X, y=y_star, model='logistic') * r
    y1hat = inverse_logit(np.dot(X1, beta))
    y0hat = inverse_logit(np.dot(X0, beta))

    # Estimate mismeasured interest parameters
    ee_delta1_w = r * ipw * a * (y_star - delta1_w)
    ee_delta0_w = r * ipw * (1-a) * (y_star - delta0_w)
    ee_delta1_g = r * (y1hat - delta1_g)
    ee_delta0_g = r * (y0hat - delta0_g)

    # Estimate interest parameters correcting for measurement error
    ee_r1_w = np.ones(y.shape) * r1_w * (sens + spec - 1) - (delta1_w + spec - 1)
    ee_r0_w = np.ones(y.shape) * r0_w * (sens + spec - 1) - (delta0_w + spec - 1)
    ee_rd_w = np.ones(y.shape) * (r1_w - r0_w) - rd_w
    ee_r1_g = np.ones(y.shape) * r1_g * (sens + spec - 1) - (delta1_g + spec - 1)
    ee_r0_g = np.ones(y.shape) * r0_g * (sens + spec - 1) - (delta0_g + spec - 1)
    ee_rd_g = np.ones(y.shape) * (r1_g - r0_g) - rd_g

    # Return stacked estimating equations
    return np.vstack([ee_sens, ee_spec,
                      ee_ps, ee_delta1_w, ee_delta0_w,
                      ee_rd_w, ee_r1_w, ee_r0_w,
                      ee_om, ee_delta1_g, ee_delta0_g,
                      ee_rd_g, ee_r1_g, ee_r0_g,
                      ])


estrf = MEstimator(psi_f, init=[0.75, 0.75,
                                0, 0, 0.5, 0.5,
                                0, 0.5, 0.5,
                                0, 0, 0, 0, 0.5, 0.5,
                                0, 0.5, 0.5])
estrf.estimate()
print("")
print("Comparing IPW and G-computation")
print("IPW:          ", estrf.theta[6])
print("G-computation:", estrf.theta[15])

# NOTE: AIPW estimator can be constructed by fitting the outcome model for g-computation weighted by IPW

# Output
#            Estimate  StdErr   LCL  UCL
# Estimator
# Study-1        -6.7     3.2 -13.0 -0.4
# Study-2        -7.0     2.5 -12.0 -2.0
# Meta           -6.9     0.0 -10.8 -3.0
# Pool           -4.7     2.2  -9.0 -0.4
# Fusion        -10.3     5.0 -20.1 -0.6
# Oracle        -10.0     2.6 -15.0 -4.9
#
# Comparing IPW and G-computation
# IPW:           -0.10347480297643738
# G-computation: -0.1034748029764375
