####################################################################################################################
# Empirical sandwich variance estimator for iterated conditional expectation g-computation
#   Illustrative application with Add Health
#
# Paul Zivich
####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.utilities import spline

from funcs import rescale, indicator_terms
from efuncs import ee_ice_gformula


######################################
# Loading Data
d = pd.read_csv("data/addhealth.csv")

# Adding intercept to data set
d['intercept'] = 1

# Scaling continuous variables as standard normals
d['height_w3'] = rescale(variable=d['height_w3'])
d['height_w1'] = rescale(variable=d['height_w1'])
d['weight_w3'] = rescale(variable=d['weight_w3'])
d['weight_w1'] = rescale(variable=d['weight_w1'])

# Generating splines
for v in ['height_w3', 'height_w1', 'weight_w3', 'weight_w1']:
    klocs = np.nanpercentile(d[v], q=[5, 33, 67, 95])
    sp_terms = spline(variable=np.asarray(d[v]), knots=klocs, power=3, restricted=True)
    d[v+"_sp1"] = sp_terms[:, 0]
    d[v+"_sp2"] = sp_terms[:, 1]
    d[v+"_sp3"] = sp_terms[:, 2]

# Generating indicator terms
indicator_terms(data=d, variable='age', values=[13, 14, 15, 16, 18, 19])
indicator_terms(data=d, variable='exercise_w1', values=[0, 1, 2])
indicator_terms(data=d, variable='exercise_w3', values=[0, 1, 2])
indicator_terms(data=d, variable='alcohol_w1', values=[1, 2, 3, 4])
indicator_terms(data=d, variable='alcohol_w3', values=[1, 2, 3, 4])
indicator_terms(data=d, variable='srh_w1', values=[1, 2, 3, 4])
indicator_terms(data=d, variable='srh_w3', values=[1, 2, 3, 4])
indicator_terms(data=d, variable='race_w1', values=[1, 2, 3])
indicator_terms(data=d, variable='race_w3', values=[1, 2, 3])
indicator_terms(data=d, variable='educ_w1', values=[7, 8, 9, 11, 12])
indicator_terms(data=d, variable='educ_w3', values=[0, 1, 3])

# Generating interaction terms
d['cigarette_w1w3'] = d['cigarette_w1']*d['cigarette_w3']
d.to_csv("data/addhealth_processed.csv", index=False)

######################################
# Creating design matrices

# Observed Outcome: Hypertension (I or II) measured at Wave IV
y = np.asarray(d['htn_w4'])

# Setting action pattern for intervention
da = d.copy()
da['cigarette_w1'] = 0
da['cigarette_w3'] = 0
da['cigarette_w1w3'] = da['cigarette_w1']*da['cigarette_w3']

# Observed Confounders Wave III
cols = ['intercept',
        'cigarette_w1', 'cigarette_w3', 'cigarette_w1w3',
        'age_13', 'age_14', 'age_15', 'age_16', 'age_18', 'age_19',
        'height_w3', 'height_w3_sp1', 'height_w3_sp2', 'height_w3_sp3',
        'weight_w3', 'weight_w3_sp1', 'weight_w3_sp2', 'weight_w3_sp3',
        'gender_w3', 'ethnic_w3',
        'race_w3_1', 'race_w3_2', 'race_w3_3',
        'educ_w3_0', 'educ_w3_1', 'educ_w3_3',
        'exercise_w3_0', 'exercise_w3_1', 'exercise_w3_2',
        'alcohol_w3_1', 'alcohol_w3_2', 'alcohol_w3_3', 'alcohol_w3_4',
        'srh_w3_1', 'srh_w3_2', 'srh_w3_3', 'srh_w3_4',
        'hbp_w3', 'hins_w3', 'tried_cigarette'
        ]
X1 = np.asarray(d[cols])
X1a = np.asarray(da[cols])
follow1 = np.asarray((1-d['cigarette_w3']) * (1-d['cigarette_w1']))

# Observed Confounders Wave I
cols = ['intercept',
        'cigarette_w1',
        'age_13', 'age_14', 'age_15', 'age_16', 'age_18', 'age_19',
        'height_w1', 'height_w1_sp1', 'height_w1_sp2', 'height_w1_sp3',
        'weight_w1', 'weight_w1_sp1', 'weight_w1_sp2', 'weight_w1_sp3',
        'gender_w1', 'ethnic_w1',
        'race_w1_1', 'race_w1_2', 'race_w1_3',
        'educ_w1_7', 'educ_w1_8', 'educ_w1_9', 'educ_w1_11', 'educ_w1_12',
        'exercise_w1_0', 'exercise_w1_1', 'exercise_w1_2',
        'alcohol_w1_1', 'alcohol_w1_2', 'alcohol_w1_3', 'alcohol_w1_4',
        'srh_w1_1', 'srh_w1_2', 'srh_w1_3', 'srh_w1_4',
        'tried_cigarette'
        ]
X0 = np.asarray(d[cols])
X0a = np.asarray(da[cols])
follow0 = np.asarray(1-d['cigarette_w1'])

######################################
# Applying ICE-g-computation


def psi_ice_prevent(theta):
    return ee_ice_gformula(theta=theta,
                           y=y,
                           X_array=[X1, X0],
                           Xa_array=[X1a, X0a],
                           )


init_vals = [0.18, ] + [0., ]*X1.shape[1] + [0., ]*X0.shape[1]
estr_nev = MEstimator(psi_ice_prevent, init=init_vals)
estr_nev.estimate(solver='lm', maxiter=20000)

print(estr_nev.theta[0])
print(np.sqrt(estr_nev.variance[0, 0]))
print(estr_nev.confidence_intervals()[0, :])
print(estr_nev.theta[1:X1.shape[1]+1])
print(estr_nev.theta[X1.shape[1]+1:])


def psi_ice_natural(theta):
    return ee_ice_gformula(theta=theta,
                           y=y,
                           X_array=[X1, X0],
                           Xa_array=[X1, X0],
                           )


init_vals = [0.18, ] + [0., ]*X1.shape[1] + [0., ]*X0.shape[1]
estr_nat = MEstimator(psi_ice_natural, init=init_vals)
estr_nat.estimate(solver='lm', maxiter=20000)

print(estr_nat.theta[0])
print(np.sqrt(estr_nat.variance[0, 0]))
print(estr_nat.confidence_intervals()[0, :])


def psi_ice_abe(theta):
    mu = theta[0]
    ndim = 1+X1.shape[1]+X0.shape[1]
    alpha = theta[1:1+ndim]
    beta = theta[1+ndim:]

    # Natural-course ICE g-formula
    ee_n = ee_ice_gformula(theta=alpha,
                           y=y,
                           X_array=[X1, X0],
                           Xa_array=[X1, X0],
                           )

    # Never smoke ICE g-formula
    ee_a = ee_ice_gformula(theta=beta,
                           y=y,
                           X_array=[X1, X0],
                           Xa_array=[X1a, X0a],
                           )

    # Average Ban Effect
    ee_m = np.ones(y.shape[0])*(alpha[0] - beta[0] + mu)

    # Return stack of estimating equations
    return np.vstack([ee_m, ee_n, ee_a])


init_vals = [0.0, ] + list(estr_nat.theta) + list(estr_nev.theta)  # Using previous model fits to speed up process
estr_abe = MEstimator(psi_ice_abe, init=init_vals)
estr_abe.estimate(solver='lm', maxiter=20000)

print(estr_abe.theta[0])
print(np.sqrt(estr_abe.variance[0, 0]))
print(estr_abe.confidence_intervals()[0, :])
