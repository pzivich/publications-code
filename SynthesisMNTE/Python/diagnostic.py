#####################################################################################################################
# Using a Synthesis of Statistical and Mathematical Models to Account for Missing Data in Public Health Research
#   This file runs the diagnostic based on the mathematical model predictions for the positive region.
#
# Paul Zivich (2024/12/17)
#####################################################################################################################

###############################################
# Loading packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import regression_predictions

from mathmodel import MathModel

np.random.seed(23905141)

###############################################################################################
# Setting up data

d = pd.read_csv("../data/nhanes.csv")
d = d.dropna(subset=['height', ])
d['height'] = d['height'] / 2.54
d['intercept'] = 1
d1 = d.loc[d['age'] >= 8].copy()  # Subset resampled data to positive region

ages = np.asarray([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
shift = 0.1

###############################################################################################
# Appendix Figure 1

###############################################
# Data

d1['age_jitter'] = d1['age'] + np.random.uniform(-0.3, 0.3, size=d1.shape[0])

plt.figure(figsize=[8, 5])
plt.plot(d1['age_jitter'], d1['sbp'], '.', alpha=0.1, color='k')


###############################################
# Statistical Model

reg_cols = []  # Storage for new regression column names
plot_cols = []
for j in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:  # For each unique age in the positive region
    age_label = 'age' + str(j)  # ... create a new column label
    reg_cols.append(age_label)  # ... add column label to the running list
    d1[age_label] = np.where(d1['age'] == j, 1, 0)  # ... create indicator variable for that specific age
    plot_cols.append(np.asarray(d1.loc[d1['age'] == j, 'sbp'].dropna()))


def psi(theta):
    # Estimating function for statistical nuisance model
    r = np.where(d1['sbp'].isna(), 0, 1)  # Indicator if the outcome was observed
    y_no_miss = np.where(r == 1, d1['sbp'], -9999)  # Replacing NaN with -9999 to avoid NaN-related errors
    return ee_regression(theta,  # Estimating functions for a regression model of
                         X=d1[reg_cols],  # ... indicator terms for age
                         y=y_no_miss,  # ... on outcomes where NaN was replaced
                         model='linear',  # ... using linear regression
                         weights=d1['sample_weight']) * r  # ... using sample weights and among complete-cases only


# Computing M-estimator
estr = MEstimator(psi, init=[100., ] * len(reg_cols))
estr.estimate()

X = np.identity(10)
preds = regression_predictions(X=X, theta=estr.theta, covariance=estr.variance)
est = preds[:, 0]
lcl = preds[:, 2]
ucl = preds[:, 3]

plt.plot(ages + shift, estr.theta, 'D', color='lightblue', markeredgecolor='k')
vparts = plt.violinplot(plot_cols, positions=ages,
                        showmeans=False, showmedians=False, showextrema=False, widths=0.75,
                        side='high')
for pc in vparts['bodies']:
    pc.set_facecolor('blue')


###############################################
# Mathematical Model

math_means = []
for age in ages:
    dx = d.loc[d['age'] == age].copy()
    ds = dx.sample(20000, replace=True, weights=dx['sample_weight'])
    math_model = MathModel()
    bp = math_model.simulate_blood_pressure(female=ds['female'], age=ds['age'], height=ds['height'])
    vparts = plt.violinplot(bp, positions=[age, ],
                            showmeans=False, showmedians=False, showextrema=False, widths=0.75,
                            side='low')
    for pc in vparts['bodies']:
        pc.set_facecolor('red')
    plt.plot([age - shift, ], [np.mean(bp), ], 'D', color='lightcoral', markeredgecolor='k')
    math_means.append(np.mean(bp))

plt.xlim([7, 18])
plt.xticks([8, 10, 12, 14, 16])
plt.ylim([65, 150])
plt.ylabel("Systolic Blood Pressure (mm Hg)")
plt.xlabel("Age (years)")
plt.tight_layout()
plt.savefig("diagnostic_plot1.png", format='png', dpi=300)
plt.close()


###############################################################################################
# Appendix Figure 2

# Storage for parameter of interest estimates
diff_hats = []

# Monte-Carlo procedure for point and confidence interval estimation
for i in range(20000):
    ds1 = d1.sample(n=d1.shape[0], replace=True)             # Resample observed data with replacement

    # Statistical model
    def psi(theta):
        # Estimating function for statistical nuisance model
        r = np.where(ds1['sbp'].isna(), 0, 1)                   # Indicator if the outcome was observed
        y_no_miss = np.where(r == 1, ds1['sbp'], -9999)         # Replacing NaN with -9999 to avoid NaN-related errors
        return ee_regression(theta,                             # Estimating functions for a regression model of
                             X=ds1[reg_cols],                   # ... indicator terms for age
                             y=y_no_miss,                       # ... on outcomes where NaN was replaced
                             model='linear',                    # ... using linear regression
                             weights=ds1['sample_weight']) * r  # ... using sample weights and among complete-cases only

    # Computing M-estimator
    estr = MEstimator(psi, init=[100., ]*len(reg_cols))
    estr.estimate()
    stat_means = np.asarray(estr.theta)

    # Mathematical model
    math_model = MathModel()                                           # Initialize the mathematical model class
    bp = math_model.simulate_blood_pressure(female=ds1['female'],      # Simulate a single SBP given gender,
                                            age=ds1['age'],            # ... age,
                                            height=ds1['height'])      # ... and height
    ds1['sbp-hat'] = bp                                                # Add simulated SBP to non-positive data

    def psi(theta):
        return ee_regression(theta,                             # Estimating functions for a regression model of
                             X=ds1[reg_cols],                   # ... indicator terms for age
                             y=ds1['sbp-hat'],                  # ... on outcomes where NaN was replaced
                             model='linear',                    # ... using linear regression
                             weights=ds1['sample_weight'])      # ... using sample weights and among complete-cases only

    estr = MEstimator(psi, init=[100., ]*len(reg_cols))
    estr.estimate()
    math_means = np.asarray(estr.theta)

    diff_hats.append(stat_means - math_means)


difference = np.median(diff_hats, axis=0)
lower = np.percentile(diff_hats, q=2.5, axis=0)
upper = np.percentile(diff_hats, q=97.5, axis=0)

###############################################
# Difference between predictions plot

plt.figure(figsize=[8, 4])
plt.axhline(0, linestyle='--', color='gray')
plt.vlines(ages, lower, upper, colors='k')
plt.plot(ages, difference, 'o', color='gray', markeredgecolor='k')
plt.yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
plt.ylim([-8, 8])
plt.xticks([8, 10, 12, 14, 16])
plt.xlim([7, 18])
plt.ylabel("Difference Between Models")
plt.xlabel("Age (years)")
plt.savefig("diagnostic_plot2.png", format='png', dpi=300)
plt.tight_layout()
plt.close()
