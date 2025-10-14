#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file runs the descriptive analyses described in the main paper.
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

np.random.seed(80941)

# Loading data
d = pd.read_csv("../data/nhanes.csv")
d = d.dropna()
d['intercept'] = 1


# Fitting a linear model for age

def psi(theta):
    return ee_regression(theta, X=d[['intercept', 'age']],
                         y=d['sbp'], model='linear')


estr = MEstimator(psi, init=[100., 0.])
estr.estimate()

# Generating predictions from the linear model for the plot
dp = pd.DataFrame()
dp['age'] = np.linspace(0.5, 24.5, num=100)
dp['intercept'] = 1

preds = regression_predictions(X=dp[['intercept', 'age']], theta=estr.theta, covariance=estr.variance)
dp['yhat'] = preds[:, 0]
dp['ylow'] = preds[:, 2]
dp['yupp'] = preds[:, 3]

# Plotting the jittered age / SBP points
d['age_jitter'] = d['age'] + np.random.uniform(-0.3, 0.3, size=d.shape[0])
plt.plot(d['age_jitter'], d['sbp'], '.', alpha=0.05, color='k')

# Drawing the extrapolation line
plt.plot(dp['age'], dp['yhat'], '-', color='blue')

# Adding nonparametric age - SBP means to the plot
for age in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
    plt.plot([age, ], [np.mean(d.loc[d['age'] == age, 'sbp']), ],
             'D', color='lightblue', markeredgecolor='k')


# Adding the mathematical model results to the plot
d = pd.read_csv("../data/nhanes.csv")
d = d.dropna(subset=['height', ])
d['height'] = d['height'] / 2.54
for age in [2, 3, 4, 5, 6, 7]:
    dx = d.loc[d['age'] == age].copy()
    ds = dx.sample(20000, replace=True, weights=dx['sample_weight'])
    math_model = MathModel()
    bp = math_model.simulate_blood_pressure(female=ds['female'], age=ds['age'], height=ds['height'])
    vparts = plt.violinplot(bp, positions=[age, ],
                            showmeans=False, showmedians=False, showextrema=False, widths=0.75)
    for pc in vparts['bodies']:
        pc.set_facecolor('red')
    plt.plot([age, ], [np.mean(bp), ], 'D', color='lightcoral', markeredgecolor='k')

# Formatting the plot
plt.xlim([1, 18])
plt.xticks([3, 5, 7, 9, 11, 13, 15, 17])
plt.ylim([60, 150])
plt.ylabel("Systolic Blood Pressure (mm Hg)")
plt.xlabel("Age (years)")
plt.tight_layout()
plt.show()
