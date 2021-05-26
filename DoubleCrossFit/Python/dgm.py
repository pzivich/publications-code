import numpy as np
import pandas as pd
from scipy.stats import logistic
from zepid.sensitivity_analysis import trapezoidal

######################################
# Setting data generation parameters
######################################
n = 3000  # size of each sample
sims = 2000  # number of repeated samples
np.random.seed(1015033030)

######################################
# Generating data
######################################
ids = []  # Generating simulation IDs
for i in range(sims):
    ids.extend([i + 1] * n)

df = pd.DataFrame()
df['sim_id'] = ids

# Creating confounders
df['age'] = np.round(trapezoidal(40, 40, 60, 75, size=n * sims), 0)  # trapezoidal distribution
df['ldl_log'] = 0.005 * df['age'] + np.random.normal(np.log(100), 0.18, size=n * sims)
df['diabetes'] = np.random.binomial(n=1, p=logistic.cdf(-4.23 + 0.03 * df['ldl_log'] - 0.02 * df['age'] +
                                                        0.0009 * df['age'] ** 2), size=n * sims)
df['frailty'] = logistic.cdf(-5.5 + 0.05 * (df['age'] - 20) + 0.001 * df['age'] ** 2 + np.random.normal(size=n * sims))
df['age_ln'] = np.log(df['age'])
df['risk_score'] = logistic.cdf(4.299 + 3.501 * df['diabetes'] - 2.07 * df['age_ln'] + 0.051 * df['age_ln']**2 +
                                4.090 * df['ldl_log'] - 1.04 * df['age_ln'] * df['ldl_log'] + 0.01 * df['frailty'])
df['risk_score_cat'] = np.where(df['risk_score'] < .05, 0, np.nan)
df['risk_score_cat'] = np.where((df['risk_score'] >= .05) & (df['risk_score'] < .075), 1, df['risk_score_cat'])
df['risk_score_cat'] = np.where((df['risk_score'] >= .075) & (df['risk_score'] < .2), 2, df['risk_score_cat'])
df['risk_score_cat'] = np.where(df['risk_score'] > .2, 3, df['risk_score_cat'])

# Treatment mechanism
df['statin_pr'] = logistic.cdf(-3.471
                               + 1.390*df['diabetes']
                               + 0.112*df['ldl_log']
                               + 0.973*np.where(df['ldl_log'] > np.log(160), 1, 0)
                               - 0.046*(df['age'] - 30)
                               + 0.003*(df['age'] - 30)**2
                               # Treatment-assignment based on risk-score
                               + 0.273 * np.where(df['risk_score_cat'] == 1, 1, 0)
                               + 1.592 * np.where(df['risk_score_cat'] == 2, 1, 0)
                               + 2.641 * np.where(df['risk_score_cat'] == 3, 1, 0)
                               )
df['statin'] = np.random.binomial(n=1, p=df['statin_pr'], size=n*sims)

# Potential outcomes
prY1 = logistic.cdf(-6.25
                    # Treatment effect
                    - 0.75
                    + 0.35 * np.where(df['ldl_log'] < np.log(130), 5-df['ldl_log'], 0)
                    # Other effects
                    + 0.45 * (np.sqrt(df['age']-39))
                    + 1.75 * df['diabetes']
                    + 0.29 * (np.exp(df['risk_score']+1))
                    + 0.14 * np.where(df['ldl_log'] > np.log(120), df['ldl_log']**2, 0)
                    )
df['Y1'] = np.random.binomial(n=1, p=prY1, size=n*sims)

prY0 = logistic.cdf(-6.25
                    # Other effects
                    + 0.45 * (np.sqrt(df['age']-39))
                    + 1.75 * df['diabetes']
                    + 0.29 * (np.exp(df['risk_score']+1))
                    + 0.14 * np.where(df['ldl_log'] > np.log(120), df['ldl_log']**2, 0)
                    )
df['Y0'] = np.random.binomial(n=1, p=prY0, size=n*sims)
df['Y'] = np.where(df['statin'] == 1, df['Y1'], df['Y0'])  # causal consistency

###############################
# Final clean before CSV
###############################
# print(np.mean(df['Y1'] - df['Y0']))
# truth = -0.1081508
columns_to_keep = ['sim_id', 'Y', 'statin', 'age', 'ldl_log', 'diabetes', 'risk_score', 'risk_score_cat']
df[columns_to_keep].to_csv("statin_sim_data.csv", index=False)
