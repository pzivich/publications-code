#####################################################################################
# Simulations for g-computation:
#   contains correct model, main-terms model, and machine learning
#
#####################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from estimators import GFormula
from super_learner import superlearnersetup


##########################
# Setting some parameters
setup = 1  # options include: 1-correct parametric model, 2-main-terms model, 3-machine learning with main-terms
bs_samples = 250  # number of bootstrap samplse to take for CL in simulations
decimal = 4  # decimal places to display in results printed to console
file_path_to_save_results = "gform_results"+str(setup)+".csv"  # file path to save all result output

##########################
# Reading in data
df = pd.read_csv("statin_sim_data.csv")
truth = -0.1081508
samples = list(df['sim_id'].unique())

if setup == 1:
    # Set-up for correct parametric model
    df['ldl_130'] = np.where(df['ldl_log'] < np.log(130), 5 - df['ldl_log'], 0)
    df['age_sqrt'] = np.sqrt(df['age'] - 39)
    df['risk_exp'] = np.exp(df['risk_score'] + 1)
    df['ldl_120'] = np.where(df['ldl_log'] > np.log(120), df['ldl_log'] ** 2, 0)

    q_model = 'statin + statin:ldl_130 + age_sqrt + diabetes + risk_exp + ldl_120'
    q_estimator = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

elif setup == 2:
    # Set-up for main-term parametric model
    q_model = 'statin + diabetes + age + risk_score + ldl_log'
    q_estimator = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

elif setup == 3:
    # Set-up for machine learning
    q_model = 'statin + diabetes + age + risk_score + ldl_log'
    q_estimator = superlearnersetup(var_type='binary', K=10)

else:
    raise ValueError("Invalid setup choice")


##########################
# Running simulation
bias = []
stderr = []
lcl = []
ucl = []

for i in samples:
    dfs = df.loc[df['sim_id'] == i].copy()

    gform = GFormula(dfs, treatment='statin', outcome='Y')
    gform.outcome_model(covariates=q_model, estimator=q_estimator)
    gform.fit()
    rd = gform.risk_difference
    bias.append(rd - truth)

    bse = []
    for j in range(bs_samples):
        dfsr = dfs.sample(n=dfs.shape[0], replace=True)
        gform = GFormula(dfsr, treatment='statin', outcome='Y')
        gform.outcome_model(covariates=q_model, estimator=q_estimator)
        gform.fit()
        bse.append(gform.risk_difference)

    stderr.append(np.std(bse, ddof=1))
    lcl.append(rd - 1.96*np.std(bse, ddof=1))
    ucl.append(rd + 1.96*np.std(bse, ddof=1))


results = pd.DataFrame()
results['bias'] = bias
results['std'] = stderr
results['lcl'] = lcl
results['ucl'] = ucl
results['cover'] = np.where((results['lcl'] < truth) & (truth < results['ucl']), 1, 0)
results['cld'] = results['ucl'] - results['lcl']

print("============================")
print("G-computation")
print("============================")
print("Mean: ", np.round(np.mean(bias), decimal))
print("RMSE: ", np.round(np.sqrt(np.mean(results['bias'])**2 + np.std(bias, ddof=1)**2), decimal))
print("ASE:  ", np.round(np.mean(stderr), decimal))
print("ESE:  ", np.round(np.std(bias, ddof=1), decimal))
print("CLD:  ", np.round(np.mean(results['cld']), decimal))
print("Cover:", np.round(np.mean(results['cover']), decimal))
print("============================")

results.to_csv(file_path_to_save_results, index=False)
