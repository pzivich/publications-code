#############################################################################
# Python code to run estimation on a single data set
#
# Paul Zivich 2020/02/26
#############################################################################

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# local imports
from estimators import IPTW, GFormula, AIPTW, TMLE, DoubleCrossfitAIPTW, DoubleCrossfitTMLE
from super_learner import superlearnersetup

##################
# Loading Data
##################
df = pd.read_csv("statin_sim_data.csv")
dfs = df.loc[df['sim_id'] == 916].copy()  # selecting out the randomly chosen data set

# neural-net can sometimes throw warnings during cross-fitting, cluttering the results
warnings.filterwarnings("ignore")

##################
# Table 1
##################
print("========================================")
print("Data description")
print("----------------------------------------")
fmt = '        Statin n={:<4}  No Statin n={:<4}'
print(fmt.format(dfs.loc[dfs['statin'] == 1].shape[0], dfs.loc[dfs['statin'] == 0].shape[0]))
print("----------------------------------------")

fmt = 'Age      {:<13}        {:<13}'
print(fmt.format(np.round(np.mean(dfs.loc[dfs['statin'] == 1, 'age'])),
                 np.round(np.mean(dfs.loc[dfs['statin'] == 0, 'age']))))
fmt = 'Age(SD)  {:<13}        {:<13}'
print(fmt.format(np.round(np.std(dfs.loc[dfs['statin'] == 1, 'age'], ddof=1), 2),
                 np.round(np.std(dfs.loc[dfs['statin'] == 0, 'age'], ddof=1), 2)))

fmt = 'Diabetes {:<13}        {:<13}'
print(fmt.format(np.round(np.mean(dfs.loc[dfs['statin'] == 1, 'diabetes']), 2),
                 np.round(np.mean(dfs.loc[dfs['statin'] == 0, 'diabetes']), 2)))

fmt = 'log(LDL) {:<13}        {:<13}'
print(fmt.format(np.round(np.mean(dfs.loc[dfs['statin'] == 1, 'ldl_log']), 2),
                 np.round(np.mean(dfs.loc[dfs['statin'] == 0, 'ldl_log']), 2)))
fmt = 'LDL(SD)  {:<13}        {:<13}'
print(fmt.format(np.round(np.std(dfs.loc[dfs['statin'] == 1, 'ldl_log'], ddof=1), 2),
                 np.round(np.std(dfs.loc[dfs['statin'] == 0, 'ldl_log'], ddof=1), 2)))

fmt = 'Risk #   {:<13}        {:<13}'
print(fmt.format(np.round(np.mean(dfs.loc[dfs['statin'] == 1, 'risk_score']), 2),
                 np.round(np.mean(dfs.loc[dfs['statin'] == 0, 'risk_score']), 2)))
fmt = 'Risk#(SD){:<13}        {:<13}'
print(fmt.format(np.round(np.std(dfs.loc[dfs['statin'] == 1, 'risk_score'], ddof=1), 2),
                 np.round(np.std(dfs.loc[dfs['statin'] == 0, 'risk_score'], ddof=1), 2)))

fmt = 'ASCVD    {:<13}        {:<13}'
print(fmt.format(np.round(np.mean(dfs.loc[dfs['statin'] == 1, 'Y']), 2),
                 np.round(np.mean(dfs.loc[dfs['statin'] == 0, 'Y']), 2)))

print("========================================")

##################
# Table 2
##################
# main effect models given to both approaches
g_model = 'diabetes + age + risk_score + ldl_log'
q_model = 'statin + diabetes + age + risk_score + ldl_log'

bs_samples = 250  # number of bootstraps used for g-computation variance
sample_splits_n = 100  # number of different sample splits to use for cross-fitting procedures


#################################
# G-formula -- Parametric
start_time = time.time()
g = GFormula(dfs, 'statin', 'Y')
g.outcome_model(q_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000))
g.fit()
rd = g.risk_difference
rd_bs = []
for i in range(bs_samples):
    s = dfs.sample(n=dfs.shape[0], replace=True)
    g = GFormula(s, 'statin', 'Y')
    g.outcome_model(q_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000))
    g.fit()
    rd_bs.append(g.risk_difference)


time_min = (time.time() - start_time) / 60
print("========================================")
print("G-formula -- Parametric")
print("----------------------------------------")
print("RD:           ", np.round(rd, 2))
sd = np.std(rd_bs, ddof=1)
print("SD(RD):       ", np.round(sd, 3))
lcl = rd - 1.96*sd
ucl = rd + 1.96*sd
print("95% CL:       ", np.round([lcl, ucl], 2))
print("CLD:       ", np.round(ucl - lcl, 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# G-formula -- Machine Learning
start_time = time.time()
g = GFormula(dfs, 'statin', 'Y')
bsl = superlearnersetup(var_type='binary', K=5)
g.outcome_model(q_model, bsl)
g.fit()
rd = g.risk_difference
rd_bs = []
for i in range(bs_samples):
    s = dfs.sample(n=dfs.shape[0], replace=True)
    g = GFormula(s, 'statin', 'Y')
    g.outcome_model(q_model, bsl)
    g.fit()
    rd_bs.append(g.risk_difference)

time_min = (time.time() - start_time) / 60
print("========================================")
print("G-formula -- Super Learner")
print("----------------------------------------")
print("RD:           ", np.round(rd, 2))
sd = np.std(rd_bs, ddof=1)
print("SD(RD):       ", np.round(sd, 3))
lcl = rd - 1.96*sd
ucl = rd + 1.96*sd
print("95% CL:       ", np.round([lcl, ucl], 2))
print("CLD:       ", np.round(ucl - lcl, 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# IPTW -- Parametric
start_time = time.time()
ipw = IPTW(dfs, 'statin', 'Y')
ipw.treatment_model(g_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000), bound=0.01)
ipw.fit()
time_min = (time.time() - start_time) / 60
print("========================================")
print("IPTW -- Parametric")
print("----------------------------------------")
print("RD:           ", np.round(ipw.risk_difference, 2))
print("SD(RD):       ", np.round(ipw.risk_difference_se, 3))
print("95% CL:       ", np.round(ipw.risk_difference_ci, 2))
print("CLD:          ", np.round(ipw.risk_difference_ci[1] - ipw.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# IPTW -- Machine Learning
start_time = time.time()
ipw = IPTW(dfs, 'statin', 'Y')
bsl = superlearnersetup(var_type='binary', K=5)
ipw.treatment_model(g_model, bsl, bound=0.01)
ipw.fit()
time_min = (time.time() - start_time) / 60
print("========================================")
print("IPTW -- Super Learner")
print("----------------------------------------")
print("RD:           ", np.round(ipw.risk_difference, 2))
print("SD(RD):       ", np.round(ipw.risk_difference_se, 3))
print("95% CL:       ", np.round(ipw.risk_difference_ci, 2))
print("CLD:          ", np.round(ipw.risk_difference_ci[1] - ipw.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# AIPTW -- Parametric
start_time = time.time()
aipw = AIPTW(dfs, 'statin', 'Y')
aipw.treatment_model(g_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000), bound=0.01)
aipw.outcome_model(q_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000))
aipw.fit()
time_min = (time.time() - start_time) / 60
print("========================================")
print("AIPTW -- Parametric")
print("----------------------------------------")
print("RD:           ", np.round(aipw.risk_difference, 2))
print("SD(RD):       ", np.round(aipw.risk_difference_se, 3))
print("95% CL:       ", np.round(aipw.risk_difference_ci, 2))
print("CLD:          ", np.round(aipw.risk_difference_ci[1] - aipw.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# AIPTW -- Machine Learning
start_time = time.time()
aipw = AIPTW(dfs, 'statin', 'Y')
bsl = superlearnersetup(var_type='binary', K=5)
aipw.treatment_model(g_model, bsl, bound=0.01)
aipw.outcome_model(q_model, bsl)
aipw.fit()
time_min = (time.time() - start_time) / 60
print("========================================")
print("AIPTW -- Super Learner")
print("----------------------------------------")
print("RD:           ", np.round(aipw.risk_difference, 2))
print("SD(RD):       ", np.round(aipw.risk_difference_se, 3))
print("95% CL:       ", np.round(aipw.risk_difference_ci, 2))
print("CLD:          ", np.round(aipw.risk_difference_ci[1] - aipw.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# TMLE -- Parametric
start_time = time.time()
tmle = TMLE(dfs, 'statin', 'Y')
tmle.treatment_model(g_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000), bound=0.01)
tmle.outcome_model(q_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000))
tmle.fit()
time_min = (time.time() - start_time) / 60
print("========================================")
print("TMLE -- Parametric")
print("----------------------------------------")
print("RD:           ", np.round(tmle.risk_difference, 2))
print("SD(RD):       ", np.round(tmle.risk_difference_se, 3))
print("95% CL:       ", np.round(tmle.risk_difference_ci, 2))
print("CLD:          ", np.round(tmle.risk_difference_ci[1] - tmle.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# TMLE -- Machine Learning
start_time = time.time()
tmle = TMLE(dfs, 'statin', 'Y')
bsl = superlearnersetup(var_type='binary', K=5)
tmle.treatment_model(g_model, bsl, bound=0.01)
tmle.outcome_model(q_model, bsl)
tmle.fit()
time_min = (time.time() - start_time) / 60
print("========================================")
print("TMLE -- Super Learner")
print("----------------------------------------")
print("RD:           ", np.round(tmle.risk_difference, 2))
print("SD(RD):       ", np.round(tmle.risk_difference_se, 3))
print("95% CL:       ", np.round(tmle.risk_difference_ci, 2))
print("CLD:          ", np.round(tmle.risk_difference_ci[1] - tmle.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# DC-AIPW -- Parametric
start_time = time.time()
dcdr = DoubleCrossfitAIPTW(dfs, 'statin', 'Y')
dcdr.treatment_model(g_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000), bound=0.01)
dcdr.outcome_model(q_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000))
dcdr.fit(resamples=sample_splits_n, method='median')
time_min = (time.time() - start_time) / 60
print("========================================")
print("DC-AIPW -- Parametric")
print("----------------------------------------")
print("RD:           ", np.round(dcdr.risk_difference, 2))
print("SD(RD):       ", np.round(dcdr.risk_difference_se, 3))
print("95% CL:       ", np.round(dcdr.risk_difference_ci, 2))
print("CLD:          ", np.round(dcdr.risk_difference_ci[1] - dcdr.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# DC-AIPW -- Super Learner
start_time = time.time()
dcdr = DoubleCrossfitAIPTW(dfs, 'statin', 'Y')
bsl = superlearnersetup(var_type='binary', K=5)
dcdr.treatment_model(g_model, bsl, bound=0.01)
dcdr.outcome_model(q_model, bsl)
dcdr.fit(resamples=sample_splits_n, method='median')
time_min = (time.time() - start_time) / 60
print("========================================")
print("DC-AIPW -- Super Learner")
print("----------------------------------------")
print("RD:           ", np.round(dcdr.risk_difference, 2))
print("SD(RD):       ", np.round(dcdr.risk_difference_se, 3))
print("95% CL:       ", np.round(dcdr.risk_difference_ci, 2))
print("CLD:          ", np.round(dcdr.risk_difference_ci[1] - dcdr.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# DC-TMLE -- Parametric
start_time = time.time()
tmle = DoubleCrossfitTMLE(dfs, 'statin', 'Y')
tmle.treatment_model(g_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000), bound=0.01)
tmle.outcome_model(q_model, LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000))
tmle.fit(resamples=sample_splits_n, method='median')
time_min = (time.time() - start_time) / 60
print("========================================")
print("DC-TMLE -- Parametric")
print("----------------------------------------")
print("RD:           ", np.round(tmle.risk_difference, 2))
print("SD(RD):       ", np.round(tmle.risk_difference_se, 3))
print("95% CL:       ", np.round(tmle.risk_difference_ci, 2))
print("CLD:          ", np.round(tmle.risk_difference_ci[1] - tmle.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")


#################################
# DC-TMLE -- Super Learner
start_time = time.time()
tmle = DoubleCrossfitTMLE(dfs, 'statin', 'Y')
bsl = superlearnersetup(var_type='binary', K=5)
tmle.treatment_model(g_model, bsl, bound=0.01)
tmle.outcome_model(q_model, bsl)
tmle.fit(resamples=sample_splits_n, method='median')
time_min = (time.time() - start_time) / 60
print("========================================")
print("DC-TMLE -- Super Learner")
print("----------------------------------------")
print("RD:           ", np.round(tmle.risk_difference, 2))
print("SD(RD):       ", np.round(tmle.risk_difference_se, 3))
print("95% CL:       ", np.round(tmle.risk_difference_ci, 2))
print("CLD:          ", np.round(tmle.risk_difference_ci[1] - tmle.risk_difference_ci[0], 2))
print("TIME ELAPSED: ", np.round(time_min, 3))
print("========================================")
