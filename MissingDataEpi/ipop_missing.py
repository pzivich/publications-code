###############################################################################
# Missing Outcome Data in Epidemiologic Studies
#		Python code to recreate the results
#
# Paul Zivich (2022/06/24)
###############################################################################

import sys
import time
from datetime import date
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import DomainWarning

np.random.seed(20220216)
bootstrap_iters = 500

print(date.today())
print("Software versions")
print("---------------------------------")
print("Python:     ", sys.version_info)
print("Numpy:      ", np.__version__)
print("pandas:     ", pd.__version__)
print("statsmodels:", sm.__version__)
print("")


######################################################
# Creating functions for later use

def gcomputation(data, model):
    family = sm.families.family.Binomial()

    # Estimating logistic regression model to data with placebo=1
    fm_p1 = smf.glm(model, data.loc[data['placebo'] == 1],
                    family=family).fit()
    pr_y_a1 = np.mean(fm_p1.predict(data))         # Predicting avg probability for full data under placebo=1

    # Estimating logistic regression model to data with placebo=1
    fm_p0 = smf.glm(model, data.loc[data['placebo'] == 0],
                    family=family).fit()
    pr_y_a0 = np.mean(fm_p0.predict(data))         # Predicting avg probability for full data under placebo=1
    return np.log(pr_y_a1 / pr_y_a0)


def gcomputation_bootstrap(data, model, reps):
    ests = []
    for i in range(reps):
        ds = data.sample(n=data.shape[0], replace=True)
        est = gcomputation(data=ds, model=model)
        ests.append(est)
    return np.std(ests, ddof=1)


######################################################
# No Missing

print("====================================")
print("No Missing")
print("====================================")

# generating data from Table 1
d = pd.DataFrame()
d['short_cervix'] = [0]*215 + [0]*222 + [1]*186 + [1]*177
d['17P'] = [0]*215 + [1]*222 + [0]*186 + [1]*177
d['placebo'] = 1 - d['17P']
d['preterm'] = [1]*15 + [0]*(215-15) + [1]*13 + [0]*(222-13) + [1]*21 + [0]*(186-21) + [1]*23 + [0]*(177-23)

# Checking data against Table 1 (should match exactly)
print('Preterm births:',
      np.sum(d['preterm']),
      "("+str(np.round(np.mean(d['preterm']), 2))+"%)\n")
print("Table 1:\n",
      pd.crosstab([d['short_cervix'], d['17P']], d['preterm']),
      "\n")

# Fitting a log-binomial model
f = sm.families.family.Binomial(sm.families.links.log())
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DomainWarning)
    full = smf.glm("preterm ~ placebo", d, family=f).fit()

print("Table 2")
print("------------------------------------")
print("Crude")
print("RR:    ", np.round(np.exp(np.asarray(full.params)[1]), 4))
print("SE:    ", np.round(np.asarray(full.bse)[1], 4))
print("95% CI:", np.round(np.exp(np.asarray(full.conf_int())[1, :]), 4))
print("------------------------------------")
print("Adjusted")
print("RR:     NA")
print("SE:     NA")
print("95% CI: NA")
print("====================================")
print("")

######################################################
# Missing Completely at Random (MCAR)

print("====================================")
print("MCAR")
print("====================================")

# generating data from Table 1
d['preterm'] = ([1]*11 + [0]*(161-11) + [np.nan]*54
                + [1]*10 + [0]*(167-10) + [np.nan]*55
                + [1]*16 + [0]*(140-16) + [np.nan]*46
                + [1]*17 + [0]*(133-17) + [np.nan]*44)

# Checking data against Table 1 (should match exactly)
print('Preterm births:',
      np.sum(d['preterm']),
      "("+str(np.round(np.mean(d['preterm']), 2))+"%)\n")
print("Table 1:\n",
      pd.crosstab([d['short_cervix'], d['17P']], d['preterm']),
      "\n")

# Fitting a log-binomial model: Crude
f = sm.families.family.Binomial(sm.families.links.log())
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DomainWarning)
    mcar_c = smf.glm("preterm ~ placebo", d, family=f).fit()

# Estimating with g-computation: Adjusted
log_rr_adj = gcomputation(data=d, model="preterm ~ short_cervix")
se_rr_adj = gcomputation_bootstrap(data=d, model="preterm ~ short_cervix",
                                   reps=bootstrap_iters)
ci_rr_adj = np.array([log_rr_adj - 1.96*se_rr_adj,
                      log_rr_adj + 1.96*se_rr_adj])

print("Table 2")
print("------------------------------------")
print("Crude")
print("RR:    ", np.round(np.exp(np.asarray(mcar_c.params)[1]), 4))
print("SE:    ", np.round(np.asarray(mcar_c.bse)[1], 4))
print("95% CI:", np.round(np.exp(np.asarray(mcar_c.conf_int())[1, :]), 4))
print("------------------------------------")
print("Adjusted")
print("RR:    ", np.round(np.exp(log_rr_adj), 4))
print("SE:    ", np.round(se_rr_adj, 4))
print("95% CI:", np.round(np.exp(ci_rr_adj), 4))
print("====================================")
print("")

######################################################
# Missing at Random (MAR) with Positivity

print("====================================")
print("MAR with Det. Positivity")
print("====================================")

# generating data from Table 1
d['preterm'] = ([1]*8 + [0]*(108-8) + [np.nan]*107
                + [1]*13 + [0]*(222-13) + [np.nan]*0
                + [1]*21 + [0]*(186-21) + [np.nan]*0
                + [1]*12 + [0]*(89-12) + [np.nan]*88)

# Checking data against Table 1 (should match exactly)
print('Preterm births:',
      np.sum(d['preterm']),
      "("+str(np.round(np.mean(d['preterm']), 2))+"%)\n")
print("Table 1:\n",
      pd.crosstab([d['short_cervix'], d['17P']], d['preterm']),
      "\n")

# Fitting a log-binomial model: Crude
f = sm.families.family.Binomial(sm.families.links.log())
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DomainWarning)
    marp_c = smf.glm("preterm ~ placebo", d, family=f).fit()

# Estimating with g-computation: Adjusted
log_rr_adj = gcomputation(data=d, model="preterm ~ short_cervix")
se_rr_adj = gcomputation_bootstrap(data=d, model="preterm ~ short_cervix",
                                   reps=bootstrap_iters)
ci_rr_adj = np.array([log_rr_adj - 1.96*se_rr_adj,
                      log_rr_adj + 1.96*se_rr_adj])

print("Table 2")
print("------------------------------------")
print("Crude")
print("RR:    ", np.round(np.exp(np.asarray(marp_c.params)[1]), 4))
print("SE:    ", np.round(np.asarray(marp_c.bse)[1], 4))
print("95% CI:", np.round(np.exp(np.asarray(marp_c.conf_int())[1, :]), 4))
print("------------------------------------")
print("Adjusted")
print("RR:    ", np.round(np.exp(log_rr_adj), 4))
print("SE:    ", np.round(se_rr_adj, 4))
print("95% CI:", np.round(np.exp(ci_rr_adj), 4))
print("====================================")
print("")

######################################################
# Missing at Random (MAR) without Positivity

print("====================================")
print("MAR without Det. Positivity")
print("====================================")

# generating data from Table 1
d['preterm'] = ([1]*15 + [0]*(215-15) + [np.nan]*0
                + [1]*13 + [0]*(222-13) + [np.nan]*0
                + [1]*21 + [0]*(186-21) + [np.nan]*0
                + [1]*0 + [0]*0 + [np.nan]*177)

# Checking data against Table 1 (should match exactly)
print('Preterm births:',
      np.sum(d['preterm']),
      "("+str(np.round(np.mean(d['preterm']), 2))+"%)\n")
print("Table 1:\n",
      pd.crosstab([d['short_cervix'], d['17P']], d['preterm']),
      "\n")

# Fitting a log-binomial model: Crude
f = sm.families.family.Binomial(sm.families.links.log())
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DomainWarning)
    marp_c = smf.glm("preterm ~ placebo", d, family=f).fit()

# Estimating with g-computation: Adjusted
log_rr_adj = gcomputation(data=d, model="preterm ~ short_cervix")
se_rr_adj = gcomputation_bootstrap(data=d, model="preterm ~ short_cervix",
                                   reps=bootstrap_iters)
ci_rr_adj = np.array([log_rr_adj - 1.96*se_rr_adj,
                      log_rr_adj + 1.96*se_rr_adj])

print("Table 2")
print("------------------------------------")
print("Crude")
print("RR:    ", np.round(np.exp(np.asarray(marp_c.params)[1]), 4))
print("SE:    ", np.round(np.asarray(marp_c.bse)[1], 4))
print("95% CI:", np.round(np.exp(np.asarray(marp_c.conf_int())[1, :]), 4))
print("------------------------------------")
print("Adjusted")
print("RR:    ", np.round(np.exp(log_rr_adj), 4))
print("SE:    ", np.round(se_rr_adj, 4))
print("95% CI:", np.round(np.exp(ci_rr_adj), 4))
print("====================================")
print("")

######################################################
# Missing Not at Random (MNAR)

print("====================================")
print("MNAR")
print("====================================")

# generating data from Table 1
d['preterm'] = ([1]*15 + [0]*(100-15) + [np.nan]*115
                + [1]*7 + [0]*(216-7) + [np.nan]*6
                + [1]*21 + [0]*(186-21) + [np.nan]*0
                + [1]*12 + [0]*(81-12) + [np.nan]*96)

# Checking data against Table 1 (should match exactly)
print('Preterm births:',
      np.sum(d['preterm']),
      "("+str(np.round(np.mean(d['preterm']), 2))+"%)\n")
print("Table 1:\n",
      pd.crosstab([d['short_cervix'], d['17P']], d['preterm']),
      "\n")

# Fitting a log-binomial model: Crude
f = sm.families.family.Binomial(sm.families.links.log())
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DomainWarning)
    marp_c = smf.glm("preterm ~ placebo", d, family=f).fit()

# Estimating with g-computation: Adjusted
log_rr_adj = gcomputation(data=d, model="preterm ~ short_cervix")
se_rr_adj = gcomputation_bootstrap(data=d, model="preterm ~ short_cervix",
                                   reps=bootstrap_iters)
ci_rr_adj = np.array([log_rr_adj - 1.96*se_rr_adj,
                      log_rr_adj + 1.96*se_rr_adj])

print("Table 2")
print("------------------------------------")
print("Crude")
print("RR:    ", np.round(np.exp(np.asarray(marp_c.params)[1]), 4))
print("SE:    ", np.round(np.asarray(marp_c.bse)[1], 4))
print("95% CI:", np.round(np.exp(np.asarray(marp_c.conf_int())[1, :]), 4))
print("------------------------------------")
print("Adjusted")
print("RR:    ", np.round(np.exp(log_rr_adj), 4))
print("SE:    ", np.round(se_rr_adj, 4))
print("95% CI:", np.round(np.exp(ci_rr_adj), 4))
print("====================================")

# END
