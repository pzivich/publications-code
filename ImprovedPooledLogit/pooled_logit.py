import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from time import time

max_follow_up_years = 5

##########################################
# Setup data

d = pd.read_csv("lau.csv")

# Recoding events and admin censoring
d['event'] = np.where(d['eventtype'] == 2, 1, 0)                              # Competing event is event of interest
d['event'] = np.where(d['t'] > max_follow_up_years, 0, d['event'])            # Administrative censoring at ...
d['t'] = np.where(d['t'] > max_follow_up_years, max_follow_up_years, d['t'])  # ... 5 years of follow-up

# Transforming time into days
d['days'] = np.ceil(d['t'] * 365.25)
d['days'] = d['days'].astype(int)
d.info()

##########################################
# Standard Implementation

#############
# Long data set
max_t = np.max(d['days'])
dl = pd.DataFrame(np.repeat(d.values, max_t, axis=0), columns=d.columns)
dl['t_in'] = dl.groupby("id")['days'].cumcount()
dl['t_out'] = dl['t_in'] + 1

dl['delta'] = np.where(dl['t_out'] == dl['days'], dl['event'], 0)
dl['delta'] = np.where(dl['t_out'] > dl['days'], np.nan, dl['delta'])
dl = dl[['id', 't_out', 'delta', 'ageatfda', 'BASEIDU', 'black', 'cd4nadir']]
dl = dl.dropna(subset=['delta', ])
dl.info()

#############
# Specify model
model = "delta ~ BASEIDU + ageatfda + black + cd4nadir + C(t_out)"

#############
# Fit model
run_times = []
for i in range(100):
    start = time()
    fam = sm.families.Binomial()
    fm = smf.glm(model, data=dl, family=fam).fit()
    run_times.append(time() - start)

print("RUNTIMES:", np.min(run_times), np.mean(run_times), np.max(run_times))


##########################################
# Proposed Implementation

#############
# Long data set
max_t = np.max(d['days'])
dl = pd.DataFrame(np.repeat(d.values, max_t, axis=0), columns=d.columns)
dl['t_in'] = dl.groupby("id")['days'].cumcount()
dl['t_out'] = dl['t_in'] + 1
dl['delta'] = np.where(dl['t_out'] == dl['days'], dl['event'], 0)
dl['delta'] = np.where(dl['t_out'] > dl['days'], np.nan, dl['delta'])
dl = dl[['id', 't_out', 'delta', 'ageatfda', 'BASEIDU', 'black', 'cd4nadir']].copy()
dl = dl.dropna(subset=['delta', ])

#############
# Restrict

unique_event_times = np.unique(d.loc[d['event'] == 1, 'days'])
dl = dl.loc[dl['t_out'].isin(unique_event_times)].copy()
dl.info()

#############
# Specify model

model = "delta ~ BASEIDU + ageatfda + black + cd4nadir + C(t_out)"

#############
# Fit model
run_times = []
for i in range(100):
    start = time()
    fam = sm.families.Binomial()
    fm = smf.glm(model, data=dl, family=fam).fit()
    run_times.append(time() - start)

print("RUNTIMES:", np.min(run_times), np.mean(run_times), np.max(run_times))
