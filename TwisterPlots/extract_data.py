############################################################################################################
# Construct data set based on Pfizer Phase 3 COVID-19 Trial
#
# Paul Zivich 2021/8/23
############################################################################################################

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

np.random.seed(587014)

# Creating empty data set to fill
data = pd.DataFrame()

# Discrete time intervals
times = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112]

#############################
# Re-Create Vaccine Arm
# at-risk numbers at each of the corresponding `times` above
at_risk = [21314, 21230, 21054, 20481, 19314, 18377, 17702, 17186,
           15464, 14083, 12169, 9591, 6403, 3374, 1463, 398, 0]
# event counts at each of the corresponding `times` above
events = [21, 37-21, 39-37, 41-39, 42-41, 42-42, 43-42, 44-43,
          47-44, 48-47, 48-48, 49-48, 49-49, 50-49, 50-50, 50-50]
# censor counts at each of the corresponding `times` above
censor = [at_risk[i] - at_risk[i+1] for i in range(len(at_risk) - 1)]

# Manipulation verifications
assert np.sum(censor) == 21314    # Verify manipulations have correct number of censored
assert np.sum(events) == 50       # Verify correct number of events

# Loop to generate simulated censor times and add to master data set
for i in range(len(censor)):  # loop through each `censor` count
    d = pd.DataFrame()                                                            # Create blank data frame
    d['t'] = np.round(np.random.random_integers(times[i]+1, times[i+1],           # Simulate censor times from uniform
                                                size=censor[i] - events[i]), 0)   # round(..., 0) keeps at 'day' level
    d['delta'] = 0                                                                # Marking as non-events
    d['V'] = 1                                                                    # Marking as the vaccine arm
    data = data.append(d, ignore_index=True)                                      # Stacking in individual data set


# Loop to generate simulated event times and add to master data set
for i in range(len(censor)):
    d = pd.DataFrame()                                                            # Create blank data frame
    d['t'] = np.round(np.random.random_integers(times[i]+1, times[i+1],           # Simulate event times from uniform
                                                size=events[i]), 0)               # round(..., 0) keeps at 'day' level
    d['delta'] = 1                                                                # Marking as events
    d['V'] = 1                                                                    # Marking as the vaccine arm
    data = data.append(d, ignore_index=True)                                      # Stacking in individual data set


#############################
# Re-Create Placebo Arm
# at-risk numbers at each of the corresponding `times` above
at_risk = [21258, 21170, 20970, 20366, 19209, 18218, 17578, 17025,
           15290, 13876, 11994, 9471, 6294, 3301, 1449, 398, 0]
# event counts at each of the corresponding `times` above
events = [25, 55-25, 73-55, 97-73, 123-97, 143-123, 166-143, 192-166,
          212-192, 235-212, 249-235, 257-249, 267-257,
          274-267, 275-274, 275-275]
# censor counts at each of the corresponding `times` above
censor = [at_risk[i] - at_risk[i+1] for i in range(len(at_risk) - 1)]

# Manipulation verifications
assert np.sum(censor) == 21258    # Verify manipulations have correct number of censored
assert np.sum(events) == 275      # Verify correct number of events

# Loop to generate simulated censor times and add to master data set
for i in range(len(censor)):
    d = pd.DataFrame()                                                            # Create blank data frame
    d['t'] = np.round(np.random.random_integers(times[i]+1, times[i+1],           # Simulate event times from uniform
                                                size=censor[i] - events[i]), 0)   # round(..., 0) keeps at 'day' level
    d['delta'] = 0                                                                # Marking as non-events
    d['V'] = 0                                                                    # Marking as the not-vaccine arm
    data = data.append(d, ignore_index=True)                                      # Stacking in individual data set


# Loop to generate simulated event times and add to master data set
for i in range(len(censor)):
    d = pd.DataFrame()                                                            # Create blank data frame
    d['t'] = np.round(np.random.random_integers(times[i]+1, times[i+1],           # Simulate event times from uniform
                                                size=events[i]), 0)               # round(..., 0) keeps at 'day' level
    d['delta'] = 1                                                                # Marking as events
    d['V'] = 0                                                                    # Marking as the not-vaccine arm
    data = data.append(d, ignore_index=True)                                      # Stacking in individual data set


#############################
# Kaplan-Meier Estimates

# Estimating Risk among vaccinated arm
data1 = data.loc[data['V'] == 1].copy()      # Restricting to vaccinated
km1 = KaplanMeierFitter()                    # KaplanMeierFitter object
km1.fit(data1['t'], data1['delta'])          # Estimating survival with Kaplan-Meier
r1 = pd.DataFrame(1 - km1.survival_function_).rename(columns={'KM_estimate': 'R1'})  # Extracting risk estimate

# Calculate Standard Errors via Greenwood's formula
table1 = km1.event_table                  # Extract nice event table
table1['St'] = km1.survival_function_     # Survival estimates
# Calculate standard error estimate
table1['se_sum'] = np.cumsum(table1['observed'] / (table1['at_risk'] * (table1['at_risk'] - table1['observed'])))**0.5
se1 = table1['St'] * table1['se_sum']
r1['R1_SE'] = se1                         # Adding SE risk estimates

# Estimating Risk among unvaccinated arm
data0 = data.loc[data['V'] == 0].copy()     # Restricting to unvaccinated
km0 = KaplanMeierFitter()                   # KaplanMeierFitter object
km0.fit(data0['t'], data0['delta'])         # Estimating survival with Kaplan-Meier
r0 = pd.DataFrame(1 - km0.survival_function_).rename(columns={'KM_estimate': 'R0'})  # Extracting risk estimate

# Calculate Standard Errors via Greenwood's formula
table0 = km0.event_table                  # Extract nice event table
table0['St'] = km0.survival_function_     # Survival estimates
# Calculate standard error estimate
table0['se_sum'] = np.cumsum(table0['observed'] / (table0['at_risk'] * (table0['at_risk'] - table0['observed'])))**0.5
se0 = table0['St'] * table0['se_sum']
r0['R0_SE'] = se0                         # Adding SE risk estimates

# Mergining risk data sets together
r = pd.merge(r1, r0, how='outer', left_index=True, right_index=True).ffill()
r['t'] = r.index                             # Setting time via the index
r['R1_LCL'] = r['R1'] - 1.96*r['R1_SE']      # Lower confidence limit for risk V=1
r['R1_UCL'] = r['R1'] + 1.96*r['R1_SE']      # Upper confidence limit for risk V=1
r['R0_LCL'] = r['R0'] - 1.96*r['R0_SE']      # Lower confidence limit for risk V=0
r['R0_UCL'] = r['R0'] + 1.96*r['R0_SE']      # Upper confidence limit for risk V=0

r['RD'] = r['R1'] - r['R0']                  # Calculating risk difference
r['RD_SE'] = np.sqrt(r['R1_SE']**2 +         # Calculating risk difference SE
                     r['R0_SE']**2)
r['RD_LCL'] = r['RD'] - 1.96*r['RD_SE']      # Lower confidence limit for risk difference
r['RD_UCL'] = r['RD'] + 1.96*r['RD_SE']      # Upper confidence limit for risk difference

r['RR'] = r['R1'] / r['R0']                               # Calculating risk ratio
r['RR_SE'] = np.sqrt((1/r['R1'])**2 *                     # Calculating risk ratio SE via the delta method
                     r['R1_SE']**2 + (1/r['R0'])**2 * r['R0_SE']**2)
r['RR_LCL'] = np.exp(np.log(r['RR']) - 1.96*r['RR_SE'])   # Lower confidence limit for risk ratio
r['RR_UCL'] = np.exp(np.log(r['RR']) + 1.96*r['RR_SE'])   # Upper confidence limit for risk ratio
for c in ['RR', 'RR_LCL', 'RR_UCL']:                      # If 'divide-by-zero' then fill-in with null
    r[c] = r[c].fillna(1)

# Saving estimates from the generate data
r[['t',
   'R1', 'R1_LCL', 'R1_UCL',
   'R0', 'R0_LCL', 'R0_UCL',
   'RD', 'RD_LCL', 'RD_UCL',
   'RR', 'RR_LCL', 'RR_UCL']].round(6  # rounding to 6 decimal places for storage purposes
                                    ).to_csv("data/data_twister.csv", index=False)
# END!
