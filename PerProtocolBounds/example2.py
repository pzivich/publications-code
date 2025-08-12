####################################################################################################################
# Example 2: ACTG 320 data
#
# Paul Zivich (2025/06/16)
####################################################################################################################

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.simplefilter('ignore', RuntimeWarning)


#########################################################################
# Loading in the data

columns = ["id", "male", "black", "hispanic", "idu", "art", "d", "drop", "r", "age", "karnof", "days", "cd4",
           "stop", "t", "delta"]
d = pd.read_csv("data/actg320.dat", sep='\s+', names=columns)
d = d[['id', 'art', 'stop', 't', 'delta']].copy()


#########################################################################
# Defining some helper functions

def compute_contrasts(results_arm1, results_arm0):
    """A function that takes the statsmodels summary output from the treatment-stratified Kaplan-Meier's and computes
    the contrasts (and their confidence intervals)"""
    # Mergining data results together
    combined = pd.merge(results_arm1, results_arm0, how='outer', left_index=True, right_index=True)
    cols = list(combined.columns)
    last_time = pd.DataFrame([[np.nan, ]*len(cols)], index=pd.Index([365, ], name="Time"), columns=cols)
    combined = pd.concat([combined, last_time])
    combined = combined.ffill()

    # Calculating risks
    combined['Risk_1'] = 1 - combined['Surv prob_x'].fillna(value=1.0)
    combined['Risk_0'] = 1 - combined['Surv prob_y'].fillna(value=1.0)

    # Calculating contrasts
    combined['RiskRatio'] = combined['Risk_1'] / combined['Risk_0']
    combined['RiskDiff'] = combined['Risk_1'] - combined['Risk_0']

    # Calculating confidence intervals
    combined['Surv prob SE_x'] = combined['Surv prob SE_x'].fillna(value=0.0)
    combined['Surv prob SE_y'] = combined['Surv prob SE_y'].fillna(value=0.0)
    combined['Var_x'] = combined['Surv prob SE_x']**2
    combined['Var_y'] = combined['Surv prob SE_y']**2

    rd_var = combined['Var_x'] + combined['Var_y']
    combined['RiskDiff_LCL'] = combined['RiskDiff'] - 1.96*np.sqrt(rd_var)
    combined['RiskDiff_UCL'] = combined['RiskDiff'] + 1.96*np.sqrt(rd_var)

    logrr_var = combined['Var_x'] / (combined['Risk_1']**2) + combined['Var_y'] / (combined['Risk_0']**2)
    combined['RiskRatio_LCL'] = np.exp(np.log(combined['RiskRatio']) - 1.96*np.sqrt(logrr_var))
    combined['RiskRatio_UCL'] = np.exp(np.log(combined['RiskRatio']) + 1.96*np.sqrt(logrr_var))

    # Returning complete data
    return combined


def twister_plot(ax, data, time, upper_bound, lower_bound, upper_ci, lower_ci, reference_line, log_scale=False,
                 treat_labs=("3-Drug", "2-Drug"), treat_labs_spacing="\t"):
    # Plotting the bounds
    ax.fill_betweenx(data[time],         # time column (no shift needed here)
                     data[upper_bound],  # upper confidence limit
                     data[lower_bound],  # lower confidence limit
                     label="Bounds",     # Sets the label in the legend
                     color='k',          # Sets the color of the shaded region (k=black)
                     alpha=1,            # Sets the transparency of the shaded region
                     step='post')

    # Plotting their confidence intervals
    ax.fill_betweenx(data[time],  # time column (no shift needed here)
                     data[upper_ci],  # upper confidence limit
                     data[lower_ci],  # lower confidence limit
                     label="95% CI",  # Sets the label in the legend
                     color='k',  # Sets the color of the shaded region (k=black)
                     alpha=0.2,  # Sets the transparency of the shaded region
                     step='post')

    # Drawing a reference line
    ax.axvline(reference_line,
               color='gray',  # Sets color to gray for the reference line
               linestyle='--',  # Sets the reference line as dashed
               label=None)  # drawing dashed reference line at RD=0

    ax.set_ylim([0, max_t])
    if log_scale:
        ax.set_xscale("log", base=np.e)

    if treat_labs is not None:
        ax2 = ax.twiny()  # Duplicate the x-axis to create a separate label
        # "test \t test".expandtabs()
        ax2.set_xlabel("Favors " + treat_labs[0]
                       + treat_labs_spacing.expandtabs()
                       + "Favors " + treat_labs[1],
                       fontdict={"size": 10})
        ax2.set_xticks([])
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 36))

    return ax


#########################################################################
# Intent-to-Treat Analysis

# Risk in the 3-drug ART arm
d1 = d.loc[d['art'] == 1].copy()
km1 = sm.SurvfuncRight(time=d1['t'], status=d1['delta'])
sf1 = km1.summary()

# Risk in the 2-drug ART arm
d0 = d.loc[d['art'] == 0].copy()
km0 = sm.SurvfuncRight(time=d0['t'], status=d0['delta'])
sf0 = km0.summary()

r_itt = compute_contrasts(sf1, sf0)
print("Intent-to-Treat Contrasts")
print(r_itt.loc[r_itt.index == 365, ['RiskDiff', 'RiskDiff_LCL', 'RiskDiff_UCL']])
print(r_itt.loc[r_itt.index == 365, ['RiskRatio', 'RiskRatio_LCL', 'RiskRatio_UCL']])


#########################################################################
# Number of Deviations

d['deviate'] = np.where(d['stop'] == '.', 0, 1)
# print(pd.crosstab(d['deviate'], d['art']))

#########################################################################
# Per-Protocol Bounds

d['t2'] = np.where(d['deviate'] == 0, d['t'], d['stop'])
d['t2'] = pd.to_numeric(d['t2'])
d['delta2'] = np.where(d['deviate'] == 0, d['delta'], 0)

# Bounds
max_t = np.max(d['t'])

#########################################################################
# Per-Protocol Upper Bound

d_upper = d.copy()
d_upper['delta2'] = np.where((d_upper['deviate'] == 1) & (d_upper['art'] == 1), 1, d_upper['delta2'])
d_upper['t2'] = np.where((d_upper['deviate'] == 1) & (d_upper['art'] == 0), max_t, d_upper['t2'])

d1 = d_upper.loc[d_upper['art'] == 1].copy()
km1 = sm.SurvfuncRight(time=d1['t2'], status=d1['delta2'])
sf1 = km1.summary()

d0 = d_upper.loc[d_upper['art'] == 0].copy()
km0 = sm.SurvfuncRight(time=d0['t2'], status=d0['delta2'])
sf0 = km0.summary()

r_upper = compute_contrasts(sf1, sf0)

#########################################################################
# Per-Protocol Lower Bound

d_lower = d.copy()
d_lower['delta2'] = np.where((d_lower['deviate'] == 1) & (d_lower['art'] == 0), 1, d_lower['delta2'])
d_lower['t2'] = np.where((d_lower['deviate'] == 1) & (d_lower['art'] == 1), max_t, d_lower['t2'])

d1 = d_lower.loc[d_lower['art'] == 1].copy()
km1 = sm.SurvfuncRight(time=d1['t2'], status=d1['delta2'])
sf1 = km1.summary()

d0 = d_lower.loc[d_lower['art'] == 0].copy()
km0 = sm.SurvfuncRight(time=d0['t2'], status=d0['delta2'])
sf0 = km0.summary()

r_lower = compute_contrasts(sf1, sf0)

#########################################################################
# Bounds at 365 days

print("\nPer-Protocol RD Bounds")
print(r_lower.loc[r_lower.index == 365, ['RiskDiff', 'RiskDiff_LCL']])
print(r_upper.loc[r_upper.index == 365, ['RiskDiff', 'RiskDiff_UCL']])

print("\nPer-Protocol RR Bounds")
print(r_lower.loc[r_lower.index == 365, ['RiskRatio', 'RiskRatio_LCL']])
print(r_upper.loc[r_upper.index == 365, ['RiskRatio', 'RiskRatio_UCL']])


#########################################################################
# Plotting Results

bounds = pd.merge(r_lower, r_upper, how='outer', left_index=True, right_index=True)
bounds = bounds.ffill()
bounds['time'] = bounds.index

fig, axs = plt.subplots(1, 2, figsize=(7, 5))  # fig_size is width by height
twister_plot(axs[0], data=bounds, time='time',
             upper_bound='RiskRatio_x', lower_bound='RiskRatio_y',
             upper_ci='RiskRatio_UCL_y', lower_ci='RiskRatio_LCL_x',
             reference_line=1, log_scale=True)
axs[0].set_xlim([1/100, 100])
# axs[0].xaxis.set_minor_formatter(mticker.ScalarFormatter())
axs[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
axs[0].set_xticks([0.01, 0.1, 1, 10, 100])
axs[0].set_xticklabels(["0.01", "0.1", "1", "10", "100"])
axs[0].set_xlabel("Risk Ratio")
axs[0].set_yticks([0, 50, 100, 150, 200, 250, 300, 365])
axs[0].set_ylabel("Time from Randomization (days)")

twister_plot(axs[1], data=bounds, time='time',
             upper_bound='RiskDiff_x', lower_bound='RiskDiff_y',
             upper_ci='RiskDiff_UCL_y', lower_ci='RiskDiff_LCL_x',
             reference_line=0, log_scale=False)
axs[1].set_xlim([-1, 1])
axs[1].set_xlabel("Risk Difference")
axs[1].set_yticks([0, 50, 100, 150, 200, 250, 300, 365])

plt.tight_layout()
plt.savefig("Figure1.png", format='png', dpi=300)
plt.show()

# plt.fill_between(bounds.index, bounds['RiskRatio_x'], bounds['RiskRatio_y'], color='k')
# plt.fill_between(bounds.index, bounds['RiskRatio_LCL_x'], bounds['RiskRatio_UCL_y'], color='k', alpha=0.2)
# plt.axhline(1., linestyle=':', color='gray')
# plt.ylim([0, 5])
# plt.xlim([0, 365])
# plt.tight_layout()
# plt.show()
#
# plt.fill_between(bounds.index, bounds['RiskDiff_x'], bounds['RiskDiff_y'], color='k')
# plt.fill_between(bounds.index, bounds['RiskDiff_LCL_x'], bounds['RiskDiff_UCL_y'], color='k', alpha=0.2)
# plt.axhline(0., linestyle=':', color='gray')
# plt.ylim([-1, 1])
# plt.xlim([0, 365])
# plt.tight_layout()
# plt.show()

# Output when running this script
#
# Intent-to-Treat Contrasts
#       RiskDiff  RiskDiff_LCL  RiskDiff_UCL
# Time
# 365  -0.087105      -0.12751       -0.0467
#       RiskRatio  RiskRatio_LCL  RiskRatio_UCL
# Time
# 365    0.454271       0.310948       0.663654
#
# Per-Protocol RD Bounds
#       RiskDiff  RiskDiff_LCL
# Time
# 365  -0.400889     -0.454242
#       RiskDiff  RiskDiff_UCL
# Time
# 365   0.064268      0.109524
#
# Per-Protocol RR Bounds
#       RiskRatio  RiskRatio_LCL
# Time
# 365    0.120105       0.081975
#       RiskRatio  RiskRatio_UCL
# Time
# 365    1.545105       2.105229
