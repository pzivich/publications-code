#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Preparing the ACTG 175 and WIHS data for the illustrative example
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd

filepath = ""

###################################
# Setup -- ACTG 175

# Reading in ACTG-175 data sets
actg = pd.read_csv(filepath+"actg175.dat", sep=' ', header=None,
                   names=["id", "age", "wtkg", "hemo", "homo", "idu", "karnof", "oprior", "z30", "zprior",
                          "preanti", "white", "male", "str2", "strat", " symptom", "offtreat", "treat", "cd4_0wk",
                          "cd4_20wk", "cd496", "r", "cd800", "cd820"])

# Restricting to women
actg = actg.loc[actg['male'] == 0].copy()

# Dropping the one obsevations with CD4 = 0
actg = actg.loc[actg['cd4_0wk'] > 100].copy()

actg = actg[['id', "age", "wtkg", "white", "cd4_0wk", "cd4_20wk"]].copy()

# Reading in ART from event data
actg_events = pd.read_csv(filepath+"actg175_events.dat", sep='\s+', header=None,
                          names=["id", "delta", "t", "art"])

actg_events['treat'] = np.where(actg_events['art'].isin([1, 2]), 1, np.nan)
actg_events['treat'] = np.where(actg_events['art'].isin([0]), 0, actg_events['treat'])
actg_events = actg_events[['id', 'treat']].copy()

# Merging (left-join)
actg = pd.merge(actg, actg_events, how='left', left_on='id', right_on='id')
actg = actg[["treat", "age", "wtkg", "white", "cd4_0wk", "cd4_20wk"]].dropna().copy()

###################################
# Setup -- WIHS

# Baseline / screening questions
wihs_base = pd.read_csv("f00.csv", encoding='ansi', low_memory=False,
                        usecols=['CASEID', 'AGE_SC', 'RACESC', 'HIVPSC', 'DATESC'])
wihs_base = wihs_base.loc[wihs_base['HIVPSC'] == 1].copy()

# Physical exam(s)
wihs_wt = pd.read_csv("f07.csv", encoding='ansi', low_memory=False,
                      usecols=['CASEID', 'VISIT', 'PCWTPE', 'PCWMPE'])
wihs_wt.sort_values(by=['CASEID', 'VISIT'], inplace=True)
wihs_wt = wihs_wt.drop_duplicates(subset='CASEID', keep='first')
wihs_wt['PCWTPE'] = np.where(wihs_wt['VISIT'] > 3, np.nan, wihs_wt['PCWTPE'])
wihs_wt['wtkg'] = np.where(wihs_wt['PCWMPE'] == 2, wihs_wt['PCWTPE'], wihs_wt['PCWTPE']*0.45359237)
wihs_wt['wtkg'] = np.where(wihs_wt['PCWMPE'] == -1, np.nan, wihs_wt['wtkg'])
wihs_wt['wtkg'] = np.where(wihs_wt['PCWMPE'] == -9, np.nan, wihs_wt['wtkg'])
wihs = pd.merge(wihs_base, wihs_wt, how='left', on='CASEID')

# Laboratory values
wihs_lab = pd.read_csv("wihs_labsum.csv")
wihs_lab = wihs_lab[['CASEID', 'VISIT', 'CD4N']].copy()
wihs_lab.sort_values(by=['CASEID', 'VISIT'], inplace=True)
wihs_lab = wihs_lab.drop_duplicates(subset='CASEID', keep='first')
wihs_lab['CD4N'] = np.where(wihs_lab['VISIT'] > 3, np.nan, wihs_lab['CD4N'])
wihs = pd.merge(wihs, wihs_lab, how='left', on='CASEID')

# Recoding variables
wihs['white'] = np.where(wihs['RACESC'] == 2, 1, 0)
wihs['white'] = np.where(wihs['RACESC'].isna(), np.nan, wihs['white'])
wihs['age'] = wihs['AGE_SC']
wihs['cd4_0wk'] = wihs['CD4N']
wihs = wihs[["age", "wtkg", "white", "cd4_0wk"]].copy()
wihs = wihs.dropna()

###################################
# Stacking Data Sources

actg['wihs'] = 0
wihs['wihs'] = 1

actg['restrict'] = 1
wihs['restrict'] = np.where((wihs['cd4_0wk'] >= np.min(actg['cd4_0wk']))
                            & (wihs['cd4_0wk'] <= np.max(actg['cd4_0wk'])), 1, 0)

data = pd.concat([actg, wihs], ignore_index=True)
data['cd4_0wk'] = data['cd4_0wk'] / 100
data['cd4_20wk'] = data['cd4_20wk'] / 100
data.to_csv("hiv.csv", index=False)
