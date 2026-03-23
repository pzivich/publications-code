######################################
# NHANES 2017-2018 cleaning data

import numpy as np
import pandas as pd


################################
# Processing demographic data
d1 = pd.read_csv("demo.csv")

d1['id'] = d1['SEQN']                                     # ID
d1['female'] = d1['RIAGENDR'] - 1                         # Indicator for gender
d1['age'] = d1['RIDAGEYR']                                # Age in years
d1['sample_weight'] = d1['WTINT2YR']                      # Sample weight
d1 = d1[['id', 'age', 'female', 'sample_weight']].copy()  # Subsetting to processed data

################################
# Processing blood pressure
d2 = pd.read_csv("bp_full.csv")
d2['id'] = d2['SEQN']
sys_cols = ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']
d2['sbp'] = d2['BPXSY1']
dia_cols = ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']
d2['dbp'] = d2['BPXDI1']
d2 = d2[['id', 'sbp', 'dbp']].copy()        # Subsetting to processed data

################################
# Processing blood pressure
d3 = pd.read_csv("body.csv")
d3['id'] = d3['SEQN']             # ID
d3['weight'] = d3['BMXWT']        # Weight (kg)
d3['height'] = d3['BMXHT']        # Standing height (cm)
d3 = d3[['id', 'weight']].copy()  # Subsetting to processed data

################################
# Saving processed data
d = pd.merge(d1, d2, how='outer', on='id')
d = pd.merge(d, d3, how='outer', on='id')

d = d.loc[(d['age'] >= 18) & (d['sbp'] >= 140)].copy()
d.to_csv("nhanes.csv", index=False)
d.info()
