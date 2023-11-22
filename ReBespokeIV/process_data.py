#####################################################################################################################
# Bespoke Instrumental Variable via two-stage regression as an M-estimator
#       Process publicly available data from Teaching of Statistics in the Health Sciences Resources Portal
#       (https://www.causeweb.org/tshs/category/dataset/) to replicate the analysis.
#
# Paul Zivich (2023/11/22)
####################################################################################################################

import numpy as np
import pandas as pd

# Reading in original data
d = pd.read_csv("OPT_Study_Person-level_Data.csv")

# Restricting to only columns needed
d = d[['Group', 'Tx comp?', 'OFIBRIN1', 'ETXU_CAT1', 'V5 %BOP']].copy()

# Recoding some variables from the excel formatting
d['assign'] = np.where(d['Group'] == 'T', 1, 0)
d['complete'] = np.where(d['Tx comp?'] == 'Yes', 1, 0)
d['OFIBRIN1'] = pd.to_numeric(d['OFIBRIN1'], errors='coerce')
d['ETXU_CAT1'] = pd.to_numeric(d['ETXU_CAT1'], errors='coerce')
d['V5 %BOP'] = d['V5 %BOP'] / 100

# Dropping observations with missing data
d = d.dropna(subset=['V5 %BOP', 'OFIBRIN1', 'ETXU_CAT1']).copy()

# Renaming variables for consistency with the paper
d['R'] = d['assign']
d['A'] = d['complete']
d['Y'] = d['V5 %BOP']
d['L1'] = d['OFIBRIN1']
d['L2'] = d['ETXU_CAT1']

# Outputting as a CSV for use across programs
d[['R', 'A', 'Y', 'L1', 'L2']].to_csv("processed.csv")
d.info()
