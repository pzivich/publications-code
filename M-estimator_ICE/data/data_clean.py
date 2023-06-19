####################################################################################################################
# Empirical sandwich variance estimator for iterated conditional expectation g-computation
#   Data Cleaning Add Health
#
# Paul Zivich (2023/06/19)
####################################################################################################################

import numpy as np
import pandas as pd

##############################################
# Wave I of Add Health

# Loading data and setting up output data
d = pd.read_sas("w1inhome.sas7bdat")         # Reading in SAS file
dc1 = pd.DataFrame()                         # Creating new data frame to store results
dc1['id'] = d['AID']                         # ID for participants in clean data frame

# Age (discrete)
dc1['age'] = d['IYEAR'] - d['H1GI1Y']        # Calculating from birth year and interview year

# Gender (0: female, 1: male)
dc1['gender_w1'] = np.where(d['BIO_SEX'] == 2, 0, np.nan)             # Female
dc1['gender_w1'] = np.where(d['BIO_SEX'] == 1, 1, dc1['gender_w1'])   # Male

# Race (0: white, 1: black, 2: native american, 3: asian or PI, 4: other)
dc1['race_w1'] = np.where(d['H1GI6A'] == 1, 1, np.nan)                                   # White
dc1['race_w1'] = np.where(d['H1GI6B'] == 1, 2, dc1['race_w1'])                           # Black or African American
dc1['race_w1'] = np.where(d['H1GI6C'] == 1, 3, dc1['race_w1'])                           # Native American
dc1['race_w1'] = np.where(d['H1GI6D'] == 1, 4, dc1['race_w1'])                           # Asian or Pacific Islander
dc1['race_w1'] = np.where(d['H1GI6E'] == 1, 5, dc1['race_w1'])                           # Other
dc1['race_w1'] = np.where(d['H1GI8'].isin([1, 2, 3, 4, 5]), d['H1GI8'], dc1['race_w1'])  # Best-described race
dc1['race_w1'] = dc1['race_w1'] - 1

# Ethnicity (0: non-hispanic, 1: hispanic)
dc1['ethnic_w1'] = np.where(d['H1GI4'] == 0, 0, np.nan)              # non-Hispanic
dc1['ethnic_w1'] = np.where(d['H1GI4'] == 1, 1, dc1['ethnic_w1'])    # Hispanic

# Education (current grade level)
dc1['educ_w1'] = np.where(d['H1GI20'].isin([7, 8, 9, 10, 11, 12]), d['H1GI20'], np.nan)

# Height self-report (inches)
dc1['height_w1'] = np.where((d['H1GH59A'].isin([96, 98])) | (d['H1GH59B'].isin([96, 98, 99])),
                            np.nan,
                            d['H1GH59A']*12 + d['H1GH59B'])

# Weight self-report (lbs)
dc1['weight_w1'] = np.where(d['H1GH60'].isin([996, 998, 999]), np.nan, d['H1GH60'])

# Exercise (0: none, 1: 1-2 times, 2: 3-4 time, 3: 5+ times)
dc1['exercise_w1'] = d['H1DA6']
dc1['exercise_w1'] = np.where(d['H1DA6'].isin([6, 8, ]), np.nan, dc1['exercise_w1'])

# Self-Rated Health (0: excellent, 1: very good, 2: good, 3: fair, 4: poor)
dc1['srh_w1'] = np.where(d['H1GH1'].isin([6, 8]), np.nan, d['H1GH1'])
dc1['srh_w1'] = dc1['srh_w1'] - 1

# Felt depressed during past week (0: never/rarely/sometimes, 1: a lot/most or all)
dc1['depr_w1'] = np.where(d['H1FS6'].isin([0, 1]), 0, np.nan)       # Never or rarely, or sometimes
dc1['depr_w1'] = np.where(d['H1FS6'].isin([2, 3]), 1, d['H1FS6'])   # a lot of the time, or most / all of the time

# Ever tried smoking a cigarette (0: no, 1: yes)
dc1['tried_cigarette'] = np.where(d['H1TO1'].isin([6, 8, 9]), np.nan, d['H1TO1'])

# Smoked cigarettes in previous 30 days (0: no, 1: yes)
dc1['cigarette_w1'] = np.where(d['H1TO5'].isin([0, 97]), 0, np.nan)
dc1['cigarette_w1'] = np.where(d['H1TO5'].isin([i for i in range(1, 31)]), 1, dc1['cigarette_w1'])

# Days drinking alcohol in past 12 months
dc1['alcohol_w1'] = np.where(d['H1TO15'].isin([97, 7]), 0, np.nan)               # Never
dc1['alcohol_w1'] = np.where(d['H1TO15'].isin([6, ]), 1, dc1['alcohol_w1'])      # 1-2 days total
dc1['alcohol_w1'] = np.where(d['H1TO15'].isin([4, 5, ]), 2, dc1['alcohol_w1'])   # 1-3 times a month
dc1['alcohol_w1'] = np.where(d['H1TO15'].isin([3, ]), 3, dc1['alcohol_w1'])      # 1-2 times a week
dc1['alcohol_w1'] = np.where(d['H1TO15'].isin([1, 2, ]), 4, dc1['alcohol_w1'])   # 3+ times a week

# Exclusion criteria
dc1['exclude'] = np.where(d['H1PL2'] == 1, 1, 0)
dc1['exclude'] = np.where(dc1['age'] == 12, 1, dc1['exclude'])
dc1['exclude'] = np.where(dc1['age'] > 19, 1, dc1['exclude'])
dc1['exclude'] = np.where(dc1['race_w1'] == 4, 1, dc1['exclude'])
dc1['exclude'] = np.where(dc1['educ_w1'].isna(), 1, dc1['exclude'])
# NOTE: only considering those currently in school AND school has grade levels
#   this excludes those graduated (33) or suspended/drop-outs/sick/pregnant

##############################################
# Wave III of Add Health

# Loading data and setting up output data
d = pd.read_sas("w3inhome.sas7bdat")         # Reading in SAS file
dc2 = pd.DataFrame()                         # Creating new data frame to store results
dc2['id'] = d['AID']                         # ID for participants in clean data frame

# Gender (0: female, 1: male)
dc2['gender_w3'] = np.where(d['BIO_SEX3'] == 2, 0, np.nan)             # Female
dc2['gender_w3'] = np.where(d['BIO_SEX3'] == 1, 1, dc2['gender_w3'])   # Male

# Race (0: white, 1: black, 2: native american, 3: asian or PI)
dc2['race_w3'] = np.where(d['H3OD4A'] == 1, 1, np.nan)                                   # White
dc2['race_w3'] = np.where(d['H3OD4B'] == 1, 2, dc2['race_w3'])                           # Black or African American
dc2['race_w3'] = np.where(d['H3OD4C'] == 1, 3, dc2['race_w3'])                           # Native American
dc2['race_w3'] = np.where(d['H3OD4D'] == 1, 4, dc2['race_w3'])                           # Asian or Pacific Islander
dc2['race_w3'] = np.where(d['H3OD6'].isin([1, 2, 3, 4]), d['H3OD6'], dc2['race_w3'])     # Best-described race
dc2['race_w3'] = dc2['race_w3'] - 1

# Ethnicity (0: non-hispanic, 1: hispanic)
dc2['ethnic_w3'] = np.where(d['H3OD2'] == 0, 0, np.nan)            # non-Hispanic
dc2['ethnic_w3'] = np.where(d['H3OD2'] == 1, 1, dc2['ethnic_w3'])  # Hispanic

# Education (highest completed grade level)
dc2['educ_w3'] = np.where(d['H3ED1'] < 12, 0, np.nan)
dc2['educ_w3'] = np.where(d['H3ED1'] == 12, 1, dc2['educ_w3'])
dc2['educ_w3'] = np.where((d['H3ED1'] > 12) & (d['H3ED1'] <= 17), 2, dc2['educ_w3'])
dc2['educ_w3'] = np.where(d['H3ED1'].isin([18, 19, 20, 21, 22]), 3, dc2['educ_w3'])

# Height measured (inches)
dc2['height_w3'] = np.where((d['H3HGT_F'].isin([96, 98])) | (d['H3HGT_I'].isin([96, 98])),
                            np.nan,
                            d['H3HGT_F']*12 + d['H3HGT_I'])

# Weight measured (lbs)
dc2['weight_w3'] = np.where(d['H3WGT'].isin([888, 996]), np.nan, d['H3WGT'])

# Exercise (0: none, 1: 1-2 times, 2: 3-4 time, 3: 5+ times)
d['exercise1'] = np.where(d['H3DA8'].isin([96, 98]), np.nan, d['H3DA8'])
d['exercise2'] = np.where(d['H3DA9'].isin([96, 98]), np.nan, d['H3DA9'])
d['exercise3'] = np.where(d['H3DA10'].isin([96, 98]), np.nan, d['H3DA10'])
d['exercise4'] = np.where(d['H3DA11'].isin([96, 98]), np.nan, d['H3DA11'])
d['exercise5'] = np.where(d['H3DA12'].isin([96, 98]), np.nan, d['H3DA12'])
d['exercise6'] = np.where(d['H3DA13'].isin([96, 98]), np.nan, d['H3DA13'])
d['exercise7'] = np.where(d['H3DA14'].isin([96, 98]), np.nan, d['H3DA14'])
d['exercise'] = (d['exercise1'] + d['exercise2'] + d['exercise3'] + d['exercise4']
                 + d['exercise5'] + d['exercise6'] + d['exercise7'])
dc2['exercise_w3'] = np.where(d['exercise'] == 0, 0, np.nan)
dc2['exercise_w3'] = np.where((d['exercise'] == 1) | (d['exercise'] == 2), 1, dc2['exercise_w3'])
dc2['exercise_w3'] = np.where((d['exercise'] == 3) | (d['exercise'] == 4), 2, dc2['exercise_w3'])
dc2['exercise_w3'] = np.where(d['exercise'] >= 5, 3, dc2['exercise_w3'])

# Self-Rated Health (0: excellent, 1: very good, 2: good, 3: fair, 4: poor)
dc2['srh_w3'] = d['H3GH1'] - 1  # No missing to re-code

# Felt depressed during past week (0: never/rarely/sometimes, 1: a lot/most or all)
dc2['depr_w3'] = np.where(d['H3SP9'].isin([0, 1]), 0, np.nan)
dc2['depr_w3'] = np.where(d['H3SP9'].isin([2, 3]), 1, dc2['depr_w3'])

# Health insurance status (0: covered, 1: no or unknown health insurance)
dc2['hins_w3'] = np.where(d['H3HS5'].isin([0, 10]), 1, np.nan)
dc2['hins_w3'] = np.where(d['H3HS5'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, dc2['hins_w3'])

# Ever diagnosed with HBP or Hypertension (0: no, 1: yes)
dc2['hbp_w3'] = np.where(d['H3ID22'].isin([0, 1]), d['H3ID22'], np.nan)

# Ever taken HBP/hypertension/heart problem medication (0: no, 1: yes)
dc2['hmed_w3'] = np.where((d['H3ID26G'] == 1) | (d['H3ID26H'] == 1), 1, 0)

# Diabetes
dc2['diabetes'] = np.where(d['H3ID17'] == 98, np.nan, d['H3ID17'])

# Smoked cigarettes in previous 30 days (0: no, 1: yes)
dc2['cigarette_w3'] = np.where(d['H3TO6'].isin([0, 7]), 0, np.nan)
dc2['cigarette_w3'] = np.where(d['H3TO6'].isin([1, ]), 1, dc2['cigarette_w3'])

# Days drinking alcohol in past 12 months
dc2['alcohol_w3'] = np.where(d['H3TO38'].isin([97, 0]), 0, np.nan)               # Never
dc2['alcohol_w3'] = np.where(d['H3TO38'].isin([1, ]), 1, dc2['alcohol_w3'])      # 1-2 days total
dc2['alcohol_w3'] = np.where(d['H3TO38'].isin([2, 3, ]), 2, dc2['alcohol_w3'])   # 1-3 times a month
dc2['alcohol_w3'] = np.where(d['H3TO38'].isin([4, ]), 3, dc2['alcohol_w3'])      # 1-2 times a week
dc2['alcohol_w3'] = np.where(d['H3TO38'].isin([5, 6, ]), 4, dc2['alcohol_w3'])   # 3+ times a week

##############################################
# Wave IV of Add Health

# Loading data and setting up output data
d = pd.read_sas("w4inhome.sas7bdat")         # Reading in SAS file
dc3 = pd.DataFrame()                         # Creating new data frame to store results
dc3['id'] = d['AID']                         # ID for participants in clean data frame

# Hypertension (0: normal / prehypertension, 1: hypertension I & II)
dc3['htn_w4'] = np.where(d['H4BPCLS'].isin([1, 2]), 0, np.nan)
dc3['htn_w4'] = np.where(d['H4BPCLS'].isin([3, 4]), 1, dc3['htn_w4'])

##############################################
# Combining Waves

# Merging processed data together
dc = pd.merge(dc1, dc2, how='left', on='id')
dc = pd.merge(dc, dc3, how='left', on='id')

# Applying exclusion criteria from Wave I
unexcluded_n = dc.shape[0]
dc = dc.loc[dc['exclude'] == 0].copy()

# Determining diabetes status (0: no, 1: yes)
# dc['diabetes_w1'] = np.where(dc['age'] >= dc['diabetes'], 1, np.nan)             # Missing data and measurement error
# dc['diabetes_w1'] = np.where(dc['age'] < dc['diabetes'], 0, dc['diabetes_w1'])   # So, not using this approach
dc['diabetes_w3'] = np.where(dc['diabetes'] < 97, 1, np.nan)
dc['diabetes_w3'] = np.where(dc['diabetes'] == 97, 0, dc['diabetes_w3'])

##############################################
# Missing Data Processing

# Restricting to W1 no missing data
clean_n = dc.shape[0]
dc = dc.dropna(subset=['race_w1', 'ethnic_w1', 'height_w1', 'weight_w1', 'exercise_w1', 'srh_w1',
                       'tried_cigarette', 'cigarette_w1', 'alcohol_w1'])
no_miss_w1_n = dc.shape[0]

# Making W3 be monotone missing (if any missing, all others missing)
w3_cols_miss = ["educ_w3", "height_w3", "weight_w3", "exercise_w3", "srh_w3", "depr_w3", "hins_w3",
                "hbp_w3", "hmed_w3", "alcohol_w3", "diabetes_w3",
                "cigarette_w3",
                "gender_w3", "race_w3", "ethnic_w3"]
for c in w3_cols_miss:
    for ci in w3_cols_miss:
        dc[ci] = np.where(dc[c].isna(), np.nan, dc[ci])
    dc['htn_w4'] = np.where(dc[c].isna(), np.nan, dc['htn_w4'])


##############################################
# Output and Post-Processing Summary

dc['id'] = dc['id'].str.decode("utf-8")
dc[['id', 'age', 'tried_cigarette',
    'gender_w1', 'race_w1', 'ethnic_w1', 'height_w1', 'weight_w1', 'educ_w1', 'exercise_w1', 'srh_w1', 'depr_w1',
    'alcohol_w1',
    'cigarette_w1',
    'gender_w3', 'race_w3', 'ethnic_w3', 'height_w3', 'weight_w3', 'educ_w3', 'exercise_w3', 'srh_w3', 'depr_w3',
    'alcohol_w3', 'hins_w3', 'hbp_w3', 'hmed_w3', 'diabetes_w3',
    'cigarette_w3',
    'htn_w4',
    ]
   ].to_csv("addhealth.csv", index=False)

print("Cleaning Metrics")
print("--------------------------")
print("Unexcluded N:", unexcluded_n)
print("Excluded N:", clean_n, np.round(1 - clean_n / unexcluded_n, 2))
print("No W1 Missing:", no_miss_w1_n, np.round(1 - no_miss_w1_n / clean_n, 2))

# Cleaning Metrics
# --------------------------
# Unexcluded N:  6504
# Excluded N:    5915 0.09
# No W1 Missing: 5657 0.04
