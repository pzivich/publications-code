#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       File to process data from Wilson et al. (2017) for paper
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd

# Data can be downloaded from
# https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002479#sec020

# Reading in directly from saved PLoS supplement file
d = pd.read_excel("S1Data.xls", sheet_name=1)

# Subsetting the desired columns
cols = ['group',      # Randomized arm
        'anytest',    # Outcome 1 (any test)
        'anydiag',    # Outcome 2 (any diagnosis)
        'gender',     # gender (male, female, transgender)
        'msm',        # MSM, other
        'age',        # age (continuous)
        'partners',   # Number of partners in <12 months
        'ethnicgrp']  # Ethnicity (5 categories)
d = d[cols].copy()

# Re-coding columns as numbers
d['group_n'] = np.where(d['group'] == 'SH:24', 1, np.nan)
d['group_n'] = np.where(d['group'] == 'Control', 0, d['group_n'])

d['gender_n'] = np.where(d['gender'] == 'Female', 0, np.nan)
d['gender_n'] = np.where(d['gender'] == 'Male', 1, d['gender_n'])
d['gender_n'] = np.where(d['gender'] == 'Transgender', 2, d['gender_n'])

d['msm_n'] = np.where(d['msm'] == 'other', 0, np.nan)
d['msm_n'] = np.where(d['msm'] == 'msm', 1, d['msm_n'])
d['msm_n'] = np.where(d['msm'] == '99', 2, d['msm_n'])

d['partners_n'] = np.where(d['partners'] == '1', 0, np.nan)
categories = ['2', '3', '4', '5', '6', '7', '8', '9', '10+']
for index in range(len(categories)):
    d['partners_n'] = np.where(d['partners'] == categories[index],
                               index, d['partners_n'])

d['ethnicgrp_n'] = np.where(d['ethnicgrp'] == 'White/ White British', 0, np.nan)
d['ethnicgrp_n'] = np.where(d['ethnicgrp'] == 'Black/ Black British', 1, d['ethnicgrp_n'])
d['ethnicgrp_n'] = np.where(d['ethnicgrp'] == 'Mixed/ Multiple ethnicity', 2, d['ethnicgrp_n'])
d['ethnicgrp_n'] = np.where(d['ethnicgrp'] == 'Asian/ Asian British', 3, d['ethnicgrp_n'])
d['ethnicgrp_n'] = np.where(d['ethnicgrp'] == 'Other', 4, d['ethnicgrp_n'])

# Dropping old columns and renaming new ones with original label
d = d.drop(columns=['group', 'gender', 'msm', 'partners', 'ethnicgrp'])
relabs = dict()
for c in cols:
    relabs[c + "_n"] = c

d = d.rename(columns=relabs)

# Outputting processed file (used by all other scripts)
d.to_csv("wilson.csv")
