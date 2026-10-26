########################################################################################################################
# Combining Randomized and Observational Data: The Case of Postnatal Prophylaxis Against HIV
#       - Data generating mechanism for the simulation experiments
#
# Paul Zivich (2026/10/26)
########################################################################################################################

# Setup
import numpy as np
import pandas as pd


def dgm(n1, n0, ratio, scenario=1):
    # Creating trial data set
    d1 = pd.DataFrame()
    d1['W'] = np.random.binomial(n=1, p=2 / 3, size=n1)
    d1['A'] = np.random.binomial(n=1, p=ratio, size=n1)
    d1['S'] = 1

    # Creating external control data set
    d0 = pd.DataFrame()
    d0['W'] = np.random.binomial(n=1, p=1 / 5, size=n0)
    d0['A'] = 0
    d0['S'] = 0

    # Stacking and preparing the specific coefficient values for the given scenario
    d = pd.concat([d1, d0], ignore_index=True)
    if scenario == 1:
        intercept = 0.01
        shift1 = 0.015
        shift2 = 0
    elif scenario == 2:
        intercept = 0.05
        shift1 = 0.075
        shift2 = 0
    elif scenario == 3:
        intercept = 0.01
        shift1 = 0
        shift2 = 0.015
    elif scenario == 4:
        intercept = 0.05
        shift1 = 0
        shift2 = 0.075
    else:
        raise ValueError("No scenario specifications available beyond 1,2,3,4.")

    # Creating the outcome variable according to the scenario mechanism
    d['Y'] = np.random.binomial(n=1, p=intercept + shift1*d['W']*(1-d['A']) + shift2*d['W'], size=d.shape[0])
    d['C'] = 1
    return d

