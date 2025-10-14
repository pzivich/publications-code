#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file provides a class object used to simulate SBP values for those in the non-positive region. There
#   are alterantive ways to implement this procedure. This is simply one implementation (note: it could be made
#   much more efficient).
#
# Paul Zivich (2024/12/17)
#####################################################################################################################

import numpy as np
import pandas as pd
from scipy.stats import norm


class MathModel:
    """Mathematical model used to simulate systolic blood pressure values for those <8 years old. Note the table used
    to create this model was stratified by gender and height (in addition to age). Therefore, this mathematical model
    also requires an individual's gender and height in order to simulate systolic blood pressure values.
    # Information extracted from Tables 4 & 5
    """
    def __init__(self):
        filepath = "../data/"
        # Height cut-points for the distributions used, stratified by gender, age
        columns = ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']
        dh = pd.read_csv(filepath+"height_params.csv")
        self.height_cuts = dh
        # self.height_cuts = {}
        # for i, row in dh.iterrows():
        #     cuts = {}
        #     for ci in range(1, 7):
        #         cuts[ci] = (row[columns[ci]] + row[columns[ci-1]]) / 2
        #     self.height_cuts[row['code']] = cuts

        # Median, 90th percentiles for each gender, age, height combination
        dbp = pd.read_csv(filepath+"sbp_params.csv")
        self.distributions = {}
        for i, row in dbp.iterrows():
            self.distributions[row['code']] = [row['median'], row['p90']]

    def simulate_blood_pressure(self, female, age, height):
        """Function to simulate SBP for given combinations of gender, age, height.
        """
        # Converting input objects to lists
        female = list(female)
        age = list(age)
        height = list(height)

        # Simulating blood pressure
        blood_pressure = []                                  # Storage for simulated SBP values
        for f, a, h in zip(female, age, height):             # For each gender, age, height combination of an individual
            code = self.encode(female=f, age=a, height=h)    # ... generate encoding used in the dictionaries in init
            sbp = self.random_draw(code=code)                # ... using encoding, take a random draw
            blood_pressure.append(sbp)                       # ... store simulated SBP

        # Return predictions as an array that can be input into a pandas DataFrame
        return np.asarray(blood_pressure).T[0]

    def encode(self, female, age, height):
        """Internal function used to generate the internal encoding by gender, age, and height for simulation
        parameters that the dictionaries use. This is an internal-only function (user does not need to call).
        """
        # Gender code
        if female:
            code = 'f'
        else:
            code = 'm'

        # Age code
        code = code + str(age)

        # Height code
        height_vals = self.height_cuts.loc[self.height_cuts['code'] == code]
        height_vals = np.asarray(height_vals[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']])[0]
        if height < height_vals[0]:           # Adding height indicator based on which interval height is within
            code = code + 'h1'
        elif height < height_vals[1]:
            code = code + 'h2'
        elif height < height_vals[2]:
            code = code + 'h3'
        elif height < height_vals[3]:
            code = code + 'h4'
        elif height < height_vals[4]:
            code = code + 'h5'
        elif height < height_vals[5]:
            code = code + 'h6'
        else:
            code = code + 'h7'

        # Return the completed code
        return code

    def random_draw(self, code):
        """Function to randomly draw a SBP given the encoded gender, age, and height for an individual observation.
        Note this function currently uses the *upper* 90th percentile to compute the standard deviation for the
        distribution.
        """
        params = self.distributions[code]                    # Pulling list of median, upper percentile for given code
        median = params[0]                                   # Extract the median as the center estimate
        sd = (params[1] - median) / norm.ppf(0.9)            # Convert from percentiles to standard deviation
        rdraw = np.random.normal(size=1)                     # Standard normal
        return sd*rdraw + median                             # Random draw from pRNG
