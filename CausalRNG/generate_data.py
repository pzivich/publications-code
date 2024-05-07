import numpy as np
import pandas as pd

from dgm import data_generation

# Setup
rng_data = np.random.default_rng(seed=9180214)
n_obs = 1000
n_data = 500  # Creating extra in case computation time wasn't terrible


# Generate all the data sets
all_data = []
for i in range(n_data):
    # Generating data set
    d = data_generation(n=n_obs, rng=rng_data)
    d['C'] = 1
    d['AW'] = d['A'] * d['W']
    d['sim_id'] = i + 1

    # Storing data copy
    all_data.append(d)


# Storing a copy of the full generated data sets
full_data = pd.concat(all_data)
full_data.to_csv("full_data.csv", index=False)
