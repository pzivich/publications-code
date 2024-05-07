from sys import argv
import numpy as np
import pandas as pd

# Setup
n_splits = int(argv[1])
n_reps = 50
n_sims = 200

# Loading and stacking each of the individual results into a list
r_all = []
for i in range(1, n_sims+1):
    d = pd.read_csv("r_s" + str(n_splits) + "r" + str(n_reps) + "i" + str(i) + ".csv")
    r_all.append(d)


# Output results as a single file for analysis
all_results = pd.concat(r_all, ignore_index=True)
all_results.to_csv("r_s" + str(n_splits) + "r" + str(n_reps) + ".csv", index=False)
