from sys import argv
from multiprocessing import Pool
import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_aipw

from estimator import SingleCrossFitTMLE

##########################################################################
# Setup
##########################################################################

# Extract hanging numbers from submission for the setup
sim_id = int(argv[1])        # Simulation ID for data set
n_reps = int(argv[2])        # Number of repetitions of procedure
n_splits = int(argv[3])      # Number of folds considered

#
truth = 0.0011909            # True value from `truth.py`
n_seeds = 50                 # Number of different seeds being considered
n_cpus = 4                   # Number of CPUs to use
rng_estr = np.random.default_rng(seed=1881912 + sim_id*10000)


def psi(theta):
    # Estimating function for AIPW
    return ee_aipw(theta=theta, y=d['Y'], A=d['A'],
                   W=d[['C', 'W']], X=d[['C', 'A', 'W', 'AW']],
                   X1=d1[['C', 'A', 'W', 'AW']], X0=d0[['C', 'A', 'W', 'AW']])


def scrossfit_tmle(rng_seed):
    # Function to apply the SCFTMLE (used by Pool to run in parallel)
    global d, n_splits, n_reps
    cftmle = SingleCrossFitTMLE(data=d, outcome='Y', action='A', seed=rng_seed)
    cftmle.nuisance_action(model="W")
    cftmle.nuisance_outcome(model="A + W")
    return cftmle.estimate(n_splits=n_splits, n_reps=n_reps)


##########################################################################
# Loading the corresponding data set
##########################################################################

dfull = pd.read_csv("data/full_data.csv")
d = dfull.loc[dfull['sim_id'] == sim_id].reset_index(drop=True).copy()
d1 = d.copy()
d1['A'] = 1
d1['AW'] = d1['A'] * d1['W']
d0 = d.copy()
d0['A'] = 0
d0['AW'] = d0['A'] * d0['W']


if __name__ == "__main__":
    # Generating storage
    columns = ["naive_bias", "naive_var", ] + [v + str(x) for x in range(n_seeds) for v in ("ml_bias_", "ml_var_")]
    results = pd.DataFrame(columns=columns)

    ##########################################################################
    # Running simulation
    ##########################################################################

    row = []

    # Parametric AIPW
    estr = MEstimator(psi, init=[0., 0.5, 0.5, ] + [0., ]*2 + [0., ]*4)
    estr.estimate()
    row.append(estr.theta[0] - truth)
    row.append(estr.variance[0, 0])

    # Single Cross-Fit TMLE
    params = list(rng_estr.integers(low=0, high=2**32-1, size=n_seeds))    # Generate list of unique seed values
    with Pool(processes=n_cpus) as pool:                                   # Run the pooling process
        output = list(pool.map(scrossfit_tmle,                             # ... call outside function to run parallel
                               params))                                    # ... provide packed input list
    row = row + [k for j in output for k in j]                             # Unpack lists of lists into single row list

    # Appending results to output
    results.loc[len(results.index)] = row
    results.to_csv("r_s"+str(n_splits)+"r"+str(n_reps)+"i"+str(sim_id)+".csv", index=False)


print("DONE! :)")
