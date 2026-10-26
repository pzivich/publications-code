########################################################################################################################
# Combining Randomized and Observational Data: The Case of Postnatal Prophylaxis Against HIV
#       - Code used to run the main simulation experiments
#
# Paul Zivich (2026/10/26)
########################################################################################################################

# Setup the environment
import numpy as np
import pandas as pd

from dgm import dgm
from efuncs import ee_naive, ee_naive_pool, ee_aipw_wr, fit_mestimator


################################################################
# Parameters to revise for the simulation setting (editable)

np.random.seed(9081234)
scenario = 1
trial_ratio = 0.75
n1, n0 = 2400, 5000


################################################################
# Run simulation scenario for 5000 iterations (don't edit)

rows = []
for i in range(5000):
    if (i + 1) % 50 == 0:
        print("running...", i+1)
    d = dgm(n1=n1, n0=n0, ratio=trial_ratio, scenario=scenario)
    # ds = d.sample(frac=1-pr_ltfu)

    y = np.asarray(d['Y'])
    a = np.asarray(d['A'])
    s = np.asarray(d['S'])
    W = np.asarray(d[['C', 'W']])
    X = np.asarray(d[['C', 'W']])

    # Naive -- trial-only
    def psi1(theta):
        return ee_naive(theta=theta, y=y, a=a, s=s)

    r_nt = fit_mestimator(psi=psi1, starting_values=[0, 0, 0.1, 0.2], scenario=scenario)

    # Naive -- pooling
    def psi2(theta):
        return ee_naive_pool(theta=theta, y=y, a=a)

    r_np = fit_mestimator(psi=psi2, starting_values=[0, 0, 0.1, 0.2], scenario=scenario)

    # AIPW -- weighted regression
    def psi4(theta):
        return ee_aipw_wr(theta=theta, y=y, a=a, s=s, W=W, X=X)

    r_wr = fit_mestimator(psi=psi4, starting_values=[0, 0, 0.1, 0.2] + [0, 0] + [0, 0], scenario=scenario)
    rows.append(r_nt + r_np + r_wr)


################################################################
# Formatting the output for the simulations
metrics = ["bias", "se", "lcl", "ucl", "cover", "pval"]
d = pd.DataFrame(rows, columns=(["nt_rd_" + x for x in metrics]
                                + ["nt_rr_" + x for x in metrics]
                                + ["np_rd_" + x for x in metrics]
                                + ["np_rr_" + x for x in metrics]
                                + ["wr_rd_" + x for x in metrics]
                                + ["wr_rr_" + x for x in metrics]
                                ))

################################################################
# Saving the results
d.to_csv("sim_results/main_scenario"+str(scenario)+".csv", index=False)
