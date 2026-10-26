########################################################################################################################
# Combining Randomized and Observational Data: The Case of Postnatal Prophylaxis Against HIV
#       - Code used to run the power as a function of the trial size and randomization ratio
#
# Paul Zivich (2026/10/26)
########################################################################################################################

# Setup the environment
import warnings
import numpy as np
import pandas as pd

from dgm import dgm
from efuncs import ee_naive, ee_aipw_wr, fit_mestimator

################################################################
# Parameters to revise for the simulation setting (editable)

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(19081234)
scenario = 3
n0 = 5000

################################################################
# Simulation run (don't edit)

for t_ratio in [1/2, 3/4]:
    for n1 in [2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        rows = []
        for i in range(5000):
            if (i + 1) % 250 == 0:
                print("running...", i+1)
            d = dgm(n1=n1, n0=n0, ratio=t_ratio, scenario=scenario)

            y = np.asarray(d['Y'])
            a = np.asarray(d['A'])
            s = np.asarray(d['S'])
            W = np.asarray(d[['C', 'W']])
            X = np.asarray(d[['C', 'W']])

            # Naive -- trial-only
            def psi1(theta):
                return ee_naive(theta=theta, y=y, a=a, s=s)

            r_nt = fit_mestimator(psi=psi1, starting_values=[0, 0, 0.1, 0.2])

            # AIPW -- weighted regression
            def psi4(theta):
                return ee_aipw_wr(theta=theta, y=y, a=a, s=s, W=W, X=X)

            r_wr = fit_mestimator(psi=psi4, starting_values=[0, 0, 0.1, 0.2] + [0, 0] + [0, 0])
            rows.append(r_nt + r_wr)

        # Compute the simulation metrics and same them
        metrics = ["bias", "se", "lcl", "ucl", "cover", "pval"]
        d = pd.DataFrame(rows, columns=(["nt_rd_" + x for x in metrics]
                                        + ["nt_rr_" + x for x in metrics]
                                        + ["wr_rd_" + x for x in metrics]
                                        + ["wr_rr_" + x for x in metrics]
                                        ))

        # Output and save the corresponding results
        d.to_csv("sim_results/power_r"+str(round(t_ratio, 2))+"_n"+str(n1)+"_scenario"+str(scenario)+".csv",
                 index=False)
        print("Completed", t_ratio, n1, "!")
