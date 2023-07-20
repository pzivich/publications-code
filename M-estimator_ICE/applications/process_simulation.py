####################################################################################################################
# Iterated Conditional Expectation (ICE) G-computation as an M-estimator
#   Processing simulation results
#
# Paul Zivich
####################################################################################################################

import numpy as np
import pandas as pd

n = [250, 500, 1000, 2000, 5000]


def metrics(bias, std, coverage):
    b = np.nanmean(bias)
    ese = np.nanstd(bias, ddof=1)
    ase = np.nanmean(std)
    ser = ase / ese
    cov = np.nanmean(coverage)
    print("Bias: ", np.round(b, 3))
    print("ESE:  ", np.round(ese, 3))
    print("ASE:  ", np.round(ase, 3))
    print("SER:  ", np.round(ser, 2))
    print("Cover:", np.round(cov, 2))
    print("Fail: ", np.sum(np.isnan(bias)))


for i in n:
    d = pd.read_csv("results/sim1_n"+str(i)+".csv")

    print("=======================================")
    print("N:", i)
    print("#######################################")
    print("a=(1, 1, 1)")
    print("#######################################")
    print("Parametric")
    print("----------------------------------------")
    metrics(bias=d['bias_g_pa1'], std=d['se_g_pa1'], coverage=d['cov_g_pa1'])
    print("----------------------------------------")
    print("Stratified")
    print("----------------------------------------")
    metrics(bias=d['bias_g_sa1'], std=d['se_g_sa1'], coverage=d['cov_g_sa1'])
    print("#######################################")
    print("a=(0, 0, 0)")
    print("#######################################")
    print("Parametric")
    print("----------------------------------------")
    metrics(bias=d['bias_g_pa0'], std=d['se_g_pa0'], coverage=d['cov_g_pa0'])
    print("----------------------------------------")
    print("Stratified")
    print("----------------------------------------")
    metrics(bias=d['bias_g_sa0'], std=d['se_g_sa0'], coverage=d['cov_g_sa0'])
    print("=======================================")
    print("")
