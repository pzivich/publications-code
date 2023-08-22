####################################################################################################################
# ACTG 320 - ACTG 175 Fusion: diagnostic results
#       This script runs the proposed diagnostics using the ACTG example. Graphical and permutation tests are
#       both demonstrated.
#
# Paul Zivich (2023/08/22)
####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chimera import SurvivalFusionIPW


# Calling __main__ since using Pool for the permutation
if __name__ == '__main__':
    # Loading pre-formatted data
    d = pd.read_csv("../data/actg_data_formatted.csv", sep=",")

    # Setting up nuisance models
    censoring = "male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat) + study"
    sampling = "male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat)"
    censoring_cd4 = ("male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat) + study "
                     "+ cd4 + cd4_rs0 + cd4_rs1")
    sampling_cd4 = "male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat) + cd4 + cd4_rs0 + cd4_rs1"

    #################################
    # Unadjusted Model
    ufipw = SurvivalFusionIPW(df=d, treatment='art', outcome='delta', time='t',
                              sample='study', censor='censor', verbose=False)
    ufipw.sampling_model("1", bound=False)
    ufipw.treatment_model(model="1", bound=False)
    ufipw.censoring_model(censoring, censor_shift=1e-4, bound=False,
                          strata='art', stratify_by_sample=False)
    u_results = ufipw.estimate(variance="bootstrap", bs_iterations=1000, n_cpus=28, seed=20230705)

    #################################
    # Adjusted Model
    afipw = SurvivalFusionIPW(df=d, treatment='art', outcome='delta', time='t',
                              sample='study', censor='censor', verbose=False)
    afipw.sampling_model(sampling, bound=False)
    afipw.treatment_model(model="1", bound=False)
    afipw.censoring_model(censoring, censor_shift=1e-4, bound=False,
                          strata='art', stratify_by_sample=False)
    a_results = afipw.estimate(variance="bootstrap", bs_iterations=1000, n_cpus=28, seed=20230704)

    #################################
    # Sampling model including CD4
    wfipw = SurvivalFusionIPW(df=d, treatment='art', outcome='delta', time='t',
                              sample='study', censor='censor', verbose=False)
    wfipw.sampling_model(sampling_cd4, bound=False)
    wfipw.treatment_model(model="1", bound=False)
    wfipw.censoring_model(censoring_cd4, censor_shift=1e-4, bound=False,
                          strata='art', stratify_by_sample=False)
    cd4_results = wfipw.estimate(variance="bootstrap", bs_iterations=1000, n_cpus=28, seed=20230706)

    #################################
    # Restricting by CD4
    dr = d.loc[(d['cd4'] >= 50) & (d['cd4'] <= 300)].copy()
    rfipw = SurvivalFusionIPW(df=dr, treatment='art', outcome='delta', time='t',
                              sample='study', censor='censor', verbose=False)
    rfipw.sampling_model(sampling, bound=False)
    rfipw.treatment_model(model="1", bound=False)
    rfipw.censoring_model(censoring, censor_shift=1e-4, bound=False,
                          strata='art', stratify_by_sample=False)
    r_results = rfipw.estimate(variance="bootstrap", bs_iterations=1000, n_cpus=28, seed=20230805)

    #################################
    # Plot Overall (.png)

    # Define figure dimensions and setup
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    # Plotting reference line across all plots
    for c in [0, 1]:
        for d in [0, 1]:
            ax[c][d].vlines(0, 0, 370,  # Dashed Reference Line at Zero
                            colors='gray', linestyles='--', label=None)

    # doing plots by-hand (not using fipw.diagnostic_plot()) since doing custom stacking together
    # Unadjusted Diagnostic Difference
    ax[0][0].step(u_results["R1D"],                             # Risk Difference column
                  u_results["t"].shift(-1).ffill(),             # time column (make sure steps occur at correct t)
                  color='k', where='post', label="Unadjusted")
    ax[0][0].fill_betweenx(u_results["t"],                      # time column (no shift needed here)
                           u_results["R1D_UCL"],                # upper confidence limit
                           u_results["R1D_LCL"],                # lower confidence limit
                           label=None, color='gray', alpha=0.5, step='post')
    ax[0][0].text(-0.22, 360, "A)", fontsize=12)

    # Adjusted Diagnostic Difference
    ax[0][1].step(a_results["R1D"],                             # Risk Difference column
                  a_results["t"].shift(-1).ffill(),             # time column (make sure steps occur at correct t)
                  color='k', where='post', label="Adjusted")
    ax[0][1].fill_betweenx(a_results["t"],                      # time column (no shift needed here)
                           a_results["R1D_UCL"],                # upper confidence limit
                           a_results["R1D_LCL"],                # lower confidence limit
                           label=None, color='gray', alpha=0.5, step='post')
    ax[0][1].text(-0.22, 360, "B)", fontsize=12)

    # Weighted with CD4 Diagnostic Difference
    ax[1][0].step(cd4_results["R1D"],                           # Risk Difference column
                  cd4_results["t"].shift(-1).ffill(),           # time column (make sure steps occur at correct t)
                  color='k', where='post', label="Adjusted")
    ax[1][0].fill_betweenx(cd4_results["t"],                    # time column (no shift needed here)
                           cd4_results["R1D_UCL"],              # upper confidence limit
                           cd4_results["R1D_LCL"],              # lower confidence limit
                           label=None, color='gray', alpha=0.5, step='post')
    ax[1][0].text(-0.22, 360, "C)", fontsize=12)

    # Restricted Diagnostic Difference
    ax[1][1].step(r_results["R1D"],                             # Risk Difference column
                  r_results["t"].shift(-1).ffill(),             # time column (make sure steps occur at correct t)
                  color='k', where='post', label="Adjusted")
    ax[1][1].fill_betweenx(r_results["t"],                      # time column (no shift needed here)
                           r_results["R1D_UCL"],                # upper confidence limit
                           r_results["R1D_LCL"],                # lower confidence limit
                           label=None, color='gray', alpha=0.5, step='post')
    ax[1][1].text(-0.22, 360, "D)", fontsize=12)

    # Updating plot parameters within a loop (so all match)
    for c in [0, 1]:
        for d in [0, 1]:
            ax[c][d].set_ylim([0, 367])
            ax[c][d].set_xlim([-0.15, 0.2])
            ax[c][d].set_xticks([-0.1, 0.0, 0.1, 0.2])
            ax[c][d].set_yticks([0, 50, 100, 150, 200, 250, 300, 365])
            ax[c][d].set_ylabel("Days")
            ax[c][d].set_xlabel("Difference in Dual")
            ax[c][d].spines['right'].set_visible(False)
            ax[c][d].spines['top'].set_visible(False)
            ax2 = ax[c][d].twiny()  # Duplicate the x-axis to create a separate label
            ax2.set_xlabel("Favors ACTG 320" + "\t".expandtabs() + "Favors ACTG 175",
                           fontdict={"size": 10})
            ax2.set_xticks([])
            ax2.xaxis.set_ticks_position('bottom')
            ax2.xaxis.set_label_position('bottom')
            ax2.spines['bottom'].set_position(('outward', 36))

    plt.tight_layout()
    plt.savefig("figure1.png", format='png', dpi=300)
    plt.close()

    #################################
    # Calculating p-values!

    # Unadjusted Model
    print("Unadjusted")
    ufipw.diagnostic_test(bs_iterations=1000, print_results=True, n_cpus=28, seed=809415)

    # Adjusted
    print("\nAdjusted")
    afipw.diagnostic_test(bs_iterations=1000, print_results=True, n_cpus=28, seed=209422)

    # CD4-adjusted
    print("\nAdjusted + CD4")
    wfipw.diagnostic_test(bs_iterations=1000, print_results=True, n_cpus=28, seed=919418)

    # Restricted by CD4
    print("\nAdjusted + restricted by CD4")
    rfipw.diagnostic_test(bs_iterations=1000, print_results=True, n_cpus=28, seed=401425)


# Unadjusted
# ======================================================================
#        Fusion Inverse Probability Weighting Diagnostic Test
# ======================================================================
# No. Bootstraps:     1000
# ----------------------------------------------------------------------
# Area:    32.496
# 95% CI:  [24.541 40.452]
# P-value: 0.0
# ======================================================================
#
# Adjusted
# ======================================================================
#        Fusion Inverse Probability Weighting Diagnostic Test
# ======================================================================
# No. Bootstraps:     1000
# ----------------------------------------------------------------------
# Area:    30.051
# 95% CI:  [20.841 39.26 ]
# P-value: 0.0
# ======================================================================
#
# Adjusted + CD4
# ======================================================================
#        Fusion Inverse Probability Weighting Diagnostic Test
# ======================================================================
# No. Bootstraps:     1000
# ----------------------------------------------------------------------
# Area:    36.367
# 95% CI:  [28.453 44.281]
# P-value: 0.0
# ======================================================================
#
# Adjusted + restricted by CD4
# ======================================================================
#        Fusion Inverse Probability Weighting Diagnostic Test
# ======================================================================
# No. Bootstraps:     1000
# ----------------------------------------------------------------------
# Area:    7.944
# 95% CI:  [-2.843 18.73 ]
# P-value: 0.149
# ======================================================================
