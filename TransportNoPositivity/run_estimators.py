#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Run the simulation study for the described estimators
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd
from delicatessen import MEstimator

from dgm import generate_data, generate_background_info
from efuncs import ee_restrict_gcomp, ee_restrict_ipw
from helper import synthesis_g_computation, synthesis_ipw

# General setup information
np.random.seed(80951342)        # Setting seed
runs = 2000                     # Number of simulation iterations
mc_iterations = 5000            # Number of iterations for Monte Carlo procedure in EACH simulation iteration
n_cpus = 15                     # Number of CPUs available to use (done on my desktop)


def prep_gcomp(data):
    """Helper function for simulations to generate the data necessary for g-computation.

    Parameters
    ----------
    data :
        Clinic and trial data

    Returns
    -------
    list
    """
    x = np.asarray(data[vars_stat_out])
    da = data.copy()
    da['A'] = 1
    xa1 = np.asarray(da[vars_stat_out])
    da['A'] = 0
    xa0 = np.asarray(da[vars_stat_out])
    return x, xa1, xa0


def psi_restrict_gcomp(theta):
    # Formatted estimating equation from deli (EE is defined in efuncs.py)
    return ee_restrict_gcomp(theta, y, s, X, Xa1, Xa0)


def psi_restrict_ipw(theta):
    # Formatted estimating equation from deli (EE is defined in efuncs.py)
    return ee_restrict_ipw(theta, y, s, a, W, null)


# Setting up variable columns to extract
vars_stat_out = ['intercept', 'A', 'V']
vars_stat_prs = ['intercept', 'V', 'V_i25']
vars_stat_non = ['intercept', ]

# Running the simulations
if __name__ == "__main__":                   # Call in case running on windows (so Pool behaves)
    truth = 0.216697                         # True value as determined in truth.py
    results = pd.DataFrame(columns=['bias_rtp_g', 'cld_rtp_g', 'cover_rtp_g', 'bias_rtp_w', 'cld_rtp_w', 'cover_rtp_w',
                                    'bias_rcs_g', 'cld_rcs_g', 'cover_rcs_g', 'bias_rcs_w', 'cld_rcs_w', 'cover_rcs_w',
                                    'bias_pr1_g', 'cld_pr1_g', 'cover_pr1_g', 'bias_pr1_w', 'cld_pr1_w', 'cover_pr1_w',
                                    'bias_pr2_g', 'cld_pr2_g', 'cover_pr2_g', 'bias_pr2_w', 'cld_pr2_w', 'cover_pr2_w',
                                    'bias_pr3_g', 'cld_pr3_g', 'cover_pr3_g', 'bias_pr3_w', 'cld_pr3_w', 'cover_pr3_w',
                                    'bias_pr4_g', 'cld_pr4_g', 'cover_pr4_g', 'bias_pr4_w', 'cld_pr4_w', 'cover_pr4_w',
                                    'bias_pr5_g', 'cld_pr5_g', 'cover_pr5_g', 'bias_pr5_w', 'cld_pr5_w', 'cover_pr5_w',
                                    ])

    # Running the simulation
    for i in range(runs):
        # print("starting", i+1)
        row = []

        ##########################################
        # Generate data
        d = generate_data(n1=1000, n0=1000)

        ##########################################
        # Restrict the Target Population
        dr = d.loc[d['W'] == 0].copy()
        X, Xa1, Xa0 = prep_gcomp(data=dr)
        W = np.asarray(dr[vars_stat_prs])
        null = np.asarray(dr[vars_stat_non])
        a = np.asarray(dr['A'])
        s = np.asarray(dr['S'])
        y = np.asarray(dr['Y'])

        # G-computation
        estr = MEstimator(psi_restrict_gcomp,
                          init=[0, 0.5, 0.5, 0, 0, 0])
        estr.estimate(solver='lm')
        ci = estr.confidence_intervals()
        row.append(estr.theta[0] - truth)
        row.append(ci[0, 1] - ci[0, 0])
        if ci[0, 0] < truth < ci[0, 1]:
            row.append(1)
        else:
            row.append(0)

        # Inverse Probability Weighting
        estr = MEstimator(psi_restrict_ipw, init=[0, 0.5, 0.5, 0, 0, 0, 0])
        estr.estimate(solver='lm')
        ci = estr.confidence_intervals()
        row.append(estr.theta[0] - truth)
        row.append(ci[0, 1] - ci[0, 0])
        if ci[0, 0] < truth < ci[0, 1]:
            row.append(1)
        else:
            row.append(0)

        ##########################################
        # Restrict the Covariate Set
        X, Xa1, Xa0 = prep_gcomp(data=d)
        W = np.asarray(d[vars_stat_prs])
        null = np.asarray(d[vars_stat_non])
        a = np.asarray(d['A'])
        s = np.asarray(d['S'])
        y = np.asarray(d['Y'])

        # G-computation
        estr = MEstimator(psi_restrict_gcomp,
                          init=[0, 0.5, 0.5, 0, 0, 0])
        estr.estimate(solver='lm')
        ci = estr.confidence_intervals()
        row.append(estr.theta[0] - truth)
        row.append(ci[0, 1] - ci[0, 0])
        if ci[0, 0] < truth < ci[0, 1]:
            row.append(1)
        else:
            row.append(0)

        # Inverse Probability Weighting
        estr = MEstimator(psi_restrict_ipw, init=[0, 0.5, 0.5, 0, 0, 0, 0])
        estr.estimate(solver='lm')
        ci = estr.confidence_intervals()
        row.append(estr.theta[0] - truth)
        row.append(ci[0, 1] - ci[0, 0])
        if ci[0, 0] < truth < ci[0, 1]:
            row.append(1)
        else:
            row.append(0)

        ##########################################
        # Stat-Sim Synthesis
        cond_params = generate_background_info(marginal=False)
        marg_params = generate_background_info(marginal=True)

        # Scenario 1: Strict Null
        point, ci = synthesis_g_computation(data=d, mc_iters=mc_iterations, setup=1, n_cpus=n_cpus)
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)
        point, ci = synthesis_ipw(data=d, mc_iters=mc_iterations, setup=1, n_cpus=n_cpus)
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)

        # Scenario 2: Uncertain Null
        point, ci = synthesis_g_computation(data=d, mc_iters=mc_iterations, setup=2, n_cpus=n_cpus)
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)
        point, ci = synthesis_ipw(data=d, mc_iters=mc_iterations, setup=2, n_cpus=n_cpus)
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)

        # Scenario 3: Accurate
        point, ci = synthesis_g_computation(data=d, mc_iters=mc_iterations, setup=3, n_cpus=n_cpus,
                                            mu=cond_params[0], cov=cond_params[1])
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)
        point, ci = synthesis_ipw(data=d, mc_iters=mc_iterations, setup=3, n_cpus=n_cpus,
                                  mu=marg_params[0], cov=marg_params[1])
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)

        # Scenario 4: Inaccurate
        point, ci = synthesis_g_computation(data=d, mc_iters=mc_iterations, setup=4, n_cpus=n_cpus,
                                            mu=-1*np.array(cond_params[0]), cov=cond_params[1])
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)
        point, ci = synthesis_ipw(data=d, mc_iters=mc_iterations, setup=4, n_cpus=n_cpus,
                                  mu=-1*np.array(marg_params[0]), cov=marg_params[1])
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)

        # Scenario 5: Accurate with Covariance
        point, ci = synthesis_g_computation(data=d, mc_iters=mc_iterations, setup=5, n_cpus=n_cpus,
                                            mu=cond_params[0], cov=cond_params[1])
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)
        point, ci = synthesis_ipw(data=d, mc_iters=mc_iterations, setup=5, n_cpus=n_cpus,
                                  mu=marg_params[0], cov=marg_params[1])
        row.append(point - truth)
        row.append(ci[1] - ci[0])
        if ci[0] < truth < ci[1]:
            row.append(1)
        else:
            row.append(0)

        ##########################################
        # Stacking Results to Output
        results.loc[len(results.index)] = row

    ##########################################
    # Saving Resuts upon Completion
    results.to_csv("sim_results.csv", index=False)
