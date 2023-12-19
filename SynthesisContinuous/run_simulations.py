#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Simulation experiments
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd
import warnings

from dgm import generate_data, calculate_truth
from statistical import StatAIPW
from synthesis import SynthesisMSM, SynthesisCACE
from helper import math_parameters_msm, math_parameters_cace, trapezoid, create_table

################################################
# General setup
warnings.filterwarnings("ignore")       # Ignoring some statsmodels GLM link-dist warnings
runs = 2000                             # Number of simulations
n1, n0 = 1000, 500                      # Sample size for each data set
mc_iters = 10000                        # Number of Monte Carlo iterations for synthesis estimators
n_cpus_avail = 30                       # Number of CPUs to parallelize the Monet Carlo with
scenario = 2                            # Scenario to apply
np.random.seed(48151623 + scenario*n0)  # Setting RNG seed

################################################
# Model Specifications
ps_model = "1"
samp_model = "V + W"
samp_rmodel = "W"
out_model = "A + V + W + A:V + A:W"
out_rmodel = "A + W + A:W"

msm_model = "A + V + A:V"
math_msm = "A:V_s300 + A:V_s800 - 1"

cace_model = "V"
math_cace = "V_s300 + V_s800 - 1"


################################################
# Running Simulation Script

if __name__ == '__main__':
    # Creating result storage
    results = pd.DataFrame(columns=['est_rt', 'low_rt', 'upp_rt',
                                    'est_rc', 'low_rc', 'upp_rc',
                                    'est_ex', 'low_ex', 'upp_ex',
                                    'est_syn_msm1', 'low_syn_msm1', 'upp_syn_msm1',
                                    'est_syn_msm2', 'low_syn_msm2', 'upp_syn_msm2',
                                    'est_syn_msm3', 'low_syn_msm3', 'upp_syn_msm3',
                                    'est_syn_msm4', 'low_syn_msm4', 'upp_syn_msm4',
                                    'est_syn_cac1', 'low_syn_cac1', 'upp_syn_cac1',
                                    'est_syn_cac2', 'low_syn_cac2', 'upp_syn_cac2',
                                    'est_syn_cac3', 'low_syn_cac3', 'upp_syn_cac3',
                                    'est_syn_cac4', 'low_syn_cac4', 'upp_syn_cac4',
                                    ])

    # Running the simulations
    for i in range(runs):
        print("starting", i+1, "...")
        row = []

        ############################################
        # Generating data
        d = generate_data(n1=n1, n0=n0, scenario=scenario)  # Generating full transport data
        dr = d.loc[d['V_star'] == 1].copy()                 # Restricting data to positive regions

        #######################################################################
        # Restricted Target Population
        aipw_rt = StatAIPW(data=dr, outcome='Y', action='A', sample='S')
        aipw_rt.action_model(ps_model)
        aipw_rt.sample_model(samp_model)
        aipw_rt.outcome_model(out_model)
        try:
            aipw_rt.estimate()
            row = row + [aipw_rt.ace, aipw_rt.ace_ci[0], aipw_rt.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Restricted Covariates
        aipw_rc = StatAIPW(data=d, outcome='Y', action='A', sample='S')
        aipw_rc.action_model(ps_model)
        aipw_rc.sample_model(samp_rmodel)
        aipw_rc.outcome_model(out_rmodel)
        try:
            aipw_rc.estimate()
            row = row + [aipw_rc.ace, aipw_rc.ace_ci[0], aipw_rc.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Extrapolation
        aipw_ex = StatAIPW(data=d, outcome='Y', action='A', sample='S')
        aipw_ex.action_model(ps_model)
        aipw_ex.sample_model(samp_model)
        aipw_ex.outcome_model(out_model)
        try:
            aipw_ex.estimate()
            row = row + [aipw_ex.ace, aipw_ex.ace_ci[0], aipw_ex.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - MSM -- normal with n=2000
        params = math_parameters_msm(n=2000, scenario=scenario)
        beta1, beta2 = params[0][0], params[1][0]
        vbeta1, vbeta2 = params[0][1], params[1][1]
        math_msm_params = [[np.random.normal(loc=beta1, scale=vbeta1),
                            np.random.normal(loc=beta2, scale=vbeta2)]
                           for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sm = SynthesisMSM(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sm.action_model(ps_model)
        aipw_sm.sample_model(samp_model)
        aipw_sm.outcome_model(out_model)
        aipw_sm.marginal_structural_model(msm_model)
        aipw_sm.math_model(model=math_msm,
                           parameters=math_msm_params)
        try:
            aipw_sm.estimate(mc_iterations=mc_iters, n_cpus=n_cpus_avail)
            row = row + [aipw_sm.ace, aipw_sm.ace_ci[0], aipw_sm.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - MSM -- normal with n=2000 trapezoid
        math_msm_params = [[trapezoid(mini=beta1 - 3*vbeta1, mode1=beta1 - vbeta1,
                                      mode2=beta1 + vbeta1, maxi=beta1 + 3*vbeta1),
                            trapezoid(mini=beta2 - 3*vbeta2, mode1=beta2 - vbeta2,
                                      mode2=beta2 + vbeta2, maxi=beta2 + 3*vbeta2)]
                           for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sm = SynthesisMSM(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sm.action_model(ps_model)
        aipw_sm.sample_model(samp_model)
        aipw_sm.outcome_model(out_model)
        aipw_sm.marginal_structural_model(msm_model)
        aipw_sm.math_model(model=math_msm,
                           parameters=math_msm_params)
        try:
            aipw_sm.estimate(mc_iterations=mc_iters, n_cpus=n_cpus_avail)
            row = row + [aipw_sm.ace, aipw_sm.ace_ci[0], aipw_sm.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - MSM -- normal with n=8000
        params = math_parameters_msm(n=8000, scenario=scenario)
        math_msm_params = [[np.random.normal(loc=params[0][0], scale=params[0][1]),
                            np.random.normal(loc=params[1][0], scale=params[1][1])]
                           for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sm = SynthesisMSM(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sm.action_model(ps_model)
        aipw_sm.sample_model(samp_model)
        aipw_sm.outcome_model(out_model)
        aipw_sm.marginal_structural_model(msm_model)
        aipw_sm.math_model(model=math_msm,
                           parameters=math_msm_params)
        try:
            aipw_sm.estimate(mc_iterations=mc_iters, n_cpus=n_cpus_avail)
            row = row + [aipw_sm.ace, aipw_sm.ace_ci[0], aipw_sm.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - MSM -- uniform null
        math_msm_params = [[np.random.uniform(-0.3, 0.3),
                            np.random.uniform(-0.3, 0.3)]
                           for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sm = SynthesisMSM(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sm.action_model(ps_model)
        aipw_sm.sample_model(samp_model)
        aipw_sm.outcome_model(out_model)
        aipw_sm.marginal_structural_model(msm_model)
        aipw_sm.math_model(model=math_msm,
                           parameters=math_msm_params)
        try:
            aipw_sm.estimate(mc_iterations=mc_iters, n_cpus=n_cpus_avail)
            row = row + [aipw_sm.ace, aipw_sm.ace_ci[0], aipw_sm.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - CACE -- normal with n=2000
        params = math_parameters_cace(n=2000, scenario=scenario)
        beta1, beta2 = params[0][0], params[1][0]
        vbeta1, vbeta2 = params[0][1], params[1][1]
        math_cace_params = [[np.random.normal(loc=beta1, scale=vbeta1),
                             np.random.normal(loc=beta2, scale=vbeta2)]
                            for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sc = SynthesisCACE(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sc.action_model(ps_model)
        aipw_sc.sample_model(samp_model)
        aipw_sc.outcome_model(out_model)
        aipw_sc.cace_model(cace_model)
        aipw_sc.math_model(model=math_cace,
                           parameters=math_cace_params)
        try:
            aipw_sc.estimate(mc_iterations=mc_iters,
                             n_cpus=n_cpus_avail)
            row = row + [aipw_sc.ace, aipw_sc.ace_ci[0], aipw_sc.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - CACE -- trapezoid with n=2000
        math_cace_params = [[trapezoid(mini=beta1 - 3*vbeta1, mode1=beta1 - vbeta1,
                                       mode2=beta1 + vbeta1, maxi=beta1 + 3*vbeta1),
                             trapezoid(mini=beta2 - 3*vbeta2, mode1=beta2 - vbeta2,
                                       mode2=beta2 + vbeta2, maxi=beta2 + 3*vbeta2)]
                            for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sc = SynthesisCACE(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sc.action_model(ps_model)
        aipw_sc.sample_model(samp_model)
        aipw_sc.outcome_model(out_model)
        aipw_sc.cace_model(cace_model)
        aipw_sc.math_model(model=math_cace,
                           parameters=math_cace_params)
        try:
            aipw_sc.estimate(mc_iterations=mc_iters,
                             n_cpus=n_cpus_avail)
            row = row + [aipw_sc.ace, aipw_sc.ace_ci[0], aipw_sc.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - CACE -- normal with n=8000
        params = math_parameters_cace(n=8000, scenario=scenario)
        math_cace_params = [[np.random.normal(loc=params[0][0], scale=params[0][1]),
                             np.random.normal(loc=params[1][0], scale=params[1][1])]
                            for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sc = SynthesisCACE(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sc.action_model(ps_model)
        aipw_sc.sample_model(samp_model)
        aipw_sc.outcome_model(out_model)
        aipw_sc.cace_model(cace_model)
        aipw_sc.math_model(model=math_cace,
                           parameters=math_cace_params)
        try:
            aipw_sc.estimate(mc_iterations=mc_iters,
                             n_cpus=n_cpus_avail)
            row = row + [aipw_sc.ace, aipw_sc.ace_ci[0], aipw_sc.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Synthesis - CACE -- uniform null
        math_cace_params = [[np.random.uniform(-0.3, 0.3),
                             np.random.uniform(-0.3, 0.3)]
                            for j in range(mc_iters)]

        # Applying synthesis estimator with math parameters
        aipw_sc = SynthesisCACE(data=d, outcome='Y', action='A', sample='S', positive_region='V_star')
        aipw_sc.action_model(ps_model)
        aipw_sc.sample_model(samp_model)
        aipw_sc.outcome_model(out_model)
        aipw_sc.cace_model(cace_model)
        aipw_sc.math_model(model=math_cace,
                           parameters=math_cace_params)
        try:
            aipw_sc.estimate(mc_iterations=mc_iters,
                             n_cpus=n_cpus_avail)
            row = row + [aipw_sc.ace, aipw_sc.ace_ci[0], aipw_sc.ace_ci[1]]
        except RuntimeError:
            row = row + [np.nan, np.nan, np.nan]

        #######################################################################
        # Appending Results
        results.loc[len(results.index)] = row

    #######################################################################
    # Saving Full Results upon Completion
    results.to_csv("s"+str(scenario)+"_results_n1-"+str(n1)+"_n0-"+str(n0)+".csv", index=False)

    #######################################################################
    # Creating Overall Summary Results
    # truth = calculate_truth(n=20000000, scenario=scenario)
    # table = create_table(data=results, truth=truth)
    # print(table.round(2))
    # table.to_csv("results_table_"+"s"+str(scenario)+"_n0-"+str(n0)+".csv")

# END
