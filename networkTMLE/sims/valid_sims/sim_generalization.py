from sys import argv
import numpy as np
import pandas as pd

from amonhen import NetworkTMLE, NetworkIPTW, NetworkGFormula
from beowulf import (sofrygin_observational, sofrygin_randomized, direct_randomized, direct_observational,
                     modified_observational, modified_randomized, continuous_observational, continuous_randomized,
                     indirect_observational, indirect_randomized, independent_observational, independent_randomized,
                     generate_sofrygin_network)
from beowulf.dgm.utils import network_to_df


script_name, network, model_setup, estimator, save_file = argv
model_setup = int(model_setup)

############################################
# Setting simulation parameters
############################################
n_mc = 2000
n_sims_truth = 5000
sample_size = 1000
np.random.seed(911)

exposure = "A"
outcome = "Y"

prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Getting network and W distribution
G = generate_sofrygin_network(n=sample_size, max_degree=2, seed=20200409)

########################################
# Running through logic from .sh script
########################################
if model_setup == 1:
    gi_model = 'W + W_sum'
    gs_model = 'A + W + W_sum'
    qi_model = 'A + A_sum + W + W_sum'
    ms = "Both Correct"
elif model_setup == 2:
    gi_model = 'W + W_sum'
    gs_model = 'A + W + W_sum'
    qi_model = 'A + W'
    ms = "Correct g-model"
elif model_setup == 3:
    gi_model = 'W'
    gs_model = 'A + W'
    qi_model = 'A + A_sum + W + W_sum'
    ms = "Correct Q-model"
elif model_setup == 4:
    gi_model = 'W'
    gs_model = 'A + W'
    qi_model = 'A + W'
    ms = "Both incorrect"
else:
    raise ValueError("Invalid Model Selection")


if network == 'sofrygin':
    random_net_gen = sofrygin_randomized
    obs_net_gen = sofrygin_observational
elif network == 'modified':
    random_net_gen = modified_randomized
    obs_net_gen = modified_observational
elif network == 'continuous':
    random_net_gen = continuous_randomized
    obs_net_gen = continuous_observational
elif network == 'direct':
    random_net_gen = direct_randomized
    obs_net_gen = direct_observational
elif network == 'indirect':
    random_net_gen = indirect_randomized
    obs_net_gen = indirect_observational
elif network == 'independent':
    random_net_gen = independent_randomized
    obs_net_gen = independent_observational
else:
    raise ValueError("Invalid network name in .sh script")


print("#############################################")
print("Sim Script:", script_name)
print("=============================================")
print("Estimator:   ", estimator)
print("Network DGM: ", network)
print("DGM:         ", exposure, '-', outcome)
print("Set-up:      ", ms)
print("#############################################")

########################################
# Calculating Truth
########################################
truth = {}
for t in prop_treated:
    ans = []
    for i in range(n_sims_truth):
        ans.append(random_net_gen(graph=G, p=t))
    truth[t] = np.mean(ans)


########################################
# Setting up storage
########################################
results = pd.DataFrame(index=range(n_mc), columns=['inc_'+exposure, 'inc_'+outcome] +
                                                  ['bias_' + str(p) for p in prop_treated] +
                                                  ['lcl_' + str(p) for p in prop_treated] +
                                                  ['ucl_' + str(p) for p in prop_treated] +
                                                  ['var_' + str(p) for p in prop_treated]
                       )


########################################
# Running simulation
########################################
for i in range(n_mc):
    # Generating Data
    H = obs_net_gen(G)
    df = network_to_df(H)
    results.loc[i, 'inc_'+exposure] = np.mean(df[exposure])
    results.loc[i, 'inc_'+outcome] = np.mean(df[outcome])

    if estimator == 'tmle':
        # Network TMLE
        ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome)
        ntmle.exposure_model(gi_model)
        ntmle.exposure_map_model(gs_model)
        ntmle.outcome_model(qi_model)
        for p in prop_treated:  # loops through all treatment plans
            try:
                ntmle.fit(p=p, bound=0.01)
                results.loc[i, 'bias_'+str(p)] = ntmle.marginal_outcome - truth[p]
                results.loc[i, 'var_'+str(p)] = ntmle.conditional_variance
                results.loc[i, 'lcl_'+str(p)] = ntmle.conditional_ci[0]
                results.loc[i, 'ucl_'+str(p)] = ntmle.conditional_ci[1]
            except:
                results.loc[i, 'bias_'+str(p)] = np.nan
                results.loc[i, 'var_'+str(p)] = np.nan
                results.loc[i, 'lcl_'+str(p)] = np.nan
                results.loc[i, 'ucl_'+str(p)] = np.nan
    elif estimator == 'iptw':
        niptw = NetworkIPTW(H, exposure=exposure, outcome=outcome, verbose=False)
        niptw.exposure_model(gi_model)
        niptw.exposure_map_model(gs_model)
        for p in prop_treated:  # loops through all treatment plans
            try:
                niptw.fit(p=p, bound=0.01)
                results.loc[i, 'bias_'+str(p)] = niptw.marginal_outcome - truth[p]
                results.loc[i, 'var_'+str(p)] = niptw.conditional_variance
                results.loc[i, 'lcl_'+str(p)] = niptw.conditional_ci[0]
                results.loc[i, 'ucl_'+str(p)] = niptw.conditional_ci[1]
            except:
                results.loc[i, 'bias_' + str(p)] = np.nan
                results.loc[i, 'var_' + str(p)] = np.nan
                results.loc[i, 'lcl_' + str(p)] = np.nan
                results.loc[i, 'ucl_' + str(p)] = np.nan
    elif estimator == 'gformula':
        gform = NetworkGFormula(H, exposure='A', outcome='Y', verbose=False)
        gform.outcome_model(qi_model)
        for p in prop_treated:  # loops through all treatment plans
            results.loc[i, 'var_' + str(p)] = np.nan
            results.loc[i, 'lcl_' + str(p)] = np.nan
            results.loc[i, 'ucl_' + str(p)] = np.nan
            try:
                gform.fit(p=p)
                results.loc[i, 'bias_'+str(p)] = gform.marginal_outcome - truth[p]
            except:
                results.loc[i, 'bias_'+str(p)] = np.nan
    else:
        raise ValueError("Invalid estimator choice")


########################################
# Summarizing results
########################################
print("\nRESULTS\n")

for p in prop_treated:
    # Confidence Interval Coverage
    results['cover_'+str(p)] = np.where((results['lcl_'+str(p)] < truth[p]) &
                                        (truth[p] < results['ucl_'+str(p)]), 1, 0)
    # Confidence Limit Difference
    results['cld_'+str(p)] = results['ucl_'+str(p)] - results['lcl_'+str(p)]

    print("===========================")
    print(p)
    print("---------------------------")
    print("Bias:", np.mean(results['bias_'+str(p)]))
    print("ESE:", np.std(results['bias_'+str(p)], ddof=1))
    print("Cover:", np.mean(results['cover_'+str(p)]))

print("===========================")

########################################
# Saving results
########################################
# results.to_csv("results/" + save_file + ".csv", index=False)
