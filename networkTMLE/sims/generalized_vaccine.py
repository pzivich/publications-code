from sys import argv
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from amonhen import StochasticTMLE, NetworkTMLE
from amonhen.utils import probability_to_odds, odds_to_probability, fast_exp_map
from beowulf import load_uniform_vaccine, load_random_vaccine, truth_values
from beowulf.dgm import vaccine_dgm
from beowulf.dgm.utils import network_to_df

############################################
# Setting simulation parameters
############################################
n_mc = 4000
np.random.seed(1927056)

exposure = "vaccine"
outcome = "D"

# StochasticTMLE models
gi_model = "A + H"
qi_model = "vaccine + A + H"

# NetworkTMLE models
gin_model = "A + H + H_mean + degree"
gsn_model = "vaccine + A + H + H_mean + degree"
qn_model = "vaccine + vaccine_mean + A + H + A_mean + H_mean + degree"
gs_dist = "poisson"
gs_measure = "sum"

########################################
# Running through logic from .sh script
########################################
script_name, network, model_setup, shift, independent, save_file = argv

# Determining if StochasticTMLE or NetworkTMLE
independent = bool(int(independent))

# Extracting needed network info
model_setup = int(model_setup)

restrict = False
degree_restrict = None
if network == 'uniform':
    G = load_uniform_vaccine()
    if model_setup == 1:
        measure_gs = None
        distribution_gs = None
        if not independent:
            raise ValueError("For independent network, only model_setup = 1 and independent = 1 is valid")
    elif model_setup == 2:
        measure_gs = None
        distribution_gs = None
    elif model_setup == 3:
        measure_gs = gs_measure
        distribution_gs = gs_dist
    else:
        raise ValueError("Invalid set-up specification for "+network+" network")

elif network == 'random':
    G = load_random_vaccine()

    gin_model += " + np.power(degree, 2)"
    gsn_model += " + np.power(degree, 2)"
    qn_model += " + np.power(degree, 2)"

    measure_gs = gs_measure
    distribution_gs = gs_dist

    if model_setup == 1:
        if not independent:
            raise ValueError("For independent network, only model_setup = 1 and independent = 1 is valid")
    elif model_setup == 2:
        pass
    elif model_setup == 3:
        degree_restrict = (0, 18)
        restrict = True
    else:
        raise ValueError("Invalid set-up specification for "+network+" network")

else:
    raise ValueError("Invalid network name in .sh script")

# Determining if shift or absolute
shift = bool(int(shift))
if shift:
    prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

    # Generating probabilities (true) to assign
    data = network_to_df(G)
    adj_matrix = nx.adjacency_matrix(G, weight=None)
    data['H_mean'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='mean')
    prob = logistic.cdf(-1.9 + 1.75*data['A'] + 0.95*data['H'] + 1.2*data['H_mean'])
    log_odds = np.log(probability_to_odds(prob))

else:
    prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

truth = truth_values(network=network, dgm=exposure, restricted_degree=restrict, shift=shift)

print("#############################################")
print("Sim Script:", script_name)
print("=============================================")
print("Network:     ", network)
print("DGM:         ", exposure, '-', outcome)
print("Independent: ", independent)
print("Shift:       ", shift)
print("Set-up:      ", model_setup)
print("#############################################")

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
    H = vaccine_dgm(network=G, restricted=restrict)
    df = network_to_df(H)
    results.loc[i, 'inc_'+exposure] = np.mean(df[exposure])
    results.loc[i, 'inc_'+outcome] = np.mean(df[outcome])

    if independent:
        # Stochastic TMLE
        stmle = StochasticTMLE(df, exposure=exposure, outcome=outcome)
        stmle.exposure_model(gi_model, bound=0.01)
        stmle.outcome_model(qi_model)
        for p in prop_treated:  # loops through all treatment plans
            try:
                if shift:
                    z = odds_to_probability(np.exp(log_odds + p))
                    stmle.fit(p=z)
                else:
                    stmle.fit(p=p)
                results.loc[i, 'bias_' + str(p)] = stmle.marginal_outcome - truth[p]
                results.loc[i, 'var_' + str(p)] = stmle.conditional_se ** 2
                results.loc[i, 'lcl_' + str(p)] = stmle.conditional_ci[0]
                results.loc[i, 'ucl_' + str(p)] = stmle.conditional_ci[1]
            except:
                results.loc[i, 'bias_' + str(p)] = np.nan
                results.loc[i, 'var_' + str(p)] = np.nan
                results.loc[i, 'lcl_' + str(p)] = np.nan
                results.loc[i, 'ucl_' + str(p)] = np.nan

    else:
        # Network TMLE
        ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict)
        ntmle.exposure_model(gin_model)
        ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs)
        ntmle.outcome_model(qn_model)
        for p in prop_treated:  # loops through all treatment plans
            try:
                if shift:
                    z = odds_to_probability(np.exp(log_odds + p))
                    ntmle.fit(p=z, bound=0.01)
                else:
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


########################################
# Summarizing results
########################################
print("RESULTS\n")

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
