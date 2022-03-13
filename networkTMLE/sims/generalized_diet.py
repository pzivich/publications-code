from sys import argv
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic
from sklearn.linear_model import Ridge

from amonhen import NetworkTMLE
from amonhen.utils import probability_to_odds, odds_to_probability, fast_exp_map
from beowulf import load_uniform_diet, load_random_diet, truth_values, simulation_setup
from beowulf.dgm import diet_dgm
from beowulf.dgm.utils import network_to_df

############################################
# Setting simulation parameters
############################################
n_mc = 500

exposure = "diet"
outcome = "bmi"

########################################
# Running through logic from .sh script
########################################
script_name, slurm_setup = argv
network, n_nodes, degree_restrict, shift, model, save = simulation_setup(slurm_id_str=slurm_setup)
sim_id = slurm_setup[4]
seed_number = 12670567 + 10000000*int(sim_id)
np.random.seed(seed_number)


# Loading correct  Network
if network == "uniform":
    G = load_uniform_diet(n=n_nodes)
if network == "random":
    G = load_random_diet(n=n_nodes)

# Marking if degree restriction is being applied
if degree_restrict is not None:
    restrict = True
else:
    restrict = False

# Setting up models
independent = False
distribution_gs = "threshold"
measure_gs = "t3"
q_estimator = None
if model == "cc":
    gin_model = "B_30 + G:E + E_mean + G_mean + degree"
    gsn_model = "diet + B_30 + G:E + E_mean + G_mean + degree"
    qn_model = "diet + diet_t3 + B + G + E + E_sum + G_sum + B_mean_dist + degree"
elif model == "cw":
    gin_model = "B_30 + G:E + E_mean + G_mean + degree"
    gsn_model = "diet + B_30 + G:E + E_mean + G_mean + degree"
    qn_model = "diet + diet_t3 + B + G + E + E_t3 + B_t30 + degree"
elif model == "wc":
    gin_model = "B_30 + G:E + E_t3 + B_t30 + degree"
    gsn_model = "diet + B_30 + G:E + E_t3 + B_t30 + degree"
    qn_model = "diet + diet_t3 + B + G + E + E_sum + G_sum + B_mean_dist + degree"
elif model == 'np':
    gin_model = "B_30 + G:E + C(E_sum_c) + C(G_sum_c) + B_mean_dist + degree"
    gsn_model = "diet + B_30 + G:E + C(E_sum_c) + C(G_sum_c) + B_mean_dist + degree"
    qn_model = "diet + diet_t3 + B + G + E + C(E_sum_c) + C(G_sum_c) + B_mean_dist + degree"
    q_estimator = Ridge(max_iter=2000)
elif model == 'ind':
    independent = True
    gi_model = "B_30 + G:E"
    qi_model = "diet + B + G + E"

# Determining if shift or absolute
if shift:
    prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

    # Generating probabilities (true) to assign
    data = network_to_df(G)
    adj_matrix = nx.adjacency_matrix(G, weight=None)
    data['E_mean'] = fast_exp_map(adj_matrix, np.array(data['E']), measure='mean')
    data['G_mean'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='mean')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(G.degree),
                                                 orient='index').rename(columns={0: 'degree'}),
                    how='left', left_index=True, right_index=True)
    prob = logistic.cdf(-1.5 + 0.05*(data['B'] - 30) + 2*data['G']*data['E']
                        + 1.*data['E_mean'] + 1.*data['G_mean']
                        + 0.05*data['degree'])
    log_odds = np.log(probability_to_odds(prob))
else:
    prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

truth = truth_values(network=network, dgm=exposure,
                     restricted_degree=restrict, shift=shift,
                     n=n_nodes)
print(truth)

print("#############################################")
print("Sim Script:", slurm_setup)
print("Seed:      ", seed_number)
print("=============================================")
print("Network:     ", network)
print("DGM:         ", exposure, '-', outcome)
print("Independent: ", independent)
print("Shift:       ", shift)
print("Set-up:      ", model)
print("=============================================")
print("results/" + exposure + "_" + save + ".csv")
print("#############################################")

########################################
# Setting up storage
########################################
if independent:
    cols = ['inc_'+exposure, 'inc_'+outcome] + ['bias_' + str(p) for p in prop_treated] + \
           ['lcl_' + str(p) for p in prop_treated] + ['ucl_' + str(p) for p in prop_treated] + \
           ['var_' + str(p) for p in prop_treated]
else:
    cols = ['inc_'+exposure, 'inc_'+outcome] + ['bias_' + str(p) for p in prop_treated] + \
           ['lcl_' + str(p) for p in prop_treated] + ['ucl_' + str(p) for p in prop_treated] + \
           ['lcll_' + str(p) for p in prop_treated] + ['ucll_' + str(p) for p in prop_treated] + \
           ['var_' + str(p) for p in prop_treated] + ['varl_' + str(p) for p in prop_treated]

results = pd.DataFrame(index=range(n_mc), columns=cols)

########################################
# Running simulation
########################################
for i in range(n_mc):
    # Generating Data
    H = diet_dgm(network=G, restricted=restrict)
    df = network_to_df(H)
    results.loc[i, 'inc_'+exposure] = np.mean(df[exposure])
    results.loc[i, 'inc_'+outcome] = np.mean(df[outcome])

    # Network TMLE
    ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict)
    ntmle.define_threshold(variable='diet', threshold=3, definition='sum')
    if model == "cw" or model == "wc":
        ntmle.define_threshold(variable='E', threshold=3, definition='sum')
        ntmle.define_threshold(variable='B', threshold=30, definition='mean')
    if model == 'np':
        if network == "uniform":
            if n_nodes == 500:
                ntmle.define_category(variable='E_sum', bins=[0, 1, 2, 3, 6], labels=False)
                ntmle.define_category(variable='G_sum', bins=[0, 1, 2, 3, 6], labels=False)
            elif n_nodes == 1000:
                ntmle.define_category(variable='E_sum', bins=[0, 1, 2, 3, 6], labels=False)
                ntmle.define_category(variable='G_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
            else:
                ntmle.define_category(variable='E_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
                ntmle.define_category(variable='G_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
        elif network == "random":
            if n_nodes == 500:
                ntmle.define_category(variable='E_sum', bins=[0, 1, 2, 5, 9, 15], labels=False)
                ntmle.define_category(variable='G_sum', bins=[0, 2, 5, 9, 16, 24], labels=False)
            elif n_nodes == 1000:
                ntmle.define_category(variable='E_sum', bins=[0, 1, 2, 3, 4, 19], labels=False)
                ntmle.define_category(variable='G_sum', bins=[0, 1, 2, 3, 4, 7, 26], labels=False)
            else:
                ntmle.define_category(variable='E_sum', bins=[0, 1, 2, 3, 4, 5, 6, 25], labels=False)
                ntmle.define_category(variable='G_sum', bins=[0, 1, 2, 3, 4, 5, 6, 9, 30], labels=False)
        else:
            raise ValueError("Invalid model-network combo")
    ntmle.exposure_model(gin_model)
    ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs)
    ntmle.outcome_model(qn_model, custom_model=q_estimator, distribution='normal')
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
            results.loc[i, 'varl_'+str(p)] = ntmle.conditional_latent_variance
            results.loc[i, 'lcll_'+str(p)] = ntmle.conditional_latent_ci[0]
            results.loc[i, 'ucll_'+str(p)] = ntmle.conditional_latent_ci[1]
        except:
            results.loc[i, 'bias_'+str(p)] = np.nan
            results.loc[i, 'var_'+str(p)] = np.nan
            results.loc[i, 'lcl_'+str(p)] = np.nan
            results.loc[i, 'ucl_'+str(p)] = np.nan
            results.loc[i, 'varl_'+str(p)] = np.nan
            results.loc[i, 'lcll_'+str(p)] = np.nan
            results.loc[i, 'ucll_'+str(p)] = np.nan


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
    if not independent:
        results['coverl_' + str(p)] = np.where((results['lcll_' + str(p)] < truth[p]) &
                                              (truth[p] < results['ucll_' + str(p)]), 1, 0)
        results['cldl_'+str(p)] = results['ucll_'+str(p)] - results['lcll_'+str(p)]

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
results.to_csv("results/" + exposure + str(sim_id) + "_" + save + ".csv", index=False)
