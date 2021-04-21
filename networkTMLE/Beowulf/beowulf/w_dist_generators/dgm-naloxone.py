################################################################################################################
# Naloxone and Mortality data-generating-mechanism
#   -Generates the baseline distribution of W
#   -Does NOT allocate treatment or the corresponding outcomes
################################################################################################################

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf import load_uniform_network, load_random_network
from beowulf.dgm.utils import exp_map


def naloxone_baseline_dgm(graph, number_of_nodes):
    """Simulates baseline variables for the naloxone & overdose data set

    G ~ Bernoulli(0.325)
    Uc ~ Bernoulli(0.65)
    P ~ Bernoulli(expit(B + B*G + B*sum(G)))
    O ~ Bernoulli(P==1: 0.1,
                  P==0: 0.3)

    Returns
    -------
    pandas DataFrame with the distribution of W
    """
    # Gender
    g = np.random.binomial(n=1, p=0.325, size=number_of_nodes)  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454335/
    for node, value in zip(graph.nodes(), g):
        graph.node[node]['G'] = value

    g_s = exp_map(graph, 'G', measure='mean')

    # Trust in authorities (unobserved variable)
    c = np.random.binomial(n=1, p=0.75, size=number_of_nodes)

    # Recently released from prison
    beta_p = {0: -1.1, 1: 0.5, 2: 0.1}  # Beta parameters
    mp = logistic.cdf(beta_p[0] + beta_p[1]*g + beta_p[2]*g_s)  # model
    p = np.random.binomial(n=1, p=mp, size=number_of_nodes)  # Generating values from above

    # Prior overdose
    beta_o = {0: -1.7, 1: 0.1, 2: 0.1, 3: 0.6}  # Beta parameters
    mo = logistic.cdf(beta_o[0] + beta_o[1]*g + beta_o[2]*g_s + beta_o[3]*p)  # model
    o = np.random.binomial(n=1, p=mo, size=number_of_nodes)  # Generating values from above

    # Output W distribution data set
    nodes = []
    for nod, d in graph.nodes(data=True):
        nodes.append(nod)

    data = pd.DataFrame()
    data['id'] = nodes
    data['G'] = g
    data['Uc'] = c
    data['P'] = p
    data['O'] = o
    return data


np.random.seed(20200123)

###################################
# Mostly-uniform Network
###################################
G = load_uniform_network()
w_dist = naloxone_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-naloxone-uniform.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network()
w_dist = naloxone_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-naloxone-cpl.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")
