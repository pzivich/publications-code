################################################################################################################
# Naloxone and Mortality data-generating-mechanism
#   -Generates the baseline distribution of W
#   -Does NOT allocate treatment or the corresponding outcomes
################################################################################################################

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf import load_uniform_network, load_random_network, load_exflu_network
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
        graph.nodes[node]['G'] = value

    g_s = exp_map(graph, 'G', measure='mean')

    # Trust in authorities (unobserved variable)
    c = np.random.binomial(n=1, p=0.75, size=number_of_nodes)

    # Recently released from prison
    mp = logistic.cdf(-1.1 + 0.5*g + 0.1*g_s)  # model
    p = np.random.binomial(n=1, p=mp, size=number_of_nodes)  # Generating values from above

    # Prior overdose
    mo = logistic.cdf(-1.7 + 0.1*g + 0.1*g_s + 0.6*p)  # model
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
# Uniform Network
###################################
G = load_uniform_network(n1k=False)
w_dist = naloxone_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-uniform.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network(n1k=False)
w_dist = naloxone_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-cpl.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")

###################################
# eX-FLU Network
###################################
N = load_exflu_network(filepath="/mnt/ARG/All_Team/ARG_Team/Paul/eXFLU/exflu_edges.csv")
w_dist = naloxone_baseline_dgm(N, number_of_nodes=nx.number_of_nodes(N)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-exflu.csv", index=False)

print("=========================================")
print("eX-FLU Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")

###################################
# Uniform Network
###################################
G = load_uniform_network(n1k=True)
w_dist = naloxone_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-uniform-1k.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network(n1k=True)
w_dist = naloxone_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-cpl-1k.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")
