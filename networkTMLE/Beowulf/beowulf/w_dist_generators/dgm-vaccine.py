################################################################################################################
# Statin and CVD data-generating-mechanism
#   -Generates the baseline distribution of W
#   -Does NOT allocate treatment or the corresponding outcomes
################################################################################################################

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf import load_uniform_network, load_random_network, load_exflu_network


def vaccine_baseline_dgm(graph, number_of_nodes):
    """Simulates baseline variables for the vaccine & infection data set

    A ~ Bernoulli(0.12)
    H ~ Bernoulli(0.65)

    Returns
    -------
    pandas DataFrame with the distribution of W
    """
    data = pd.DataFrame()
    nodes = []
    for nod, d in graph.nodes(data=True):
        nodes.append(nod)

    data['id'] = nodes

    # Asthma
    a = np.random.binomial(n=1, p=0.15, size=number_of_nodes)
    data['A'] = a

    # Hand hygiene
    d = np.random.binomial(n=1, p=logistic.cdf(-0.15 + 0.1*a), size=number_of_nodes)
    data['H'] = d

    # Output W distribution data set
    return data


np.random.seed(20200503)

###################################
# Mostly-uniform Network
###################################
G = load_uniform_network(n1k=False)
w_dist = vaccine_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-vaccine-uniform.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network(n1k=False)
w_dist = vaccine_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-vaccine-cpl.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")

###################################
# eX-FLU Network
###################################
N = load_exflu_network(filepath="/mnt/ARG/All_Team/ARG_Team/Paul/eXFLU/exflu_edges.csv")
w_dist = vaccine_baseline_dgm(N, number_of_nodes=nx.number_of_nodes(N))
w_dist.to_csv("dgm-vaccine-exflu.csv", index=False)

print("=========================================")
print("eX-FLU Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")

###################################
# Mostly-uniform Network
###################################
G = load_uniform_network(n1k=True)
w_dist = vaccine_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-vaccine-uniform-1k.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network(n1k=True)
w_dist = vaccine_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-vaccine-cpl-1k.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")
