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


def statin_baseline_dgm(graph, number_of_nodes):
    """Simulates baseline variables for the naloxone & overdose data set

    A ~ Bernoulli(0.325)
    L ~ Bernoulli(0.65)
    D ~ Bernoulli(expit(B + B*G + B*sum(G)))
    R ~ Bernoulli(P==1: 0.1, P==0: 0.3)

    Returns
    -------
    pandas DataFrame with the distribution of W
    """
    data = pd.DataFrame()
    nodes = []
    for nod, d in graph.nodes(data=True):
        nodes.append(nod)

    data['id'] = nodes

    # Age
    a = np.random.randint(40, 61, size=number_of_nodes)
    data['A'] = a

    # LDL
    l = 0.005*a + np.random.normal(np.log(100), 0.18, size=number_of_nodes)
    data['L'] = l

    # Diabetes
    d = np.random.binomial(n=1, p=logistic.cdf(-4.23 + 0.03*l - 0.02*a + 0.0009*(a**2)), size=number_of_nodes)
    data['D'] = d

    # Frailty
    f = logistic.cdf(-5.5 + 0.05*(a-20) + 0.001*(a**2) + np.random.normal(size=number_of_nodes))

    # Risk Scores
    ln_a = np.log(a)
    r = logistic.cdf(4.299 + 3.501*d - 2.07*ln_a + 0.051*ln_a**2 + 4.090*l - 1.04*ln_a*l + 0.01*f)
    data['R'] = r

    # Output W distribution data set
    return data


np.random.seed(20200123)

###################################
# Uniform Network
###################################
G = load_uniform_network(n1k=False)
w_dist = statin_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G)).sort_values(by='id')
w_dist.to_csv("dgm-statin-uniform.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network(n1k=False)
w_dist = statin_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H)).sort_values(by='id')
w_dist.to_csv("dgm-statin-cpl.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")

###################################
# eX-FLU Network
###################################
N = load_exflu_network(filepath="/mnt/ARG/All_Team/ARG_Team/Paul/eXFLU/exflu_edges.csv")
w_dist = statin_baseline_dgm(N, number_of_nodes=nx.number_of_nodes(N)).sort_values(by='id')
w_dist.to_csv("dgm-statin-exflu.csv", index=False)

print("=========================================")
print("eX-FLU Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")

###################################
# Uniform Network (1K)
###################################
U = load_uniform_network(n1k=True)
w_dist = statin_baseline_dgm(U, number_of_nodes=nx.number_of_nodes(U)).sort_values(by='id')
w_dist.to_csv("dgm-statin-uniform-1k.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network (1K)
###################################
V = load_random_network(n1k=True)
w_dist = statin_baseline_dgm(V, number_of_nodes=nx.number_of_nodes(V)).sort_values(by='id')
w_dist.to_csv("dgm-statin-cpl-1k.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")
