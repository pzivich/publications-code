################################################################################################################
# Diet and BMI data-generating-mechanism
#   -Generates the baseline distribution of W
#   -Does NOT allocate treatment or the corresponding outcomes
################################################################################################################

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf import load_uniform_network, load_random_network


def diet_baseline_dgm(graph, number_of_nodes):
    """Simulates baseline variables for the diet & BMI data set

    Returns
    -------
    pandas DataFrame with the distribution of W
    """
    # Gender
    g = np.random.binomial(n=1, p=0.5, size=number_of_nodes)

    # Baseline BMI
    b = np.random.lognormal(3.4, sigma=0.2, size=number_of_nodes)

    # Exercise
    pe = logistic.cdf(-0.25)  # logistic.cdf(-0.25 + 0.3*g + -0.0515*b + 0.001*b*b)
    e = np.random.binomial(n=1, p=pe, size=number_of_nodes)  # Generating values from above

    # Output W distribution data set
    nodes = []
    for nod, d in graph.nodes(data=True):
        nodes.append(nod)

    data = pd.DataFrame()
    data['id'] = nodes
    data['G'] = g
    data['B'] = b
    data['E'] = e
    return data


np.random.seed(4072020)

###################################
# Mostly-uniform Network
###################################
G = load_uniform_network()
w_dist = diet_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-diet-uniform.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network()
w_dist = diet_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-diet-cpl.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E']].describe())
print("=========================================")
