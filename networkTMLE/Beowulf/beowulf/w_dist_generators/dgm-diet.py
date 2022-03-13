################################################################################################################
# Diet and BMI data-generating-mechanism
#   -Generates the baseline distribution of W
#   -Does NOT allocate treatment or the corresponding outcomes
################################################################################################################

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf import load_uniform_network, load_random_network, load_exflu_network
from beowulf.dgm.utils import exp_map


def diet_baseline_dgm(graph, number_of_nodes):
    """Simulates baseline variables for the diet & BMI data set

    Returns
    -------
    pandas DataFrame with the distribution of W
    """
    # Unobserved factor to induce assortativity by BMI
    u = np.random.binomial(n=1, p=0.5, size=number_of_nodes)
    for node, value in zip(graph.nodes(), u):
        graph.nodes[node]['U'] = value
    u_s = exp_map(graph, 'U', measure='mean')

    # Gender
    g = np.random.binomial(n=1, p=0.4 + 0.5*u*np.where(u_s>0.5, 1, 0), size=number_of_nodes)

    # Baseline BMI
    # https://onlinelibrary.wiley.com/doi/full/10.1038/oby.2008.492
    b = np.random.lognormal(mean=3.35 + 0.25*u*np.where(u_s>0.5, 1, 0), sigma=0.2, size=number_of_nodes)

    # Exercise
    pe = logistic.cdf(-0.1 + 0.3*g + -0.0515*b + 0.001*b*b)
    e = np.random.binomial(n=1, p=pe, size=number_of_nodes)  # Generating values from above

    # Proximity to Work
    p = np.random.uniform(0, 1, size=number_of_nodes)

    # Output W distribution data set
    nodes = []
    for nod, d in graph.nodes(data=True):
        nodes.append(nod)

    data = pd.DataFrame()
    data['id'] = nodes
    data['G'] = g
    data['B'] = b
    data['E'] = e
    data['P'] = p
    return data


np.random.seed(4072020)

###################################
# Uniform Network
###################################
G = load_uniform_network(n1k=False)
w_dist = diet_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-diet-uniform.csv", index=False)
for n in G.nodes():
    G.nodes[n]['B'] = int(w_dist.loc[w_dist['id'] == n, 'B'].values)
    G.nodes[n]['G'] = int(w_dist.loc[w_dist['id'] == n, 'G'].values)

print(nx.numeric_assortativity_coefficient(G, attribute='B'))
print(nx.attribute_assortativity_coefficient(G, attribute='G'))

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E', 'P']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_random_network(n1k=False)
w_dist = diet_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-diet-cpl.csv", index=False)
for n in H.nodes():
    H.nodes[n]['B'] = int(w_dist.loc[w_dist['id'] == n, 'B'].values)
    H.nodes[n]['G'] = int(w_dist.loc[w_dist['id'] == n, 'G'].values)

print(nx.numeric_assortativity_coefficient(H, attribute='B'))
print(nx.attribute_assortativity_coefficient(H, attribute='G'))

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E', 'P']].describe())
print("=========================================")

###################################
# eX-FLU Network
###################################
N = load_exflu_network(filepath="/mnt/ARG/All_Team/ARG_Team/Paul/eXFLU/exflu_edges.csv")
w_dist = diet_baseline_dgm(N, number_of_nodes=nx.number_of_nodes(N))
w_dist.to_csv("dgm-diet-exflu.csv", index=False)
for n in N.nodes():
    N.nodes[n]['B'] = int(w_dist.loc[w_dist['id'] == n, 'B'].values)
    N.nodes[n]['G'] = int(w_dist.loc[w_dist['id'] == n, 'G'].values)

print(nx.numeric_assortativity_coefficient(N, attribute='B'))
print(nx.attribute_assortativity_coefficient(N, attribute='G'))

print("=========================================")
print("eX-FLU Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E', 'P']].describe())
print("=========================================")

###################################
# Uniform Network - 1K
###################################
G = load_uniform_network(n1k=True)
w_dist = diet_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-diet-uniform-1k.csv", index=False)
for n in G.nodes():
    G.nodes[n]['B'] = int(w_dist.loc[w_dist['id'] == n, 'B'].values)
    G.nodes[n]['G'] = int(w_dist.loc[w_dist['id'] == n, 'G'].values)

print(nx.numeric_assortativity_coefficient(G, attribute='B'))
print(nx.attribute_assortativity_coefficient(G, attribute='G'))

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E', 'P']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network - 1K
###################################
H = load_random_network(n1k=True)
w_dist = diet_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-diet-cpl-1k.csv", index=False)
for n in H.nodes():
    H.nodes[n]['B'] = int(w_dist.loc[w_dist['id'] == n, 'B'].values)
    H.nodes[n]['G'] = int(w_dist.loc[w_dist['id'] == n, 'G'].values)

print(nx.numeric_assortativity_coefficient(H, attribute='B'))
print(nx.attribute_assortativity_coefficient(H, attribute='G'))

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'B', 'E', 'P']].describe())
print("=========================================")
