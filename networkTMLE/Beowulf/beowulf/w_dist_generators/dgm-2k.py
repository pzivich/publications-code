import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf import load_2k_network
from beowulf.dgm.utils import exp_map

np.random.seed(20220107)

################################################################################################################
# Statin and CVD data-generating-mechanism
################################################################################################################


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


###################################
# Uniform Network (1K)
###################################
U = load_2k_network(network='uniform')
w_dist = statin_baseline_dgm(U, number_of_nodes=nx.number_of_nodes(U)).sort_values(by='id')
w_dist.to_csv("dgm-statin-uniform-2k.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network (1K)
###################################
V = load_2k_network(network='random')
w_dist = statin_baseline_dgm(V, number_of_nodes=nx.number_of_nodes(V)).sort_values(by='id')
w_dist.to_csv("dgm-statin-cpl-2k.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['A', 'L', 'D', 'R']].describe())
print("=========================================")

################################################################################################################
# Naloxone and Overdose data-generating-mechanism
################################################################################################################


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


###################################
# Uniform Network
###################################
G = load_2k_network(network='uniform')
w_dist = naloxone_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-uniform-2k.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_2k_network(network='random')
w_dist = naloxone_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H)).sort_values(by='id')
w_dist.to_csv("dgm-naloxone-cpl-2k.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['G', 'Uc', 'P', 'O']].describe())
print("=========================================")


################################################################################################################
# Diet and BMI data-generating-mechanism
################################################################################################################


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


###################################
# Uniform Network

G = load_2k_network(network='uniform')
w_dist = diet_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-diet-uniform-2k.csv", index=False)
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

H = load_2k_network(network='random')
w_dist = diet_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-diet-cpl-2k.csv", index=False)
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


################################################################################################################
# Vaccine and Infection data-generating-mechanism
################################################################################################################


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


###################################
# Uniform Network
###################################
G = load_2k_network(network='uniform')
w_dist = vaccine_baseline_dgm(G, number_of_nodes=nx.number_of_nodes(G))
w_dist.to_csv("dgm-vaccine-uniform-2k.csv", index=False)

print("=========================================")
print("Uniform Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")

###################################
# Clustered Power-Law Network
###################################
H = load_2k_network(network='random')
w_dist = vaccine_baseline_dgm(H, number_of_nodes=nx.number_of_nodes(H))
w_dist.to_csv("dgm-vaccine-cpl-2k.csv", index=False)

print("=========================================")
print("Clustered Power-Law Network")
print("-----------------------------------------")
print(w_dist[['A', 'H']].describe())
print("=========================================")
