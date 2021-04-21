################################################################################################################
# Mostly uniform degree distribution network
#   -Generates a network following a specific degree distribution
#
# Paul Zivich 2019/10/31
################################################################################################################

import numpy as np
import networkx as nx

# Parameters
n = 512
seed = 20200122
np.random.seed(seed)

# Generating compatible degree distribution
is_it_odd = 1
while is_it_odd % 2 != 0:
    # Uniform probability distribution
    probs = np.random.uniform(0, 1, size=n)
    # Setting degree based on mostly uniform distribution
    degree_dist = np.where(probs < 1/7, 1, np.nan)
    degree_dist = np.where((2/6 > probs) & (probs > 1/6), 2, degree_dist)
    degree_dist = np.where((3/6 > probs) & (probs > 2/6), 3, degree_dist)
    degree_dist = np.where((4/6 > probs) & (probs > 3/6), 4, degree_dist)
    degree_dist = np.where((5/6 > probs) & (probs > 4/6), 5, degree_dist)
    degree_dist = np.where(probs > 5/6, 6, degree_dist)
    degree_dist = degree_dist.astype(int)
    # Checking whether odd degree (if odd, then configuration model won't work)
    is_it_odd = np.sum(degree_dist)

# Generating network following degree distribution
G = nx.configuration_model(degree_dist, seed=seed)

# Removing multiple edges!
G = nx.Graph(G)

# Removing self-loops
G.remove_edges_from(G.selfloop_edges())

# Removing the few isolates that exist
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

# saving graph edge-list
df = nx.to_pandas_edgelist(G)
df.to_csv("network-uniform.csv", index=False)
