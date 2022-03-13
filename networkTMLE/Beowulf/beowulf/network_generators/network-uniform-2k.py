################################################################################################################
# Mostly uniform degree distribution network
#   -Generates a network following a specific degree distribution
#   -Creates 2K nodes
#
# Paul Zivich 2021/8/2
################################################################################################################

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
n = 2073
seed = 20220107
np.random.seed(seed)

# Generating compatible degree distribution
captured = False

while not captured:
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
    G.remove_edges_from(nx.selfloop_edges(G))

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])

    print(G0.number_of_nodes())
    # nx.draw(G0, node_size=20)
    # plt.show()

    if G0.number_of_nodes() == 2000:
        captured = True



# saving graph edge-list
df = nx.to_pandas_edgelist(G0)
df.to_csv("network-uniform-2k.csv", index=False)
