################################################################################################################
# Cluster-Powerlaw Network
#   -Generates a network following a clustered powerlaw with a community structure
#
# Paul Zivich 2019/10/31
################################################################################################################

import numpy as np
import networkx as nx

# Parameters
n_comms = [50, 40, 60, 50, 120, 30, 80, 70]
add_edges = 3
p_cluster = 0.75
p_between = 0.0007
seed = 20200122
np.random.seed(seed)

# Creating network of the disconnected components
N = nx.Graph()

for i in range(len(n_comms)):
    # Generate the component
    G = nx.powerlaw_cluster_graph(int(n_comms[i]), m=add_edges, p=p_cluster, seed=int(seed+(i+1)*10000))

    # Re-label nodes so no corresponding overlaps between node labels
    if i == 0:
        start_label = 0
    else:
        start_label = np.sum(n_comms[:i])
    mapping = {}
    for j in range(n_comms[i]):
        mapping[j] = start_label + j
    H = nx.relabel_nodes(G, mapping)

    # Adding component to overall network
    N.add_nodes_from(H.nodes)
    N.add_edges_from(H.edges)


# Creating some random connections across groups
for i in range(len(n_comms)):
    # Gettings IDs
    first_id = int(np.sum(n_comms[:i]))
    last_id = int(np.sum(n_comms[:i+1]))

    # Only adding edges to > last_id
    for j in range(first_id+1, last_id+1):
        for n in list(N.nodes()):
            if n > last_id:
                if np.random.uniform(0, 1) < p_between:
                    N.add_edge(j, n)


# saving graph edge-list
df = nx.to_pandas_edgelist(N)
df.to_csv("network-random.csv", index=False)
