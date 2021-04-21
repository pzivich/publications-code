import numpy as np
import pandas as pd
import networkx as nx
from pkg_resources import resource_filename


def network_generator(edgelist, source, target, label):
    """Reads in a NetworkX graph object from an edgelist

    IDs need to be sequential from 0 to n_max for this function to behave as expected (and add nodes that have no edges)
    """
    graph = nx.Graph(label=label)

    # adding edges
    for i, j in zip(edgelist[source], edgelist[target]):
        graph.add_edge(i, j)

    return graph


def load_uniform_network():
    # file path to uniform network.
    edgelist = pd.read_csv(resource_filename('beowulf', 'data_files/network-uniform.csv'), index_col=False)

    # call network_generator function
    graph = network_generator(edgelist, source='source', target='target', label='uniform')
    return graph


def load_random_network():
    # file path to uniform network.
    edgelist = pd.read_csv(resource_filename('beowulf', 'data_files/network-random.csv'), index_col=False)

    # call network_generator function
    graph = network_generator(edgelist, source='source', target='target', label='random')
    return graph


def generate_sofrygin_network(n, max_degree=2, seed=None):
    """Generates a network following Sofrygin & van der Laan 2017. Network follows a uniform degree distribution

    Returns
    -------
    Network object
    """
    np.random.seed(seed)

    # checking if even sum for degrees, since needed
    sum = 1
    while sum % 2 != 0:
        degree_dist = list(np.random.randint(0, max_degree+1, size=n))
        sum = np.sum(degree_dist)

    G = nx.configuration_model(degree_dist, seed=seed)
    # Removing multiple edges!
    G = nx.Graph(G)
    # Removing self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Generating W and adding to network
    w = np.random.binomial(n=1, p=0.35, size=n)
    for node in G.nodes():
        G.nodes[node]['W'] = w[node]

    return G


def load_uniform_naloxone():
    graph = load_uniform_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-naloxone-uniform.csv'), index_col=False)
    for n in graph.nodes():
        graph.nodes[n]['G'] = int(attrs.loc[attrs['id'] == n, 'G'].values)
        graph.nodes[n]['Uc'] = int(attrs.loc[attrs['id'] == n, 'Uc'].values)
        graph.nodes[n]['P'] = int(attrs.loc[attrs['id'] == n, 'P'].values)
        graph.nodes[n]['O'] = int(attrs.loc[attrs['id'] == n, 'O'].values)
    return nx.convert_node_labels_to_integers(graph)


def load_uniform_statin():
    graph = load_uniform_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-statin-uniform.csv'), index_col=False)
    attrs['R_1'] = np.where((attrs['R'] >= .05) & (attrs['R'] < .075), 1, 0)
    attrs['R_2'] = np.where((attrs['R'] >= .075) & (attrs['R'] < .2), 1, 0)
    attrs['R_3'] = np.where(attrs['R'] >= .2, 1, 0)
    attrs['A_30'] = attrs['A'] - 30
    attrs['A_sqrt'] = np.sqrt(attrs['A']-39.9)

    for n in graph.nodes():
        graph.nodes[n]['A'] = int(attrs.loc[attrs['id'] == n, 'A'].values)
        graph.nodes[n]['L'] = float(attrs.loc[attrs['id'] == n, 'L'].values)
        graph.nodes[n]['R'] = float(attrs.loc[attrs['id'] == n, 'R'].values)
        graph.nodes[n]['R_1'] = int(attrs.loc[attrs['id'] == n, 'R_1'].values)
        graph.nodes[n]['R_2'] = int(attrs.loc[attrs['id'] == n, 'R_2'].values)
        graph.nodes[n]['R_3'] = int(attrs.loc[attrs['id'] == n, 'R_3'].values)
        graph.nodes[n]['A_30'] = int(attrs.loc[attrs['id'] == n, 'A_30'].values)
        graph.nodes[n]['A_sqrt'] = float(attrs.loc[attrs['id'] == n, 'A_sqrt'].values)

    return nx.convert_node_labels_to_integers(graph)


def load_uniform_diet():
    graph = load_uniform_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-diet-uniform.csv'), index_col=False)
    for n in graph.nodes():
        graph.nodes[n]['G'] = int(attrs.loc[attrs['id'] == n, 'G'].values)
        graph.nodes[n]['B'] = int(attrs.loc[attrs['id'] == n, 'B'].values)
        graph.nodes[n]['B_30'] = int(attrs.loc[attrs['id'] == n, 'B'].values) - 30
        graph.nodes[n]['E'] = int(attrs.loc[attrs['id'] == n, 'E'].values)
    return nx.convert_node_labels_to_integers(graph)


def load_uniform_vaccine():
    graph = load_uniform_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-vaccine-uniform.csv'), index_col=False)
    for n in graph.nodes():
        graph.nodes[n]['A'] = int(attrs.loc[attrs['id'] == n, 'A'].values)
        graph.nodes[n]['H'] = int(attrs.loc[attrs['id'] == n, 'H'].values)
    return nx.convert_node_labels_to_integers(graph)


def load_random_naloxone():
    graph = load_random_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-naloxone-cpl.csv'), index_col=False)
    for n in graph.nodes():
        graph.nodes[n]['G'] = int(attrs.loc[attrs['id'] == n, 'G'].values)
        graph.nodes[n]['Uc'] = int(attrs.loc[attrs['id'] == n, 'Uc'].values)
        graph.nodes[n]['P'] = int(attrs.loc[attrs['id'] == n, 'P'].values)
        graph.nodes[n]['O'] = int(attrs.loc[attrs['id'] == n, 'O'].values)
    return nx.convert_node_labels_to_integers(graph)


def load_random_statin():
    graph = load_random_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-statin-cpl.csv'), index_col=False)
    attrs['R_1'] = np.where((attrs['R'] >= .05) & (attrs['R'] < .075), 1, 0)
    attrs['R_2'] = np.where((attrs['R'] >= .075) & (attrs['R'] < .2), 1, 0)
    attrs['R_3'] = np.where(attrs['R'] >= .2, 1, 0)
    attrs['A_30'] = attrs['A'] - 30
    attrs['A_sqrt'] = np.sqrt(attrs['A']-39)

    for n in graph.nodes():
        graph.nodes[n]['A'] = int(attrs.loc[attrs['id'] == n, 'A'].values)
        graph.nodes[n]['L'] = float(attrs.loc[attrs['id'] == n, 'L'].values)
        graph.nodes[n]['R'] = float(attrs.loc[attrs['id'] == n, 'R'].values)
        graph.nodes[n]['R_1'] = int(attrs.loc[attrs['id'] == n, 'R_1'].values)
        graph.nodes[n]['R_2'] = int(attrs.loc[attrs['id'] == n, 'R_2'].values)
        graph.nodes[n]['R_3'] = int(attrs.loc[attrs['id'] == n, 'R_3'].values)
        graph.nodes[n]['A_30'] = int(attrs.loc[attrs['id'] == n, 'A_30'].values)
        graph.nodes[n]['A_sqrt'] = float(attrs.loc[attrs['id'] == n, 'A_sqrt'].values)

    return nx.convert_node_labels_to_integers(graph)


def load_random_diet():
    graph = load_random_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-diet-cpl.csv'), index_col=False)
    for n in graph.nodes():
        graph.nodes[n]['G'] = int(attrs.loc[attrs['id'] == n, 'G'].values)
        graph.nodes[n]['B'] = int(attrs.loc[attrs['id'] == n, 'B'].values)
        graph.nodes[n]['B_30'] = int(attrs.loc[attrs['id'] == n, 'B'].values) - 30
        graph.nodes[n]['E'] = int(attrs.loc[attrs['id'] == n, 'E'].values)
    return nx.convert_node_labels_to_integers(graph)


def load_random_vaccine():
    graph = load_random_network()

    # adding attributes to the network
    attrs = pd.read_csv(resource_filename('beowulf', 'data_files/dgm-vaccine-cpl.csv'), index_col=False)
    for n in graph.nodes():
        graph.nodes[n]['A'] = int(attrs.loc[attrs['id'] == n, 'A'].values)
        graph.nodes[n]['H'] = int(attrs.loc[attrs['id'] == n, 'H'].values)
    return nx.convert_node_labels_to_integers(graph)

