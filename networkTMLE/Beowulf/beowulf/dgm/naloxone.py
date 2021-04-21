import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)


def naloxone_dgm(network, restricted=False):
    """
    Parameters
    ----------
    network:
        input network
    restricted:
        whether to use the restricted treatment assignment
    """
    graph = network.copy()
    data = network_to_df(graph)

    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['O_sum'] = fast_exp_map(adj_matrix, np.array(data['O']), measure='sum')
    data['O_mean'] = fast_exp_map(adj_matrix, np.array(data['O']), measure='mean')
    data['G_sum'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='sum')
    data['G_mean'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='mean')

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(-1.3 - 1.5*data['P'] + 1.5*data['P']*data['G'] +
                        0.95*data['O_mean'] + 0.95*data['G_mean'])
    naloxone = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['naloxone'] = naloxone  # https://www.sciencedirect.com/science/article/pii/S074054721730301X (30%)
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='naloxone')
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['naloxone']))

    data['naloxone_sum'] = fast_exp_map(adj_matrix, np.array(data['naloxone']), measure='sum')

    # Running Data Generating Mechanism for Y
    pr_y = logistic.cdf(-1.1 - 0.2*data['naloxone_sum'] +
                        1.7*data['P'] - 0.9*data['G'] + 0.75*data['O_mean'] - 0.75*data['G_mean'])
    overdose = np.random.binomial(n=1, p=pr_y, size=nx.number_of_nodes(graph))
    data['overdose'] = overdose  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5832501/?report=classic (20%)
    # print(data[['naloxone', 'overdose']].describe())

    # Adding node information back to graph
    for n in graph.nodes():
        graph.nodes[n]['naloxone'] = int(data.loc[data.index == n, 'naloxone'].values)
        graph.nodes[n]['overdose'] = int(data.loc[data.index == n, 'overdose'].values)

    return graph


def naloxone_dgm_truth(network, pr_a, shift=False, restricted=False):
    graph = network.copy()
    data = network_to_df(graph)
    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['O_sum'] = fast_exp_map(adj_matrix, np.array(data['O']), measure='sum')
    data['O_mean'] = fast_exp_map(adj_matrix, np.array(data['O']), measure='mean')
    data['G_sum'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='sum')
    data['G_mean'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='mean')

    # Running Data Generating Mechanism for A
    if shift:  # If a shift in the Odds distribution is instead specified
        prob = logistic.cdf(-1.3 - 1.5*data['P'] + 1.5*data['P']*data['G'] + 0.95*data['O_mean'] + 0.95*data['G_mean'])
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    naloxone = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['naloxone'] = naloxone
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='naloxone')
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['naloxone']))
        exclude = list(attrs.keys())

    # Creating network summary variables
    data['naloxone_sum'] = fast_exp_map(adj_matrix, np.array(data['naloxone']), measure='sum')

    # Running Data Generating Mechanism for Y
    pr_y = logistic.cdf(-1.1 - 0.2*data['naloxone_sum'] +
                        1.7*data['P'] - 0.9*data['G'] + 0.75*data['O_mean'] - 0.75*data['G_mean'])
    overdose = np.random.binomial(n=1, p=pr_y, size=nx.number_of_nodes(graph))
    if restricted:
        data['overdose'] = overdose
        data = data.loc[~data.index.isin(exclude)].copy()
        overdose = np.array(data['overdose'])

    return np.mean(overdose)
