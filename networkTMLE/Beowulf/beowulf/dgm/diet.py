import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)


def diet_dgm(network, restricted=False):
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
    data['G_mean'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='mean')
    data['G_sum'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='sum')
    data['E_mean'] = fast_exp_map(adj_matrix, np.array(data['E']), measure='mean')
    data['E_sum'] = fast_exp_map(adj_matrix, np.array(data['E']), measure='sum')
    data['B_mean_dist'] = fast_exp_map(adj_matrix, np.array(data['B']), measure='mean_dist')
    data['B_mean'] = fast_exp_map(adj_matrix, np.array(data['B']), measure='mean')
    data['P_mean'] = fast_exp_map(adj_matrix, np.array(data['P']), measure='mean')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                 orient='index').rename(columns={0: 'F'}),
                    how='left', left_index=True, right_index=True)

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(-1.5 + 0.05*(data['B'] - 30) + 2*data['G']*data['E']
                        + 1.*data['E_mean'] + 1.*data['G_mean'] + 0.05*data['F'])
    diet = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['diet'] = diet
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='diet',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['diet']))

    data['diet_sum'] = fast_exp_map(adj_matrix, np.array(data['diet']), measure='sum')
    data['diet_t3'] = np.where(data['diet_sum'] > 3, 1, 0)

    # Running Data Generating Mechanism for Y
    bmi = (3.9 + data['B'] - 3*data['diet'] - 2*data['diet_t3'] - 2*data['P']
           + 2*data['G'] - 2*data['E'] - 1.*data['E_sum'] - 0.75*data['G_sum']
           + data['B_mean_dist'] + 0.2*data['F']
           + 3*np.where(data['P_mean'] > 0.4, 1, 0)  # latent variable for contacts
           + np.random.normal(0, scale=1, size=nx.number_of_nodes(graph)))
    data['bmi'] = bmi

    # Adding node information back to graph
    for n in graph.nodes():
        graph.nodes[n]['diet'] = int(data.loc[data.index == n, 'diet'].values)
        graph.nodes[n]['bmi'] = float(data.loc[data.index == n, 'bmi'].values)

    return graph


def diet_dgm_truth(network, pr_a, restricted=False, shift=False):
    graph = network.copy()
    data = network_to_df(graph)
    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['B_mean_dist'] = fast_exp_map(adj_matrix, np.array(data['B']), measure='mean_dist')
    data['G_mean'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='mean')
    data['G_sum'] = fast_exp_map(adj_matrix, np.array(data['G']), measure='sum')
    data['E_sum'] = fast_exp_map(adj_matrix, np.array(data['E']), measure='sum')
    data['E_mean'] = fast_exp_map(adj_matrix, np.array(data['E']), measure='mean')
    data['P_mean'] = fast_exp_map(adj_matrix, np.array(data['P']), measure='mean')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                 orient='index').rename(columns={0: 'F'}),
                    how='left', left_index=True, right_index=True)

    if shift:  # If a shift in the Odds distribution is instead specified
        prob = logistic.cdf(-1.5 + 0.05 * (data['B'] - 30) + 2 * data['G'] * data['E']
                            + 1. * data['E_mean'] + 1. * data['G_mean'] + 0.05 * data['F'])
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    diet = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['diet'] = diet
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='diet',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['diet']))
        exclude = list(attrs.keys())

    # Running Data Generating Mechanism for Y
    data['diet_sum'] = fast_exp_map(adj_matrix, np.asarray(data['diet']), measure='sum')
    data['diet_t3'] = np.where(data['diet_sum'] > 3, 1, 0)

    bmi = (3.9 + data['B'] - 3*data['diet'] - 2*data['diet_t3'] - 2*data['P']
           + 2*data['G'] - 2*data['E'] - 1.*data['E_sum'] - 0.75*data['G_sum']
           + data['B_mean_dist'] + 0.2*data['F']
           + 3*np.where(data['P_mean'] > 0.4, 1, 0)  # latent variable for contacts
           + np.random.normal(0, scale=1, size=nx.number_of_nodes(graph)))

    if restricted:
        data['bmi'] = bmi
        data = data.loc[~data.index.isin(exclude)].copy()
        bmi = np.array(data['bmi'])

    return np.mean(bmi)
