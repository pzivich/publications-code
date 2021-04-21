import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)


def statin_dgm(network, restricted=False):
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

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(-5.3 + 0.2*data['L'] + 0.15*(data['A'] - 30)
                        + 0.4 * np.where(data['R_1'] == 1, 1, 0)
                        + 0.9 * np.where(data['R_2'] == 2, 1, 0)
                        + 1.5 * np.where(data['R_3'] == 3, 1, 0))
    statin = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['statin'] = statin

    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='statin')
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['statin']))

    # Running Data Generating Mechanism for Y
    pr_y = logistic.cdf(-5.05 - 0.8*data['statin'] + 0.37*(np.sqrt(data['A']-39.9))
                        + 0.75*data['R'] + 0.75*data['L'])
    cvd = np.random.binomial(n=1, p=pr_y, size=nx.number_of_nodes(graph))
    data['cvd'] = cvd

    # Adding node information back to graph
    for n in graph.nodes():
        graph.nodes[n]['statin'] = int(data.loc[data.index == n, 'statin'].values)
        graph.nodes[n]['cvd'] = float(data.loc[data.index == n, 'cvd'].values)

    return graph


def statin_dgm_truth(network, pr_a, shift=False, restricted=False):
    graph = network.copy()
    data = network_to_df(graph)

    # Running Data Generating Mechanism for A
    if shift:  # If a shift in the Odds distribution is instead specified
        prob = logistic.cdf(-5.3 + 0.2 * data['L'] + 0.15 * (data['A'] - 30)
                            + 0.4 * np.where(data['R_1'] == 1, 1, 0)
                            + 0.9 * np.where(data['R_2'] == 2, 1, 0)
                            + 1.5 * np.where(data['R_3'] == 3, 1, 0))
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    statin = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['statin'] = statin

    if restricted:  # removing other observations from the restricted set
        attrs = exposure_restrictions(network=network.graph['label'], exposure='statin')
        exclude = list(attrs.keys())
        data = data.loc[~data.index.isin(exclude)].copy()

    # Running Data Generating Mechanism for Y
    pr_y = logistic.cdf(-5.05 - 0.8*data['statin'] + 0.37*(np.sqrt(data['A']-39.9))
                        + 0.75*data['R'] + 0.75*data['L'])
    cvd = np.random.binomial(n=1, p=pr_y, size=data.shape[0])
    return np.mean(cvd)
