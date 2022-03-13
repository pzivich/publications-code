import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)


def vaccine_dgm(network, restricted=False,
                time_limit=10, inf_duration=5):
    """
    Parameters
    ----------
    network:
        input network
    restricted:
        whether to use the restricted treatment assignment
    time_limit:
        maximum time to let the outbreak go through
    inf_duration:
        duration of infection status in time-steps
    """
    graph = network.copy()
    data = network_to_df(graph)

    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                 orient='index').rename(columns={0: 'F'}),
                    how='left', left_index=True, right_index=True)

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H']
                        + 1.*data['H_sum'] + 1.3*data['A_sum'] - 0.65*data['F'])
    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running outbreak simulation
    graph = _outbreak_(graph, duration=inf_duration, limit=time_limit)
    return graph


def vaccine_dgm_truth(network, pr_a, shift=False, restricted=False,
                      time_limit=10, inf_duration=5):
    graph = network.copy()
    data = network_to_df(graph)

    # Running Data Generating Mechanism for A
    if shift:  # If a shift in the Odds distribution is instead specified
        adj_matrix = nx.adjacency_matrix(graph, weight=None)
        data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
        data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
        data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
        data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                     orient='index').rename(columns={0: 'F'}),
                        how='left', left_index=True, right_index=True)
        prob = logistic.cdf(-1.9 + 1.75 * data['A'] + 1. * data['H']
                            + 1. * data['H_sum'] + 1.3 * data['A_sum'] - 0.65 * data['F'])
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running Data Generating Mechanism for Y
    graph = _outbreak_(graph, duration=inf_duration, limit=time_limit)
    dis = []
    for nod, d in graph.nodes(data=True):
        dis.append(d['D'])
    return np.mean(dis)


def _outbreak_(graph, duration, limit):
    """Outbreak simulation script in a single function"""
    # Adding node attributes
    for n, d in graph.nodes(data=True):
        d['D'] = 0
        d['R'] = 0
        d['t'] = 0

    # Selecting initial infections
    all_ids = [n for n in graph.nodes()]
    # infected = random.sample(all_ids, 5)
    if len(all_ids) <= 500:
        infected = [4, 36, 256, 305, 443]
    elif len(all_ids) == 1000:
        infected = [4, 36, 256, 305, 443, 552, 741, 803, 825, 946]
    elif len(all_ids) == 2000:
        infected = [4, 36, 256, 305, 443, 552, 741, 803, 825, 946,
                    1112, 1204, 1243, 1253, 1283, 1339, 1352, 1376, 1558, 1702]
    else:
        raise ValueError("Invalid network IDs")

    # Running through infection cycle
    time = 0
    while time < limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        for inf in sorted(infected, key=lambda _: random.random()):
            # Book-keeping for infected nodes
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1
            if graph.nodes[inf]['t'] > duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)

            # Attempt infections of neighbors
            for contact in nx.neighbors(graph, inf):
                if graph.nodes[contact]["D"] == 1:
                    pass
                else:
                    pr_y = logistic.cdf(- 2.5
                                        - 1.0*graph.nodes[contact]['vaccine']
                                        - 0.2*graph.nodes[inf]['vaccine']
                                        + 1.0*graph.nodes[contact]['A']
                                        - 0.2*graph.nodes[contact]['H'])
                    if np.random.binomial(n=1, p=pr_y, size=1):
                        graph.nodes[contact]['I'] = 1
                        graph.nodes[contact]["D"] = 1
                        infected.append(contact)

    return graph
