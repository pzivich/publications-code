import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)


def vaccine_dgm(network, restricted=False, n_init_infect=7, time_limit=10, inf_duration=5):
    """
    Parameters
    ----------
    network:
        input network
    restricted:
        whether to use the restricted treatment assignment
    n_init_infect:
        number of initial infections to start with
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
    data['H_mean'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='mean')

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 0.95*data['H'] + 1.2*data['H_mean'])
    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine')
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running outbreak simulation
    graph = _outbreak_(graph, n_init_infect, duration=inf_duration, limit=time_limit)
    return graph


def vaccine_dgm_truth(network, pr_a, shift=False, restricted=False,
                      n_init_infect=7, time_limit=10, inf_duration=5):
    graph = network.copy()
    data = network_to_df(graph)

    # Running Data Generating Mechanism for A
    if shift:  # If a shift in the Odds distribution is instead specified
        adj_matrix = nx.adjacency_matrix(graph, weight=None)
        data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
        data['H_mean'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='mean')
        prob = logistic.cdf(-1.9 + 1.75*data['A'] + 0.95*data['H'] + 1.2*data['H_mean'])
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine')
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running Data Generating Mechanism for Y
    graph = _outbreak_(graph, n_init_infect, duration=inf_duration, limit=time_limit)
    dis = []
    for nod, d in graph.nodes(data=True):
        dis.append(d['D'])
    return np.mean(dis)


def _outbreak_(graph, n_init_infect, duration=5, limit=25):
    """Outbreak simulation script in a single function"""
    all_ids = [n for n in graph.nodes()]
    prev = 0
    while (prev > 0.95) | (prev < 0.05):
        # Selecting initial infections
        init_inf = random.sample(all_ids, n_init_infect)

        # Adding node attributes
        for n, d in graph.nodes(data=True):
            d['R'] = 0
            d['t'] = 0
            if n in init_inf:  # Set initial infected nodes as infected
                d['I'] = 1
                d['D'] = 1
            else:
                d['I'] = 0  # Set all other nodes as uninfected
                d['D'] = 0

        # Running through infection cycle
        time = 0
        while time < limit:  # Simulate outbreaks until time-step limit is reached
            time += 1

            for n, d in sorted(graph.nodes(data=True), key=lambda x: random.random()):
                # Checking node infection / disease status
                if d['I'] == 1:
                    d['D'] = 1  # Disease status is yes
                    d['t'] += 1  # Increase infection duration counter
                # Checking duration of disease
                if d['t'] > duration:
                    d['I'] = 0  # Node is no longer infectious
                    d['R'] = 1  # Node switches to Recovered

                # Node "tries" to transmit infection to direct contacts
                for neighbor in graph[n]:
                    pr_y = logistic.cdf(-2.4
                                        - 1.5*graph.nodes[neighbor]['vaccine'] - 0.4*d['vaccine']
                                        + 1.5*graph.nodes[neighbor]['A']
                                        - 0.4*graph.nodes[neighbor]['H'])
                    graph.nodes[neighbor]['I'] = int(np.random.binomial(n=1, p=pr_y, size=1))

        dis = []
        for nod, d in graph.nodes(data=True):
            dis.append(d['D'])
        prev = np.mean(dis)

    return graph
