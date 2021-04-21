import numpy as np
import networkx as nx
from scipy.stats import logistic

from .utils import exp_map


def sofrygin_observational(graph):
    """Simulates the exposure and outcome according to the mechanisms specified in Sofrygin & van der Laan 2017

    A ~ Bernoulli(expit(-1.2 + 1.5*W + 0.6*map(W)))
    Y ~ Bernoulli(expit(-2.5 + 1.5*W + 0.5*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 1.5*w + 0.6*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-2.5 + 1.5*w + 0.5*a + 1.5*a_s + 1.5*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def sofrygin_randomized(graph, p):
    """Obtains the truth values via simulations. Takes the generated graph and W distribution and applies treatment
    randomly

    A ~ Bernoulli(p)
    Y ~ Bernoulli(expit(-2.5 + 1.5*W + 0.5*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-2.5 + 1.5*w + 0.5*a + 1.5*a_s + 1.5*w_s), size=n)
    return np.mean(y)


def modified_observational(graph):
    """Simulates the exposure and outcome according to the mechanism similar to Sofrygin & van der Laan 2017

    A ~ Bernoulli(expit(-0.6 - 0.9*W + 0.8*map(W)))
    Y ~ Bernoulli(expit(-1.75 + 1.5*W - 0.5*A + 1.5*map(A) - 0.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a = np.random.binomial(n=1, p=logistic.cdf(-0.6 - 0.9*w + 0.8*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 1.5*w - 1.5*a + 1.5*a_s - 0.5*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def modified_randomized(graph, p):
    """Obtains the truth values via simulations. Takes the generated graph and W distribution and applies treatment
    randomly

    A ~ Bernoulli(p)
    Y ~ Bernoulli(expit(-2.5 + 1.5*W + 0.5*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 1.5*w - 1.5*a + 1.5*a_s - 0.5*w_s), size=n)
    return np.mean(y)


def continuous_observational(graph):
    """Simulates the exposure and outcome according to the mechanism similar to Sofrygin & van der Laan 2017

    A ~ Bernoulli(expit(-1.2 + 1.5*W + 0.8*map(W)))
    Y ~ 5 - 1.5*W + 1.5*A + 1.5*map(A) - 1.5*map(W) + N(0,1)

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 1.5*w + 0.4*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = exp_map(graph, 'A', measure='sum')
    y = 20 - 5*w + 5*a + 1.5*a_s + 1.5*w_s + np.random.normal(size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def continuous_randomized(graph, p):
    """Obtains the truth values via simulations. Takes the generated graph and W distribution and applies treatment
    randomly

    A ~ Bernoulli(p)
    Y ~ 5 - 1.5*W + 1.5*A + 1.5*map(A) - 1.5*map(W) + N(0,1)

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a_s = exp_map(graph, 'A', measure='sum')
    y = 20 - 5*w + 5*a + 1.5*a_s + 1.5*w_s + np.random.normal(size=n)
    return np.mean(y)


def direct_observational(graph):
    """Simulates the exposure and outcome according to the mechanism similar to Sofrygin & van der Laan 2017

    A ~ Bernoulli(expit(-0.6 - 0.9*W + 0.8*map(W)))
    Y ~ Bernoulli(expit(-1.75 + 1.5*W - 1.5*A + 0*map(A) - 0.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a = np.random.binomial(n=1, p=logistic.cdf(-0.6 - 0.9*w + 0.8*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 1.5*w - 1.75*a + 0*a_s + 1.5*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def direct_randomized(graph, p):
    """Obtains the truth values via simulations. Takes the generated graph and W distribution and applies treatment
    randomly

    A ~ Bernoulli(p)
    Y ~ Bernoulli(expit(-2.5 + 1.5*W - 1.5*A + 0*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 1.5*w - 1.75*a + 0*a_s + 1.5*w_s), size=n)
    return np.mean(y)


def indirect_observational(graph):
    """Simulates the exposure and outcome according to the mechanism similar to Sofrygin & van der Laan 2017

    A ~ Bernoulli(expit(-0.6 - 0.9*W + 0.8*map(W)))
    Y ~ Bernoulli(expit(-1.75 + 1.5*W - 0*A + 1.5*map(A) - 0.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a = np.random.binomial(n=1, p=logistic.cdf(-0.6 - 0.9*w + 0.8*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 1.5*w - 0*a - 1.5*a_s + 1.5*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def indirect_randomized(graph, p):
    """Obtains the truth values via simulations. Takes the generated graph and W distribution and applies treatment
    randomly

    A ~ Bernoulli(p)
    Y ~ Bernoulli(expit(-2.5 + 1.5*W - 0*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a_s = exp_map(graph, 'A', measure='sum')
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 1.5*w - 0*a - 1.5*a_s + 1.5*w_s), size=n)
    return np.mean(y)


def independent_observational(graph):
    """Generates data that is independent (i.e. network-TMLE is not necessary)
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=logistic.cdf(-1.3*w), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 2.0*w - 2.0*a), size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def independent_randomized(graph, p):
    """Generates randomized data that is independent (i.e. network-TMLE is not necessary)
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    y = np.random.binomial(n=1, p=logistic.cdf(-1.75 + 2.0*w - 2.0*a), size=n)
    return np.mean(y)


def threshold_observational(graph):
    """Simulates the exposure and outcome according to the mechanisms specified in Sofrygin & van der Laan 2017

    A ~ Bernoulli(expit(-1.2 + 1.5*W + 0.6*map(W)))
    Y ~ Bernoulli(expit(-2.5 + 1.5*W + 0.5*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a = np.random.binomial(n=1, p=logistic.cdf(-1.2 + 1.5*w + 0.6*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    a_s = exp_map(graph, 'A', measure='sum')
    a_s = np.where(a_s > 2, 1, 0)
    y = np.random.binomial(n=1, p=logistic.cdf(-2.5 + 1.5*w + 0.5*a + 1.5*a_s + 1.5*w_s), size=n)
    for node in graph.nodes():
        graph.node[node]['Y'] = y[node]

    return graph


def threshold_randomized(graph, p):
    """Obtains the truth values via simulations. Takes the generated graph and W distribution and applies treatment
    randomly

    A ~ Bernoulli(p)
    Y ~ Bernoulli(expit(-2.5 + 1.5*W + 0.5*A + 1.5*map(A) + 1.5*map(W)))

    Returns
    -------
    Network object with node attributes
    """
    n = len(graph.nodes())
    w = np.array([d['W'] for n, d in graph.nodes(data=True)])

    # Calculating map(W), generating A, and adding to network
    a = np.random.binomial(n=1, p=p, size=n)
    for node in graph.nodes():
        graph.node[node]['A'] = a[node]

    # Calculating map(A), generating Y, and adding to network
    w_s = exp_map(graph, 'W', measure='sum')
    a_s = exp_map(graph, 'A', measure='sum')
    a_s = np.where(a_s > 2, 1, 0)
    y = np.random.binomial(n=1, p=logistic.cdf(-2.5 + 1.5*w + 0.5*a + 1.5*a_s + 1.5*w_s), size=n)
    return np.mean(y)
