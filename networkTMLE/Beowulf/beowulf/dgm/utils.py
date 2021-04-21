import warnings
import numpy as np
import pandas as pd
import networkx as nx


def probability_to_odds(prob):
    r"""Converts given probability (proportion) to odds
    """
    return prob / (1 - prob)


def odds_to_probability(odds):
    r"""Converts given odds to probability (proportion)
    """
    return odds / (1 + odds)


def exp_map(graph, var, measure):
    """Does the exposure mapping functionality with potentially different measures
    """
    # get adjacency matrix
    matrix = nx.adjacency_matrix(graph, weight=None)
    # get node attributes
    y_vector = np.array(list(nx.get_node_attributes(graph, name=var).values()))
    # multiply the weight matrix by node attributes
    if measure.lower() == 'sum':
        wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
        return np.asarray(wy_matrix).flatten()
    elif measure.lower() == 'mean':
        rowsum_vector = np.sum(matrix, axis=1)  # calculate row-sum (denominator / degree)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            weight_matrix = matrix / rowsum_vector.reshape((matrix.shape[0]), 1)  # calculate each nodes weight
        wy_matrix = weight_matrix * y_vector.reshape((matrix.shape[0]), 1)  # multiply matrix by node attributes
        return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...
    else:
        raise ValueError("The summary measure mapping" + str(measure) + "is not available")


def fast_exp_map(matrix, y_vector, measure):
    """Improved exposure mapping speed (doesn't need to parse adj matrix every time)"""
    if measure.lower() == 'sum':
        # multiply the weight matrix by node attributes
        wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
        return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...
    elif measure.lower() == 'mean':
        rowsum_vector = np.sum(matrix, axis=1)  # calculate row-sum (denominator / degree)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            weight_matrix = matrix / rowsum_vector.reshape((matrix.shape[0]), 1)  # calculate each nodes weight
        wy_matrix = weight_matrix * y_vector.reshape((matrix.shape[0]), 1)  # multiply matrix by node attributes
        return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...
    elif measure.lower() == 'var':
        a = matrix.toarray()  # Convert matrix to array
        a = np.where(a == 0, np.nan, a)  # filling non-edges with NaN's
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(a * y_vector, axis=1)
    elif measure.lower() == 'mean_dist':
        a = matrix.toarray()  # Convert matrix to array
        a = np.where(a == 0, np.nan, a)  # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector  # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(c.transpose(), axis=1)
    elif measure.lower() == 'var_dist':
        a = matrix.toarray()  # Convert matrix to array
        a = np.where(a == 0, np.nan, a)  # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector  # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(c.transpose(), axis=1)
    else:
        raise ValueError("The summary measure mapping" + str(measure) + "is not available")


def network_to_df(graph):
    """Convert node attributes to pandas dataframe
    """
    return pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')


def exposure_restrictions(network, exposure):
    if network == 'uniform':
        raise ValueError("No support for restricted-by-degree for the uniform network")

    elif network == 'random':
        if exposure == 'statin':
            attrs = {3: {exposure: 1}, 93: {exposure: 0}, 98: {exposure: 0}, 153: {exposure: 1}, 201: {exposure: 0},
                     202: {exposure: 0}, 203: {exposure: 0}, 350: {exposure: 1}, 353: {exposure: 1}, 358: {exposure: 1}}
        elif exposure == 'naloxone':
            attrs = {3: {exposure: 0}, 93: {exposure: 0}, 98: {exposure: 0}, 153: {exposure: 1}, 201: {exposure: 1},
                     202: {exposure: 1}, 203: {exposure: 0}, 350: {exposure: 1}, 353: {exposure: 0}, 358: {exposure: 1}}
        elif exposure == 'diet':
            attrs = {3: {exposure: 0}, 93: {exposure: 0}, 98: {exposure: 0}, 153: {exposure: 0}, 201: {exposure: 1},
                     202: {exposure: 0}, 203: {exposure: 0}, 350: {exposure: 1}, 353: {exposure: 0}, 358: {exposure: 0}}
        elif exposure == 'vaccine':
            attrs = {3: {exposure: 0}, 93: {exposure: 0}, 98: {exposure: 0}, 153: {exposure: 1}, 201: {exposure: 1},
                     202: {exposure: 1}, 203: {exposure: 0}, 350: {exposure: 1}, 353: {exposure: 0}, 358: {exposure: 1}}
        else:
            raise ValueError("Invalid exposure argument")

    else:
        raise ValueError("Invalid network label")
    return attrs
