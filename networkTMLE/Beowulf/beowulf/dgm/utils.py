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


def exposure_restrictions(network, exposure, n):
    if network == 'uniform':
        raise ValueError("No support for restricted-by-degree for the uniform network")

    elif network == 'random':
        if exposure == 'statin':
            if n == 500:
                attrs = {3: {exposure: 1}, 121: {exposure: 0}, 140: {exposure: 0}, 149: {exposure: 0}, 196: {exposure: 0},
                         247: {exposure: 1}, 279: {exposure: 1}, 381: {exposure: 1}, 390: {exposure: 1}, 391: {exposure: 1}, }
            elif n == 1000:
                attrs = {1: {exposure: 0}, 72: {exposure: 0}, 311: {exposure: 1}, 327: {exposure: 1}, 331: {exposure: 1},
                         334: {exposure: 0}, 517: {exposure: 0}, 613: {exposure: 1}, 618: {exposure: 1}, 658: {exposure: 1},
                         783: {exposure: 1}, 816: {exposure: 0}, 820: {exposure: 0}, 910: {exposure: 1}, 960: {exposure: 1}, }
            elif n == 2000:
                attrs = {1: {exposure: 0}, 118: {exposure: 0}, 306: {exposure: 1}, 321: {exposure: 0},
                         410: {exposure: 0}, 411: {exposure: 1}, 450: {exposure: 0}, 530: {exposure: 0},
                         567: {exposure: 0}, 571: {exposure: 0}, 602: {exposure: 0}, 606: {exposure: 0},
                         641: {exposure: 0}, 661: {exposure: 0}, 681: {exposure: 1}, 690: {exposure: 1},
                         691: {exposure: 1}, 729: {exposure: 1}, 821: {exposure: 0}, 836: {exposure: 0},
                         858: {exposure: 0}, 901: {exposure: 0}, 939: {exposure: 1}, 973: {exposure: 0},
                         1006: {exposure: 1}, 1038: {exposure: 0}, 1062: {exposure: 0}, 1202: {exposure: 0},
                         1225: {exposure: 0}, 1347: {exposure: 0}, 1348: {exposure: 0}, 1423: {exposure: 0},
                         1424: {exposure: 0}, 1679: {exposure: 1}, 1755: {exposure: 0}, 1957: {exposure: 0}}
            else:
                raise ValueError("Invalid N for the network")
        elif exposure == 'naloxone':
            if n == 500:
                attrs = {3: {exposure: 0}, 121: {exposure: 1}, 140: {exposure: 0}, 149: {exposure: 1}, 196: {exposure: 1},
                         247: {exposure: 0}, 279: {exposure: 1}, 381: {exposure: 1}, 390: {exposure: 0}, 391: {exposure: 1}, }
            elif n == 1000:
                attrs = {1: {exposure: 0}, 72: {exposure: 0}, 311: {exposure: 0}, 327: {exposure: 0}, 331: {exposure: 0},
                         334: {exposure: 0}, 517: {exposure: 1}, 613: {exposure: 1}, 618: {exposure: 1}, 658: {exposure: 0},
                         783: {exposure: 0}, 816: {exposure: 1}, 820: {exposure: 0}, 910: {exposure: 0}, 960: {exposure: 0}, }
            elif n == 2000:
                attrs = {1: {exposure: 0}, 118: {exposure: 1}, 306: {exposure: 1}, 321: {exposure: 0},
                         410: {exposure: 0}, 411: {exposure: 0}, 450: {exposure: 0}, 530: {exposure: 0},
                         567: {exposure: 0}, 571: {exposure: 1}, 602: {exposure: 0}, 606: {exposure: 0},
                         641: {exposure: 1}, 661: {exposure: 0}, 681: {exposure: 1}, 690: {exposure: 0},
                         691: {exposure: 0}, 729: {exposure: 1}, 821: {exposure: 0}, 836: {exposure: 0},
                         858: {exposure: 0}, 901: {exposure: 0}, 939: {exposure: 0}, 973: {exposure: 0},
                         1006: {exposure: 0}, 1038: {exposure: 1}, 1062: {exposure: 0}, 1202: {exposure: 1},
                         1225: {exposure: 1}, 1347: {exposure: 1}, 1348: {exposure: 0}, 1423: {exposure: 0},
                         1424: {exposure: 0}, 1679: {exposure: 0}, 1755: {exposure: 0}, 1957: {exposure: 0}}
            else:
                raise ValueError("Invalid N for the network")
        elif exposure == 'diet':
            if n == 500:
                attrs = {3: {exposure: 0}, 121: {exposure: 1}, 140: {exposure: 0}, 149: {exposure: 1}, 196: {exposure: 1},
                         247: {exposure: 0}, 279: {exposure: 0}, 381: {exposure: 1}, 390: {exposure: 1}, 391: {exposure: 0}, }
            elif n == 1000:
                attrs = {1: {exposure: 1}, 72: {exposure: 0}, 311: {exposure: 0}, 327: {exposure: 1}, 331: {exposure: 1},
                         334: {exposure: 1}, 517: {exposure: 0}, 613: {exposure: 1}, 618: {exposure: 0}, 658: {exposure: 0},
                         783: {exposure: 1}, 816: {exposure: 1}, 820: {exposure: 1}, 910: {exposure: 1}, 960: {exposure: 0}, }
            elif n == 2000:
                attrs = {1: {exposure: 0}, 118: {exposure: 0}, 306: {exposure: 0}, 321: {exposure: 0},
                         410: {exposure: 0}, 411: {exposure: 0}, 450: {exposure: 1}, 530: {exposure: 1},
                         567: {exposure: 1}, 571: {exposure: 1}, 602: {exposure: 1}, 606: {exposure: 0},
                         641: {exposure: 0}, 661: {exposure: 0}, 681: {exposure: 1}, 690: {exposure: 1},
                         691: {exposure: 1}, 729: {exposure: 0}, 821: {exposure: 1}, 836: {exposure: 1},
                         858: {exposure: 1}, 901: {exposure: 0}, 939: {exposure: 1}, 973: {exposure: 0},
                         1006: {exposure: 1}, 1038: {exposure: 0}, 1062: {exposure: 1}, 1202: {exposure: 0},
                         1225: {exposure: 0}, 1347: {exposure: 0}, 1348: {exposure: 1}, 1423: {exposure: 1},
                         1424: {exposure: 1}, 1679: {exposure: 0}, 1755: {exposure: 0}, 1957: {exposure: 1}}
            else:
                raise ValueError("Invalid N for the network")
        elif exposure == 'vaccine':
            if n == 500:
                attrs = {3: {exposure: 1}, 121: {exposure: 1}, 140: {exposure: 0}, 149: {exposure: 0}, 196: {exposure: 1},
                         247: {exposure: 1}, 279: {exposure: 0}, 381: {exposure: 1}, 390: {exposure: 0}, 391: {exposure: 0}, }
            elif n == 1000:
                attrs = {1: {exposure: 0}, 72: {exposure: 1}, 311: {exposure: 1}, 327: {exposure: 0}, 331: {exposure: 0},
                         334: {exposure: 0}, 517: {exposure: 1}, 613: {exposure: 0}, 618: {exposure: 0}, 658: {exposure: 0},
                         783: {exposure: 1}, 816: {exposure: 0}, 820: {exposure: 1}, 910: {exposure: 1}, 960: {exposure: 0}, }
            elif n == 2000:
                attrs = {1: {exposure: 1}, 118: {exposure: 1}, 306: {exposure: 0}, 321: {exposure: 1},
                         410: {exposure: 0}, 411: {exposure: 0}, 450: {exposure: 0}, 530: {exposure: 0},
                         567: {exposure: 0}, 571: {exposure: 0}, 602: {exposure: 0}, 606: {exposure: 0},
                         641: {exposure: 0}, 661: {exposure: 0}, 681: {exposure: 0}, 690: {exposure: 1},
                         691: {exposure: 1}, 729: {exposure: 0}, 821: {exposure: 0}, 836: {exposure: 1},
                         858: {exposure: 0}, 901: {exposure: 0}, 939: {exposure: 1}, 973: {exposure: 1},
                         1006: {exposure: 1}, 1038: {exposure: 0}, 1062: {exposure: 0}, 1202: {exposure: 0},
                         1225: {exposure: 0}, 1347: {exposure: 0}, 1348: {exposure: 0}, 1423: {exposure: 0},
                         1424: {exposure: 0}, 1679: {exposure: 0}, 1755: {exposure: 1}, 1957: {exposure: 0}}
            else:
                raise ValueError("Invalid N for the network")
        else:
            raise ValueError("Invalid exposure argument")
    else:
        raise ValueError("Invalid network label")
    return attrs
