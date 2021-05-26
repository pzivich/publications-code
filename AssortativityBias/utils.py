###################################################################################################################
# Python Code for "Treatment assortativity in epidemiologic studies of contagious outcomes"
#
# File containing background functions for simulations
#
# Outputs
#   None
#
###################################################################################################################

import random
import numpy as np
import pandas as pd
import networkx as nx


def induce_homophily(comm_data, g, percent):
    vaccine = []
    values = np.ndarray.tolist(comm_data[g].unique())  # Extracts the unique community values
    random.shuffle(values)  # Randomly order the number of communities
    for x, vr in zip(values, percent):  # For loop through communities and assigned percentages
        ss = comm_data.loc[comm_data[g] == x]  # Subset out only nodes within that community
        vnodes = ss.sample(frac=vr).index  # Randomly select node IDs by the corresponding fraction
        vnodes = list(vnodes)  # Convert to list object
        vaccine += vnodes  # Append list object to the entire list of node IDs determined to be vaccinated
    return vaccine


def one_step(graph, label, var):
    # get adjacency matrix
    matrix = nx.adjacency_matrix(graph, weight=None)
    # calculate row-sum (denominator for one-step)
    rowsum_vector = np.sum(matrix, axis=1)
    # calculate each nodes weight (divide by row-sum
    weight_matrix = matrix / rowsum_vector.reshape((matrix.shape[0]), 1)
    # get node attributes
    y_vector = np.array(list(nx.get_node_attributes(graph, name=var).values()))
    # multiply the weight matrix by node attributes
    wy_matrix = weight_matrix * y_vector.reshape((matrix.shape[0]), 1)
    return pd.DataFrame(wy_matrix, index=(graph.nodes()), columns=[label])


def two_step(graph, var, label):
    # Dictionary object to store all results
    degstr = {el: 0 for el in graph.nodes()}

    # Looping through all potential nodes
    for n in graph.nodes():
        # Select all neighbors of node i
        G_neighbors = graph[n]
        zv2 = []
        # Looping through all neighbors connected to node i
        for n2 in G_neighbors:
            # all node j neighbors
            G2_neighbors = graph[n2]
            v2 = 0
            t2 = 0
            # Looping through all neighbors of node j
            for n3 in G2_neighbors:
                # if node k is vaccinated and not node i
                if n3 != n and graph.node[n3][var] == 1:
                    # Adds to numerator and denominator
                    v2 += 1
                    t2 += 1
                # if node k is not node i (and not vaccinated)
                elif n3 != n:
                    # Adds to denominator only
                    t2 += 1
                # ignore the remaining nodes
                else:
                    pass
            # Try to divide numerator by denominator
            try:
                zv2.append((v2 / t2))
            # If can't divide, then add np.nan since denominator = 0
            except:
                zv2.append(np.nan)
        # Take the mean of all neighbors
        degstr[n] = np.nanmean(zv2)
    twostep = pd.DataFrame(index=graph.nodes())
    twostep[label] = pd.Series(degstr)
    return twostep


def spline(df, var, knots=None, term=1):
    if knots is None:
        knots = [0.05, 0.35, 0.65, 0.95]
        pts = list(df[var].quantile(q=knots))
    else:
        pts = knots
    colnames = []
    sf = df.copy()
    for i in range(len(pts)):
        colnames.append('spline' + str(i))
        sf['spline' + str(i)] = np.where(sf[var] > pts[i], (sf[var] - pts[i]) ** term, 0)
        sf['spline' + str(i)] = np.where(sf[var].isnull(), np.nan, sf['spline' + str(i)])
    rsf = sf.copy()
    colnames = []
    for i in range(len(pts) - 1):
        colnames.append('rspline' + str(i))
        rsf['rspline' + str(i)] = np.where(rsf[var] > pts[i],
                                           rsf['spline' + str(i)] - rsf['spline' + str(len(pts) - 1)], 0)
        rsf['rspline' + str(i)] = np.where(rsf[var].isnull(), np.nan, rsf['rspline' + str(i)])
    return rsf[colnames]


