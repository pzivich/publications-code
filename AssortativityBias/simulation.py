###################################################################################################################
# Python Code for "Assortativity can lead to bias in epidemiologic studies of contagious outcomes: a simulated
#                  example in the context of vaccination"
#
# Paul N Zivich, Alexander Volfovsky, James Moody, Allison E Aiello
#
# Simulation Parameters
#   PrI:            prob transmission down a single edge at each time point for unvaccinated
#   bVE:            relative reduction in transmission prob for single exposure comparing vaccinated to unvac
#   duration:       times steps infected person is infectious for unvaccinated
#   duration_v:     times steps infected person is infectious for vaccinated
#   V_redinf:       relative reduction in infectiousness of vaccinated-but-infected cases compared to
#                   1 indicates the same between vac & unvac
#                   <1 is % reduction in infectiousness
#   limit:          total number of time steps to run outbreak simulation
#   sim:            total number of simulated outbreaks to run
#   perc_cover:     percentage of people to randomly vaccinate in the population
#   true_direct_ve: true direct effect. bias is calculated relative to this number
#   homophily:      whether to induce homophily/assorativity or vaccinate randomly
#   perc_cover:     if randomly vaccinating, the percent to randomly vaccinate
#   comm_cover:     if assortative vaccinating, the coverage for each community
#
# Files
#   netw_file:  points to the file containing the edge-list for the network
#   comm_file:  points to the file containing the community IDs for nodes
#   save_file:  file to save simulation results to
#
# Outputs
#   Data set with: bias, direct effect, standard error, confidence limit ratio, confidence interval coverage
#
# Paul Zivich
###################################################################################################################

import random
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import induce_homophily, one_step, two_step, spline

##################################################
# Setting simulation parameters                  #
##################################################

# Infection simulation parameters
PrI = 0.07
bVE = 0.60
duration = 5
duration_v = 3
V_redinf = 0.75

# Vaccination distribution parameters
homophily = True
perc_cover = 0.25
comm_cover = [0.7, 0.3, 0.3, 0.3, 0.2, 0.2, 0.125, 0.1, 0.05]

# Meta simulation parameters
limit = 20
sim = 10000

# True value to compare to
true_direct_ve = 0.502

# File path for networks and where to save results
netw_file = 'stochblock.dat'
comm_file = 'stochblock_groups.dat'
save_file = 'saved_results.csv'

##################################################
# Data and variable preparation                  #
##################################################

# Calculating prob transmission down a single edge at each time point for vaccinated (based on `bVE` and `PrI`)
PrI_v = PrI - (PrI * bVE)

# Calculating true direct effect
true_direct_ve = np.log(true_direct_ve)

# Containers for simulation result storage
model_1 = [[], [], [], []]
model_2 = [[], [], [], []]
model_3 = [[], [], [], []]
model_4 = [[], [], [], []]
prevalence = []
assort = []
coverage = []

# Reading in edgelist data
dfg = pd.read_csv(netw_file, delimiter='\s', engine='python', names=['pid', 'cid'])

# Creating NetworkX Graph Object (undirect, single edge, no self-loops)
Q = nx.Graph()
Q.add_nodes_from(dfg['pid'])
Q.add_nodes_from(dfg['cid'])
for obs, con in zip(dfg['pid'], dfg['cid']):
    Q.add_edge(obs, con)

# Pulling graph data for usage in simulation loop
all_ids = [n for n in Q.nodes()]  # all nodes in network
size = max(Q.nodes())  # number of nodes in network

# Reading in community ID file
part = pd.read_csv(comm_file, sep='|')
part.set_index('id', inplace=True)

# Creating empty pandas dataframe with centrality measure
cent = pd.DataFrame(index=range(0, size))
for key, value in dict(Q.degree()).items():
    cent.loc[(cent.index == key), 'degree'] = value


meta = part.merge(cent, left_index=True, right_index=True)
failure = 0  # counting number of failures for outbreak thresholds


# Function to fit the regression models
def modeler(model, lists, linkdist=sm.families.family.Poisson()):
    global df, true_direct_ve
    try:
        # Modified Poisson Regression Model
        ind = sm.cov_struct.Independence()
        log = smf.gee(model, 'id', df, family=linkdist, cov_struct=ind).fit()

        # Estimated Direct Effect
        dvebeta = log.params[1]

        # Estimated Standard Error
        dvese = log.bse[1]

        # Estimated Confidence Intervals
        dlcl = log.conf_int().loc['Vac'][0]
        ducl = log.conf_int().loc['Vac'][1]
        if ((dlcl < true_direct_ve) & (ducl > true_direct_ve)):
            dciv = 1
        else:
            dciv = 0
        dclr = np.exp(ducl) / np.exp(dlcl)

        # Adding results to the end of storage lists
        lists[0].append(dvebeta)
        lists[1].append(dvese)
        lists[2].append(dciv)
        lists[3].append(dclr)

    # If model doesn't converge, add NaN to list
    except:
        lists[0].append(np.nan)
        lists[1].append(np.nan)
        lists[2].append(np.nan)
        lists[3].append(np.nan)


##################################################
# Outbreak Simulations                           #
##################################################

# Generating splines for degree
meta[['d1', 'd2']] = spline(meta, 'degree', knots=[3, 7, 12], term=2)

# Number of loops/simulations to conduct based on `sim`
for i in range(sim):
    # Set init prevalence to zero
    prev = 0

    # Outbreaks below a certain level are ignored for results
    while (prev > 0.95) | (prev < 0.05):
        # Simulation set-up
        if homophily:  # induce homophily by vaccination status
            v_node = induce_homophily(comm_data=meta, g='community', percent=comm_cover)
        else:  # randomly vaccinate with no treatment pattern
            v_node = random.sample(all_ids, int(size*perc_cover))

        r_node = random.sample(all_ids, 2)  # Randomly select two nodes as initial infections
        X = Q.copy()  # Copy of graph to simulate on
        for node, data in X.nodes(data=True):  # Set initial node attributes
            data['R'] = 0  # No nodes are recovered status
            data['t'] = 0  # Infection duration is zero
            if node in r_node:  # Set initial infected nodes as infected
                data['I'] = 1
                data['D'] = 1
            else:
                data['I'] = 0  # Set all other nodes as uninfected
                data['D'] = 0
            if node in v_node:  # Set selected nodes to be vaccinated
                data['V'] = 1
            else:
                data['V'] = 0  # Set all other nodes as unvaccinated

        # Outbreak simulation Procedure
        time = 0
        while time < limit:  # Simulate outbreaks until time-step limit is reached
            time += 1

            # Loop through each node and try to infect nodes in a random order
            for nod, d in sorted(X.nodes(data=True), key=lambda x: random.random()):

                # If the node is infected
                if d['I'] == 1:
                    d['D'] = 1  # Disease status is yes
                    d['t'] += 1  # Increase infection duration counter

                    # If infected node is unvaccinated
                    if d['V'] == 0:
                        # If the disease duration is at/above the max for unvaccinated
                        if d['t'] >= duration:
                            d['I'] = 0  # Node is no longer infectious
                            d['R'] = 1  # Node switches to Recovered
                    # If infected node is vaccinated
                    if d['V'] == 1:
                        # If the disease duration is at/above the max for vaccinated
                        if d['t'] >= duration_v:
                            d['I'] = 0  # Node is no longer infectious
                            d['R'] = 1  # Node becomes Recovered

                    # Node "tries" to transmit infection to contacts
                    neighbor = X[nod]  # Select all the neighbors of infected node
                    for neigh in neighbor:
                        # If infected node is vaccinated
                        if d['V'] == 1:
                            # If the neighbor is not infected/recovered
                            if (X.nodes[neigh]['I'] == 0) & (X.nodes[neigh]['R'] == 0):
                                # If neighbor is unvaccinated
                                if X.nodes[neigh]['V'] == 0:
                                    # See if random number is less than probability of becoming infected
                                    if np.random.uniform() / V_redinf < PrI:
                                        X.nodes[neigh]['I'] = 1  # Node becomes infected
                                # If neighbor is vaccinated
                                if X.nodes[neigh]['V'] == 1:
                                    # See if random number is less than probability of becoming infected
                                    if np.random.uniform() / V_redinf < PrI_v:
                                        X.nodes[neigh]['I'] = 1  # Node becomes infected
                        # If infected node is unvaccinated
                        if d['V'] == 0:
                            # If the neighbor is not infected/recovered
                            if (X.nodes[neigh]['I'] == 0) & (X.nodes[neigh]['R'] == 0):
                                # If the neighbor is unvaccinated
                                if X.nodes[neigh]['V'] == 0:
                                    # See if random number is less than probability of becoming infected
                                    if np.random.uniform() < PrI:
                                        X.nodes[neigh]['I'] = 1  # Node becomes infected
                                # If the neighbor is vaccinated
                                if X.nodes[neigh]['V'] == 1:
                                    # See if random number is less than probability of becoming infected
                                    if np.random.uniform() < PrI_v:
                                        X.nodes[neigh]['I'] = 1  # Node becomes infected

        # Count up the prevalence of Disease after time-step limit reached
        dis = []
        for nod, d in X.nodes(data=True):
            dis.append(d['D'])
        prev = sum(dis) / len(dis)  # Calculates the prevalence

    # Operations on outbreak past the threshold criteria #
    prevalence.append(prev)

    # Extracting information from all nodes (ID, vaccination status, ever-infected status)
    nodes = []
    ill = []
    vac = []
    for nod, d in X.nodes(data=True):
        nodes.append(nod)
        ill.append(d['D'])
        vac.append(d['V'])
    df = pd.DataFrame(index=nodes)  # Create pandas dataframe to store extracted node attributes
    df['Ill'] = ill
    df['Vac'] = vac

    # Network calculations
    dnv = one_step(X, 'zv_deg1', 'V')
    dnv2_2 = two_step(graph=X, var='V', label='zv2_2')

    # Data prep phase
    df = df.merge(dnv, left_index=True, right_index=True)
    df = df.merge(dnv2_2, left_index=True, right_index=True)
    df = df.merge(meta, left_index=True, right_index=True)
    df['zv_deg1'] *= 10
    df['zv2_2'] *= 10
    df[['osv2', 'osv3', 'osv4']] = spline(df, 'zv_deg1', term=2)
    df[['tsv2', 'tsv3', 'tsv4']] = spline(df, 'zv2_2', term=2)
    df['id'] = df.index
    dfdumm = pd.get_dummies(df['community'], 'g', prefix_sep='', drop_first=True)
    df = pd.concat([df, dfdumm], axis=1)

    # Model fitting phase
    modeler(model='Ill ~ Vac + zv_deg1 + osv2 + osv3 + osv4 + zv2_2 + tsv2 + tsv3 + tsv4 + degree + d1 + d2',
            lists=[model_1[0], model_1[1], model_1[2], model_1[3]])
    modeler(model='Ill ~ Vac + zv_deg1 + osv2 + osv3 + osv4 + degree + d1 + d2',
            lists=[model_2[0], model_2[1], model_2[2], model_2[3]])
    modeler(model='Ill ~ Vac + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8',
            lists=[model_3[0], model_3[1], model_3[2], model_3[3]])
    modeler(model='Ill ~ Vac',
            lists=[model_4[0], model_4[1], model_4[2], model_4[3]])

    # Saving simulation run meta data
    assort.append(nx.attribute_assortativity_coefficient(X, 'V'))
    coverage.append(np.mean(df['Vac']))


##################################################
# Compiling Results                              #
##################################################
print('========================================')
print('Assortativity:', np.mean(assort))
print('Coverage:', np.mean(coverage))
print('========================================')

dfr = pd.DataFrame()
dfr['m1_direct'] = model_1[0]
dfr['m1_dse'] = model_1[1]
dfr['m1_cover'] = model_1[2]
dfr['m1_clr'] = model_1[3]
dfr['m1_bias'] = dfr['m1_direct'] - true_direct_ve

dfr['m2_direct'] = model_2[0]
dfr['m2_dse'] = model_2[1]
dfr['m2_cover'] = model_2[2]
dfr['m2_clr'] = model_2[3]
dfr['m2_bias'] = dfr['m2_direct'] - true_direct_ve

dfr['m3_direct'] = model_3[0]
dfr['m3_dse'] = model_3[1]
dfr['m3_cover'] = model_3[2]
dfr['m3_clr'] = model_3[3]
dfr['m3_bias'] = dfr['m3_direct'] - true_direct_ve

dfr['m4_direct'] = model_4[0]
dfr['m4_dse'] = model_4[1]
dfr['m4_cover'] = model_4[2]
dfr['m4_clr'] = model_4[3]
dfr['m4_bias'] = dfr['m4_direct'] - true_direct_ve

print(dfr[['m1_bias', 'm2_bias', 'm3_bias', 'm4_bias']].describe())
print(dfr[['m1_cover', 'm2_cover', 'm3_cover', 'm4_cover']].describe())
print('========================================')

# Output results to file
dfr.to_csv(save_file, index=False)
