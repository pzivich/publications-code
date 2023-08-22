#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Process simulation results from saved file into a table then output as a .csv
#
# Paul Zivich
#######################################################################################################################


import numpy as np
import pandas as pd


def calculate_metrics(data, estimator):
    """Calculate the metrics of interest for the simulation

    Parameters
    ----------
    data : pandas.DataFrame
        Simulation output from the `run_estimators.py` file
    estimator :
        Column indicator for scenario (string defined below)

    Returns
    -------
    list
    """
    bias = np.mean(data['bias_' + estimator])
    cld = np.mean(data['cld_' + estimator])
    cover = np.mean(data['cover_' + estimator])
    return bias, cld, cover


def create_table(data):
    """Function to create a table of the simulation results

    Parameters
    ----------
    data : pandas.DataFrame
        Simulation output from the `run_estimators.py` file

    Returns
    -------
    pandas.DataFrame
    """
    # Creating blank DataFrame
    table = pd.DataFrame(columns=['Estimator', 'Bias', 'CLD', 'Coverage'])

    # String indicators for all scenarios explored
    estrs = ['rtp_g', 'rtp_w', 'rcs_g', 'rcs_w',
             'pr1_g', 'pr1_w', 'pr2_g', 'pr2_w',
             'pr3_g', 'pr3_w', 'pr4_g', 'pr4_w',
             'pr5_g', 'pr5_w', 'pr6_g', 'pr6_w',
             'pr7_g', 'pr7_w', 'pr8_g', 'pr8_w',
             'pr9_g', 'pr9_w', 'pr10_g', 'pr10_w',
             'pr11_g', 'pr11_w', 'pr12_g', 'pr12_w']

    # Calculating metrics for each estimator, then adding as a row in the table
    for estr in estrs:
        bias, cld, cover = calculate_metrics(data=data, estimator=estr)
        table.loc[len(table.index)] = [estr, bias, cld, cover]

    # Return processed simulation results
    return table


# Running simulation processing
for n0 in ["500", "1000"]:
    d = pd.read_csv("sim_results_n1-1000_n0-"+n0+".csv")
    tb1 = create_table(data=d)
    tb1.to_csv("results_table_n0-"+n0+".csv", index=False)
