#####################################################################################################################
# Code to generate summary tables of the simulation results
#####################################################################################################################

import numpy as np
import pandas as pd


def sim_metrics(result, bias, se, coverage):
    b = np.mean(result[bias])
    ese = np.var(result[bias]) ** 0.5
    ase = np.mean(result[se]**2) ** 0.5
    ser = ase / ese
    c = np.mean(result[coverage])
    return b, ese, ser, c


def sim_results_table(result, estimators, bias, se, coverage):
    results = pd.DataFrame(columns=['bias', 'ese', 'ser', 'coverage'])
    for j in range(len(estimators)):
        row = sim_metrics(result=result, bias=bias[j], se=se[j], coverage=coverage[j])
        results.loc[len(results.index)] = row
    results['estimator'] = estimators
    return results.set_index('estimator')
