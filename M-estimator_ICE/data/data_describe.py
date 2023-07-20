####################################################################################################################
# Empirical sandwich variance estimator for iterated conditional expectation g-computation
#   Generate descriptive table for Add Health data
#
# Paul Zivich
####################################################################################################################

import numpy as np
import pandas as pd


def descriptive_stats(variable, type):
    """Helper function for descriptive statistics.
    """
    miss = np.sum(variable.isna())
    if type == 'binary':
        count = np.sum(variable)
        percent = count / np.sum(1 - variable.isna())
        return count, percent, miss
    if type == 'continuous':
        median = np.nanpercentile(variable, q=50)
        p25 = np.nanpercentile(variable, q=25)
        p75 = np.nanpercentile(variable, q=75)
        return median, p25, p75, miss
    if type == 'categorical':
        unique = np.unique(variable)
        total = np.sum(1 - variable.isna())
        output = []
        for u in unique:
            count = np.sum(variable == u)
            percent = count / total
            output.append((count, percent, u))
        return output, miss


def generate_table(data, variables, types):
    """Helper function to generate a table of descriptive statistics for a data set.
    """
    table_rows = range(len(variables))
    column_labs = ["_", "N (%)"]
    table = pd.DataFrame(columns=column_labs)

    for i in table_rows:
        v = variables[i]
        t = types[i]
        out = descriptive_stats(variable=data[v], type=t)
        fmt_m = "{0}"
        if t == 'binary':
            # Adding counts/% and then missing
            fmt = "{0:.0f} ({1:.0f}%)"
            row = pd.DataFrame([[v, fmt.format(out[0], out[1]*100)],
                                [v+"_miss", fmt_m.format(out[2])]],
                               columns=column_labs)
            table = pd.concat([table, row])
        if t == 'continuous':
            # Median [P25, P75]
            fmt = "{0:.0f} [{1:.0f}, {2:.0f}]"
            row = pd.DataFrame([[v, fmt.format(out[0], out[1], out[2])],
                                [v+"_miss", fmt_m.format(out[3])]],
                               columns=column_labs)
            table = pd.concat([table, row])
        if t == 'categorical':
            rows, miss = out[0], out[1]
            fmt = "{0:.0f} ({1:.0f}%)"
            store = []
            for r in rows:
                store.append([v+"_"+str(r[2]), fmt.format(r[0], r[1]*100)])
            store.append([v+"_miss", fmt_m.format(miss)])
            row = pd.DataFrame(store, columns=column_labs)
            table = pd.concat([table, row])

    # Formatting output table
    table = table.loc[table["N (%)"] != "0"].copy()          # Dropping missing rows for vars with no miss
    return table.sort_values(by="_").reset_index(drop=True)  # Sorting by alphabetical order to align miss


######################################
# Loading Data
d = pd.read_csv("addhealth.csv")

######################################
# Descriptive table

variables = {"age": "continuous",
             "tried_cigarette": "binary",
             "gender_w1": "binary",
             "race_w1": "categorical",
             "ethnic_w1": "binary",
             "educ_w1": "categorical",
             "height_w1": "continuous",
             "weight_w1": "continuous",
             "exercise_w1": "categorical",
             "srh_w1": "categorical",
             "depr_w1": "categorical",
             "alcohol_w1": "categorical",
             "cigarette_w1": "binary",
             }
k, v = list(variables.keys()), list(variables.values())
dtable_w1 = generate_table(data=d, variables=k, types=v)

variables = {"gender_w3": "binary",
             "race_w3": "categorical",
             "ethnic_w3": "binary",
             "educ_w3": "categorical",
             "height_w3": "continuous",
             "weight_w3": "continuous",
             "exercise_w3": "categorical",
             "srh_w3": "categorical",
             "depr_w3": "categorical",
             "alcohol_w3": "categorical",
             "hins_w3": "binary",
             "hbp_w3": "binary",
             "hmed_w3": "binary",
             "diabetes_w3": "binary",
             "cigarette_w3": "binary",
             }
k, v = list(variables.keys()), list(variables.values())
dtable_w3 = generate_table(data=d, variables=k, types=v)

variables = {"htn_w4": "binary",
             }
k, v = list(variables.keys()), list(variables.values())
dtable_w4 = generate_table(data=d, variables=k, types=v)

dtable = pd.concat([dtable_w1, dtable_w3, dtable_w4], ignore_index=True)

dtable.to_csv("data/table1.csv", index=False)
