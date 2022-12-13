import csv
import numpy as np

from chimera import SurvivalFusionIPW
from sim_utils import generate_dataset, generate_truth, generate_super_population, create_empty_csv

# Simulation details
n1 = 1000
n0 = 1000
np.random.seed(60610712 + n1)

if __name__ == "__main__":
    # Calculating truth with super-population
    d = generate_super_population(n=20000000)
    ds1 = d.loc[d['S'] == 1]
    truth_91 = generate_truth(data=ds1, t_end=91)
    truth_183 = generate_truth(data=ds1, t_end=183)
    truth_274 = generate_truth(data=ds1, t_end=274)
    truth_365 = generate_truth(data=ds1, t_end=365)
    # print("Truth:", truth_91, truth_183, truth_274, truth_365)

    # Generating empty csv to store results
    create_empty_csv(n1=n1, n0=n0)

    # Running simulations
    for i in range(1000):
        # list storage of results to output to csv
        iter_results = []

        # Creating observed data set
        d = generate_super_population(n=3*(n1+n0))
        data = generate_dataset(data_s1=d.loc[d['S'] == 1].copy(),
                                data_s0=d.loc[d['S'] == 0].copy(),
                                n_1=n1, n_0=n0)

        # Going through estimator sampling model combinations
        for j in ["cd4", "idu + cd4"]:
            fipw = SurvivalFusionIPW(df=data, treatment='art',
                                     outcome='delta', time='t',
                                     sample='study', censor='censor', verbose=False)
            fipw.sampling_model(j, bound=0.01)
            fipw.treatment_model(model="1", bound=None)
            fipw.censoring_model("", censor_shift=1e-1, bound=None, stratify_by_sample=False)
            ans = fipw.estimate(variance="bootstrap", bs_iterations=500, n_cpus=30)
            pvalue = fipw.permutation_test(permutation_n=1000, n_cpus=30,
                                           print_results=False, plot_results=False)

            # Storing results for selected time points
            for day, tr in zip([91, 183, 274, 365], [truth_91, truth_183, truth_274, truth_365]):
                rdd = ans.loc[ans['t'] <= day]

                # Risk difference
                iter_results.append(rdd.iloc[-1, 1] - tr[0])  # Adding bias
                iter_results.append(rdd.iloc[-1, 2])          # Adding variance
                ucl = rdd.iloc[-1, 4]                         # Extract upper confidence limit
                lcl = rdd.iloc[-1, 3]                         # Extract lower confidence limit
                iter_results.append(ucl - lcl)                # Adding confidence limit difference
                if lcl <= tr[0] <= ucl:                       # Calculating confidence interval coverage
                    iter_results.append(1)
                else:
                    iter_results.append(0)

            # adding p-value (only once since for entire fusion)
            iter_results.append(pvalue)

        # Each iteration ends with adding to the last list of a csv file (less things needs to be actively in memory)
        with open("results/sim_n1-"+str(n1)+"_n0-"+str(n0)+".csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(iter_results)

        # Option to keep track of progress (takes long time to run)
        # print("COMPLETED:", i+1)
