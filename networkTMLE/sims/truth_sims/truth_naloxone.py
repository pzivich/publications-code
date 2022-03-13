from sys import argv
import numpy as np
import pandas as pd

from beowulf.dgm import naloxone_dgm_truth
from beowulf import load_uniform_naloxone, load_random_naloxone, simulation_setup

n_sims_truth = 10000
np.random.seed(20220109)

########################################
# Running through logic from .sh script
########################################
script_name, slurm_setup = argv
network, n_nodes, degree_restrict, shift, model, save = simulation_setup(slurm_id_str=slurm_setup)

# Loading correct  Network
if network == "uniform":
    G = load_uniform_naloxone(n=n_nodes)
if network == "random":
    G = load_random_naloxone(n=n_nodes)

# Marking if degree restriction is being applied
if degree_restrict is not None:
    restrict = True
else:
    restrict = False

if shift:
    treat_plan = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
else:
    treat_plan = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


print("#############################################")
print("Truth Sim Script:", slurm_setup)
print("=============================================")
print("Network:     ", network)
print("N-nodes:     ", n_nodes)
print("DGM:         ", 'naloxone')
print("Restricted:  ", restrict)
print("Shift:       ", shift)
print("#############################################")

########################################
# Running Truth
########################################

truth = {}
for t in treat_plan:
    ans = []
    for i in range(n_sims_truth):
        ans.append(naloxone_dgm_truth(network=G, pr_a=t, shift=shift, restricted=restrict))

    truth[t] = np.mean(ans)
    print(truth)

print("#############################################")
print(truth)
print("#############################################")
