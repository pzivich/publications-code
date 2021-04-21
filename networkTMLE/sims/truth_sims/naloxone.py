import numpy as np

from beowulf.dgm import naloxone_dgm_truth
from beowulf import load_uniform_naloxone, load_random_naloxone

n_sims_truth = 10000
treat_plan = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
              0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
np.random.seed(30202001)

################################
# Uniform
################################
truth = {}
G = load_uniform_naloxone()
for t in treat_plan:
    ans = []
    for i in range(n_sims_truth):
        ans.append(naloxone_dgm_truth(network=G, pr_a=t))

    truth[t] = np.mean(ans)

print("===============================")
print("Uniform")
print("-------------------------------")
print(truth)
print("===============================")

################################
# Clustered Power-Law
################################
truth = {}
G = load_random_naloxone()
for t in treat_plan:
    ans = []
    for i in range(n_sims_truth):
        ans.append(naloxone_dgm_truth(network=G, pr_a=t))

    truth[t] = np.mean(ans)

print("===============================")
print("Clustered Power-Law")
print("-------------------------------")
print(truth)
print("===============================")
