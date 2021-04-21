import numpy as np

from beowulf.dgm import naloxone_dgm_truth
from beowulf import load_uniform_naloxone, load_random_naloxone

n_sims_truth = 10000
treat_plan = [-2.5, -2.0, -1.5, -1.0, -0.5,
              0.5, 1.0, 1.5, 2.0, 2.5]
np.random.seed(30202001)

################################
# Uniform
################################
truth = {}
G = load_uniform_naloxone()
for t in treat_plan:
    ans = []
    for i in range(n_sims_truth):
        ans.append(naloxone_dgm_truth(network=G, pr_a=t, shift=True))

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
        ans.append(naloxone_dgm_truth(network=G, pr_a=t, shift=True))

    truth[t] = np.mean(ans)

print("===============================")
print("Clustered Power-Law")
print("-------------------------------")
print(truth)
print("===============================")
