#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Simulating true average causal effect of action in the clinic population for simulation experiment
#
# Paul Zivich
#######################################################################################################################

import numpy as np
from dgm import calculate_truth

np.random.seed(8091141)

# Calculating true average causal effect
truth = calculate_truth(n=10000000)
print(truth)                          # 0.216697
