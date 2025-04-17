######################################################################################################################
# Code to simulate the true values for the simulation experiments
#
# Paul Zivich (Last update: 2025/4/17)
######################################################################################################################

import numpy as np
from lifelines import KaplanMeierFitter

from dgm import dgm

np.random.seed(999878)

d = dgm(n=10000000, truth=True)

km1 = KaplanMeierFitter()
km1.fit(d['T1_star'], d['delta1'])
risk1 = 1-km1.survival_function_at_times([10, 20, 30])
km0 = KaplanMeierFitter()
km0.fit(d['T0_star'], d['delta0'])
risk0 = 1-km0.survival_function_at_times([10, 20, 30])
true_rd = risk1 - risk0
print(risk1)
print(risk0)
print(true_rd)
true_rd.to_csv("truth.csv")

# from the .csv
# ,KM_estimate
# 10,0.131868799999999
# 20,0.11401150000000015
# 30,0.056044699999999836
