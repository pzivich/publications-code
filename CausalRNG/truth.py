import numpy as np

from dgm import data_generation

# Setup the RNG
rng = np.random.default_rng(seed=480214)

# Compute the truth with 10 million observations
truth = data_generation(n=10000000, rng=rng, truth=True)
print(truth)
