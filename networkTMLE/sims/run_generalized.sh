#!/bin/bash

# .sh script to run any of the previous generalized_*.py files
# To run the corresponding file, the last line needs to be changed to the generalized_*.py of interest

# Specifies the network to use. Options are: uniform, random
network="uniform"
# Setup for the simulation. Options:
#                                 1 - runs the IID-TMLE estimator
#                                 2 - if uniform network-TMLE with iterative logit models, if random then unrestricted
#                                 3 - if uniform network-TMLE with single model, if random then restricted by degree
setup=1
# Whether to set all probabilities to a constant (0), or look at shifts in the probabilities (1)
shift=0
# Double check that you IID-TMLE is desired (set to 0 to look at network-TMLE)
ind=1
# File name to save results as. Save output is disabled in the last line currently
save="save_file_name"

python -u generalized_statin.py "$network" "$setup" "$shift" "$ind" "$save"
