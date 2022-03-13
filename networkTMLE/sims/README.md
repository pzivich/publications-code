# Simulation scripts

The `generalization_*.py` files are meant to be called using the correspond `run_generalized.sh` file. The `.sh` file is
written for use with slurm array submissions. The corresponding array ID is processed by an internal logic script
to setup the scenario. This was to make it easy to run multiples of scenarios in parallel and submit all the different
scenarios with ease.

The slurm array ID's consist of 5-digit numbers with the following logic:
- The first number designates which network to use {1: uniform (n=500), 2: cpl (n=500), 4:uniform (n=1000),
  5: cpl (n=1000), 6: uniform (n=2000), 7: uniform (n=2000)}
- The second number designates whether a degree-restricted version of the network is used {0: no, 1: yes}
- The third number designates the policy type {0: set to single value, 1: shifts in log-odds}
- The fourth number designates the model specification {1: both correct, 2: exposure-only, 3: outcome-only, 4: flexible}
- The fifth number corresponds to the ID of the sub parallel process. These should range from 0-5
