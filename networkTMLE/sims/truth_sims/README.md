# Truth simulation scripts

Calculate the true values using logic based via slurm array IDs.

The slurm array ID's consist of 4-digit numbers with the following logic:
- The first number designates which network to use {1: uniform (n=500), 2: cpl (n=500), 4:uniform (n=1000),
  5: cpl (n=1000), 6: uniform (n=2000), 7: uniform (n=2000)}
- The second number designates whether a degree-restricted version of the network is used {0: no, 1: yes}
- The third number designates the policy type {0: set to single value, 1: shifts in log-odds}
- The fourth number is only needed as a placeholder for the logic (just always leave set as 1)
