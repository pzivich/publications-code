#!/bin/bash

# Specify the exposure and outcome model. Options include:
#                                             sofrygin : original dgm from Sofrygin & van der Laan 2017
#                                             modified : a modified version of sofrygin (different coefficients)
#                                             continuous : continuous Y
#                                             direct : unit-treatment effect only (still some network dependence)
#                                             indirect : spillover-effect only
#                                             independent : unit-treatment and NO network dependence
network="continuous"
# Specification of the models. Options include:
#                                             1 : both models are correctly specified
#                                             2 : only exposure models correctly specified
#                                             3 : only outcome model correctly specified
#                                             4 : neither of the models are correctly specified
setup=1
# Estimator for network data to use. Options include: gformula, iptw, tmle
estimator="gformula"
# Save file name
save="save_file_name"

module add python
export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:$PYTHONPATH
python3 -u sim_generalization.py "$network" "$setup" "$estimator" "$save"
