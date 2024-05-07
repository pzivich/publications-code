# Invited Commentary: The Seedy Side of Causal Effect Estimation with Machine Learning

### Paul N Zivich

--------------------------------

Due to the long computation time for each estimator, these simulations were structured to run massively in parallel.
Each simulation was ran as an independent job on a cluster using SLURM. The provided code follows this structure, but
one could adapt the provided code to run all simulations in sequence (would take weeks/months) or for other cluster
designs. The following is a description of the workflow.

To replicate the simulations, one would need to:
- Run `generate_data.py` to create all the data sets.
- Run the `run_iterations.sbatch` file on a SLURM-based cluster. This script runs the `single_scftmle.py` file for the
  requested data sets and SCFTMLE setup. In the `python3` call, the `SLURM_ARRAY_TASK_ID` designates which of the
  generated data sets to load, the second number designates the number of repetitions to use, and the third number
  designates the number of folds. So, the provided scripts creates 200 jobs (one for each simulated data set) and runs
  SCFTMLE for 30 repetitions with 5-folds. This script outputs a .csv file of the results across 50 applications to
  the same data set for each job (i.e., 200 .csv files are generated).
- Run `stack_iterations.py` to compile the 200 .csv files generated in the previous step into a single overall .csv
- Run `summarize.py` to create the figure. This script assumes that you have run all combinations of repetitions and
  folds for `run_iterations.sbatch` described in the main paper.

## Dependencies

You will need to have the following open-source libraries installed to replicate the simulations
- NumPy, SciPy, pandas, sci-kit learn, statsmodels, delicatessen, matplotlib

## File Manifesto

`dgm.py`
- Contains the data generating mechanism and the approximation of the Weierstrass function.

`estimator.py`
- Implementation of a single cross-fit targeted maximum likelihood estimator (SCFTMLE). This estimator uses a built-in
  super learner that uses 10-fold cross-validation with linear regression, LASSO, decision trees, and a neural network
  consisting of 3 hidden layers (with 10, 10, and 5 nodes, respectively). Note that this implementation is not meant for
  general use (i.e., it was designed for the simulation experiment specifically).

`generate_data.py`
- Generates each of the corresponding simulated data sets. Outputs a stacked .csv file of all data sets for the
  simulations. These data sets are accessed by `single_scftmle.py`.

`nuisance_estimators.py`
- Implementation of super-learner in Python and logistic regression that does not apply regularization and is
  compatible with `SuperLearner`.

`run_iterations.sbatch`
- File to run simulations for a particular number of folds and repetitions. This script is for a SLURM-based cluster
  submission system.

`single_scftmle.py`
- Runs the SCFTMLE procedure on a single generated data set from `generate_data.py`. This script is called by
  `run_iterations.sbatch` for each setup. Note that this script expects 4 CPUs to be available, as the 50 different
  RNG seeds are run in parallel to save time.

`stack_iterations.py`
- Takes output results from `single_scftmle.py` and combines them into a single .csv file.

`summarize.py`
- Generates the summary figure provided in the commentary.

`truth.py`
- Computes the true value from the data generating mechanism using 10 million observations.

