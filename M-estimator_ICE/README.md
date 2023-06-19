# Empirical sandwich variance estimator for iterated conditional expectation g-computation

### Paul N Zivich, Rachael K Ross, Bonnie E Shook-Sa, Stephen R Cole, Jessie K Edwards

--------------------------------

## Abstract

Iterated conditional expectation (ICE) g-computation is an estimation approach for addressing time-varying confounding
for both longitudinal and time-to-event data. Unlike other g-computation implementations, ICE avoids the need to specify
models for each time-varying covariate. For variance estimation, previous work has suggested the bootstrap. However,
bootstrapping can be computationally intense and sensitive to the number of resamples used. Here, we present ICE
g-computation as a set of stacked estimating equations. Therefore, the variance for the ICE g-computation estimator
can be estimated using the empirical sandwich variance estimator. Performance of the variance estimator was evaluated
empirically with a simulation study. The proposed approach is also demonstrated with an illustrative example on the
effect of cigarette smoking on the prevalence of hypertension. In the simulation study, the empirical sandwich variance
estimator appropriately estimated the variance. When comparing runtimes between the sandwich variance estimator and the
bootstrap for the applied example, the sandwich estimator was substantially faster, even when bootstraps were run in
parallel. The empirical sandwich variance estimator is a viable option for variance estimation with ICE g-computation.

--------------------------------

## File Manifesto

### applications/

`data_example.py`
- Apply the ICE g-computation estimator for the applied Add Health example in Python. The first step also generates the
  fully formatted data (`addhealth_processed.csv`) which is used in the `.R` file.

`data_example.R`
- Apply the ICE g-computation estimator for the applied Add Health example in R.

`data_timed.py`
- Apply the ICE g-computation estimator for the applied Add Health example and compare runtimes for variance estimation
  between the emprical sandwich variance estimator and the nonparametric bootstrap.

`efuncs.py`
- Define the estimating functions for ICE g-computation used in the applied example and the simulations.

`funcs.py`
- Define utility functions for data processing for the applied example and simulations.

`process_simulation.py`
- Process and summarize the generated simulation result files.

`simulation.py`
- Run the simulation experiment for the designated sample size.

### data/

`data_clean.py`
- Processes the raw data from the publicly-available Add Health data sets into the data used for the applied example.
  Data is available at https://dataverse.unc.edu/dataverse/odum

`data_describe.py`
- Generate the descriptive tables from the cleaned data set.
