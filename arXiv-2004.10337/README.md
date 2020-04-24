# Machine learning for causal inference: on the use of cross-fit estimators 

## Paul N Zivich, Alexander Breskin

### ABSTRACT
Modern causal inference methods allow machine learning to be used to weaken parametric modeling assumptions. However, 
the use of machine learning may result in bias and incorrect inferences due to overfitting. Cross-fit estimators have 
been proposed to eliminate this bias and yield better statistical properties.

We conducted a simulation study to assess the performance of several different estimators for the average causal 
effect (ACE). The data generating mechanisms for the simulated treatment and outcome included log-transforms, 
polynomial terms, and discontinuities. We compared singly-robust estimators (g-computation, inverse probability 
weighting) and doubly-robust estimators (augmented inverse probability weighting, targeted maximum likelihood 
estimation). Nuisance functions were estimated with parametric models and ensemble machine learning, separately. 
We further assessed cross-fit doubly-robust estimators.

With correctly specified parametric models, all of the estimators were unbiased and confidence intervals achieved 
nominal coverage. When used with machine learning, the cross-fit estimators substantially outperformed all of the 
other estimators in terms of bias, variance, and confidence interval coverage.

Due to the difficulty of properly specifying parametric models in high dimensional data, doubly-robust estimators with 
ensemble learning and cross-fitting may be the preferred approach for estimation of the ACE in most epidemiologic 
studies. However, these approaches may require larger sample sizes to avoid finite-sample issues.

**Full text:** 
https://arxiv.org/abs/2004.10337

## Code Structure

### Python
The main simulation files are contained in the `Python/` path. Simulations presented in the paper were completed using 
Python 3.5.1. The data generating mechanism is included in `Python/dgm.py` and the data was originally generated with 
Windows 10. Due to file size limits on GitHub, the original data is not included in this repository. 
`Python/super_learner.py` contains a generalized implementation of super-learner along with a function to stack the 
unfit estimators ready to use in simulations. All implemented estimators are contained and called from the 
`Python/estimators.py` file. Estimators are compatible with `sklearn`-style class objects to estimate nuisance 
functions. These estimators were only tested in the simulated data and may not be generalizable to all input data sets.
Check the `zepid` library for generalized versions.

Simulation files are labeled according to the estimator. `sim_single_example.py` runs the simulation on a single 
randomly selected  data set from all the generated data sets. `sim_gform.py`, `sim_iptw.py`, `sim_aipw.py`, 
`sim_tmle.py`, `sim_dcaipw.py`, and `sim_dctmle.py` run the corresponding simulation over the 2000 generated data sets. 
Each file runs the selected set-up (correct parametric model, main-terms parametric model, super-learner) and this is 
controlled via the `setup` variable. The double cross-fit estimators take a long time and should be broken into pieces
to run in parallel due to the extensive runtimes (I used `multiprocessing.Pool` on Linux).

#### Dependencies
Dependencies are provided for what the original simulations were completed with
- `numpy v1.17.2`
- `pandas v0.24.2`
- `sklearn v0.22.1`
- `patsy v0.5.1`
- `statsmodels v0.10.0`
- `scipy v1.3.1`
- `pygam v0.8.0`
- `zepid v0.8.1`

### R
Earlier versions of an R implementation of the estimators are available in the `R/` directory. These were not used to
run and conduct the final simulations. However, they were used in early versions of double-coding and checking the
various implementations.

#### Dependencies
- `SuperLearner`
- `purrr`
- `furrr`
- `dplyr`
- `tidyr`
- `tibble`
