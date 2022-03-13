# Targeted maximum likelihood estimation of causal effects with interference: a simulation study

## Paul N Zivich, Michael G Hudgens, M Alan Brookhart, James Moody, David J Weber, Allison E Aiello

### ABSTRACT

Interference, the dependency of an individual’s potential outcome on the exposure of other individuals, is a common
occurrence in medicine and public health. Recently, targeted maximum likelihood estimation (TMLE) has been extended to
settings of interference, including in the context of estimation of the overall mean of an outcome under a specified
distribution of exposure, referred to as a policy. This paper summarizes how TMLE for independent data is extended to
general interference (network-TMLE). An extensive simulation study is presented of network-TMLE, consisting of four
data generating mechanisms (unit-treatment effect only, spillover effects only, unit-treatment and spillover effects,
infection transmission) in networks of varying structures. Simulations show that network-TMLE performs well across
scenarios with interference but issues manifest when policies are not well-supported by the observed data, potentially
leading to poor confidence interval coverage. Guidance for practical application, freely available software, and areas
of future work are provided.

## Code Structure

`sims/` contains the code to replicate the simulations. 
- `sims/truth_sims/` contains the simulations used to generate the true values for each mechanism
- `sims/valid_sims/` contains the replications of Sofrygin & van der Laan (2017) and similar mechanisms
- Other files are generalized scripts to run simulations for each of the mechanisms

`AmonHen/` includes the implementation of TMLE for dependent data (`NetworkTMLE`).

`Beowulf/` includes the various data generating mechanisms. 

### Dependencies

Dependencies for `NetworkTMLE` consists of
- `numpy v1.15+`
- `pandas v0.18+`
- `statsmodels v0.9+`
- `patsy v0.5+`
- `scipy`
- `NetworkX v2.0+`

*NOTE:* `NetworkTMLE` may not work with versions of NetworkX before 2.0

### Install

To use `NetworkTMLE`, download this folder path of the git repository. Be sure to install the
previous dependences (the package doesn't install those libraries by default). Install both the 
`amonhen` and `beowulf` via the following:

```
python -m pip install AmonHen
python -m pip install Beowulf
```

These are both custom libraries which contain all the functionalities for network-TMLE (`amonhen`)
and management of the simulated data (`beowulf`).

### Use

For example use of `NetworkTMLE` and the data generating mechanisms see the `example_usage.py` 
file. Also see documentation within each function, or by calling `help(NetworkTMLE)`

### References

Sofrygin O & van der Laan MJ. (2017). Semi-parametric estimation and inference for the mean outcome of
the single time-point intervention in a causally connected population. Journal of Causal Inference, 5(1).