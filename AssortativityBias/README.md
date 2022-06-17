# Assortativity and Bias in Epidemiologic Studies of Contagious Outcomes: A Simulated Example in the Context of Vaccination

### Paul N Zivich, Alexander Volfovsky, James Moody, Allison E Aiello

--------------------------------

## Abstract

Assortativity is the tendency of individuals connected in a network to share traits and behaviors. Through simulations,
we demonstrated the potential for bias resulting from assortativity by vaccination, where vaccinated individuals are 
more likely to be connected with other vaccinated individuals. We simulated outbreaks of a hypothetical infectious 
disease and a vaccine on a randomly generated network and a contact network of university students living on-campus. We 
varied protection of the vaccine to the individual, transmission potential of vaccinated-but-infected individuals, and 
assortativity by vaccination. We compared a traditional approach, which ignores the structural features of a network, to 
simple approaches which summarized information from the network. The traditional approach resulted in biased estimates 
if the unit-treatment effect when there was assortativity by vaccination. Several different approaches that included 
summary measures of the network reduced bias and improved confidence interval coverage. Through simulations, we showed
pitfalls of ignoring assortativity by vaccination. While our example is described in terms of vaccines, our results 
apply more widely to exposures for contagious outcomes. Assortativity should be considered when evaluating exposures for
contagious outcomes.

--------------------------------

This folder contains the python code used to generate the simulations as well as the stochastic block network data 
(both the network and the designated communities according to Louvain's algorithm).

## File Manifesto
`simulation.py`
- Python file for running simulations for the set parameters. Parameters for the simulation can be manually set the 
  start of the simulation file
    
`stochblock.dat`
- Edge list for stochastic block network data

`stochblock.dat`
- Data by node ID and unique community ID. Communities were identified through Louvain's algorithm

`utils.py`
- Python file containing background functions for simulations.


### Simulation dependencies
Simulations were originally conducted with Python 3.5.1 along with the following libraries: `numpy` 1.16.0, 
`pandas` 0.23.4, `networkx` 2.2, `statsmodels` 0.8.0.

