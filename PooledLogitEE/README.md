# Estimating equations for survival analysis with pooled logistic regression

### Paul N Zivich, Stephen R Cole, Bonnie E Shook-Sa, Justin B DeMonte, Jessie K Edwards

arXiv:2504.13291

--------------------------------

NOTE: to recreate the results, you will need to copy the `efuncs.py` file into the folder you want to recreate. This
is a limitation of how the relative importing works and the code organization.

## File Manifesto

`data/`
- `collett.dat`: Data for the example from Collett
- `lau.csv`: Data for the example from Lau et al.

`plots/`
- `collett.py`: Code for the publication figures for the data from Collett
- `lau.py`: Code for the publication figures for the data from Lau et al.

`sims/`
- `dgm.py`: Code for the data generating mechanism
- `estimators.py`: Simplified class object implementation of estimators to simplify simulation code
- `simulation.py`: Script to run the reported simulations
- `truth.py`: Script to generate the true values using 10 million simulated observations

`time-trials/`
- `collett.py`: Code for the run-time results for Collett
- `lau.py`: Code for the run-time results for Lau et al.
- `standard.py`: Python class object implementing the standard implementation of pooled logistic regression models.

`appendix.py`
- By-hand example reviewed in the Appendix.

`Collett.ipynb`
- A walkthrough of the application to the data from Collett. See `plots/` or `time-trials/` for those results

`efuncs.py`
- Estimating functions and associated utilities for pooled logistic regression models.

`Lau.ipynb`
- A walkthrough of the application to the data from Lau et al. See `plots/` or `time-trials/` for those results


--------------------------------

Package versions used

```
NumPy:        1.25.2
pandas:       1.4.1
SciPy:        1.11.2
delicatessen: 3.0
matplotlib:   3.9.2
lifeslines:   0.27.3
statsmodels:  0.14.1
```