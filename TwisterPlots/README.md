# Twister Plots for Time-to-Event Studies

### Paul N Zivich, Stephen R Cole, Alexander Breskin

-----------------------------------

The folder consists of the SAS (`twister.sas`), R (`twister.R`), and Python (`twister.py`) files to generate the 
example twister plot presented in the figure. Risk differences and associated confidence intervals estimated from 
the reconstructed data set are available in `data_twister.csv`.

-----------------------------------

## File Manifesto

`extract_data.py`
- Python code to construct the simulated Pfizer data to calculate the corresponding risks. Outputs the risk, risk
  difference, and risk ratio estimates and confidence intervals as `data_twister.csv`. For storage reasons, estimates
  are output to 6 decimal places (but can be updated in the script).

`data_twister.csv`
- Estimated risk differences and risk ratios (vaccine minus placebo) and associated confidence 
  intervals for each unique event time from the recreated Pfizer data.

`twister.sas`
- SAS code to generate twister plots. Demonstrated with applied example

`twister.R`
- R code to generate twister plots. Consists of a generalized function and example of function in use
- Dependencies: `ggplot2`

`twister.py`
- Python 3.6+ code to generate twister plots. Consists of a generalized function and example of
  function in use
- Dependencies: `numpy`, `pandas`, `matplotlib`
