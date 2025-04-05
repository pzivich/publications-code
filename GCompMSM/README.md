# Estimating Marginal Structural Model Parameters for Time-Fixed, Binary Actions with g-computation and Estimating Equations

### Paul N Zivich, Bonnie E Shook-Sa

--------------------------------

## File Manifesto

`actg.csv`
- Pre-processed data from the publicly-available ACTG 175 data sets for the applied example. Variables are
  ID `id`,
  age (years) `age`,
  white versus non-white `white`,
  history of injection drug use `idu`,
  Karnofsky score categories `karnof`,
  baseline CD4 `cd4_0wk`,
  20-week CD4 `cd4_20wk`,
  gender `male`,
  assigned dual ART `treat`,
  centered and standardized age `agec`,
  restricted quadratic spline term for centered age `age_rs1`,
  restricted quadratic spline term for centered age `age_rs2`,
  restricted quadratic spline term for centered age `age_rs3`,
  centered and standardized baseline CD4 `cd4c_0wk`,
  restricted quadratic spline term for centered baseline CD4 `cd4_rs1`,
  restricted quadratic spline term for centered baseline CD4 `cd4_rs2`, and
  restricted quadratic spline term for centered baseline CD4 `cd4_rs3`.

`application.py`
- Apply the g-computation and IPW estimators of the MSM for the applied example in Python.

`application.R`
- Apply the g-computation and IPW estimators of the MSM for the applied example in R.

`application.sas`
- Apply the g-computation and IPW estimators of the MSM for the applied example in SAS.
