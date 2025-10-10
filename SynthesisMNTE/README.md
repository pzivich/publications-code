# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models

### Paul N Zivich, Bonnie E Shook-Sa, Stephen R Cole, Eric T Lofgren, Jessie K Edwards


--------------------------------

## File Manifesto

`data/`
- `height_params.csv`: Parameters for height cut points for mathematical model
- `nhanes.csv`: Pre-processed NHANES data
- `sbp_params.csv`: Parameters for systolic blood pressure distributions for mathematical model

`Python/`
- `analysis.py`: Runs the main analyses reported in the main paper
- `appendix2.py`: Runs the parametric statistical model for the example described in the Appendix
- `appendix3.py`: Runs the AIPW estimator for the example described in the Appendix
- `bounds.py`: Runs the positivity bounds for the example
- `diagnostic.py`: Runs the diagnostic procedure for the example
- `figure.py`: Creates descriptive figures for SBP by age
- `mathmodel.py`: Mathematical model to impute SBP in other scripts

`R/`
- `analysis.py`: Runs the main analyses reported in the main paper
- `diagnostic.py`: Runs the diagnostic procedure for the example
# TODO appendix2, appendix3, bounds

--------------------------------

## System Details

Python: 3.9.4
- Dependencies (version): NumPy (1.25.2), SciPy (1.11.2), formulaic (0.5.2), pandas (1.4.1), matplotlib (3.9.2),
  delicatessen (3.0), statsmodels (0.14.1)

R: 4.4.1
- Dependencies (version): tidyverse (2.0.0), rootSolve (1.8.2.4)
