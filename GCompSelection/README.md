# Constructing g-computation estimators: two case studies in selection bias

### Paul N Zivich, Haidong Lu

--------------------------------

## File Manifesto

`data/`
- Single simulated data sets from each data generating mechanism described in the paper.
    - `example1.csv`: data set from the first case study
    - `example2.csv`: data set from the second case study

`simulations/`
- Python code to replicate the simulation results for each case study in the paper
    - `dgm.py`: data generating mechanisms
    - `estfun.py`: estimating functions to simply calls
    - `postprocess.py`: simulation result processing helper functions
    - `sim_case-study1.py`: simulation experiment for the first case study
    - `sim_case-study2.py`: simulation experiment for the second case study

`examples.ipynb`
- Python notebook walking through the estimators applied for the case study data sets in `data/`

`examples.Rmd`
- R markdown walking through the estimators applied for the case study data sets in `data/`

--------------------------------

## System Details

Python: 3.9.4
- Dependencies (used version): NumPy (1.25.2), SciPy (1.11.2), pandas (1.4.1), delicatessen (3.0)

R: 4.4.1
- Dependencies (used version): dplyr (1.1.4), geex (1.1.1), data.table (1.15.4)
