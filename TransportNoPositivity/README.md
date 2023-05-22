# Transportability without positivity: a synthesis of statistical and simulation modeling

### Paul N Zivich, Jessie K Edwards, Eric T Lofgren, Stephen R Cole, Bonnie E Shook-Sa, Justin Lessler

-----------------------------------

The folder consists of the Python code to recreate the analysis and simulations presented in the corresponding
publication. Data from Wilson et al. (2017) is available from

https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002479

Wilson E, Free C, Morris TP, et al. (2017). Internet-accessed sexually transmitted infection (e-STI) testing and
results service: a randomised, single-blind, controlled trial. *PLoS Medicine*, 14(12), e1002479.

-----------------------------------

## File Manifesto

`continuous.py`
- Computes the example of non-positivity by a continuous variable as presented in the appendix.

`dgm.py`
- Data generating mechanisms for the simulation experiments.

`efuncs.py`
- Estimating functions for the restricted approaches.

`helper.py`
- Helper functions for simulations and the illustrative example.

`process_results.py`
- Takes the output file from `run_estimators.py` (the simulation experiments) and generates Appendix Tables 1 & 2.

`process_wilson.py`
- Takes the files from the supplement of Wilson et al. (2017) and sets it up for illustrative experiment.

`run_estimators.py`
- Runs the simulation experiment detailed in the appendix of the paper.

`solution1.py`
- Runs solution 1, restrict the target population, for the illustrative example.

`solution2.py`
- Runs solution 2, restrict the covariate set, for the illustrative example.

`solution3_gcomp.py`
- Runs solution 3, model synthesis, for the illustrative example and g-computation estimator.

`solution3_ipw.py`
- Runs solution 3, model synthesis, for the illustrative example and IPW estimator.

`truth.py`
- Simulates the true average causal effect for the simulation experiment.
