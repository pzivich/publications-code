# Pooled multinomial logistic regression for parametric g-computation in the presence of competing events

### Lucas M Neuroth, Monica E Swilley-Martinez, Paul N Zivich, Jessie K Edwards

--------------------------------

## File Manifesto

`data/`
- `lau.csv`: Data for the example from Lau et al.

`R/`
- `01_comparing_approaches.R`: sets up the example, runs all estimators, and generates the plots
- `01a_naive_AJ.R`: runs the Aalen-Johansen estimator
- `01b_multiple_pooled_logit.R`: runs the multiple pooled logistic estimator at several time interval widths
- `01c_pooled_multinomial_logit.R`: runs the pooled multinomial logistic estimator at several time interval widths
- `01d_merge_results.R`: aligns all the results
- `02_applied_examples.R`: g-computation application

`SAS/`
- `00_multinomial_exposure_macro.sas`: macro to estimate the CIF with multinomial exposures
- `01_comparing_approaches.sas`: sets up the example, runs all estimators, and generates the plots
- `01a_naive_AJ.sas`: runs the Aalen-Johansen estimator
- `01b_multiple_pooled_logit.sas`: runs the multiple pooled logistic estimator at several time interval widths
- `01c_pooled_multinomial_logit.sas`: runs the pooled multinomial logistic estimator at several time interval widths
- `01d_merge_results.sas`: aligns all the results
- `02_applied_examples.sas`: g-computation application

--------------------------------

Package versions used

R (4.4.1)
```
readr:     2.1.5
janitor:   2.2.0
dplyr:     1.1.4
tidyr:     1.3.1
survival:  3.6-4
VGAM:      1.1-13
ggplot2:   3.5.1
patchwork: 1.2.0
```
