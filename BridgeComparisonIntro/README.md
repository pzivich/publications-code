# Bridged treatment comparisons: an illustrative application in HIV treatment

### Paul N Zivich, Stephen R Cole, Jessie K Edwards, Bonnie E Shook-Sa, Alexander Breskin, Michael G Hudgens

**Citation**: Zivich PN, Cole SR, Edwards JK, Shook-Sa BE, Breskin A, and Hudgens MG. "Bridged treatment comparisons: an
illustrative application in HIV treatment." *arXiv* arXiv:2206.04445.

--------------------------------

## Abstract

Comparisons of treatments or exposures are of central interest in epidemiology, but direct comparisons are not always
possible due to practical or ethical reasons. Here, we detail a data fusion approach that allows bridged treatment
comparisons across studies. The motivating example entails comparing the risk of the composite outcome of death, AIDS,
or greater than a 50% CD4 cell count decline in people with HIV when assigned triple versus mono antiretroviral therapy,
using data from the AIDS Clinical Trial Group (ACTG) 175 (mono versus dual therapy) and ACTG 320 (dual versus triple
therapy). We review a set of identification assumptions and estimate the risk difference using an inverse probability
weighting estimator that leverages the shared trial arms (dual therapy). A fusion diagnostic based on comparing the
shared arms is proposed that may indicate violation of the identification assumptions. Application of the data fusion
estimator and diagnostic to the ACTG trials indicates triple therapy results in a reduction in risk compared to
mono therapy in individuals with baseline CD4 counts between 50 and 300 cells/mm3. Bridged treatment comparisons address
questions that none of the constituent data sources could address alone, but valid fusion-based inference requires
careful consideration.

--------------------------------

## File Manifesto

### Python
The `Python/` path includes the bridged inverse probability weighting estimator and the associated diagnostics within
a local package called `chimera`. Installation instructions are provided in the Python README. There are three files,
`versions.py`, `actg_diagnostic.py`, and `actg_analysis.py`. `version.py` is provided for easy checking of installed
versions of dependencies. `actg_diagnostic.py` runs the associated diagnostics and re-creates Figure 1.
`actg_analysis.py` conducts the analysis on the CD4 restricted data and re-creates Figure 2.

### R
The `R/` path includes the bridged inverse probability weighting estimator in the `Chimera.R` file that can be loaded.
Additionally, there are two files to recreate the analyses presented in the paper: `actg_diagnostic.R`, and
`actg_analysis.R`. `actg_diagnostic.py` runs the associated diagnostics and generates the plots that make up Figure 1.
`actg_analysis.py` conducts the analysis on the CD4 restricted data and re-creates Figure 2.

### SAS
To be added...
