# Sensitivity Analyses for Means or Proportions with Missing Data

### Stephen R Cole, Paul N Zivich, Jessie K Edwards, Bonnie E Shook-Sa, Michael G Hudgens

-----------------------------------

The folder consists of the SAS (`sensitivity_analysis.sas`) and Python (`sensitivity_analysis.ipynb`) code to recreate
the examples presented in the corresponding publication. Data from Lau et al. is provided in the `lau_wihs.dat` file.

-----------------------------------

## File Manifesto

`lau_wihs.dat`
- Data from Lau et al. 2009, which originates from the Women's Interagency HIV Study (WIHS). Variables consist of ID
  (`id`), race (`black`), age (`age`), the original CD4 (`cd4`), CD4 with alpha=0 (`cd41`), CD4 with alpha=0.01
  (`cd42`), CD4 with alpha=-0.01 (`cd43`), and  CD4 with alpha=0.01 with a covariate (`cd44`),

`sensitivity_analysis.ipynb`
- Python code to replicate examples 1-3. Dependencies include NumPy, pandas, SciPy, delicatessen, matplotlib. Results
  are displayed inline in the notebook.

`sensitivity_analysis.sas`
- SAS code to replicate examples 1-3.
