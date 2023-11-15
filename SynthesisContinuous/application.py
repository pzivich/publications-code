#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   ACTG 175 and WIHS illustrative example
#
# Paul Zivich
#######################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from delicatessen import MEstimator
from delicatessen.utilities import spline

from statistical import StatAIPW
from synthesis import SynthesisCACE

########################################################################
# Loading data
d = pd.read_csv("data/hiv.csv")
d[['cd4_sp1', 'cd4_sp2', 'cd4_sp3']] = spline(variable=d['cd4_0wk'],
                                              knots=[1.6, 3, 5, 6.5],
                                              power=2, restricted=True)
d[['age_sp1', 'age_sp2', 'age_sp3']] = spline(variable=d['age'],
                                              knots=[24, 33, 39, 50],
                                              power=2, restricted=True)
d[['wtkg_sp1', 'wtkg_sp2', 'wtkg_sp3']] = spline(variable=d['wtkg'],
                                                 knots=[48, 61, 71, 100],
                                                 power=2, restricted=True)
actg_lcd4 = np.min(d.loc[d['wihs'] == 0, 'cd4_0wk'])
actg_ucd4 = np.max(d.loc[d['wihs'] == 0, 'cd4_0wk'])
d['cd4_np_l'] = np.where(d['cd4_0wk'] < actg_lcd4, 1, 0)
d['cd4_np_u'] = np.where(d['cd4_0wk'] > actg_ucd4, 1, 0)

########################################################################
# Model specifications
s_model = ('white + age + age_sp1 + age_sp2 + age_sp3 + wtkg + wtkg_sp1 + wtkg_sp2 + wtkg_sp3 '
           '+ cd4_0wk + cd4_sp1 + cd4_sp2 + cd4_sp3')
y_model = ('treat*(cd4_0wk + cd4_sp1 + cd4_sp2 + cd4_sp3) '
           '+ white + age + age_sp1 + age_sp2 + age_sp3 + wtkg + wtkg_sp1 + wtkg_sp2 + wtkg_sp3')

sr_model = 'white + age + age_sp1 + age_sp2 + age_sp3 + wtkg + wtkg_sp1 + wtkg_sp2 + wtkg_sp3'
yr_model = 'treat + white + age + age_sp1 + age_sp2 + age_sp3 + wtkg + wtkg_sp1 + wtkg_sp2 + wtkg_sp3'

cace_model = 'restrict + restrict:cd4_0wk + restrict:cd4_sp1 + restrict:cd4_sp2 + restrict:cd4_sp3 - 1'
math_model = 'cd4_np_l + cd4_np_u - 1'

########################################################################
# Storage for results
label = []
est_id = []
point_est = []
lcl_est = []
ucl_est = []
bound_est = []

########################################################################
# Naive
d0 = d.loc[d['wihs'] == 0].copy()
y = np.asarray(d0['cd4_20wk'])
a = np.asarray(d0['treat'])


def psi(theta):
    ace, mu1, mu0 = theta[0], theta[1], theta[2]
    ee_mu1 = a*(y - mu1)
    ee_mu0 = (1-a)*(y - mu0)
    ee_ace = np.ones(y.shape[0]) * ((mu1 - mu0) - ace)
    return np.vstack([ee_ace, ee_mu1, ee_mu0])


estr = MEstimator(psi, init=[0, 5, 5])
estr.estimate(solver='hybr')
label.append("Naive")
est_id.append(0)
point_est.append(estr.theta[0] * 100)
lcl_est.append(estr.confidence_intervals()[0, 0] * 100)
ucl_est.append(estr.confidence_intervals()[0, 1] * 100)

########################################################################
# Restricted Target Population

dr = d.loc[d['restrict'] == 1].copy()

aipw_rs = StatAIPW(dr, outcome='cd4_20wk', action='treat', sample='wihs')
aipw_rs.action_model("1")
aipw_rs.sample_model(s_model)
aipw_rs.outcome_model(y_model)
aipw_rs.estimate(solver='hybr')
label.append("Restricted \n Population")
est_id.append(1)
point_est.append(aipw_rs.ace * 100)
lcl_est.append(aipw_rs.ace_ci[0] * 100)
ucl_est.append(aipw_rs.ace_ci[1] * 100)


########################################################################
# Restricted Covariate Set

aipw_rc = StatAIPW(d, outcome='cd4_20wk', action='treat', sample='wihs')
aipw_rc.action_model("1")
aipw_rc.sample_model(sr_model)
aipw_rc.outcome_model(yr_model)
aipw_rc.estimate(solver='hybr')
label.append("Restricted \n Covariates")
est_id.append(2)
point_est.append(aipw_rc.ace * 100)
lcl_est.append(aipw_rc.ace_ci[0] * 100)
ucl_est.append(aipw_rc.ace_ci[1] * 100)

########################################################################
# Extrapolation

aipw_ex = StatAIPW(d, outcome='cd4_20wk', action='treat', sample='wihs')
aipw_ex.action_model("1")
aipw_ex.sample_model(s_model)
aipw_ex.outcome_model(y_model)
aipw_ex.estimate(solver='hybr')
label.append("Extrapolation")
est_id.append(3)
point_est.append(aipw_ex.ace * 100)
lcl_est.append(aipw_ex.ace_ci[0] * 100)
ucl_est.append(aipw_ex.ace_ci[1] * 100)

########################################################################
# Synthesis CACE

aipw_syn = SynthesisCACE(d, outcome='cd4_20wk', action='treat', sample='wihs', positive_region='restrict')
aipw_syn.action_model("1")
aipw_syn.sample_model(s_model)
aipw_syn.outcome_model(y_model)
aipw_syn.cace_model(cace_model)
aipw_syn.math_model(math_model, parameters=None)
aipw_syn.estimate_bounds(lower=[-0.2, -0.2], upper=[1.5, 1.], solver='hybr')
label.append("Synthesis CACE")
est_id.append(4)
lowerb = aipw_syn.bounds[0] * 100
upperb = aipw_syn.bounds[1] * 100
point_est.append((lowerb+upperb) / 2)
lcl_est.append(aipw_syn.bounds_ci[0] * 100)
ucl_est.append(aipw_syn.bounds_ci[1] * 100)

########################################################################
# Turning into a Figure

plt.figure(figsize=(6.5, 3.5))
gspec = gridspec.GridSpec(1, 6)
fplot = plt.subplot(gspec[0, :4])
write = plt.subplot(gspec[0, 4:])

point_est = np.asarray(point_est)
lcl_est = np.asarray(lcl_est)
ucl_est = np.asarray(ucl_est)

# Forest Plot Part
# Confidence intervals
fplot.errorbar(point_est, est_id,
               xerr=(point_est - lcl_est, ucl_est - point_est),
               marker='None', zorder=2, ecolor='dimgray', elinewidth=1.,
               capsize=3, capthick=1., linewidth=0)
# Point Estimates
fplot.scatter(point_est, est_id, c='k',
              s=60, marker="o", zorder=3, edgecolors='k')
# Bounds
fplot.fill_between([lowerb, upperb], [4-0.15, 4-0.15], [4+0.15, 4+0.15],
                   color='k', alpha=1, zorder=3)
# Extra parts
fplot.axvline(0, color="k", linestyle=':', zorder=1, linewidth=0.85)
# Axis manipulations
fplot.set_xlim([-60, 170])
fplot.set_xticks([-50, 0, 50, 100, 150])
fplot.set_ylim(-1, len(est_id))  # Spacing out y-axis properly
fplot.set_yticks(est_id)  # Setting y-axis ticks at each index
fplot.set_yticklabels(label)  # Then setting y-axis ticks at the input labels
fplot.invert_yaxis()  # Invert y-axis to align values with input order (descending)
fplot.set_xlabel("Average Causal Effect")

# Table Part
# Formatting values to input into the table
cell_vals = []
fdec = "{:.0f}"
for i in est_id:
    p_est = fdec.format(point_est[i])
    lcl_f = fdec.format(lcl_est[i])
    ucl_f = fdec.format(ucl_est[i])
    ci = "(" + lcl_f + ", " + ucl_f + ")"
    cell_vals.append([p_est, ci])

cell_vals[-1][0] = "["+fdec.format(lowerb)+", "+fdec.format(upperb)+"]"
# Displaying Table
write.axis('off')
tb = write.table(cellText=cell_vals,
                 cellLoc='center', loc='right',
                 colLabels=["Estimate", "95% CI"], bbox=[0, 0.08, 1, 1])
tb.auto_set_font_size(False)
tb.set_fontsize(11)
for key, cell in tb.get_celld().items():
    cell.set_linewidth(0)

plt.tight_layout()
plt.savefig("results/forest_plot.png", format='png', dpi=300)
plt.close()
