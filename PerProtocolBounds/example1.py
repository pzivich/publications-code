####################################################################################################################
# Example 1: IPOP data from Cole et al. 2023
#
# Paul Zivich (2025/06/16)
####################################################################################################################

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import DomainWarning


#########################################################################
# Loading in the data

d = pd.read_csv("data/ipop.dat", sep='\s+', names=["pid", "cervix", "p17", "preterm", "adhere"])

#########################################################################
# Estimating the ITT

with warnings.catch_warnings():
    # Silence the DomainWarnings for non-canonical specifications with statsmodels
    warnings.simplefilter('ignore', DomainWarning)

    # Risk Ratio bounds and Confidence Intervals
    fm_upper = smf.glm("preterm ~ p17", data=d, family=sm.families.Binomial(sm.families.links.Log())).fit()
    print("RR ITT")
    print("RR:    ", np.exp(fm_upper.params[1]))
    print("95% CI:", np.exp(np.asarray(fm_upper.conf_int()))[1, :])
    print()

    # Risk Difference bounds and Confidence Intervals
    fm_upper = smf.glm("preterm ~ p17", data=d, family=sm.families.Binomial(sm.families.links.Identity())).fit()
    print("RD ITT")
    print("RD:    ", fm_upper.params[1])
    print("95% CI:", np.asarray(fm_upper.conf_int())[1, :])
    print()


#########################################################################
# Estimating the PP by naively censoring

ds = d.loc[d['adhere'] == 1].copy()

with warnings.catch_warnings():
    # Silence the DomainWarnings for non-canonical specifications with statsmodels
    warnings.simplefilter('ignore', DomainWarning)

    # Risk Ratio bounds and Confidence Intervals
    fm_upper = smf.glm("preterm ~ p17", data=ds, family=sm.families.Binomial(sm.families.links.Log())).fit()
    print("RR Per-Protocol")
    print("RR:    ", np.exp(fm_upper.params[1]))
    print("95% CI:", np.exp(np.asarray(fm_upper.conf_int()))[1, :])
    print()

    # Risk Difference bounds and Confidence Intervals
    fm_upper = smf.glm("preterm ~ p17", data=ds, family=sm.families.Binomial(sm.families.links.Identity())).fit()
    print("RD Per-Protocol")
    print("RD:    ", fm_upper.params[1])
    print("95% CI:", np.asarray(fm_upper.conf_int())[1, :])
    print()


#########################################################################
# Constructing the upper bounds

d_upper = d.copy()
d_upper['preterm'] = np.where((d_upper['adhere'] == 0) & (d_upper['p17'] == 1), 1, d_upper['preterm'])
d_upper['preterm'] = np.where((d_upper['adhere'] == 0) & (d_upper['p17'] == 0), 0, d_upper['preterm'])
p17_upper = np.mean(d_upper.loc[d_upper['p17'] == 1, 'preterm'])
plc_lower = np.mean(d_upper.loc[d_upper['p17'] == 0, 'preterm'])
print("Upper Bounds")
print('RR:', p17_upper / plc_lower)
print('RD:', p17_upper - plc_lower)

#########################################################################
# Constructing the lower bounds

d_lower = d.copy()
d_lower['preterm'] = np.where((d_lower['adhere'] == 0) & (d_lower['p17'] == 1), 0, d_lower['preterm'])
d_lower['preterm'] = np.where((d_lower['adhere'] == 0) & (d_lower['p17'] == 0), 1, d_lower['preterm'])
p17_lower = np.mean(d_lower.loc[d_lower['p17'] == 1, 'preterm'])
plc_upper = np.mean(d_lower.loc[d_lower['p17'] == 0, 'preterm'])
print("Lower Bounds")
print('RR:', p17_lower / plc_upper)
print('RD:', p17_lower - plc_upper)
print()

#########################################################################
# Estimating confidence intervals for bounds using regression

with warnings.catch_warnings():
    # Silence the DomainWarnings for non-canonical specifications with statsmodels
    warnings.simplefilter('ignore', DomainWarning)

    # Risk Ratio bounds and Confidence Intervals
    fm_upper = smf.glm("preterm ~ p17", data=d_upper, family=sm.families.Binomial(sm.families.links.Log())).fit()
    print("RR Bounds")
    print("Upper 95% CI:", np.exp(np.asarray(fm_upper.conf_int()))[1, 1])
    print("Upper:       ", np.exp(fm_upper.params[1]))

    fm_lower = smf.glm("preterm ~ p17", data=d_lower, family=sm.families.Binomial(sm.families.links.Log())).fit()
    print("Lower:       ", np.exp(fm_lower.params[1]))
    print("Lower 95% CI:", np.exp(np.asarray(fm_lower.conf_int()))[1, 0])
    print()

    # Risk Difference bounds and Confidence Intervals
    fm_upper = smf.glm("preterm ~ p17", data=d_upper, family=sm.families.Binomial(sm.families.links.Identity())).fit()
    print("RD Bounds")
    print("Upper 95% CI:", np.asarray(fm_upper.conf_int())[1, 1])
    print("Upper:       ", fm_upper.params[1])

    fm_lower = smf.glm("preterm ~ p17", data=d_lower, family=sm.families.Binomial(sm.families.links.Identity())).fit()
    print("Lower:       ", fm_lower.params[1])
    print("Lower 95% CI:", np.asarray(fm_lower.conf_int())[1, 0])


# Output when running this script
#
# RR ITT
# RR:     0.9950124688299863
# 95% CI: [0.64038445 1.54602413]
#
# RD ITT
# RD:     -0.00045000281251758763
# 95% CI: [-0.04011238  0.03921237]
#
# RR Per-Protocol
# RR:     0.7367514002585024
# 95% CI: [0.4101535  1.32341337]
#
# RD Per-Protocol
# RD:     -0.02540118067681048
# 95% CI: [-0.07210767  0.02130531]
#
# Upper Bounds
# RR: 6.181138063931082
# RD: 0.4285151782198639
# Lower Bounds
# RR: 0.16583541147132172
# RD: -0.18815742598391239
#
# RR Bounds
# Upper 95% CI: 8.688483832415677
# Upper:        6.181138063274826
# Lower:        0.16583541147132055
# Lower 95% CI: 0.09773399428822288
#
# RD Bounds
# Upper 95% CI: 0.4844091656598309
# Upper:        0.4285151782198646
# Lower:        -0.18815742598391255
# Lower 95% CI: -0.2331769160046414
