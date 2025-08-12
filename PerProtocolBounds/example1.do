******************

* Example 1: IPOP data from Cole et al. 2023


******************

clear all

* Loading in the data

import delimited "data\ipop.csv", delimiter(comma, collapse) varnames(1) case(preserve)


******************

* Estimating the ITT

display "ITT Estimates"

** Risk Ratio bounds and Confidence Intervals

quietly binreg preterm p17, rr

matrix rtable_itt_rr = r(table)

display "RR: " rtable_itt_rr[1,1]
display "RR Lower 95% CL: " rtable_itt_rr[5,1]
display "RR Upper 95% CL: " rtable_itt_rr[6,1]


** Risk Difference bounds and Confidence Intervals

quietly binreg preterm p17, rd

matrix rtable_itt_rd = r(table)

display "RD: " rtable_itt_rd[1,1]
display "RD Lower 95% CL: " rtable_itt_rd[5,1]
display "RD Upper 95% CL: " rtable_itt_rd[6,1]

******************

* Estimating the PP by naively censoring

display "Naive PP Estimates"

** Risk Ratio bounds and Confidence Intervals

quietly binreg preterm p17 if adhere == 1, rr

matrix rtable_naivepp_rr = r(table)

display "RR: " rtable_naivepp_rr[1,1]
display "RR Lower 95% CL: " rtable_naivepp_rr[5,1]
display "RR Upper 95% CL: " rtable_naivepp_rr[6,1]



** Risk Difference bounds and Confidence Intervals

quietly binreg preterm p17 if adhere == 1, rd

matrix rtable_naivepp_rd = r(table)

display "RD: " rtable_naivepp_rd[1,1]
display "RD Lower 95% CL: " rtable_naivepp_rd[5,1]
display "RD Upper 95% CL: " rtable_naivepp_rd[6,1]

******************

* Constructing the upper bounds

generate preterm_upper = preterm if adhere == 1
replace preterm_upper = 1 if adhere == 0 & p17 == 1
replace preterm_upper = 0 if adhere == 0 & p17 == 0

quietly mean preterm_upper, over(p17)
matrix rtable_upper = r(table)

display "Upper Bounds"
display "RR: " rtable_upper[1,2] / rtable_upper[1,1]
display "RD: " rtable_upper[1,2] - rtable_upper[1,1]


******************

* Constructing the lower bounds

generate preterm_lower = preterm if adhere == 1
replace preterm_lower = 1 if adhere == 0 & p17 == 0
replace preterm_lower = 0 if adhere == 0 & p17 == 1

quietly mean preterm_lower, over(p17)
matrix rtable_lower = r(table)

display "Lower Bounds"
display "RR: " rtable_lower[1,2] / rtable_lower[1,1]
display "RD: " rtable_lower[1,2] - rtable_lower[1,1]


******************

* Estimating confidence intervals for bounds using regression

** Risk Ratio bounds and Confidence Intervals

quietly binreg preterm_upper p17, rr

matrix rtable_upperpp_rr = r(table)

quietly binreg preterm_lower p17, rr

matrix rtable_lowerpp_rr = r(table)

display "RR Bounds"
display "Upper 95% CL: " rtable_upperpp_rr[6,1]
display "Upper:        " rtable_upperpp_rr[1,1]
display "Lower:        " rtable_lowerpp_rr[1,1]
display "Lower 95% CL: " rtable_lowerpp_rr[5,1]

** Risk Difference bounds and Confidence Intervals

quietly binreg preterm_upper p17, rd

matrix rtable_upperpp_rd = r(table)

quietly binreg preterm_lower p17, rd

matrix rtable_lowerpp_rd = r(table)

display "RD Bounds"
display "Upper 95% CL: " rtable_upperpp_rd[6,1]
display "Upper:        " rtable_upperpp_rd[1,1]
display "Lower:        " rtable_lowerpp_rd[1,1]
display "Lower 95% CL: " rtable_lowerpp_rd[5,1]


******************


* Display when running this script
*
* ITT Estimates
* RR: .99501247
* RR Lower 95% CL: .64038445
* RR Upper 95% CL: 1.5460241
* RD: -.00045
* RD Lower 95% CL: -.04011238
* RD Upper 95% CL: .03921237
*
* Naive PP Estimates
* RR: .7367514
* RR Lower 95% CL: .41015607
* RR Upper 95% CL: 1.3234051
* RD: -.02540118
* RD Lower 95% CL: -.07210767
* RD Upper 95% CL: .02130531
*
* Upper Bounds
* RR: 6.1811381
* RD: .42851518
*
* Lower Bounds
* RR: .16583541
* RD: -.18815743
*
* RR Bounds
* Upper 95% CL:  8.6884838
* Upper:         6.1811381
* Lower:         .16583541
* Lower 95% CL:  .09773399
*
* RD Bounds
* Upper 95% CL:  .48440917
* Upper:         .42851518
* Lower:        -.18815743
* Lower 95% CL: -.23317692