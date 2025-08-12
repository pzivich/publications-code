******************

* Example 2: ACTG 320 data


******************

clear all

* Loading in the data

import delimited "data\actg320.csv", delimiter(comma, collapse) varnames(1) case(preserve)

keep id art stop t delta

generate origin = -1

quietly summarize t, detail
scalar followup = r(max)


* Convert to stset (survival data) 

quietly stset t, failure(delta) id(id) origin(origin)



* Intent-to-Treat Analysis

quietly sts list, failure risktable(365) by(art) saving(survival_itt_followup.dta, replace)

frame create survival_itt_followup

frame survival_itt_followup: use survival_itt_followup.dta

frame change survival_itt_followup

sort art

generate variance = std_err^2




scalar risk_diff_itt = failure[2] - failure[1]
scalar rd_var_itt = variance[1] + variance[2]

scalar risk_ratio_itt = failure[2]/failure[1]
scalar logrr_var_itt = variance[1]/failure[1]^2 + variance[2]/failure[2]^2

display "Intent to Treat Contrast"
display "Time: " followup
display "RD: " risk_diff_itt
display "RD Lower 95% CL: " risk_diff_itt - 1.96*sqrt(rd_var_itt)
display "RD Upper 95% CL: " risk_diff_itt + 1.96*sqrt(rd_var_itt)
display "RR: " risk_ratio_itt
display "RR Lower 95% CL: " exp(ln(risk_ratio_itt) - 1.96*sqrt(logrr_var_itt))
display "RR Upper 95% CL: " exp(ln(risk_ratio_itt) + 1.96*sqrt(logrr_var_itt))

frame change default


******************

* Create deviations variable


generate deviate = 0 if stop == .
replace deviate = 1 if deviate == .

display "Protocol Deviations"
table (art) (deviate)

******************

* Per-Protocol Bounds

generate t2 = t if deviate == 0
replace t2 = stop if deviate == 1

generate delta2 = delta if deviate == 0
replace delta2 = 0 if deviate == 1


* Bounds

quietly summarize t, detail
scalar max_t = r(max)

******************

* Per-Protocol Upper Bound

generate delta2_upper = 1 if deviate == 1 & art == 1
replace delta2_upper = delta2 if delta2_upper == .

generate t2_upper = max_t if deviate == 1 & art == 0
replace t2_upper = t2 if t2_upper == .


quietly stset t2_upper, failure(delta2_upper) id(id) origin(origin)


quietly sts list, failure risktable(365) by(art) saving(survival_upperpp_followup.dta, replace)
quietly sts list if art == 0, failure risktable(0(1)365) saving(survival_upperpp_art0.dta, replace)
quietly sts list if art == 1, failure risktable(0(1)365) saving(survival_upperpp_art1.dta, replace)

frame create survival_upperpp_followup

frame survival_upperpp_followup: use survival_upperpp_followup.dta

frame change survival_upperpp_followup

sort art

generate variance = std_err^2


scalar risk_diff_upperpp = failure[2] - failure[1]
scalar rd_var_upperpp = variance[1] + variance[2]

scalar risk_ratio_upperpp = failure[2]/failure[1]
scalar logrr_var_upperpp = variance[1]/(failure[1]^2) + variance[2]/(failure[2]^2)


frame change default


******************

* Per-Protocol Lower Bound
generate delta2_lower = 1 if deviate == 1 & art == 0
replace delta2_lower = delta2 if delta2_lower == .

generate t2_lower = max_t if deviate == 1 & art == 1
replace t2_lower = t2 if t2_lower == .


quietly stset t2_lower, failure(delta2_lower) id(id) origin(origin)


quietly sts list, failure risktable(365) by(art) saving(survival_lowerpp_followup.dta, replace)
quietly sts list if art == 0, failure risktable(0(1)365) saving(survival_lowerpp_art0.dta, replace)
quietly sts list if art == 1, failure risktable(0(1)365) saving(survival_lowerpp_art1.dta, replace)

frame create survival_lowerpp_followup

frame survival_lowerpp_followup: use survival_lowerpp_followup.dta

frame change survival_lowerpp_followup

sort art

generate variance = std_err^2


scalar risk_diff_lowerpp = failure[2] - failure[1]
scalar rd_var_lowerpp = variance[1] + variance[2]

scalar risk_ratio_lowerpp = failure[2]/failure[1]
scalar logrr_var_lowerpp = variance[1]/failure[1]^2 + variance[2]/failure[2]^2

display "Per-Protocol RD Bounds"
display "Time: " followup
display "Upper 95% CL: " risk_diff_upperpp + 1.96*sqrt(rd_var_upperpp)
display "Upper:        " risk_diff_upperpp
display "Lower:        " risk_diff_lowerpp
display "Lower 95% CL: " risk_diff_lowerpp - 1.96*sqrt(rd_var_lowerpp)


display "Per-Protocol RR Bounds"
display "Time: " followup 
display "Upper 95% CL: " exp(ln(risk_ratio_upperpp) + 1.96*sqrt(logrr_var_upperpp))
display "Upper:        " risk_ratio_upperpp
display "Lower:        " risk_ratio_lowerpp
display "Lower 95% CL: " exp(ln(risk_ratio_lowerpp) - 1.96*sqrt(logrr_var_lowerpp))




******************

* Plotting results

frame create survival_upperpp_art0
frame survival_upperpp_art0: use survival_upperpp_art0.dta
frame change survival_upperpp_art0
generate variance_upperpp_art0 = std_err^2
replace variance_upperpp_art0 = 0 if variance_upperpp_art0 == .
rename failure failure_upperpp_art0
keep time failure_upperpp_art0 variance_upperpp_art0 
save survival_upperpp_art0, replace

frame create survival_upperpp_art1
frame survival_upperpp_art1: use survival_upperpp_art1.dta
frame change survival_upperpp_art1
generate variance_upperpp_art1 = std_err^2
replace variance_upperpp_art1 = 0 if variance_upperpp_art1 == .
rename failure failure_upperpp_art1
keep time failure_upperpp_art1 variance_upperpp_art1
save survival_upperpp_art1, replace

frame create survival_lowerpp_art0
frame survival_lowerpp_art0: use survival_lowerpp_art0.dta
frame change survival_lowerpp_art0
generate variance_lowerpp_art0 = std_err^2
replace variance_lowerpp_art0 = 0 if variance_lowerpp_art0 == .
rename failure failure_lowerpp_art0
keep time failure_lowerpp_art0 variance_lowerpp_art0
save survival_lowerpp_art0, replace


frame create survival_lowerpp_art1
frame survival_lowerpp_art1: use survival_lowerpp_art1.dta
frame change survival_lowerpp_art1
generate variance_lowerpp_art1 = std_err^2
replace variance_lowerpp_art1 = 0 if variance_lowerpp_art1 == .
rename failure failure_lowerpp_art1
keep time failure_lowerpp_art1 variance_lowerpp_art1
save survival_lowerpp_art1, replace


* Merge data

use survival_lowerpp_art1

merge 1:1 time using survival_lowerpp_art0
drop _merge
merge 1:1 time using survival_upperpp_art1
drop _merge
merge 1:1 time using survival_upperpp_art0
drop _merge

save survival_merged, replace

frame create survival_merged

frame survival_merged: use survival_merged.dta

frame change survival_merged

replace failure_upperpp_art1 = 1 if failure_upperpp_art1 == .
replace failure_upperpp_art0 = 1 if failure_upperpp_art0 == .
replace failure_lowerpp_art1 = 1 if failure_lowerpp_art1 == .
replace failure_lowerpp_art0 = 1 if failure_lowerpp_art0 == .


replace variance_upperpp_art1 = 0 if variance_upperpp_art1 == .
replace variance_upperpp_art0 = 0 if variance_upperpp_art0 == .
replace variance_lowerpp_art1 = 0 if variance_lowerpp_art1 == .
replace variance_lowerpp_art0 = 0 if variance_lowerpp_art0 == .

generate risk_diff_upperpp = failure_upperpp_art1 - failure_upperpp_art0
generate rd_var_upperpp = variance_upperpp_art1 + variance_upperpp_art0

generate risk_ratio_upperpp = failure_upperpp_art1 / failure_upperpp_art0
generate logrr_var_upperpp = variance_upperpp_art1 /failure_upperpp_art1^2 + variance_upperpp_art0 / failure_upperpp_art0^2



generate risk_diff_upperpp95 = risk_diff_upperpp + 1.96*sqrt(rd_var_upperpp)
generate risk_ratio_upperpp95 = exp(ln(risk_ratio_upperpp) + 1.96*sqrt(logrr_var_upperpp))

generate risk_diff_lowerpp = failure_lowerpp_art1 - failure_lowerpp_art0
generate rd_var_lowerpp = variance_lowerpp_art1 + variance_lowerpp_art0

generate risk_ratio_lowerpp = failure_lowerpp_art1 / failure_lowerpp_art0
replace risk_ratio_lowerpp = 0.01 if risk_ratio_lowerpp < 0.01

generate logrr_var_lowerpp = variance_lowerpp_art1 /failure_lowerpp_art1^2 + variance_lowerpp_art0 / failure_lowerpp_art0^2

generate risk_diff_lowerpp95 = risk_diff_lowerpp - 1.96*sqrt(rd_var_lowerpp)

generate risk_ratio_lowerpp95 = exp(ln(risk_ratio_lowerpp) - 1.96*sqrt(logrr_var_lowerpp))
replace risk_ratio_lowerpp95 = 0.01 if risk_ratio_lowerpp95 < 0.01


keep time risk_diff_upperpp risk_ratio_upperpp risk_diff_upperpp95 risk_ratio_upperpp95 risk_diff_lowerpp risk_ratio_lowerpp risk_diff_lowerpp95 risk_ratio_lowerpp95


twoway rarea risk_diff_upperpp95 risk_diff_lowerpp95 time, horizontal color(gs11) xlabel(-1(.5)1, nogrid) ylabel(0 50 100 150 200 250 300 365, nogrid) connect(stairstep) legend(off) || rarea risk_diff_upperpp risk_diff_lowerpp time, horizontal color(black) connect(stairstep) legend(off) || scatteri 0 0 365 0, recast(line) lcolor(gs7) lpattern(dash) legend(off) ytitle("Time (days)") xtitle("Risk Difference") text(-35 -1 "Favors 3-drug", place(e) size(small)) text(-35 1 "Favors 2-drug", place(w) size(small)) ysize(7) xsize(5)


twoway rarea risk_ratio_upperpp95 risk_ratio_lowerpp95 time, horizontal color(gs11) xlabel(0.01 0.1 1 10 100, nogrid) xscale(log) ylabel(0 50 100 150 200 250 300 365, nogrid) connect(stairstep) legend(off) || rarea risk_ratio_upperpp risk_ratio_lowerpp time, horizontal color(black) connect(stairstep) legend(off) || scatteri 0 1 365 1, recast(line) lcolor(gs7) lpattern(dash) legend(off) ytitle("Time (days)") xtitle("Risk Ratio") ysize(7) xsize(5) text(-35 0.01 "Favors 3-drug", place(e) size(small)) text(-35 100 "Favors 2-drug", place(w) size(small))



******************



* Display when running this script
* 
* Intent to Treat Contrast
* Time: 365
* RD: -.08710501
* RD Lower 95% CL: -.12750966
* RD Upper 95% CL: -.04670035
* RR: .45427103
* RR Lower 95% CL: .31094827
* RR Upper 95% CL: .66365434
*
* Protocol Deviations
* ----------------------------
*         |       deviate     
*         |    0     1   Total
* --------+-------------------
* art     |                   
*   0     |  426   153     579
*   1     |  513    64     577
*   Total |  939   217   1,156
* ----------------------------
*
* Per-Protocol RD Bounds
* Time: 365
* Upper 95% CL: .10952361
* Upper:        .06426779
* Lower:        -.40088908
* Lower 95% CL: -.45424219
*
* Per-Protocol RR Bounds
* Time: 365
* Upper 95% CL: 2.1052288
* Upper:        1.5451055
* Lower:        .1201052
* Lower 95% CL: .08197495