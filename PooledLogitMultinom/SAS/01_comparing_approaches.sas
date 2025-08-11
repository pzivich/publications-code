*********************************************************************                                                                    
*  Title:         01_comparing_approaches.sas
*  Date:          08/06/2025
*  Author:        Lucas Neuroth 
*------------------------------------------------------------------- 
*  Purpose: Read in example data, call approach scripts, append and 
*           plot results                                                                
********************************************************************;

/*********************** Set-up (global macro variables) ***********************/
* Specify data filepath;
%let datapath = C:\Users\lucasneuroth\OneDrive - University of North Carolina at Chapel Hill\UNC GRA\data\lau2009\;
* Specify code filepath;
%let codepath = C:\Users\lucasneuroth\OneDrive - University of North Carolina at Chapel Hill\UNC GRA\analysis\01_code\09_multinomial_logit_g_formula\SAS\;
* Specify follow-up (years);
%let tau = 2;
* Specify discretized timescale for g-comp. approaches (can set days/months/weeks, with weeks corresponding to analysis presented in manuscript);
%let timescale = 'weeks';

/*********************** Reading in the example dataset ***********************/
proc import datafile="&datapath.lau2009.csv" 
    out=wide
    dbms=CSV
    REPLACE;
    getnames=YES;
    datarow=2;
* Inspect dataset;
proc contents data=wide;
run;

/* Continuous and discretized time */
* Time will be treated contunuously (in days) for the naive Aalen-Johansen estimator, 
  and discretized into weeks for both parametric g-computation approaches;
data wide;
	set wide;
	* Administratively censor individuals with t>tau (observed event time and indicator);
	if t <= &tau. then t_star = t;
		else t_star = &tau.; 
	if t <= &tau. then eventtype_star = eventtype;
		else eventtype_star = 0;
	* Time from years to days, discretized into weeks/months;
	t_days = ceil(t_star*365.25);
	t_weeks = ceil(t_star*365.25/7);
	t_months = ceil(t_star*365.25/30.44);
	* Indicate timescale for g-comp.;
	timescale = &timescale.;
	rename eventtype_star=d;
run;

/* Approach 1: naive Aalen Johansen */
%include "&codepath.01a_naive_AJ.sas";
/* Approach 2: Multiple pooled logistic regression (multiple logit) g-computation */
%include "&codepath.01b_multiple_pooled_logit.sas";
/* Approach 3: Pooled multinomial logistic regression (multinomial) g-computation */
%include "&codepath.01c_pooled_multinomial_logit.sas";
/* Merging output from each approach for comparisons */
%include "&codepath.01d_merge_results.sas";

/* Appendix 1. Absolute value of difference */
data appendix1;
	set results;
	* AJ vs. Multiple Logit;
	r1_aj_multi = abs(r1_aj-r1_multi);
	r2_aj_multi = abs(r2_aj-r2_multi);
	* AJ vs. Multinomial;
	r1_aj_multinomial = abs(r1_aj-r1_multinomial);
	r2_aj_multinomial = abs(r2_aj-r2_multinomial);
	* Multiple Logit vs. Multinomial;
	r1_multi_multinomial = abs(r1_multi -r1_multinomial);
	r2_multi_multinomial = abs(r2_multi-r2_multinomial);
run;
* Maximum absolute value of differences;
proc means data=appendix1 max;
	var r1_aj_: r2_aj_: r1_multi_multinomial r2_multi_multinomial; * Matches R output!;
run;

/* Figure 1 - Comparing approaches */
* Cumulative incidence of HAART initiaion;
proc sgplot data = results noborder noautolegend;
	step x = t_days y = r1_aj / name = "risk1_aj" legendlabel = "Aalen-Johansen" lineattrs=(pattern = solid color=green thickness=1);
	step x = t_days y = r1_multi / name = "risk1_multi" legendlabel = "Multiple Logit" lineattrs=(pattern = longdash color=red thickness=1);
	step x = t_days y = r1_multinomial / name = "risk1_multinomial" legendlabel = "Multinomial" lineattrs=(pattern = shortdash color=blue thickness=1);
	yaxis label = "Risk (HAART Initiation)" values = (0 to 0.35 by .05);
	keylegend "risk1_aj" "risk1_multi" "risk1_multinomial" / position=topleft across=1 location = inside;
run;
* Cumulative incidence of AIDS/Death prior to HAART initiation;
proc sgplot data = results noborder noautolegend;
	step x = t_days y = r2_aj / name = "risk2_aj" legendlabel = "Aalen-Johansen" lineattrs=(pattern = solid color=green thickness=1);
	step x = t_days y = r2_multi / name = "risk2_multi" legendlabel = "Multiple Logit" lineattrs=(pattern = longdash color=red thickness=1);
	step x = t_days y = r2_multinomial / name = "risk2_multinomial" legendlabel = "Multinomial" lineattrs=(pattern = shortdash color=blue thickness=1);
	yaxis label = "Risk (AIDS/Death)" values = (0 to 0.35 by .05);
	keylegend "risk2_aj" "risk2_multi" "risk2_multinomial" / position=topleft across=1 location = inside;
run;
