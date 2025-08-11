*********************************************************************                                                                    
*  Title:         02_applied_example.sas
*  Date:          06/30/2025
*  Author:        Lucas Neuroth 
*------------------------------------------------------------------- 
*  Purpose: Read in example data, fit pooled multinomial logit,
            estimate cumulative incidence under various exposure levels 
********************************************************************;

/*********************** Set-up (global macro variables) ***********************/
* Specify data filepath;
%let datapath = C:\Users\lucasneuroth\OneDrive - University of North Carolina at Chapel Hill\UNC GRA\data\lau2009\;
* Specify code filepath;
%let codepath = C:\Users\lucasneuroth\OneDrive - University of North Carolina at Chapel Hill\UNC GRA\analysis\01_code\09_multinomial_logit_g_formula\SAS\;
* Specify follow-up (years);
%let tau = 2;
* Specify discretized timescale for g-comp. approaches (weeks corresponds to analysis presented in manuscript);
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

/*********************** Step 1: Convert wide dataset to long dataset    ***********************/
* Wide data to initial intermediate dataset, rename time;
data qqint_a;
	set wide;
	* setting discretized timescale for long dataset (time_up);
	if timescale = 'days' then time_up = t_days;
	else if timescale = 'weeks' then time_up = t_weeks;
	else if timescale = 'months' then time_up = t_months;
run;

* Storing maximum discretized time (max_t) and number of event types (J) as global macro variables;
proc sql noprint; 
   select max(time_up) into: max_t
   from qqint_a;
   select max(d) into: J
   from qqint_a;
quit;
%put &max_t.;
%put &J.;

* Creating initial elongated dataset: max_t copies for each id;
data qqint_b (drop=i);
	set qqint_a;
	do i=1 to &max_t;
		output;
	end;
run;

* Each row (interval) has an in-time and out-time. We'll be modeling the out-time;
data qqint_c;
	set qqint_b;
	by id;
	retain t_out;
	if first.id then t_out = 1; * First interval is (0-1];
	else t_out = t_out + 1;
run;

* Interval-specific event indicator;
* Event is 0 until the interval an event occurs (i.e. time_up);
* When t_out = time_up the the indicator will be >=1 for an event and 0 if censored;
* Time after time_up will have missing value (excludes them from modeling step);
data long;
	set qqint_c;
	if t_out < time_up then event = 0;
	else if t_out = time_up then event = d;
	else if t_out > time_up then event = .;
run;

* Remove intermediate datasets;
proc datasets lib=work nolist;
   delete qqint:;
run;
quit;

/*********************** Step 2: Fit pooled multinomial logit model ***********************/
* NOTE: Took approximately 10 minutes to fit the model;
proc logistic data=long noprint;
	/* Time: Disjoint indicator coding */
	class event(ref='0') baseidu(ref='0') t_out(ref='1') black(ref='0') / param=ref;
	model event = baseidu|t_out baseidu|black baseidu|cd4nadir|cd4nadir|cd4nadir baseidu|ageatfda|ageatfda|ageatfda / link = glogit;
	/* Time: Cubic polynommal          */
	*class event(ref='0') baseidu(ref='0') black(ref='0') / param=ref;
	*model event = baseidu|t_out|t_out|t_out baseidu|black baseidu|cd4nadir|cd4nadir|cd4nadir baseidu|ageatfda|ageatfda|ageatfda / link = glogit;
	* Store model coefficients for later use;
	store multinomial;
run;

/*********************** Steps 3-5: Estimate cumulative incidence under each treatment plan ***********************/
%include "&codepath.00_multinomial_exposure_macro.SAS";
* All women had a history of injection drug use (IDU) at baseline (baseidu=1);
%gcomp(input = long, id_var = id, time_var = t_out,
	   exposure_var = baseidu, exposure_lvl = 1,
	   model = multinomial);
* No women had a history of injection drug use (non-IDU) at baseline (baseidu=0);
%gcomp(input = long, id_var = id, time_var = t_out,
	   exposure_var = baseidu, exposure_lvl = 0,
	   model = multinomial);
* Natural course (distribution of history of injection drug use at baseline);
%gcomp(input = long, id_var = id, time_var = t_out,
	   exposure_var = baseidu, exposure_lvl = baseidu,
	   model = multinomial);

/*********************** Plotting cumulative incidence curves (IDU vs. non-IDU) ***********************/
data cumulative_incidence;
	merge cuminc_1 (rename=(mu1=r1a1 mu2=r2a1)) cuminc_0 (rename=(mu1=r1a0 mu2=r2a0));
	by t_out;
	* RDs contrasting all idu (a1) with all non-idu (a0);
	rd1 = r1a1-r1a0;
	rd2 = r2a1-r2a0;
	keep t_out r1a1 r1a0 rd1 r2a1 r2a0 rd2;
run;

* Cumulative incidence of HAART initiaion;
proc sgplot data = cumulative_incidence noborder noautolegend;
	step x = t_out y = r1a1 / name = "riska1" legendlabel = "IDU" lineattrs=(pattern = solid color=blue thickness=1);
	step x = t_out y = r1a0 / name = "riska0" legendlabel = "non-IDU" lineattrs=(pattern = shortdash color=blue thickness=1);
	xaxis label = &timescale.;
	yaxis label = "Risk (HAART Initiation)" values = (0 to 0.4 by .1);
	keylegend "riska1" "riska0" / position=topleft across=1 location = inside;
run;
* Cumulative incidence of AIDS/Death prior to HAART initiation;
proc sgplot data = cumulative_incidence noborder noautolegend;
	step x = t_out y = r2a1 / name = "riska1" legendlabel = "IDU" lineattrs=(pattern = solid color=red thickness=1);
	step x = t_out y = r2a0 / name = "riska0" legendlabel = "non-IDU" lineattrs=(pattern = shortdash color=red thickness=1);
	xaxis label = &timescale.;
	yaxis label = "Risk (AIDS/Death)" values = (0 to 0.4 by .1);
	keylegend "riska1" "riska0" / position=topleft across=1 location = inside;
run;
* Risk difference plot;
proc sgplot data = cumulative_incidence noborder noautolegend;
	step x = t_out y = rd1 / name = "risk1" legendlabel = "HAART Initiation" lineattrs=(pattern = solid color=blue thickness=1);
	step x = t_out y = rd2 / name = "risk2" legendlabel = "AIDS/Death" lineattrs=(pattern = solid color=red thickness=1);
	refline 0 / axis = y;
	xaxis label = &timescale.;
	yaxis label = "Risk Difference (IDU vs. non-IDU)" values = (-0.15 to 0.15 by .05);
	keylegend "risk1" "risk2" / position=topleft across=1 location = inside;
run;
