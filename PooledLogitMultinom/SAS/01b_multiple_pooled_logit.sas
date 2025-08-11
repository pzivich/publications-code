*********************************************************************                                                                    
*  Title:         01b_multiple_pooled_logit.sas
*  Date:          08/06/2025
*  Author:        Lucas Neuroth 
*------------------------------------------------------------------- 
*  Purpose: Implement multiple pooled logistic regression (multiple
*           logit) g-computation with example data 
********************************************************************;

/* Approach 2: Multiple pooled logistic regression (multiple logit) g-computation */
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

/*********************** Step 2: Fit multiple pooled logistic regression models ***********************/
* We assume the following order to maintain risk sets: event2 occurs, event1 occurs, censoring occurs, end of interval;
data qqint_a;
	set long;
	if event = 2 then event2 = 1;
		else if event in (1,0) then event2 = 0;
	if event = 1 then event1 = 1;
		else if event=0 then event1 = 0;
		else if event=2 then event1 = .; * Will exclude those with event2 from event1 model (maintain risk set);
run;
* Event type 1 conditional logistic regression model;
proc logistic data=qqint_a noprint;
	class event1(ref='0') t_out(ref='1') / param=ref; * Time modeled with disjoint indicators;
	model event1 = t_out / link = logit;
	store multi1; * Storing model coefficients;
run;
* Event type 2 conditional logistic regression model;
proc logistic data=qqint_a noprint;
	class event2(ref='0') t_out(ref='1') / param=ref; * Time will be modeled with disjoint indicators;
	model event2 = t_out / link = logit;
	store multi2; * Storing model coefficients;
run;

/*********************** Step 3: Estimate conditional discrete-time hazards ***********************/
* Estimated conditional hazard of specified event type for a given interval;
proc plm source=multi1 noprint; * event type 1 model;
	score data=qqint_a out=qqint_b pred=h1 / ilink; * conditional discrete-time hazard saved as h1;
proc plm source=multi2 noprint; * event type 2 model;
	score data=qqint_b out=qqint_b pred=h2 / ilink; * conditional discrete-time hazard saved as h2;
run;
* Estimated probability of no event (p0) is the product of the complement of each conditional hazard;
data qqint_b;
	set qqint_b;
	p0 = (1-h1)*(1-h2); 
run;

/*********************** Step 4a: Get predicted survival up to time t ***********************/
* Predicted event-free survival is the cumulative product of p0;
data qqint_c;
	set qqint_b;
	by id;
	retain s;
	if first.id then s = p0; 
	else s = s*(p0); 
	lags = lag(s);
run;

/*********************** Step 4b: Get predicted outcome up to time t ***********************/
data qqint_d;
	set qqint_c;
	by id;
	if first.id then lags = 1;
	r1 = h1*(1-h2)*lags; * for event-type 1, probability of making it to k (lagged survival), not having j=2 (1-h2), having j=1 (h1);
	r2 = h2*lags; * for event type 1, probability of making it to k (lagged survival), having j=2 (h2);
run;
* Cumulative incidence;
data qqint_d;
	set qqint_d;
	by id;
	array risk [&J.]; * cumulative incidence of j=1,...,J event types;
	array r [*] r:; * j=1,...,J individual cumulative outcomes;
	retain risk;
	do i=1 to &J.;
		if first.id then risk[i] = r[i]; * first row: cumulative incidence = discrete-time hazard;
		else risk[i] = risk[i] + r[i]; * subsequent rows: cumulative incidence = cumulative sum;
	end;
	drop i;
run;

/*********************** Step 5: Marginalize over all the observations for each interval ***********************/
proc means data=qqint_d nway noprint;
	var s risk:; * mean survival, cumulative incidence;
	class t_out; * for each time interval;
	output out=qqint_surv mean=;
run;

* Outputting results;
data results_multi;
	set qqint_surv;
	rename s=surv_multi risk1=r1_multi risk2=r2_multi;
	drop _type_ _freq_;
run;

/* Remove intermediate datasets */
proc datasets lib=work nolist;
   delete qqint:;
run;
quit;
