*********************************************************************                                                                    
*  Title:         01c__pooled_multinomial_logit.sas
*  Date:          08/06/2025
*  Author:        Lucas Neuroth 
*------------------------------------------------------------------- 
*  Purpose: Implement pooled multinomial logistic regression (multi-
*           nomial) g-computation with example data 
********************************************************************;

/* Approach 3: Pooled multinomial logistic regression (multinomial) g-computation */
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
proc logistic data=long noprint;
	class event(ref='0') t_out(ref='1') / param=ref; * Discretized time (t_out) will be modeled with disjoint indicators;
	model event = t_out / link = glogit; * glogit for multinomial distribution;
	store multinomial; * storing model coefficients;
run;

/*********************** Step 3: Estimate unconditional discrete-time hazards ***********************/
* Initialize intermediate datasets;
data qqint_a;
	set long;
* Use fit multinomial model to predict hazard of each event type;
proc plm source=multinomial noprint; * source is the multinomial model fit above;
	score data=qqint_a out=qqint_b pred=h / ilink; * predicted probability of outcome saved as h;
run;
* NOTE: proc plm with a multinomial model will output a row for each event type (j=1,...,J) and no event (0);
* For each row h0--hJ will sum to 1 (sanity check);
	* We'll transpose those rows into columns...;
	proc transpose data=qqint_b out=prob(drop=_:) prefix=h ;
	     by id t_out;
	     var h;
	     id _level_;
	run;
	* ...and merge them back into the intermediate dataset;
	data qqint_b;
	  merge qqint_a prob;
	  by id t_out;
	run;

/*********************** Step 4a: Calculate survival to interval k ***********************/
data qqint_c;
	set qqint_b;
	by id;
	retain s;
	if first.id then s = h0; * because p_0 is probability of no event;
	else s = s*(h0); * recall recursive formula for survival;
	lags = lag(s);
run;

/*********************** Step 4b: Get predicted outcome up to time t ***********************/
data qqint_d;
	set qqint_c;
	by id;
	array mu [&J.]; * Will generate j=1,...,J predicted outcomes up to K;
	array h [*] h:; * Using j+1 hazards estimated above;
	retain mu;
	do i=1 to &J.; * Repeat for each event type;
		if first.id then mu[i] = h[i+1]; * First interval: mu_j = hazard j;
		else mu[i] = mu[i] + h[i+1]*lags; * Subsequent intervals: mu_j = mu_j(k-1) + h_j*surv(k-1);
	end;
	drop i;
run;

/*********************** Step 5: Marginalize over all the observations for each interval ***********************/
proc means data=qqint_d nway noprint;
	var s mu:; * mean survival, cumulative incidence;
	class t_out; * for each time interval;
	output out=qqint_surv mean=;
run;

* Outputting results;
data results_multinomial;
	set qqint_surv;
	rename s=surv_multinomial mu1=r1_multinomial mu2=r2_multinomial;
	drop _type_ _freq_;
run;

/* Remove intermediate datasets */
proc datasets lib=work nolist;
   delete qqint:;
run;
quit;
