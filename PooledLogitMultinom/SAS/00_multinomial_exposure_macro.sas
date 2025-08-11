*********************************************************************                                                                    
*  Title:         00_multinomial_exposure_macro.sas
*  Date:          06/30/2025
*  Author:        Lucas Neuroth 
*------------------------------------------------------------------- 
*  Purpose: Estimate cumulative incidence under specified exposure 
            levels (macro)
********************************************************************;

%macro gcomp(input,id_var,time_var,exposure_var,exposure_lvl,model);
/* Step 3a: Assign exposure level (A) for the long dataset */
data qqint_a;
	set &input.; * input (long) dataset;
	&exposure_var. = &exposure_lvl.; * setting exposure (A) to level in plan (a,1,0,etc.);
run;
/* Step 3b: Predicted discrete-time hazards under exposure level */
proc plm source=&model. noprint; * source is the pooled logistic regression from step 1;
	score data=qqint_a out=qqint_b pred=h / ilink; 
run;
	* Multinomial outputs predicted probabilities as (J+1) rows, transposing back to columns;
	proc transpose data=qqint_b out=prob(drop=_:) prefix=h ;
	     by &id_var. &time_var.;
	     var h;
	     id _level_;
	run;
	* Merging predicted probabilities back to dataset;
	data qqint_b;
	  merge qqint_a prob;
	  by &id_var. &time_var.;
	run;
/* Step 4a: Predicted survival up to interval k */
data qqint_c;
	set qqint_b;
	by &id_var.;
	retain surv;
	if first.&id_var. then surv = h0;
	else surv = surv*(h0);
	lags = lag(surv);
run;
/* Step 4b: Predicted outcomes up to interval k */
data qqint_d;
	set qqint_c;
	by &id_var.;
	* create an array of J cumulative incidence variables r1 to rJ;
	* NOTE: this uses the maximum value of the event indicator J assuming 0 to J coding scheme;
	array mu [&J.];
	* array for the predicted discrete-time hazards. each p[i+1] will correspond to r[i] because p[1]=p_0;
	array h [*] h:;
	* individual cumulative incidence for i=1 to i=J event types;
	retain mu;
	do i=1 to &J.;
		if first.&id_var. then mu[i] = h[i+1];
		else mu[i] = mu[i] + h[i+1]*lags; 
	end;
	drop i;
run;
/* Step 4: Marginalize over all the observations for each interval */
proc means data=qqint_d nway noprint;
	var surv mu:; * mean survival, cumulative incidence;
	class &time_var.; * for each time interval;
	output out=qqint_surv mean=;
run;

/* Output results for exposure level */
data cuminc_&exposure_lvl.;
	set qqint_surv;
	&exposure_var. = "&exposure_lvl.";
	drop _type_ _freq_;
run;

/* Remove intermediate datasets */
proc datasets lib=work nolist;
   delete qqint:;
run;
quit;
%mend;
