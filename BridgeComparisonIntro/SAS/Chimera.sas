/***********************************************************************************
* ACTG 320 - ACTG 175 Fusion: bridge estimator macros
*       This script contains macros to make the bridge estimation less 'bulky' in the 
*       corresponding actg_diagnostic.sas and actg_analysis files. 
*
* Paul Zivich (2023/08/22)
* adapted from Steve Cole
***********************************************************************************/

%MACRO bridge (data=, treatment=, outcome=, time=, sample=, censor=, sample_model=, censor_model=, class_vars=, bootstrap_n=200);
	/*****************************************************
	MACRO to compute the estimates with the bridge.
		Runs the point estimation and bootstraps to 
		estimate the variance.

	data:          input data

	treatment:     treatment column name

	outcome:       outcome indicator column name

	time:          follow-up time column name

	sample:        sample indicator name

	censor:        indicator for loss-to-follow-up

	sample_model:  model specification for sampling model

	censor_model:  model specification for censoring model

	class_vars:    variables to include in the CLASS 
 			       statement for the nuisance models

	bootstrap_n:   number of resamples to use for the bootstrap
	*****************************************************/	


	/*Computing point estimates*/
	%bridge_point(data=&data, 
                  treatment=&treatment, 
                  outcome=&outcome, 
                  time=&time,  
                  sample=&sample, 
                  censor=&censor, 
                  sample_model=&sample_model, 
                  censor_model=&censor_model, 
                  class_vars=&class_vars)
  	DATA results;
		SET result;
	RUN;

	/*Computing variance estimates*/
	* Storage for bootstrap iterations;
	DATA boot;
		SET result;
		KEEP time;
	RUN;

	*Suppressing log output (to prevent it filling up);
	OPTIONS NOSOURCE NONOTES;

	* Running bootstrap iterations;
	%DO i = 1 %TO &bootstrap_n %BY 1;
		/*Resampling with replacement*/
		PROC SURVEYSELECT DATA=&data OUT=bdat 
									 SEED=0
                                     METHOD=URS
                                     SAMPRATE=1
                                     OUTHITS
                                     rep=1 NOPRINT;        
		RUN;

		/*Calculating point estimate*/
		%bridge_point(data=bdat, 
                      treatment=&treatment, 
                      outcome=&outcome, 
                      time=&time,  
                      sample=&sample, 
                      censor=&censor, 
                      sample_model=&sample_model, 
                      censor_model=&censor_model, 
                      class_vars=&class_vars)
		RUN;
		DATA result;
			SET result;
			RENAME rd=rd&i;
		RUN;

		/*Mergining columns together*/
		DATA boot;
			MERGE boot(in=T1) result(in=T2);
			IF T1;
			BY time;
		RUN;
		
		/*Forward-filling risk difference*/
		DATA boot;
			SET boot;
			retain _rd&i;
			IF _n_=1 AND MISSING(rd&i) THEN rd&i=0;
			IF NOT MISSING(rd&i) THEN _rd&i=rd&i;
			ELSE IF MISSING(rd&i) THEN rd&i=_rd&i;
			DROP _rd&i;
		RUN;
	%END;

	*Unsuppressing log output;
	OPTIONS SOURCE NOTES;

	/*Calculating variance estimates*/
	PROC TRANSPOSE DATA=boot out=boot_t name=t prefix=time;
		VAR rd1-rd&bootstrap_n;
	RUN;
	PROC MEANS DATA=boot_t NOPRINT;
		OUTPUT OUT=var STD=;
	RUN;
	DATA var;
		SET var;
		DROP _TYPE_ _FREQ_;
	RUN;
	PROC TRANSPOSE DATA=var out=var prefix=sd;
	RUN;
	DATA var;
		SET var;
		se = sd1;
		KEEP se;
	RUN;

	/*Putting all the results together*/
	DATA results;
		MERGE results var;
		lo = rd - 1.96*se;   * Lower 95% confidence interval;
		hi = rd + 1.96*se;	
	RUN;

	/*Removing intermediate data sets*/
	PROC DATASETS LIBRARY=WORK NOLIST;
	    DELETE result boot boot_t bdat var fusion;
	QUIT;
	RUN;
%MEND;

%MACRO bridge_point (data=, treatment=, outcome=, time=, sample=, censor=, sample_model=, censor_model=, class_vars=);
	/*****************************************************
	MACRO to compute the point estimates with the bridge.
		Runs the point estimation procedure. Used for both
		point and bootstrap variance estimation.

	data:          input data

	treatment:     treatment column name

	outcome:       outcome indicator column name

	time:          follow-up time column name

	sample:        sample indicator name

	censor:        indicator for loss-to-follow-up

	sample_model:  model specification for sampling model

	censor_model:  model specification for censoring model

	class_vars:    variables to include in the CLASS 
 			       statement for the nuisance models
	*****************************************************/	

	/*Estimating Nuisance Models*/
	* Sampling weights;
	PROC LOGISTIC DATA=&data DESC NOPRINT;    * Pr(S | W);
		CLASS &class_vars / PARAM=ref DESC;
		MODEL &sample = &sample_model;
		OUTPUT OUT=fusion P=pr_s_w;           * output as new data set;
	RUN;
	DATA fusion;                              * Getting IOSW;
		SET fusion;
		IF study THEN pis = 1;                * =1 if in 320;
		ELSE pis = (1-pr_s_w) / pr_s_w;       * =Pr(S=0 | W)/Pr(S=1 | W) otherwise;
		LABEL pr_s_w = ;
		DROP _level_;
	RUN;

	* Treatment weights;
	PROC LOGISTIC DATA=fusion DESC NOPRINT;
		MODEL &treatment = ;                * null model for Pr(A | S);
		BY &sample;                         * by study to condition on S;
		OUTPUT OUT=fusion P=pir;            * output as new data set;
	RUN;
	DATA fusion;                            * assigning Pr(A=a | S) given A_i;
		SET fusion;
		IF &treatment=1 AND &sample=1 THEN pir = 1 - pir;
		IF &treatment=0 AND &sample=0 THEN pir = 1 - pir;
		DROP _level_;
	RUN;

	* Censoring weights;
	PROC PHREG DATA=fusion NOPRINT;
		CLASS &class_vars / DESC;                   * Indicator variable for karnof categories;
		MODEL &time*&censor(0) = &censor_model / 
			CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
		STRATA &treatment;                          * stratified by ART;
		OUTPUT OUT=fusion SURVIVAL=pid / METHOD=BRESLOW;
	RUN;
	
	/*Data Prep after nuisance*/
	PROC SORT DATA=fusion;                    * Re-sorting to ensure correct order;
		BY &sample &treatment &time;
	RUN;

	/*Estimating Parameter of Interest*/
	PROC IML;
		* Reading in data for fusion estimator;
		USE fusion;
			READ all VAR {&sample}    INTO s;                 * sample indicator;
			READ all VAR {&treatment} INTO a;                 * treatment indicator;
			READ all VAR {&outcome}   INTO y;                 * outcome indicator;
			READ all VAR {&time}      INTO t;                 * time;
			READ all VAR {pir}        INTO pir;               * Pr(A | S);
			READ all VAR {pis}        INTO pis;               * Pr(C > T | A,W);
			READ all VAR {pid}        INTO pid;               * Pr(S=1 | W) / Pr(S=0 | W);
		CLOSE fusion;

		* Divisor (n) for the summations;
		n1 = sum(s / pis);                                    * Number of observations in S=1;
		n0 = sum((1 - s) / pis);                              * Number of observations in S=0;
		
		*Corresponding calculations for the risks;
		times = t[loc((t # y) > 0)];                          * all event times;
		times = insert(times, max(t), 1);                     * adding max t (last time in data);
		call sort(times);                                     * has duplicates, not a problem but could slow code down if lots;
		r_a2_s1 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=2,S=1);
		r_a1_s1 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=1,S=1);
		r_a1_s0 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=1,S=0);
		r_a0_s0 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=0,S=0);
		do count = 1 to nrow(times);                          * loop over event times;
			* Calculating component risks at each t;
			r_a2_s1[count] = (1/n1) # sum((s=1) # (a=2) # y # (t <= times[count]) / (pir # pid # pis));
			r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
			r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
			r_a0_s0[count] = (1/n0) # sum((s=0) # (a=0) # y # (t <= times[count]) / (pir # pid # pis));
		end;

		* Calculating the bridged RD;
		rd = (r_a2_s1` - r_a1_s1`) + (r_a1_s0` - r_a0_s0`);	  * Calculating bridged risk difference;

		* Creating results data set;
		varnames = {"time", "rd"};                            * Column names;
		create result from times rd[colname = varnames];      * Creating output dataset;
			append from times rd;                             * Appending in the corresponding order;
		close result;                                         * Close the dataset;
		QUIT;
	RUN;

	/*Removing any duplicated times happening*/
	PROC SORT DATA=result out=result NODUPKEY;
    	BY time;
	RUN;
%MEND;

%MACRO diagnostic_plot (data=, treatment=, outcome=, time=, sample=, censor=, sample_model=, censor_model=, class_vars=, bootstrap_n=200);
	/*****************************************************
	MACRO to compute the diagnostic estimates.
		Runs the point estimation and bootstraps to 
		estimate the variance.

	data:          input data

	treatment:     treatment column name

	outcome:       outcome indicator column name

	time:          follow-up time column name

	sample:        sample indicator name

	censor:        indicator for loss-to-follow-up

	sample_model:  model specification for sampling model

	censor_model:  model specification for censoring model

	class_vars:    variables to include in the CLASS 
 			       statement for the nuisance models

	bootstrap_n:   number of resamples to use for the bootstrap
	*****************************************************/	


	/*Computing point estimates*/
	%diagnostic_point(data=&data, 
                      treatment=&treatment, 
                      outcome=&outcome, 
                      time=&time,  
                      sample=&sample, 
                      censor=&censor, 
                      sample_model=&sample_model, 
                      censor_model=&censor_model, 
                      class_vars=&class_vars)
  	DATA results;
		SET result;
	RUN;

	/*Computing variance estimates*/
	* Storage for bootstrap iterations;
	DATA boot;
		SET result;
		KEEP time;
	RUN;

	*Suppressing log output (to prevent it filling up);
	OPTIONS NOSOURCE NONOTES;

	* Running bootstrap iterations;
	%DO i = 1 %TO &bootstrap_n %BY 1;
		/*Resampling with replacement*/
		PROC SURVEYSELECT DATA=&data OUT=bdat 
									 SEED=0
                                     METHOD=URS
                                     SAMPRATE=1
                                     OUTHITS
                                     rep=1 NOPRINT;        
		RUN;

		/*Calculating point estimate*/
		%diagnostic_point(data=bdat, 
                          treatment=&treatment, 
                          outcome=&outcome, 
                          time=&time,  
                          sample=&sample, 
                          censor=&censor, 
                          sample_model=&sample_model, 
                          censor_model=&censor_model, 
                          class_vars=&class_vars)
		RUN;
		DATA result;
			SET result;
			RENAME shared=shared&i;
		RUN;

		/*Mergining columns together*/
		DATA boot;
			MERGE boot(in=T1) result(in=T2);
			IF T1;
			BY time;
		RUN;
		
		/*Forward-filling risk difference*/
		DATA boot;
			SET boot;
			retain _shared&i;
			IF _n_=1 AND MISSING(shared&i) THEN shared&i=0;
			IF NOT MISSING(shared&i) THEN _shared&i=shared&i;
			ELSE IF MISSING(shared&i) THEN shared&i=_shared&i;
			DROP _shared&i;
		RUN;
	%END;

	*Unsuppressing log output;
	OPTIONS SOURCE NOTES;

	/*Calculating variance estimates*/
	PROC TRANSPOSE DATA=boot out=boot_t name=t prefix=time;
		VAR shared1-shared&bootstrap_n;
	RUN;
	PROC MEANS DATA=boot_t NOPRINT;
		OUTPUT OUT=var STD=;
	RUN;
	DATA var;
		SET var;
		DROP _TYPE_ _FREQ_;
	RUN;
	PROC TRANSPOSE DATA=var out=var prefix=sd;
	RUN;
	DATA var;
		SET var;
		se = sd1;
		KEEP se;
	RUN;

	/*Putting all the results together*/
	DATA results;
		MERGE results var;
		lo = shared - 1.96*se;   * Lower 95% confidence interval;
		hi = shared + 1.96*se;	
	RUN;

	/*Removing intermediate data sets*/
	PROC DATASETS LIBRARY=WORK NOLIST;
	    DELETE result boot boot_t bdat var;
	QUIT;
	RUN;
%MEND;


%MACRO diagnostic_test (data=, treatment=, outcome=, time=, sample=, censor=, sample_model=, censor_model=, class_vars=, bootstrap_n=200);

	*Suppressing log output (to prevent it filling up);
	OPTIONS NOSOURCE NONOTES;

	/*Computing point estimates*/
	%diagnostic_test_iter(data=&data, 
                          treatment=&treatment, 
                          outcome=&outcome, 
                          time=&time,  
                          sample=&sample, 
                          censor=&censor, 
                          sample_model=&sample_model, 
                          censor_model=&censor_model, 
                          class_vars=&class_vars)
	RUN;
  	DATA area_results;
		SET area;
	RUN;

	/*Computing variance estimates*/
	* Storage for bootstrap iterations;
	DATA boot;
		SET area;
	RUN;

	* Running bootstrap iterations;
	%DO i = 1 %TO &bootstrap_n %BY 1;
		/*Resampling with replacement*/
		PROC SURVEYSELECT DATA=&data OUT=bdat 
									 SEED=0
                                     METHOD=URS
                                     SAMPRATE=1
                                     OUTHITS
                                     rep=1 NOPRINT;        
		RUN;

		/*Calculating point estimate*/
		%diagnostic_test_iter(data=bdat, 
                      treatment=&treatment, 
                      outcome=&outcome, 
                      time=&time,  
                      sample=&sample, 
                      censor=&censor, 
                      sample_model=&sample_model, 
                      censor_model=&censor_model, 
                      class_vars=&class_vars)
		RUN;

		PROC APPEND BASE=boot DATA=area;
		RUN;
	%END;

	*Unsuppressing log output;
	OPTIONS SOURCE NOTES;

	/*Calculating variance estimates*/
	PROC MEANS DATA=boot NOPRINT;
		OUTPUT OUT=var STD=;
	RUN;
	DATA var;
		SET var;
		DROP _TYPE_ _FREQ_;
	RUN;
	DATA var;
		SET var;
		se = tarea;
		KEEP se;
	RUN;

	/*Putting all the results together*/
	DATA area_results;
		MERGE area_results var;
		lo = tarea - 1.96*se;   * Lower 95% confidence interval;
		hi = tarea + 1.96*se;	
		zscore = tarea / se;
		pvalue = 2*(1-cdf("Normal", abs(zscore)));
	RUN;

	/*Displaying diagnostic test results*/
	PROC PRINT DATA=area_results;
	RUN;

	/*Removing intermediate data sets*/
	PROC DATASETS LIBRARY=WORK NOLIST;
	    DELETE result boot boot_t bdat var fusion;
	QUIT;
	RUN;
	
%MEND;


%MACRO diagnostic_test_iter (data=, treatment=, outcome=, time=, sample=, censor=, sample_model=, censor_model=, class_vars=);
	/*****************************************************
	MACRO to compute the diagnostic test.
		Runs the point estimation and bootstraps to 
		estimate the variance for the integrated risk
		difference.

	data:          input data

	treatment:     treatment column name

	outcome:       outcome indicator column name

	time:          follow-up time column name

	sample:        sample indicator name

	censor:        indicator for loss-to-follow-up

	sample_model:  model specification for sampling model

	censor_model:  model specification for censoring model

	class_vars:    variables to include in the CLASS 
 			       statement for the nuisance models

	bootstrap_n:   number of resamples to use for the bootstrap
	*****************************************************/	

	/*Computing point estimates*/
	%diagnostic_point(data=&data, 
                      treatment=&treatment, 
                      outcome=&outcome, 
                      time=&time,  
                      sample=&sample, 
                      censor=&censor, 
                      sample_model=&sample_model, 
                      censor_model=&censor_model, 
                      class_vars=&class_vars)
	RUN;
  	DATA results;
		SET result;
	RUN;

	%area_calc(data=results);
	RUN;
%MEND;


%MACRO diagnostic_point (data=, treatment=, outcome=, time=, sample=, censor=, sample_model=, censor_model=, class_vars=);
	/*****************************************************
	MACRO to compute the diagnostic point estimates.
		Runs the point estimation procedure. Used for both
		point and bootstrap variance estimation.

	data:          input data

	treatment:     treatment column name

	outcome:       outcome indicator column name

	time:          follow-up time column name

	sample:        sample indicator name

	censor:        indicator for loss-to-follow-up

	sample_model:  model specification for sampling model

	censor_model:  model specification for censoring model

	class_vars:    variables to include in the CLASS 
 			       statement for the nuisance models
	*****************************************************/	

	/*Estimating Nuisance Models*/
	* Sampling weights;
	PROC LOGISTIC DATA=&data DESC NOPRINT;    * Pr(S | W);
		CLASS &class_vars / PARAM=ref DESC;
		MODEL &sample = &sample_model;
		OUTPUT OUT=fusion P=pr_s_w;           * output as new data set;
	RUN;
	DATA fusion;                              * Getting IOSW;
		SET fusion;
		IF study THEN pis = 1;                * =1 if in 320;
		ELSE pis = (1-pr_s_w) / pr_s_w;       * =Pr(S=0 | W)/Pr(S=1 | W) otherwise;
		LABEL pr_s_w = ;
		DROP _level_;
	RUN;

	* Treatment weights;
	PROC LOGISTIC DATA=fusion DESC NOPRINT;
		MODEL &treatment = ;                * null model for Pr(A | S);
		BY &sample;                         * by study to condition on S;
		OUTPUT OUT=fusion P=pir;            * output as new data set;
	RUN;
	DATA fusion;                            * assigning Pr(A=a | S) given A_i;
		SET fusion;
		IF &treatment=1 AND &sample=1 THEN pir = 1 - pir;
		IF &treatment=0 AND &sample=0 THEN pir = 1 - pir;
		DROP _level_;
	RUN;

	* Censoring weights;
	PROC PHREG DATA=fusion NOPRINT;
		CLASS &class_vars / DESC;                   * Indicator variable for karnof categories;
		MODEL &time*&censor(0) = &censor_model / 
			CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
		STRATA &treatment;                          * stratified by ART;
		OUTPUT OUT=fusion SURVIVAL=pid / METHOD=BRESLOW;
	RUN;
	
	/*Data Prep after nuisance*/
	PROC SORT DATA=fusion;                    * Re-sorting to ensure correct order;
		BY &sample &treatment &time;
	RUN;

	/*Estimating Parameter of Interest*/
	PROC IML;
		* Reading in data for fusion estimator;
		USE fusion;
			READ all VAR {&sample}    INTO s;                 * sample indicator;
			READ all VAR {&treatment} INTO a;                 * treatment indicator;
			READ all VAR {&outcome}   INTO y;                 * outcome indicator;
			READ all VAR {&time}      INTO t;                 * time;
			READ all VAR {pir}        INTO pir;               * Pr(A | S);
			READ all VAR {pis}        INTO pis;               * Pr(C > T | A,W);
			READ all VAR {pid}        INTO pid;               * Pr(S=1 | W) / Pr(S=0 | W);
		CLOSE fusion;

		* Divisor (n) for the summations;
		n1 = sum(s / pis);                                    * Number of observations in S=1;
		n0 = sum((1 - s) / pis);                              * Number of observations in S=0;
		
		*Corresponding calculations for the risks;
		times = t[loc((t # y) > 0)];                          * all event times;
		times = insert(times, max(t), 1);                     * adding max t (last time in data);
		call sort(times);                                     * has duplicates, not a problem but could slow code down if lots;
		r_a1_s1 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=1,S=1);
		r_a1_s0 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=1,S=0);
		do count = 1 to nrow(times);                          * loop over event times;
			* Calculating component risks at each t;
			r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
			r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
		end;

		* Calculating the bridged RD;
		shared = r_a1_s1` - r_a1_s0`;	                      * Calculating shared arm difference;

		* Creating results data set;
		varnames = {"time", "shared"};                        * Column names;
		create result from times shared[colname = varnames];  * Creating output dataset;
			append from times shared;                         * Appending in the corresponding order;
		close result;                                         * Close the dataset;
		QUIT;
	RUN;

	/*Removing any duplicated times happening*/
	PROC SORT DATA=result out=result NODUPKEY;
    	BY time;
	RUN;

	/*Removing intermediate data sets*/
	PROC DATASETS LIBRARY=WORK NOLIST;
	    DELETE fusion;
	QUIT;
	RUN;
%MEND;

%MACRO area_calc (data=);
	/*****************************************************
	MACRO to calculate the area between risk functions
		Runs the calculation of the area and returns the 
		total area result in the area data set.
		Steps here refer to the algorithm in the Appendix 
		of the paper.

	data:          input data consisting of risks and time
	*****************************************************/	
	DATA area;
		SET &data END=eof;
		/*Step 2: time difference*/
		%IF eof = 1 %THEN %DO; * IF ELSE statement mimics a LEAD() function;
			tdiff = 0;
	  		%END;
		%ELSE %DO; 
			pt = _N_ + 1;
	  		SET &data (KEEP=time RENAME=(time=next_t)) POINT=pt;
	  		tdiff = next_t - time;
			%END;

		/*Step 3: calculate area*/
		area = shared*tdiff;

		/*Step 4: cumulative sum*/
		tarea + area;

		/*Keep only total area as output*/
		IF eof = 1 THEN output;
		KEEP tarea;
	RUN;
%MEND;
