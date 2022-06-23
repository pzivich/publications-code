/***********************************************************************************
* ACTG 320 - ACTG 175 Fusion: main analysis
*       This script runs the procedure for the estimation of the risk difference
*
* Paul Zivich (2022/6/23)
* adapted from Steve Cole
***********************************************************************************/

/*** DATA SETUP***/

/*Import ACTG data*/
PROC IMPORT OUT=dat 
            /* DATAFILE="publications-code\BridgeComparisonIntro\data\actg_data_formatted.csv" */
            DATAFILE="C:\Users\zivic\Documents\open-source\publications-code\BridgeComparisonIntro\data\actg_data_formatted.csv" 
            DBMS=CSV REPLACE;
	GETNAMES=YES;
    DATAROW=2; 
RUN;

/*Variable transformations*/
DATA dat;
	SET dat;
	WHERE CD4 >= 50 and CD4 <= 300;  * Restricting data by baseline CD4;
	IF censor = 1 THEN t + 0.0001;   * Shifting censor to prevent any ties;
RUN;

PROC SORT DATA=dat;                  * Sorting by location, treatment, and time;
	BY study art t;
RUN;

/*** Bridged Treatment Comparison Fusion Estimator ***/

/* Nuisance models */

* Sampling weights;
PROC LOGISTIC DATA=dat DESC;            * Pr(S | W);
	CLASS karnof_cat / PARAM=ref DESC;
	MODEL study = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat;
	OUTPUT OUT=dat_pr_s P=pr_s_w;       * output as new data set;
RUN;
DATA dat_pr_s;                          * Getting IOSW;
	SET dat_pr_s;
	IF study THEN pis = 1;              * =1 if in 320;
	ELSE pis = (1-pr_s_w) / pr_s_w;     * =Pr(S=0 | W)/Pr(S=1 | W) otherwise;
	LABEL pr_s_w = ;
	DROP _level_;
RUN;

* Treatment weights;
PROC LOGISTIC DATA=dat_pr_s DESC;
	MODEL art = ;                       * null model for Pr(A | S);
	BY study;                           * by study to condition on S;
	OUTPUT OUT=dat_pr_sa P=pir;         * output as new data set;
RUN;
DATA dat_pr_sa;                         * assigning Pr(A=a | S) given A_i;
	SET dat_pr_sa;
	IF art=1 AND study=1 THEN pir = 1 - pir;
	IF art=0 AND study=0 THEN pir = 1 - pir;
	DROP _level_;
RUN;

* Censoring weights;
PROC PHREG DATA=dat_pr_sa;
	WHERE study=0;                       * Restrict model to ACTG 175;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat / 
		CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
	STRATA art;                          * stratified by ART;
	OUTPUT OUT=c175 SURVIVAL=pid / METHOD=BRESLOW;
RUN;

PROC PHREG DATA=dat_pr_sa;
	WHERE study=1;                       * Restrict model to ACTG 320;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat / 
		CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
	STRATA art;                          * stratified by ART;
	OUTPUT OUT=c320 SURVIVAL=pid / METHOD=BRESLOW;
RUN;

* Combining everything back into the fusion data set;
DATA fusion;
	SET c175 c320;
RUN;

/* FUSION ESTIMATOR */
PROC SORT DATA=fusion;                    * Re-sorting to ensure correct order;
	BY study art t;
RUN;

PROC IML;
	* Reading in data for fusion estimator;
	USE fusion;
		READ all VAR {study} INTO s;                      * sample indicator;
		READ all VAR {art}   INTO a;                      * treatment indicator;
		READ all VAR {delta} INTO y;                      * outcome indicator;
		READ all VAR {t}  INTO t;                         * time;
		READ all VAR {pir}   INTO pir;                    * Pr(A | S);
		READ all VAR {pis}   INTO pis;                    * Pr(C > T | A,W);
		READ all VAR {pid}   INTO pid;                    * Pr(S=1 | W) / Pr(S=0 | W);
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

	* Calculating the SE(RD) via Breskin et al. (2021);
	sse = j(1, nrow(times));                              * Storage for SE(RD(t));
	do count = 1 to nrow(times);                          * loop over event times again for SE;
		sse[count] = sum( 
			( 	(s=1) # (a=2) # y # (t <= times[count] ) / (pir # pid # pis)  - 
			 	(s=1) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis)  + 
			 	(s=0) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis)  - 
				(s=0) # (a=0) # y # (t <= times[count] ) / (pir # pid # pis)
			- rd[count] )
		##2);
	end;
	se = sqrt( sse / n1##2) `;                            * standard error calculation;

	* Creating results data set;
	varnames = {"time", "rd", "se"};                      * Column names;
	create results from times rd se[colname = varnames];  * Creating output dataset;
		append from times rd se;                          * Appending in the corresponding order;
	close results;                                        * Close the dataset;
	QUIT;
RUN;

* Calculating confidence intervals;
DATA results;
	SET results;
	lo = rd - 1.96*se;   * Lower 95% confidence interval;
	hi = rd + 1.96*se;   * Upper 95% confidence interval;
RUN;

* Risk Difference at t=365;
DATA est365;
	SET results nobs=nobs;
	IF _n_=nobs;            * Selecting out last observation;
RUN;
PROC PRINT DATA=est365;     * Printing the data with only the last obs;
RUN;

*Twister plot of main results;
PROC SGPLOT DATA=results NOAUTOLEGEND NOBORDER;
	TITLE;                                                                           * Blank title;
	XAXIS LABEL="Risk Difference" VALUES=(-.4 to .4 by .2) OFFSETMIN=0 OFFSETMAX=0;  * X-axis setup;
	YAXIS LABEL= "Days" VALUES=(0 to 400 by 50) OFFSETMIN=0 OFFSETMAX=0;             * Y-axis setup;
	BAND Y=time LOWER=lo UPPER=hi / FILLATTRS=(COLOR=LIGGR);                         * 95% CI shaded region;
	STEP X=rd Y=time / JUSTIFY=center LINEATTRS=(COLOR=BL);                          * Step function for point estimate;
	REFLINE 0 / AXIS=x LINEATTRS=(PATTERN=shortdash COLOR=GRAY);                     * Adding reference line at RD=0;
	INSET (" " = "Favors Triple ART") / POSITION=topleft NOBORDER;                   * Inlay favors triple label;
	INSET (" " = "Favors Single ART") / POSITION=topright NOBORDER;                  * Inlay favors mono label;
RUN;
