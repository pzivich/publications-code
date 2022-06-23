/***********************************************************************************
* ACTG 320 - ACTG 175 Fusion: diagnostic application
*       This script runs the proposed diagnostics using the ACTG example. Graphical 
*		and permutation tests are both demonstrated.
*
* Paul Zivich (2022/6/23)
* adapted from Steve Cole
***********************************************************************************/

/*** DATA SETUP***/

/*Loading permutation macro*/
%INCLUDE 'C:\Users\zivic\Documents\open-source\publications-code\BridgeComparisonIntro\SAS\permute_tools.sas';

/*Import ACTG data*/
PROC IMPORT OUT=dat 
            DATAFILE="C:\Users\zivic\Documents\open-source\publications-code\BridgeComparisonIntro\data\actg_data_formatted.csv" 
            DBMS=CSV REPLACE;
	GETNAMES=YES;
    DATAROW=2; 
RUN;

/*Variable transformations*/
PROC SORT DATA=dat;  * Sorting by location, treatment, and time;
	BY study art t;
RUN;

DATA dat;
	SET dat;
	IF censor = 1 THEN t + 0.0001;   * Shifting censor to prevent any ties;
RUN;

DATA rdat;                           * Restricted data set for later;
	SET dat;
	WHERE CD4 >= 50 and CD4 <= 300;  * Restricting data by baseline CD4;
RUN;

/*** DIAGNOSTIC: unadjusted***/

/* Nuisance models */

* Sampling weights;
PROC LOGISTIC DATA=dat DESC NOPRINT;    * Pr(S | W);
	CLASS karnof_cat / PARAM=ref DESC;
	MODEL study = ;                     * Unadjusted is the null;
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
PROC LOGISTIC DATA=dat_pr_s DESC NOPRINT;
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
PROC PHREG DATA=dat_pr_sa NOPRINT;
	WHERE study=0;                       * Restrict model to ACTG 175;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat / 
		CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
	STRATA art;                          * stratified by ART;
	OUTPUT OUT=c175 SURVIVAL=pid / METHOD=BRESLOW;
RUN;

PROC PHREG DATA=dat_pr_sa NOPRINT;
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
		READ all VAR {study} INTO s;       * sample indicator;
		READ all VAR {art}   INTO a;       * treatment indicator;
		READ all VAR {delta} INTO y;       * outcome indicator;
		READ all VAR {t}  INTO t;          * time;
		READ all VAR {pir}   INTO pir;     * Pr(A | S);
		READ all VAR {pis}   INTO pis;     * Pr(C > T | A,W);
		READ all VAR {pid}   INTO pid;     * Pr(S=1 | W) / Pr(S=0 | W);
	CLOSE fusion;

	* Divisor for the summations;
	n1 = sum(s / pis);                     * Number of observations in S=1;
	n0 = sum((1 - s) / pis);               * Number of observations in S=0;
	
	*RD;
	times = t[loc((t # y) > 0)];           * all event times;
	times = insert(times, max(t), 1);      * adding max t (last time in data);
	call sort(times);                      * has duplicates, not a problem but could slow code down if lots;
	r_a1_s1 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=1);
	r_a1_s0 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=0);
	do count = 1 to nrow(times);           * loop over event times;
		* Calculating component risks at each t;
		r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
		r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
	end;
	shared = r_a1_s1` - r_a1_s0`;          * Calculating bridge diagnostic;
	sse = j(1, nrow(times));               * Storage for SE(RD(t));
	do count = 1 to nrow(times);           * loop over event times again for SE;
		sse[count] = sum( 
			( 	(s=1) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis) - 
			 	(s=0) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis)
			- shared[count] )
		##2);
	end;
	se = sqrt( sse / n1##2) `;             * standard error calculation;

	* Creating results data set;
	varnames = {"time", "shared", "se", "lo", "hi"};
	create diag from times shared se [colname=varnames];
		append from times shared se ;
	close diag;
	QUIT;
RUN;

DATA diag;
	SET diag;
	lo = shared - 1.96*se;
	hi = shared + 1.96*se;
RUN;

*Twister plot of unadjusted diagnostic;
PROC SGPLOT DATA=diag NOAUTOLEGEND NOBORDER;
	TITLE;
	XAXIS LABEL="Risk Difference" VALUES=(-.2 to .2 by .1) OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL= "Days" VALUES=(0 to 400 by 50) OFFSETMIN=0 OFFSETMAX=0;
	BAND Y=time LOWER=lo UPPER=hi / FILLATTRS=(COLOR=LIGGR);
	STEP X=shared Y=time / JUSTIFY=center LINEATTRS=(COLOR=BL);
	REFLINE 0 / AXIS=x LINEATTRS=(PATTERN=shortdash COLOR=GRAY);
	INSET (" " = "Favors Triple ART") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Single ART") / POSITION=topright NOBORDER;
RUN;

/*Running permutation test*/
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 32.43, P-value < 0.001*/

*clearing log of permutation ouputs;
DM "log; clear; ";

/*** DIAGNOSTIC: adjusted***/

/* Nuisance models */

* Sampling weights;
PROC LOGISTIC DATA=dat DESC NOPRINT;            * Pr(S | W);
	CLASS karnof_cat / PARAM=ref DESC;
	MODEL study = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat;    * Unadjusted is the null;
	OUTPUT OUT=dat_pr_s P=pr_s_w;                                           * output as new data set;
RUN;
DATA dat_pr_s;                          * Getting IOSW;
	SET dat_pr_s;
	IF study THEN pis = 1;              * =1 if in 320;
	ELSE pis = (1-pr_s_w) / pr_s_w;     * =Pr(S=0 | W)/Pr(S=1 | W) otherwise;
	LABEL pr_s_w = ;
	DROP _level_;
RUN;

* Treatment weights;
PROC LOGISTIC DATA=dat_pr_s DESC NOPRINT;
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
PROC PHREG DATA=dat_pr_sa NOPRINT;
	WHERE study=0;                       * Restrict model to ACTG 175;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat / 
		CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
	STRATA art;                          * stratified by ART;
	OUTPUT OUT=c175 SURVIVAL=pid / METHOD=BRESLOW;
RUN;

PROC PHREG DATA=dat_pr_sa NOPRINT;
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
		READ all VAR {study} INTO s;       * sample indicator;
		READ all VAR {art}   INTO a;       * treatment indicator;
		READ all VAR {delta} INTO y;       * outcome indicator;
		READ all VAR {t}  INTO t;          * time;
		READ all VAR {pir}   INTO pir;     * Pr(A | S);
		READ all VAR {pis}   INTO pis;     * Pr(C > T | A,W);
		READ all VAR {pid}   INTO pid;     * Pr(S=1 | W) / Pr(S=0 | W);
	CLOSE fusion;

	* Divisor for the summations;
	n1 = sum(s / pis);                     * Number of observations in S=1;
	n0 = sum((1 - s) / pis);               * Number of observations in S=0;
	
	*RD;
	times = t[loc((t # y) > 0)];           * all event times;
	times = insert(times, max(t), 1);      * adding max t (last time in data);
	call sort(times);                      * has duplicates, not a problem but could slow code down if lots;
	r_a1_s1 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=1);
	r_a1_s0 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=0);
	do count = 1 to nrow(times);           * loop over event times;
		* Calculating component risks at each t;
		r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
		r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
	end;
	shared = r_a1_s1` - r_a1_s0`;          * Calculating bridge diagnostic;
	sse = j(1, nrow(times));               * Storage for SE(RD(t));
	do count = 1 to nrow(times);           * loop over event times again for SE;
		sse[count] = sum( 
			( 	(s=1) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis) - 
			 	(s=0) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis)
			- shared[count] )
		##2);
	end;
	se = sqrt( sse / n1##2) `;             * standard error calculation;

	* Creating results data set;
	varnames = {"time", "shared", "se", "lo", "hi"};
	create diag from times shared se [colname=varnames];
		append from times shared se ;
	close diag;
	QUIT;
RUN;

DATA diag;
	SET diag;
	lo = shared - 1.96*se;
	hi = shared + 1.96*se;
RUN;

*Twister plot of unadjusted diagnostic;
PROC SGPLOT DATA=diag NOAUTOLEGEND NOBORDER;
	TITLE;
	XAXIS LABEL="Risk Difference" VALUES=(-.2 to .2 by .1) OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL= "Days" VALUES=(0 to 400 by 50) OFFSETMIN=0 OFFSETMAX=0;
	BAND Y=time LOWER=lo UPPER=hi / FILLATTRS=(COLOR=LIGGR);
	STEP X=shared Y=time / JUSTIFY=center LINEATTRS=(COLOR=BL);
	REFLINE 0 / AXIS=x LINEATTRS=(PATTERN=shortdash COLOR=GRAY);
	INSET (" " = "Favors Triple ART") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Single ART") / POSITION=topright NOBORDER;
RUN;

/*Running permutation test*/
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 29.99, P-value < 0.001*/

*clearing log of permutation ouputs;
DM "log; clear; ";


/*** DIAGNOSTIC: adjusted + CD4***/

/* Nuisance models */

* Sampling weights;
PROC LOGISTIC DATA=dat DESC NOPRINT;            * Pr(S | W);
	CLASS karnof_cat / PARAM=ref DESC;
	MODEL study = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat     /* Unadjusted is the null*/
                  cd4 cd4_rs0 cd4_rs1;    
	OUTPUT OUT=dat_pr_s P=pr_s_w;                                           * output as new data set;
RUN;
DATA dat_pr_s;                          * Getting IOSW;
	SET dat_pr_s;
	IF study THEN pis = 1;              * =1 if in 320;
	ELSE pis = (1-pr_s_w) / pr_s_w;     * =Pr(S=0 | W)/Pr(S=1 | W) otherwise;
	LABEL pr_s_w = ;
	DROP _level_;
RUN;

* Treatment weights;
PROC LOGISTIC DATA=dat_pr_s DESC NOPRINT;
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
PROC PHREG DATA=dat_pr_sa NOPRINT;
	WHERE study=0;                       * Restrict model to ACTG 175;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat
                        cd4 cd4_rs0 cd4_rs1 / 
		CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
	STRATA art;                          * stratified by ART;
	OUTPUT OUT=c175 SURVIVAL=pid / METHOD=BRESLOW;
RUN;

PROC PHREG DATA=dat_pr_sa NOPRINT;
	WHERE study=1;                       * Restrict model to ACTG 320;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat
                        cd4 cd4_rs0 cd4_rs1 / 
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
		READ all VAR {study} INTO s;       * sample indicator;
		READ all VAR {art}   INTO a;       * treatment indicator;
		READ all VAR {delta} INTO y;       * outcome indicator;
		READ all VAR {t}  INTO t;          * time;
		READ all VAR {pir}   INTO pir;     * Pr(A | S);
		READ all VAR {pis}   INTO pis;     * Pr(C > T | A,W);
		READ all VAR {pid}   INTO pid;     * Pr(S=1 | W) / Pr(S=0 | W);
	CLOSE fusion;

	* Divisor for the summations;
	n1 = sum(s / pis);                     * Number of observations in S=1;
	n0 = sum((1 - s) / pis);               * Number of observations in S=0;
	
	*RD;
	times = t[loc((t # y) > 0)];           * all event times;
	times = insert(times, max(t), 1);      * adding max t (last time in data);
	call sort(times);                      * has duplicates, not a problem but could slow code down if lots;
	r_a1_s1 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=1);
	r_a1_s0 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=0);
	do count = 1 to nrow(times);           * loop over event times;
		* Calculating component risks at each t;
		r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
		r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
	end;
	shared = r_a1_s1` - r_a1_s0`;          * Calculating bridge diagnostic;
	sse = j(1, nrow(times));               * Storage for SE(RD(t));
	do count = 1 to nrow(times);           * loop over event times again for SE;
		sse[count] = sum( 
			( 	(s=1) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis) - 
			 	(s=0) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis)
			- shared[count] )
		##2);
	end;
	se = sqrt( sse / n1##2) `;             * standard error calculation;

	* Creating results data set;
	varnames = {"time", "shared", "se", "lo", "hi"};
	create diag from times shared se [colname=varnames];
		append from times shared se ;
	close diag;
	QUIT;
RUN;

DATA diag;
	SET diag;
	lo = shared - 1.96*se;
	hi = shared + 1.96*se;
RUN;

*Twister plot of unadjusted diagnostic;
PROC SGPLOT DATA=diag NOAUTOLEGEND NOBORDER;
	TITLE;
	XAXIS LABEL="Risk Difference" VALUES=(-.4 to .4 by .2) OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL= "Days" VALUES=(0 to 400 by 50) OFFSETMIN=0 OFFSETMAX=0;
	BAND Y=time LOWER=lo UPPER=hi / FILLATTRS=(COLOR=LIGGR);
	STEP X=shared Y=time / JUSTIFY=center LINEATTRS=(COLOR=BL);
	REFLINE 0 / AXIS=x LINEATTRS=(PATTERN=shortdash COLOR=GRAY);
	INSET (" " = "Favors Triple ART") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Single ART") / POSITION=topright NOBORDER;
RUN;

/*Running permutation test*/
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 36.33, P-value < 0.001*/

*clearing log of permutation ouputs;
DM "log; clear; ";


/*** DIAGNOSTIC: CD4-restricted***/

/* Nuisance models */

* Sampling weights;
PROC LOGISTIC DATA=rdat DESC NOPRINT;            * Pr(S | W);
	CLASS karnof_cat / PARAM=ref DESC;
	MODEL study = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat;    * Unadjusted is the null;
	OUTPUT OUT=dat_pr_s P=pr_s_w;                                           * output as new data set;
RUN;
DATA dat_pr_s;                          * Getting IOSW;
	SET dat_pr_s;
	IF study THEN pis = 1;              * =1 if in 320;
	ELSE pis = (1-pr_s_w) / pr_s_w;     * =Pr(S=0 | W)/Pr(S=1 | W) otherwise;
	LABEL pr_s_w = ;
	DROP _level_;
RUN;

* Treatment weights;
PROC LOGISTIC DATA=dat_pr_s DESC NOPRINT;
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
PROC PHREG DATA=dat_pr_sa NOPRINT;
	WHERE study=0;                       * Restrict model to ACTG 175;
	CLASS karnof_cat / DESC;             * Indicator variable for karnof categories;
	MODEL t*censor(0) = male black idu age age_rs0 age_rs1 age_rs2 karnof_cat / 
		CONVERGELIKE=1e-8 FCONV=1e-8 GCONV=1e-8; 
	STRATA art;                          * stratified by ART;
	OUTPUT OUT=c175 SURVIVAL=pid / METHOD=BRESLOW;
RUN;

PROC PHREG DATA=dat_pr_sa NOPRINT;
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
		READ all VAR {study} INTO s;       * sample indicator;
		READ all VAR {art}   INTO a;       * treatment indicator;
		READ all VAR {delta} INTO y;       * outcome indicator;
		READ all VAR {t}  INTO t;          * time;
		READ all VAR {pir}   INTO pir;     * Pr(A | S);
		READ all VAR {pis}   INTO pis;     * Pr(C > T | A,W);
		READ all VAR {pid}   INTO pid;     * Pr(S=1 | W) / Pr(S=0 | W);
	CLOSE fusion;

	* Divisor for the summations;
	n1 = sum(s / pis);                     * Number of observations in S=1;
	n0 = sum((1 - s) / pis);               * Number of observations in S=0;
	
	*RD;
	times = t[loc((t # y) > 0)];           * all event times;
	times = insert(times, max(t), 1);      * adding max t (last time in data);
	call sort(times);                      * has duplicates, not a problem but could slow code down if lots;
	r_a1_s1 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=1);
	r_a1_s0 = j(1, nrow(times));           * empty storage for Pr(T < t | A=1,S=0);
	do count = 1 to nrow(times);           * loop over event times;
		* Calculating component risks at each t;
		r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
		r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
	end;
	shared = r_a1_s1` - r_a1_s0`;          * Calculating bridge diagnostic;
	sse = j(1, nrow(times));               * Storage for SE(RD(t));
	do count = 1 to nrow(times);           * loop over event times again for SE;
		sse[count] = sum( 
			( 	(s=1) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis) - 
			 	(s=0) # (a=1) # y # (t <= times[count] ) / (pir # pid # pis)
			- shared[count] )
		##2);
	end;
	se = sqrt( sse / n1##2) `;             * standard error calculation;

	* Creating results data set;
	varnames = {"time", "shared", "se", "lo", "hi"};
	create diag from times shared se [colname=varnames];
		append from times shared se ;
	close diag;
	QUIT;
RUN;

DATA diag;
	SET diag;
	lo = shared - 1.96*se;
	hi = shared + 1.96*se;
RUN;

*Twister plot of unadjusted diagnostic;
PROC SGPLOT DATA=diag NOAUTOLEGEND NOBORDER;
	TITLE;
	XAXIS LABEL="Risk Difference" VALUES=(-.2 to .2 by .1) OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL= "Days" VALUES=(0 to 400 by 50) OFFSETMIN=0 OFFSETMAX=0;
	BAND Y=time LOWER=lo UPPER=hi / FILLATTRS=(COLOR=LIGGR);
	STEP X=shared Y=time / JUSTIFY=center LINEATTRS=(COLOR=BL);
	REFLINE 0 / AXIS=x LINEATTRS=(PATTERN=shortdash COLOR=GRAY);
	INSET (" " = "Favors Triple ART") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Single ART") / POSITION=topright NOBORDER;
RUN;

/*Running permutation test*/
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 9.56, P-value = 0.11*/
