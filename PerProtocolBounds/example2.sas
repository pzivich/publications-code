/******************************************************************************************
* Example 2: ACTG 320 data
*
* Paul Zivich
******************************************************************************************/

filename in "C:\Users\zivic\Documents\open-source\publications-code\PerProtocolBounds\data\actg320.dat";

ODS GRAPHICS OFF;

/**************************************
Import ACTG data
***************************************/
DATA dat;
    INFILE in ;                                       * reading in from above path;
    INPUT id male black hispanic idu art 
          d drop r age karnof days cd4 stop t delta;  * column names;
	PUT _infile_ ;
RUN;

/*Some data processing*/
DATA dat;
	SET dat;
	KEEP id art t delta stop deviate;
	IF stop = . THEN deviate = 0;
		ELSE deviate = 1;
	* IF stop = 0 THEN stop = 1;
RUN;


/**************************************
Helper Function to Process Output
***************************************/

%MACRO get_risks (surv=, time=, r_name=, v_name=);
	/*Macro to take lifetest output and process as needed*/

	*Getting rid of the censor rows;
	DATA &surv;
		SET &surv;
		WHERE _censor_ = 0 OR &time = 0;
	RUN;

	*Deduplicating rows by time;
	DATA &surv;
		SET &surv;
	  	BY &time;
	  		IF last.&time;
	RUN;

	* Outputting risk and variance only;
	DATA &surv;
		SET &surv;
		KEEP &time &r_name &v_name;
		&r_name = 1 - survival;
		&v_name = ((survival - sdf_lcl) / 1.96) ** 2;
	RUN;
%MEND;

%MACRO ffill (data=, var=);
	DATA &data;
		SET &data;
		RETAIN _v1;
		IF not missing(&var) THEN _v1=&var;
			ELSE IF missing(&var) THEN &var=_v1;
		DROP _v1;
	RUN;
%MEND;

%MACRO get_contrasts (surv1=, surv0=, time=, output=);
	/*Macro to take lifetest output and convert to risk difference and risk ratio*/
	
	*Processing survival data for ART = 1;
	%get_risks(surv=&surv1, r_name=r1, time=&time, v_name=var1);

	*Processing survival data for ART = 0;
	%get_risks(surv=&surv0, r_name=r0, time=&time, v_name=var0);
	
	*Merging risk function information together;
	DATA &output;
		MERGE &surv0 &surv1;
		BY &time;
	RUN;
	
	*Forward filling after data merger;
	%ffill(data=&output, var=r1);
	%ffill(data=&output, var=var1);
	%ffill(data=&output, var=r0);
	%ffill(data=&output, var=var0);
	/*
	DATA &output;
		SET &output;
		RETAIN _r1 _var1 _r0 _var0;
		IF not missing(r1) THEN _r1=r1;
			ELSE IF missing(r1) THEN r1=_r1;
		IF not missing(var1) THEN _var1=var1;
			ELSE IF missing(var1) THEN var1=_var1;
		IF not missing(r0) THEN _r0=r0;
			ELSE IF missing(r0) THEN r0=_r0;
		IF not missing(var0) THEN _var0=var0;
			ELSE IF missing(var0) THEN var0=_var0;
		DROP _r1 _var1 _r0 _var0;
	RUN;
	*/

	*Computing Risk Differences and Risk Ratios with 95% CI;
	DATA &output;
		SET &output;
		*Formula comes from delta method;
		rd = r1 - r0;
		se_rd = (var1 + var0) ** 0.5;
		*Formula comes from delta method;
		rr = r1 / r0;
		se_lrr = (var1 / (r1**2) + var0 / (r0**2)) ** 0.5;
		*95% CI from delta method variance;
		rd_lcl = rd - 1.96*se_rd;
		rd_ucl = rd + 1.96*se_rd;
		rr_lcl = exp(log(rr) - 1.96*se_lrr);
		rr_ucl = exp(log(rr) + 1.96*se_lrr);
		*Placeholder constant to make output easy;
		c = 1;
	RUN;
%MEND;


/**************************************
Intent-to-Treat Analysis
***************************************/

* Survival function for ART = 1;
PROC LIFETEST DATA=dat OUTSURV=s1_itt NOPRINT CONFTYPE=LINEAR;
	TIME t*delta(0);
	WHERE art = 1;
RUN;
	
* Survival function for ART = 1;
PROC LIFETEST DATA=dat OUTS=s0_itt NOPRINT CONFTYPE=LINEAR;
	TIME t*delta(0);
	WHERE art = 0;
RUN;

* Displaying results for the final time;
DATA itt_result;
	SET itt;
	BY c;
	if last.c;
RUN;

PROC PRINT DATA=itt_result;
	TITLE "Intent-to-Treat : Risk Difference";
	VAR t rd rd_lcl rd_ucl;
RUN;

PROC PRINT DATA=itt_result;
	TITLE "Intent-to-Treat : Risk Ratio";
	VAR t rr rr_lcl rr_ucl;
RUN;


/**************************************
Per Protocol Bounds
***************************************/

/*Setting up data for bounds*/
DATA bound;
	SET dat;
	*Set follow-up time based on whether deviated;
	IF deviate = 0 THEN t2 = t;
		ELSE t2 = stop;
	*Set events based on whether deviated;
	IF deviate = 0 THEN delta2 = delta;
		ELSE delta2 = 0;
RUN;


/**************************************
Upper Bound
***************************************/

DATA upper;
	SET bound;
	IF deviate = 1 AND art = 1 THEN delta2 = 1;
	IF deviate = 1 AND art = 0 THEN t2 = 365;
RUN;

* Survival function for ART = 1;
PROC LIFETEST DATA=upper OUTSURV=s1_u NOPRINT CONFTYPE=LINEAR;
	TIME t2*delta2(0);
	WHERE art = 1;
RUN;

* Survival function for ART = 1;
PROC LIFETEST DATA=upper OUTS=s0_u NOPRINT CONFTYPE=LINEAR;
	TIME t2*delta2(0);
	WHERE art = 0;
RUN;

*Computing Contrasts using the helper function; 
%get_contrasts(surv1=s1_u, surv0=s0_u, time=t2, output=upp);


/**************************************
Lower Bound
***************************************/

DATA lower;
	SET bound;
	IF deviate = 1 AND art = 1 THEN t2 = 365;
	IF deviate = 1 AND art = 0 THEN delta2 = 1;
RUN;

* Survival function for ART = 1;
PROC LIFETEST DATA=lower OUTSURV=s1_l NOPRINT CONFTYPE=LINEAR;
	TIME t2*delta2(0);
	WHERE art = 1;
RUN;

* Survival function for ART = 1;
PROC LIFETEST DATA=lower OUTS=s0_l NOPRINT CONFTYPE=LINEAR;
	TIME t2*delta2(0);
	WHERE art = 0;
RUN;

*Computing Contrasts using the helper function; 
%get_contrasts(surv1=s1_l, surv0=s0_l, time=t2, output=low);


/**************************************
Mergining Bound Data Together
***************************************/

DATA low;
	SET low;
	KEEP t rd_lower rd_lower_cl rr_lower rr_lower_cl c;
	t = t2;
	rd_lower_cl = rd_lcl;
	rd_lower = rd;
	rr_lower_cl = rr_lcl;
	rr_lower = rr;
RUN;

DATA upp;
	SET upp;
	KEEP t rd_upper rd_upper_cl rr_upper rr_upper_cl;
	t = t2;
	rd_upper = rd;
	rd_upper_cl = rd_ucl;
	rr_upper = rr;
	rr_upper_cl = rr_ucl;
RUN;

DATA ppb;
	MERGE low upp;
	BY t;
RUN;

/*Forward filling in missing measures*/
%ffill(data=ppb, var=c);
%ffill(data=ppb, var=rd_lower);
%ffill(data=ppb, var=rd_upper);
%ffill(data=ppb, var=rd_upper_cl);
%ffill(data=ppb, var=rd_lower_cl);
%ffill(data=ppb, var=rd_lower);
%ffill(data=ppb, var=rd_upper);
%ffill(data=ppb, var=rd_upper_cl);
%ffill(data=ppb, var=rr_lower_cl);
%ffill(data=ppb, var=rr_lower);
%ffill(data=ppb, var=rr_upper);
%ffill(data=ppb, var=rr_upper_cl);

/*Displaying results for the final time*/
DATA ppb_result;
	SET ppb;
	BY c;
	if last.c;
RUN;

PROC PRINT DATA=ppb_result;
	TITLE "Per-Protocol : Risk Difference";
	VAR t rd_lower_cl rd_lower rd_upper rd_upper_cl;
RUN;

PROC PRINT DATA=ppb_result;
	TITLE "Per-Protocol : Risk Ratio";
	VAR t rr_lower_cl rr_lower rr_upper rr_upper_cl;
RUN;


/**************************************
Twister Plot
***************************************/

ODS GRAPHICS ON;

* Some setup for the plot;
DATA last_time;
	input t ;
	cards;
365
;
RUN;

DATA bplot;
	SET ppb last_time;
RUN;

%ffill(data=bplot, var=rd_lower);
%ffill(data=bplot, var=rd_upper);
%ffill(data=bplot, var=rd_upper_cl);
%ffill(data=bplot, var=rd_lower_cl);
%ffill(data=bplot, var=rd_lower);
%ffill(data=bplot, var=rd_upper);
%ffill(data=bplot, var=rd_upper_cl);
%ffill(data=bplot, var=rr_lower_cl);
%ffill(data=bplot, var=rr_lower);
%ffill(data=bplot, var=rr_upper);
%ffill(data=bplot, var=rr_upper_cl);


* Code pulled from: https://github.com/pzivich/publications-code/blob/master/TwisterPlots/twister.sas;
* 	because I don't want to write from scratch;

PROC SGPLOT DATA=bplot NOAUTOLEGEND NOBORDER;
	TITLE;  /*No title displayed*/
	BAND Y=t /*time column*/
		LOWER=rd_lower_cl  /*lower confidence limit*/
		UPPER=rd_upper_cl / /*upper confidence limit*/
		FILLATTRS=(COLOR=black TRANSPARENCY=0.4); /*Set the fill color and transparency*/
	BAND Y=t /*time column*/
		LOWER=rd_lower  /*lower confidence limit*/
		UPPER=rd_upper / /*upper confidence limit*/
		FILLATTRS=(COLOR=black); /*Set the fill color and transparency*/
	REFLINE 0 / AXIS=X /*Sets a reference line at RD=0*/
		LINEATTRS=(PATTERN=shortdash COLOR=gray); /*Sets as a dashed gray line*/
	XAXIS LABEL="Risk Difference"  /*Sets the x-label*/
		VALUES=(-1. TO 1. BY 0.25) /*Sets the x-axis marks*/
		OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL="Days" /*Sets the y-label*/
		VALUES=(0 TO 400 BY 25)  /*Defining y-axis marks*/
		OFFSETMIN=0 OFFSETMAX=0;
	/*Top a-axis label for 'favors'*/
	INSET (" " = "Favors 3-Drug") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors 2-Drug") / POSITION=topright NOBORDER;
RUN;


PROC SGPLOT DATA=bplot NOAUTOLEGEND NOBORDER;
	TITLE;  /*No title displayed*/
	BAND Y=t /*time column*/
		LOWER=rr_lower_cl  /*lower confidence limit*/
		UPPER=rr_upper_cl / /*upper confidence limit*/
		FILLATTRS=(COLOR=black TRANSPARENCY=0.4); /*Set the fill color and transparency*/
	BAND Y=t /*time column*/
		LOWER=rr_lower  /*lower confidence limit*/
		UPPER=rr_upper / /*upper confidence limit*/
		FILLATTRS=(COLOR=black); /*Set the fill color and transparency*/
	REFLINE 1 / AXIS=X /*Sets a reference line at RD=0*/
		LINEATTRS=(PATTERN=shortdash COLOR=gray); /*Sets as a dashed gray line*/
	XAXIS LABEL="Risk Ratio"  /*Sets the x-label*/
		MIN=0.01 MAX=5
		/* TYPE=LOG LOGSTYLE=LOGEXPAND LOGBASE=e */
		/*SAS is unhappy with above line since it can't resolve ln(0)*/
		OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL="Days" /*Sets the y-label*/
		VALUES=(0 TO 400 BY 25)  /*Defining y-axis marks*/
		OFFSETMIN=0 OFFSETMAX=0;
	/*Top a-axis label for 'favors'*/
	INSET (" " = "Favors 3-Drug") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors 2-Drug") / POSITION=topright NOBORDER;
RUN;
