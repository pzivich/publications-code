/***********************************************************************************
* ACTG 320 - ACTG 175 Fusion: main analysis
*       This script runs the procedure for the estimation of the risk difference
*
* Paul Zivich (2023/08/22)
* adapted from Steve Cole
***********************************************************************************/

/*** DATA SETUP***/

%INCLUDE 'C:\Users\zivic\Documents\open-source\publications-code\BridgeComparisonIntro\SAS\Chimera.sas';

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
%bridge(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
        sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat, 
        censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
        class_vars=karnof_cat, bootstrap_n=1000)

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
