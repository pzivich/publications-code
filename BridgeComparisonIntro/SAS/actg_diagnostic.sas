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
%INCLUDE 'C:\Users\zivic\Documents\open-source\publications-code\BridgeComparisonIntro\SAS\Chimera.sas';

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

/*Twister plot diagnostic*/
%diagnostic(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
            sample_model= , 
            censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
            class_vars=karnof_cat, bootstrap_n=1000)
PROC SGPLOT DATA=results NOAUTOLEGEND NOBORDER;
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
%bridge_point(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
              sample_model= , 
              censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
              class_vars=karnof_cat)
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 32.50, P-value < 0.001*/

*clearing log of permutation ouputs;
DM "log; clear; ";


/*** DIAGNOSTIC: adjusted***/

/*Twister plot diagnostic*/
%diagnostic(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
            sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat, 
            censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
            class_vars=karnof_cat, bootstrap_n=1000)
PROC SGPLOT DATA=results NOAUTOLEGEND NOBORDER;
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
%bridge_point(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
              sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat, 
              censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
              class_vars=karnof_cat)
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 30.05, P-value < 0.001*/

*clearing log of permutation ouputs;
DM "log; clear; ";


/*** DIAGNOSTIC: adjusted + CD4***/

/*Twister plot diagnostic*/
%diagnostic(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
            sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat cd4 cd4_rs0 cd4_rs1, 
            censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study cd4 cd4_rs0 cd4_rs1, 
            class_vars=karnof_cat, bootstrap_n=1000)
PROC SGPLOT DATA=results NOAUTOLEGEND NOBORDER;
	TITLE;
	XAXIS LABEL="Risk Difference" VALUES=(-.2 to .2 by .2) OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL= "Days" VALUES=(0 to 400 by 50) OFFSETMIN=0 OFFSETMAX=0;
	BAND Y=time LOWER=lo UPPER=hi / FILLATTRS=(COLOR=LIGGR);
	STEP X=shared Y=time / JUSTIFY=center LINEATTRS=(COLOR=BL);
	REFLINE 0 / AXIS=x LINEATTRS=(PATTERN=shortdash COLOR=GRAY);
	INSET (" " = "Favors Triple ART") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Single ART") / POSITION=topright NOBORDER;
RUN;

/*Running permutation test*/
%bridge_point(data=dat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
              sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat cd4 cd4_rs0 cd4_rs1, 
              censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study cd4 cd4_rs0 cd4_rs1, 
              class_vars=karnof_cat)
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 36.37, P-value < 0.001*/

*clearing log of permutation ouputs;
DM "log; clear; ";


/*** DIAGNOSTIC: CD4-restricted***/

/*Twister plot diagnostic*/
%diagnostic(data=rdat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
            sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat, 
            censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
            class_vars=karnof_cat, bootstrap_n=1000)
PROC SGPLOT DATA=results NOAUTOLEGEND NOBORDER;
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
%bridge_point(data=rdat, treatment=art, outcome=delta, time=t, sample=study, censor=censor, 
              sample_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat, 
              censor_model=male black idu age age_rs0 age_rs1 age_rs2 karnof_cat study, 
              class_vars=karnof_cat)
%permutation(data=fusion, permutation_n=10000);
RUN; /*Area: 9.52, P-value = 0.11*/
