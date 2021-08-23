/**************************************************************************************************
* Twister Plots
* 
* Paul Zivich (2021/6/29) & Stephen Cole (2021/5/13)
**************************************************************************************************/

/***Setup data***/
%LET file=file/path/to/data_twister.csv;
%LET dir=file/path/to/save/;

ODS LISTING GPATH="&dir" IMAGE_DPI=600;
ODS GRAPHICS / RESET IMAGENAME="twister_plot" BORDER=OFF IMAGEFMT=PNG HEIGHT=7IN WIDTH=5IN;

* Reading in data;
PROC IMPORT DATAFILE="&file" OUT=data DBMS=CSV REPLACE;
	GETNAMES=YES;
RUN;

* Lagging RD, RD_LCL, RD_UCL so jumps in curve are at correct spots;
DATA data;
	SET data;
	srd = lag(rd);  srd_lcl = lag(rd_lcl); srd_ucl = lag(rd_ucl); 
	srr = lag(rr);  srr_lcl = lag(rr_lcl); srr_ucl = lag(rr_ucl); 
	IF t = 0 THEN DO;
		srd = 0;  srd_lcl = 0;  srd_ucl = 0; 
		srr = 1;  srr_lcl = 1;  srr_ucl = 1; 
		END;
RUN;

/***Twister Plot: Risk Difference***/
PROC SGPLOT DATA=data NOAUTOLEGEND NOBORDER;
	TITLE;  /*No title displayed*/
	BAND Y=t /*time column*/
		LOWER=srd_lcl  /*lower confidence limit*/
		UPPER=srd_ucl / /*upper confidence limit*/
		FILLATTRS=(COLOR=black TRANSPARENCY=0.8); /*Set the fill color and transparency*/
	STEP X=srd /*Risk difference column*/
		Y=t / /*time column*/
		LINEATTRS=(COLOR=black); /*Sets the line color*/
	REFLINE 0 / AXIS=X /*Sets a reference line at RD=0*/
		LINEATTRS=(PATTERN=shortdash COLOR=gray); /*Sets as a dashed gray line*/
	XAXIS LABEL="Risk Difference"  /*Sets the x-label*/
		VALUES=(-0.03 TO 0.03 BY 0.01) /*Sets the x-axis marks*/
		OFFSETMIN=0 OFFSETMAX=0;
	YAXIS LABEL="Days" /*Sets the y-label*/
		VALUES=(0 TO 125 BY 7)  /*Defining y-axis marks*/
		OFFSETMIN=0 OFFSETMAX=0;
	/*Top a-axis label for 'favors'*/
	INSET (" " = "Favors Vaccine") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Placebo") / POSITION=topright NOBORDER;
RUN;

/***Twister Plot: Risk Ratio***/
PROC SGPLOT DATA=data NOAUTOLEGEND NOBORDER;
	TITLE;  /*No title displayed*/
	BAND Y=t /*time column*/
		LOWER=srr_lcl  /*lower confidence limit*/
		UPPER=srr_ucl / /*upper confidence limit*/
		FILLATTRS=(COLOR=black TRANSPARENCY=0.8); /*Set the fill color and transparency*/
	STEP X=srr /*Risk difference column*/
		Y=t / /*time column*/
		LINEATTRS=(COLOR=black); /*Sets the line color*/
	REFLINE 1 / AXIS=X /*Sets a reference line at RD=0*/
		LINEATTRS=(PATTERN=shortdash COLOR=gray); /*Sets as a dashed gray line*/
	XAXIS LABEL="Risk Ratio"  /*Sets the x-label*/
		VALUES=(0.1 0.25 0.5 1 2 5 10) /*Sets the x-axis marks*/
		LOGBASE=e  LOGSTYLE=LOGEXPAND TYPE=log /*Log x-scale*/
		OFFSETMIN=0.05 OFFSETMAX=0.05;
	YAXIS LABEL="Days" /*Sets the y-label*/
		VALUES=(0 TO 125 BY 7)  /*Defining y-axis marks*/
		OFFSETMIN=0 OFFSETMAX=0;
	/*Top a-axis label for 'favors'*/
	INSET (" " = "Favors Vaccine") / POSITION=topleft NOBORDER;
	INSET (" " = "Favors Placebo") / POSITION=topright NOBORDER;
RUN;
