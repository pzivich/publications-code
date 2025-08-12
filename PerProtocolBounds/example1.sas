/******************************************************************************************
* Example 1: IPOP data from Cole et al. 2023
*
* Paul Zivich
******************************************************************************************/

filename in "C:\path\to\data\ipop.dat";

/**************************************
Import IPOP data
***************************************/
DATA dat;
    INFILE in ;                          * reading in from above path;
    INPUT id cervix p17 preterm adhere;  * column names;
	PUT _infile_ ;
RUN;


/**************************************
Estimating the ITT
***************************************/
PROC GENMOD DATA=dat DESCENDING;
	MODEL preterm = p17 / LINK=log DIST=binomial;  * log-binomial model for the risk ratio ;
	TITLE "Intent-to-Treat";
	ESTIMATE 'RR' p17 1;
RUN;

PROC GENMOD DATA=dat DESCENDING;
	MODEL preterm = p17 / LINK=identity DIST=binomial; * identity-binomial model for the risk difference ;
	TITLE "Intent-to-Treat";
	ESTIMATE 'RD' p17 1;
RUN;


/**************************************
Per-Protocol by Naively Censoring
***************************************/
PROC GENMOD DATA=dat DESCENDING;
	MODEL preterm = p17 / LINK=log DIST=binomial; * log-binomial model for the risk ratio ;
	WHERE adhere = 1;                             * restricted to those who adhered;
	TITLE "Per-Protocol : Naive";
	ESTIMATE 'RR' p17 1;
RUN;

PROC GENMOD DATA=dat DESCENDING;
	MODEL preterm = p17 / LINK=identity DIST=binomial; * identity-binomial model for the risk difference ;
	WHERE adhere = 1;                                  * restricted to those who adhered;
	TITLE "Per-Protocol : Naive";
	ESTIMATE 'RD' p17 1;
RUN;


/**************************************
Estimating the Upper Bounds
***************************************/
DATA upper;
	SET dat;
	IF adhere = 0 AND p17 = 1 THEN preterm = 1; * set those with treatment and non-adherence to have worst outcome ;
	IF adhere = 0 AND p17 = 0 THEN preterm = 0; * set those with no treatment and non-adherence to have best outcome ;
RUN;


PROC GENMOD DATA=upper DESCENDING;
	MODEL preterm = p17 / LINK=log DIST=binomial; * log-binomial model for the risk ratio ;
	TITLE "Per-Protocol : Upper Bound";
	ESTIMATE 'RR' p17 1;
	ODS OUTPUT ParameterEstimates=upper_rr;
RUN;

PROC GENMOD DATA=upper DESCENDING;
	MODEL preterm = p17 / LINK=identity DIST=binomial; * identity-binomial model for the risk difference ;
	TITLE "Per-Protocol : Upper Bound";
	ESTIMATE 'RD' p17 1;
	ODS OUTPUT ParameterEstimates=upper_rd;
RUN;


/**************************************
Estimating the Lower Bounds
***************************************/
DATA lower;
	SET dat;
	IF adhere = 0 AND p17 = 1 THEN preterm = 0; * set those with treatment and non-adherence to have best outcome ;
	IF adhere = 0 AND p17 = 0 THEN preterm = 1; * set those with no treatment and non-adherence to have worst outcome ;
RUN;


PROC GENMOD DATA=lower DESCENDING;
	MODEL preterm = p17 / LINK=log DIST=binomial; * log-binomial model for the risk ratio ;
	TITLE "Per-Protocol : Lower Bound";
	ESTIMATE 'RR' p17 1;
	ODS OUTPUT ParameterEstimates=lower_rr;
RUN;

PROC GENMOD DATA=lower DESCENDING;
	MODEL preterm = p17 / LINK=identity DIST=binomial; * identity-binomial model for the risk difference ;
	TITLE "Per-Protocol : Lower Bound";
	ESTIMATE 'RD' p17 1;
	ODS OUTPUT ParameterEstimates=lower_rd;
RUN;


/**************************************
Cleaning up bound presentation
***************************************/
DATA upper_rr;
	SET upper_rr;
	* Keeping a subset of the output and applying link function;
	KEEP parameter bound_u cl_u;
	bound_u = exp(estimate);
	cl_u = exp(upperwaldcl);
	WHERE parameter = 'p17';
RUN;

DATA lower_rr;
	SET lower_rr;
	KEEP parameter cl_l bound_l;
	* Keeping a subset of the output and applying link function;
	cl_l = exp(lowerwaldcl);
	bound_l = exp(estimate);
	WHERE parameter = 'p17';
RUN;

DATA bound_rr;
	MERGE lower_rr upper_rr;
	* Merging output together;
	BY parameter;
RUN;

PROC PRINT DATA=bound_rr;
	TITLE "Bound for the Risk Ratio";
RUN;

DATA upper_rd;
	SET upper_rd;
	* Keeping a subset of the output and applying link function;
	KEEP parameter bound_u cl_u;
	bound_u = estimate;
	cl_u = upperwaldcl;
	WHERE parameter = 'p17';
RUN;

DATA lower_rd;
	SET lower_rd;
	* Keeping a subset of the output and applying link function;
	KEEP parameter cl_l bound_l;
	cl_l = lowerwaldcl;
	bound_l = estimate;
	WHERE parameter = 'p17';
RUN;

DATA bound_rd;
	MERGE lower_rd upper_rd;
	BY parameter;
RUN;

PROC PRINT DATA=bound_rd;
	* Merging output together;
	TITLE "Bound for the Risk Difference";
RUN;
