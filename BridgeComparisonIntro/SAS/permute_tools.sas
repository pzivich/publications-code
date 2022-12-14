/***********************************************************************************
* ACTG 320 - ACTG 175 Fusion: permutation macros
*       This script contains macros to make the permutation calculations less 
*		'bulky' in the corresponding actg_diagnostic.sas file. Notice that these 
*		macros are not fully general (they require the column naming conventions of
*		the ACTG example).
*
* Paul Zivich (2022/6/23)
* adapted from Steve Cole
***********************************************************************************/


%MACRO permutation (data=, permutation_n=10000);
	/*****************************************************
	MACRO to conduct the permutation test.
		Runs the permutation procedure and prints the 
		resulting P-value of the procedure to the Viewer

	data:          input data with estimated probabilities
	
	permutation_n: number of permutation iterations. 
                   Default is 10000
	*****************************************************/	

	/*Suppressing log output (to prevent it filling up)*/
	OPTIONS NOSOURCE NONOTES;

	/*Calculating observed area*/
	%permute_iter(data=&data, permute=0)
	RUN;
	DATA area;
   		SET area end=lr;
   		IF lr THEN CALL SYMPUTX('obsarea', tarea);
	RUN;

	/*Creating empty data for permutation storage*/
	DATA perm;
		SET area;
		STOP;
	RUN;
	
	/*Calculating area under permutations*/
	%DO i = 0 %TO &permutation_n %BY 1;
		%permute_iter(data=&data, permute=1)
		RUN;

		DATA perm;
			SET perm area;
		RUN;
	%END;

	/*Calculating bigger values*/
	DATA perm;
		SET perm;
		obs_area = &obsarea;
		IF tarea > obs_area THEN DO;
			pvalue = 1;
			END;
		ELSE DO;
			pvalue = 0;
			END;
	RUN;

	/*Unsuppressing log output*/
	OPTIONS SOURCE NOTES;

	/*Displaying the permutation results*/
	PROC MEANS DATA=perm MEAN;
		VAR pvalue obs_area;
		TITLE "Permutation Results (mean is the P-value)";
	RUN;

	/*Removing intermediate data sets*/
	PROC DATASETS LIBRARY=WORK NOLIST;
	    DELETE permn pres shuffled s_vals;
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

		/*Step 3: shared risk difference*/
		sdiff = rs1 - rs0;

		/*Step 4: calculate area*/
		area = ABS(sdiff)*tdiff;

		/*Step 5: cumulative sum*/
		tarea + area;

		/*Keep only total area as output*/
		IF eof = 1 THEN output;
		KEEP tarea;
	RUN;
%MEND;

%MACRO permute_iter (data=, permute=1);
	/*****************************************************
	MACRO to calculate the area for a single permutation
		Runs the calculation of the risk functions and then
		calls the area calculation macro. 
		Options include calculating either the observed
		area or the area when S is random shuffled/permuted

	data:     input data consisting of risks and time
	permute:  whether to permute the S indicator (1: yes,
			  0: no) in the call of the macro
	*****************************************************/	
    %IF &permute = 1 %THEN %DO;
		/*Extracting S indicator*/
		DATA s_vals;
			SET &data;
			KEEP study;
			RENAME study=s_star;
		RUN;
		/*Shuffling order of rows*/
		PROC SURVEYSELECT DATA=s_vals OUT=shuffled rate=1 OUTORDER=random NOPRINT;
		RUN;
		/*Adding shuffled rows back in*/
		DATA permn;
			MERGE &data shuffled;
		RUN;
	%END;
	%ELSE %DO;
		/*Updating label for later*/
		DATA permn;
			SET &data;
			RENAME study=s_star;
		RUN;		
	%END;

	/*Calculating Risks under permuted S*/
	PROC IML;
		* Reading in data for fusion estimator;
		USE permn;
			READ all VAR {s_star} INTO s;                     * sample indicator;
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
		r_a1_s1 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=1,S=1);
		r_a1_s0 = j(1, nrow(times));                          * empty storage for Pr(T < t | A=1,S=0);
		do count = 1 to nrow(times);                          * loop over event times;
			* Calculating component risks at each t;
			r_a1_s1[count] = (1/n1) # sum((s=1) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
			r_a1_s0[count] = (1/n0) # sum((s=0) # (a=1) # y # (t <= times[count]) / (pir # pid # pis));
		end;

		* Calculating shared RD;
		rs1 = r_a1_s1`;                                      * Risk under A=1 among S=1;
		rs0 = r_a1_s0`;                                      * Risk under A=1 among S=0;

		* Creating results data set;
		varnames = {"time", "rs1", "rs0"};                      * Column names;
		create pres from times rs1 rs0[colname = varnames];  * Creating output dataset;
			append from times rs1 rs0;                          * Appending in the corresponding order;
		close pres;                                          * Close the dataset;
		QUIT;
	RUN;
	
	/*Calculating the area between risk functions*/
	%area_calc(data=pres);
%MEND;
