*********************************************************************                                                                    
*  Title:         01a_naive_AJ.sas
*  Date:          08/06/2025
*  Author:        Lucas Neuroth 
*------------------------------------------------------------------- 
*  Purpose: Implement naive Aalen-Johansen estimator on example data                                                               
********************************************************************;

/* Approach 1: naive Aalen Johansen */
* NOTE: Using proc lifetest to implement AJ estimator;
proc lifetest data=wide outcif=aj noprint;
	time t_days*d(0) / failcode = 1 2;
run;

proc sort data=aj;
	by t_days;
proc transpose data=aj(where=(CONFTYPE^="")) out=aj2(drop = _name_ _label_) prefix=r;
	by t_days;
	var cif;
	id failcode;
run;

* Save results;
data results_aj;
	set aj2;
	surv_aj = 1 - r1 - r2;
	rename r1=r1_aj r2=r2_aj;
run;

* Remove intermediate datasets;
proc datasets lib=work nolist;
   delete aj:;
run;
quit;
