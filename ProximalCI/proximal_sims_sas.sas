/********************************************************************************************************************
* Introducing Proximal Causal Inference for Epidemiologists
*
*   SAS code the the described simulations
*
* (2022/08/02)
********************************************************************************************************************/

*Write log to file;
proc printto log="&filepath.\ProximalCI_log.log"; run;

%let k = 4000;    *Number of simulations;
%let n = 500;     *Sample size;
%let seed = 15;   *RNG Seed;
%let truth = -1;  *True ACE;

%macro odsoff(); ods graphics off; ods exclude all; ods noresults; %mend;
%macro odson(); ods graphics on; ods exclude none; ods results; %mend;
options minoperator;

********************************************************;
*Generate data;
%odsoff; options nonotes;
data ds;                                       *Generate a data set with n records for this scenario (1, 2, or 3);
	call streaminit(&seed.);
	do simnumber=1 to &k.;
		do Scenario = 1 to 3;
			if scenario = 1 then do;
				beta_u = 0;
				beta_z = 0;
				end;
			else if scenario = 2 then do;
				beta_u = 1;
				beta_z = 0;
				end;
			else if scenario = 3 then do;
				beta_u = 1;
				beta_z = 1;
				end;
			do i = 1 to &n.;
				x  = rand("normal");			                       /* Generate traditional observed confounder (age) */
				u = x + rand("normal"); 		                       /* Generate unobserved confounder (immune function) */		
				z  = u + rand("normal");		                       /* Generate treatment/action proxy (CD4 cell count) */
				w = u + rand("normal");	                               /* Generate outcome proxy (self-rated health) */

				/* Generate action of interest */
				a = rand("bernoulli", 1/(1+exp(-1*(z + beta_u*u + x))));

				/* Generate potential outcomes */
				y0 = w + beta_u*u + x + beta_z*z + rand("normal");     /* Generate potential outcome where a = 0 */
				y1 = y0 + &truth.;                                     /* Generate potential outcome where a = 1 */

				/* Generate (observed) outcome of interest */
				if a = 1 then y = y1;
					else if a = 0 then y = y0;

				output;

			end;                                                      * n loop;
		end;                                                          * scenario loop;
	end;                                                              * simnumber loop;
run; 

proc sort data=ds; by simnumber scenario; run;


*********************************************************;
*Define macro for g computation examples using M estimation;
%macro m_est (approach);

data res_&approach.; set _null_; run;

%do scen=1 %to 3;
%do simn=1 %to &k.;

data this_res_&approach.; set _null_; run;

data iml_ds; set ds (where=(scenario=&scen. and simnumber=&simn.)); run;

proc iml;
	*Read data;
		use iml_ds;
			read all var {a} into a;
			read all var {y} into y;
			read all var {v} into v; 
			read all var {z} into z; 
			read all var {w} into w;
			read all var {x} into x;
		close iml_ds;
	*Sample size;
		n = nrow(y);

	%if &approach. = minstdgcomp %then %do;
		q = 4; %end;                        * 4 parameters in y_hat model;
	%if &approach. = stdgcomp %then %do;
		q = 5; %end;                        * 5 parameters in y_hat model (includes z variable);
	%if &approach. = proxygcomp %then %do;
		q = 8; %end;                        * 4 parameters in w_hat model plus 4 parameters in y_hat model;

	start ef(beta) global(n, a, y, v, z, w, x, w_hat, y_hat);	

		*Create design matrices and define estimating functions;
		%if &approach. = minstdgcomp %then %do;
			y_matrix = j(n, 1, 1)|| a || w || x;
			y_hat = beta[1] + beta[2] *a + beta[3]*w + beta[4]*x; *Multiply the data matrix and parameter values to compute predicted values for y; %end;

		%if &approach. = stdgcomp %then %do;
			y_matrix = j(n, 1, 1)|| a || w || x || z;
			y_hat = beta[1] + beta[2]*a + beta[3]*w + beta[4]*x + beta[5]*z; *Compute predicted values for y; %end;

		%if &approach. = proxygcomp %then %do;
			w_matrix = j(n, 1, 1)|| a || z || x; *Create design matrix for w model variables;
			w_hat = beta[1] + beta[2]*a + beta[3]*z + beta[4]*x; *Multiply the data matrix and parameter values to compute predicted values for w;
			w_f = w_matrix` * (w-w_hat);
			y_matrix = j(n, 1, 1)|| a || x || w_hat; *Create design matrix for y model variables;		
			y_hat =  beta[5] + beta[6]*a + beta[7]*x + beta[8]*w_hat; *Compute predicted values for y; %end;

		y_f = y_matrix` * (y-y_hat);

		%if &approach. IN minstdgcomp stdgcomp %then %do;
			return(y_f); %end;
		%if &approach. = proxygcomp %then %do;
			*Stack functions;
				f = w_f // y_f;
			return(f); %end;

	finish ef;
	*Find the roots (b_hat);
		beta = j(1, q, 0.05); *Initial parameter values;
		optn = q || 1; *q roots/parameters;
		tc = j(1, 12, .); *Missing values set to defaults;
		tc[6] = 1e-8; *Default is 1e-5;
		call nlplm(rc, b_hat, "ef", beta, optn,, tc); *Levenberg-Marquardt least squares method;

	*Bread;
		par = j(1, 3, .); *par is a length 3 vector of details;
		par[1] = q; *tell FD we have q parameters;
		call nlpfdd(func, bread, hess, "ef", b_hat, par);
		*Evaluate derivative at b_hat;	
		bread = - (bread) / n; *negative derivative, averaged;

	*Meat;

		*Mimics estimating functions, but uses b_hat instead of beta and outputs n x p rather than p x 1;
		%if &approach. = minstdgcomp %then %do;
			y_matrix = j(n, 1, 1)|| a || w || x;
			y_hat = b_hat[1] + b_hat[2]*a + b_hat[3]*w + b_hat[4]*x; *Compute predicted values for y;	%end;

		%if &approach. = stdgcomp %then %do;
			y_matrix = j(n, 1, 1)|| a || w || x || z;
			y_hat = b_hat[1] + b_hat[2]*a + b_hat[3]*w + b_hat[4]*x + b_hat[5]*z; *Compute predicted values for y; %end;

		%if &approach. = proxygcomp %then %do;
			w_matrix = j(n, 1, 1)|| a || z || x;
			w_hat = b_hat[1] + b_hat[2]*a + b_hat[3]*z + b_hat[4]*x; *Multiply the data matrix and parameter values to compute predicted values for w;
			w_f = w_matrix # (w-w_hat);
			y_matrix = j(n, 1, 1)|| a || x || w_hat;		
			y_hat =  b_hat[5] + b_hat[6]*a + b_hat[7]*x + b_hat[8]*w_hat; *Compute predicted values for y; %end;

		y_f = y_matrix # (y-y_hat);

		meat   =	j(q, q, 0);
		ef_hat =	j(q, 1, 0);
		temp   =	j(n, 1, 0);

		%if &approach. IN minstdgcomp stdgcomp %then %do;
			meat = (y_f` * y_f) / n; %end;
		%if &approach. = proxygcomp %then %do;
			*Stack w & y functions horizontally;
				f = w_f || y_f; *Results in an n x 8 matrix;
				meat = (f` * f) / n; %end;

	*Sandwich;
		sandwich = (inv(bread) * meat * inv(bread)` ) / n; * / n sizes it to your study;
		beta = b_hat`;
		stderr = sqrt(vecdiag(sandwich));
		lower = beta - 1.96 # stderr;
		upper = beta + 1.96 # stderr;

	*Create data set;
	create this_res_&approach.
		var {beta stderr lower upper}; 
		append;
		close this_res_&approach.;
	quit;
run;

*Add scenario & simulation number to result;
data this_res_&approach.2;
	set this_res_&approach.;
	scenario=&scen.;
	simnumber=&simn.;
	approach="&approach.";
	%if &approach. IN minstdgcomp stdgcomp %then %do;
		if _N_=2 then output; *Keep beta2 (coefficient of a in y model); %end;
	%if &approach. = proxygcomp %then %do;
 		if _N_=6 then output; *Keep beta6 (coefficient of a in y model); %end;
run;

*Append results;
data res_&approach.;
	set res_&approach. this_res_&approach.2;
run;

%end;
%end;

%mend m_est;

%m_est(approach = minstdgcomp);
%m_est(approach = stdgcomp);
%m_est(approach = proxygcomp);

*Compile results;
options notes; %odson;

data compare (drop=upper lower);
	set res_minstdgcomp res_stdgcomp res_proxygcomp;
		Bias = beta - &truth.; *beta is the coefficient of a;
run;

proc means data = compare mean std nway;
	class scenario approach;
	var bias;
	output out = bias_res (drop = _:) mean = Bias std = ESE;
run;

proc means data = compare mean nway;
	class scenario approach;
	var stderr;
	output out = ase_res (drop = _:) mean = ASE;
run;

data coverage;
	set compare;
	lcl = beta - 1.96*stderr;
	ucl = beta + 1.96*stderr;
	if lcl < &truth. < ucl then cov=1; 
		else cov=0;
run;

proc means data = coverage mean nway;
	class scenario approach;
	var cov;
	output out = cov_res (drop = _:) mean = Coverage;
run;

data mean_res;
	merge bias_res ase_res cov_res;
	by scenario approach;
run;

proc format;
	value $approachf
	minstdgcomp="Minimal standard g-computation"
	stdgcomp="Standard g-computation"
	proxygcomp="Proxy g-computation";
run;

data compare_3; 
	set mean_res;
	RMSE=sqrt(bias*bias + ese*ese);
	SER=ase/ese;
	format approach $approachf.;
	if approach="minstdgcomp" then approachorder=1;
		else if approach="stdgcomp" then approachorder=2;
		else if approach="proxygcomp" then approachorder=3;
run;

proc sort data = compare_3 out = results;
	by scenario approachorder;
run;

proc printto log=log; run;

*Export results to table;
ods tagsets.ExcelXP file="&filepath.\Results_SAS.xls" style=journal;
proc print data = results noobs;
	var scenario approach;
	var bias ESE / style={TAGATTR='format:0.000'};
	var SER / style={TAGATTR='format:0.00'};
	var RMSE / style={TAGATTR='format:0.000'};
	var coverage / style={TAGATTR='format:0.0%'};
run;

ods tagsets.ExcelXP close;
