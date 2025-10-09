/******************************************************************************************************************
* Confidence Regions for Multiple Outcomes, Effect Modifiers, and Other Multiple Comparisons
*
* Stephen Cole (with formatting modifications from Paul Zivich)
******************************************************************************************************************/

/* Setup */
options pagesize = 60 linesize = 80 nodate pageno = 1;
dm "log; clear; out; clear;";
%LET dir = C:\Users\file\path\;
ods listing gpath = "&dir.\" image_dpi = 600;

PROC IMPORT DATAFILE = "&dir.actg.csv"
    OUT = a
    DBMS = csv
    REPLACE;
    GUESSINGROWS = max;
    GETNAMES = yes;
DATA a;
	SET a;
	y1 = cd4_20wk;
	y2 = cd8_20wk;
	a = treat;
	KEEP y1 y2 a male cd4c_0wk cd4_rs1 cd4_rs2 cd4_rs3;
RUN;


/* Example 1 */

PROC IML;
	TITLE "Confidence Bands by supt";
	TITLE2 "Example 1";
	USE a;
		read all var {a} into a;
		read all var {y1} into y1;
		read all var {y2} into y2;
	CLOSE a;
	n = nrow(y1);
	q = 6;

	* Defining estimating functions;
	START f(beta) global(n, a, y1, y2);
		f1 = a # (y1 - beta[1]);
		f2 = (1 - a) # (y1 - beta[2]);
		f3 = a # (y2 - beta[3]);
		f4 = (1 - a) # (y2 - beta[4]);
		f5 = j(n, 1, (beta[1] - beta[2]) - beta[5]);
		f6 = j(n, 1, (beta[3] - beta[4]) - beta[6]);
		f = f1 || f2 || f3 || f4 || f5 || f6;
		return(f);
	FINISH f;

    * M-estimation procedure;
	START ee(beta);
		ef = f(beta);
		return(ef[+,]);       								
	FINISH ee;

	initial = j(q, 1, 0);
	optn = q || 0;
	tc = j(1, 12, .);
	tc[1:2] = 999; tc[3:12] = 1e-12;
	call nlplm(rc, betahat, "ee", initial, optn, , tc);
	par = q||.||.;
	call nlpfdd(func, deriv, na, "ee", betahat, par);
	bread = - (deriv) / n;
	residuals = f(betahat);
	outerprod = residuals` * residuals;
	filling = outerprod / n;
	sandwich = ( inv(bread) * filling * inv(bread)` ) / n;
	variable = {"E(Y1|A=1)","E(Y1|A=0)","E(Y2|A=1)","E(Y2|A=0)","RD Y1","RD Y2"};

    * Estimation;
	est = betahat`;
	se = sqrt(vecdiag(sandwich));
	alpha = .05;
	k = 2;

	* Standard CI;
	c = probit(1 - alpha / 2);
	lcl = est - c * se;
	ucl = est + c * se;
	pvalue = 1 - cdf("chisquare", ((est - 0) / se) ## 2, 1);
	print variable est se lcl ucl pvalue;

	* Bonferroni;
	cb = probit(1 - alpha / (2 * k));
	lclb = est[5:6] - cb * se[5:6];
	uclb = est[5:6] + cb * se[5:6];
	print cb, lclb uclb;

	* supt;
	cov = sandwich[5:6, 5:6];
	se2 = sqrt(vecdiag(cov));
	call randseed(1);
	m = 100000;
    x = abs(randnormal(m, {0, 0}, cov) / se2`);
	maxx = x[,<>]; * max col for each sim;
	* No built-in function for quantiles in IML!;
	call sort(maxx); * sort ascending;
	pos = 1 + (1 - alpha) * (m - 1);
	lo = floor(pos);
   	hi = ceil(pos);
   	frac = pos - lo;
	if lo = hi then qval = maxx[lo];
   	else cs = (1 - frac) * maxx[lo] + frac * maxx[hi];        * linear interpolation;
	lcls = est[5:6] - cs * se[5:6];
	ucls = est[5:6] + cs * se[5:6];
	print cs, lcls ucls;
QUIT;
RUN;


/* Example 2 */

PROC IML;
	TITLE "Confidence Bands by supt";
	TITLE2 "Example 2";
	USE a;
		read all var {a} into a;
		read all var {y1} into y;
		read all var {male} into w;
	CLOSE a;
	n = nrow(y);
	q = 4;
	aw = a#w;
	x = j(n, 1, 1) || a || w || aw;                      * design matrix;

	* Defining estimating functions;
	START f(beta) global(n, w, a, y, x);
		mu = x * beta[1:4];
		* f = (x` * (y - mu))`;
		f = x # repeat(y-mu, 1, ncol(x));                * Gives n x q, sum over n later;
	    return(f);
	FINISH f;

    * M-estimation procedure;
	START ee(beta);
		ef = f(beta);
		return(ef[+,]);       								
	FINISH ee;

	initial = j(q, 1, 0);
	optn = q || 0;
	tc = j(1, 12, .);
	tc[1:2] = 999; tc[3:12] = 1e-12;
	call nlplm(rc, betahat, "ee", initial, optn, , tc);
	par = q||.||.;
	call nlpfdd(func, deriv, na, "ee", betahat, par);
	bread = - (deriv) / n;
	residuals = f(betahat);
	outerprod = residuals` * residuals;
	filling = outerprod / n;
	sandwich = ( inv(bread) * filling * inv(bread)` ) / n;
	variable = {"Int","a","w","aw"};

    * Estimation;
	est = betahat`;
	se = sqrt(vecdiag(sandwich));
	alpha = .05;

	* Standard CI;
	c = probit(1 - alpha / 2);
	lcl = est - c * se;
	ucl = est + c * se;
	pvalue = 1 - cdf("chisquare", ((est - 0) / se) ## 2, 1);
	print variable est se lcl ucl pvalue;

	* Bonferroni;
	k = 4;
	cb = probit(1 - alpha / (2 * k));
	lclb = est - cb * se;
	uclb = est + cb * se;
	print cb, lclb uclb;

	* supt;
	call randseed(1);
	m = 100000;
    x = abs(randnormal(m, {0, 0, 0, 0}, sandwich) / se`);
	maxx = x[,<>]; * max col for each sim;
	* No built-in function for quantiles in IML!;
	call sort(maxx); * sort ascending;
	pos = 1 + (1 - alpha) * (m - 1);
	lo = floor(pos);
   	hi = ceil(pos);
   	frac = pos - lo;
	if lo = hi then qval = maxx[lo];
   	else cs = (1 - frac) * maxx[lo] + frac * maxx[hi];           * linear interpolation;
	lcls = est - cs * se;
	ucls = est + cs * se;
	print cs, lcls ucls;
quit;
run;


/* Example 3 */

PROC IMPORT DATAFILE = "&dir.actg_predict_g50.csv"
    OUT = b
    DBMS = csv
    REPLACE;
    GUESSINGROWS = max;
    GETNAMES = yes;

DATA b;
    SET b;
    a = treat;
    drop treat;
RUN;

PROC IML;
	TITLE "Confidence Bands by supt";
	TITLE2 "Example 3";
	USE a;
		read all var {a} into a;
		read all var {y1} into y;
		read all var {cd4c_0wk} into w0;
		read all var {cd4_rs1} into w1;
		read all var {cd4_rs2} into w2;
		read all var {cd4_rs3} into w3;
	CLOSE a;
	n = nrow(y);
	q = 10;
	x = j(n, 1, 1) || a || w0 || w1 || w2 || w3 || 
		a#w0 || a#w1 || a#w2 || a#w3;                         * design matrix;

	* Defining estimating functions;
	START f(beta) global(n, a, y, x);
		mu = x * beta[1:10];
		f = x # repeat(y-mu, 1, ncol(x));                     * Gives n x q, sum over n later;
    	return(f);
	FINISH f;

    * M-estimation procedure;
	START ee(beta);
		ef = f(beta);
		return(ef[+,]);       								
	FINISH ee;

	initial = j(q, 1, 0);
	optn = q || 0;
	tc = j(1, 12, .);
	tc[1:2] = 999; tc[3:12] = 1e-12;
	call nlplm(rc, betahat, "ee", initial, optn, , tc);
	par = q||.||.;
	call nlpfdd(func, deriv, na, "ee", betahat, par);
	bread = - (deriv) / n;
	residuals = f(betahat);
	outerprod = residuals`                                       * residuals;
	filling = outerprod / n;
	sandwich = ( inv(bread) * filling * inv(bread)` ) / n;
	variable = {"Int","a","w0","w1","w2","w3","a*w0","a*w1","a*w2","a*w3"};

    * Estimation;
	est = betahat`;
	se = sqrt(vecdiag(sandwich));
	alpha = .05;

	* Make predictions on 50 point grid;
	* Read prediction data, make predictions based on est, get cov...;
	USE b;
		read all var {cd4_0wk} into pred;
		read all var {a} into a2;
		read all var {cd4c_0wk} into w02;
		read all var {cd4_rs1} into w12;
		read all var {cd4_rs2} into w22;
		read all var {cd4_rs3} into w32;
	CLOSE b;
	xp = j(50, 1, 1) || a2 || w02 || w12 || w22 || w32 || 
		a2#w02 || a2#w12 || a2#w22 || a2#w32;
	yhat = xp * est;
	ycov = xp * sandwich * xp`;
	yse = sqrt(vecdiag(ycov));
	
	* Standard CI;
	c = probit(1 - alpha / 2);
	lcl = yhat - c * yse;
	ucl = yhat + c * yse;
	print c, lcl ucl;

	* Bonferroni;
	k = 50;
	cb = probit(1 - alpha / (2 * k));
	lclb = yhat - cb * yse;
	uclb = yhat + cb * yse;
	print cb, lclb uclb;

	* supt;
	call randseed(1);
	m = 100000;
	zeros = j(50, 1, 0);
	call eigen(val, vec, ycov);                         * Get eigen values;
   	val = choose(val < 1e-8, 1e-8, val);                * Clip negatives to just > 0;
	ycovpsd = vec * diag(val) * vec`;                   * Reconstruct covariance to be positive semi definite;
	* ycov2 = ycov + 1e-8 * I(nrow(ycov));              * Alternatively, could add a ridge to diagonal;
    x2 = abs(randnormal(m, zeros, ycovpsd) / yse`);
	maxx = x2[,<>]; * max col for each sim;
	* No built-in function for quantiles in IML!;
	call sort(maxx); * sort ascending;
	pos = 1 + (1 - alpha) * (m - 1);
	lo = floor(pos);
   	hi = ceil(pos);
   	frac = pos - lo;
	cs = (1 - frac) * maxx[lo] + frac * maxx[hi];       * linear interpolation;
	lcls = yhat - cs * yse;
	ucls = yhat + cs * yse;
	print cs, lcls ucls;
QUIT;
RUN;
