/****************************************************************************************************************
Fusion: Sato et al example

Stephen R Cole (2026/02/09)
****************************************************************************************************************/

options pagesize = 60 linesize = 80 nodate pageno = 1;
dm "log; clear; out; clear;";
%LET dir = C:\Users\colesr\Dropbox\Cole\Research\Semiparametrics\Fusion 5\;
ods listing gpath = "&dir.\" image_dpi = 600;

* 4901 women with breast cancer contributed data on the possible effect of tamoxifen use (A=1) 
on breast cancer recurrence (V=1), but were also classified as having positive lymph node metastasis (W=1) 
at surgery before tamoxifen use (3). Crude 1.00 (95% CI: 0.90, 1.14) and the W-adjusted RR was 0.85 (0.76, 0.95);

data a;
	input w a v n;
	cards;
0	0	0	1421
0	0	1	171
0	1	0	1238
0	1	1	96
1	0	0	507
1	0	1	253
1	1	0	847
1	1	1	368
;
data b;
	set a;
	do i = 1 to n; output; end;
	drop i n;
proc means data = b;
	var w a v;
	title "Sato 2003 BC data";
proc genmod data = b desc;
	model v = a / d = b link = log;
	estimate "RR" a 1 / exp;
	ods select modelinfo estimates;
	title2 "Crude RR";
proc genmod data = b desc;
	model v = a w / d = b link = log;	
	estimate "RR" a 1 / exp;
	ods select modelinfo estimates;
	title2 "Adjusted RR";
run;

* Step 1, misclassify V as Y;
data c;
	set b;
	call streaminit(31);
	if v = 1 then y = rand("bernoulli", .9); * sensitivity;
		else if v = 0 then y = rand("bernoulli", .3); * 1 - specificity;
	u = rand("uniform"); * used below to split data;
proc genmod data = c desc;
	model y = a / d = b link = log;
	estimate "RR" a 1 / exp;
	ods select modelinfo estimates;
	title2 "Crude RR, V misclassified as Y";
proc genmod data = c desc;
	model y = a w / d = b link = log;
	estimate "RR" a 1 / exp;
	ods select modelinfo estimates;
	title2 "Adjusted RR, V misclassified as Y";
run;

* Step 2, hide W for half and V for half;
data d;
	set c;
	if u > .5 then r = 1; else r = 0;
	if r = 0 then w = .;
		else if r = 1 then v = .;

* Check it works, yep;
proc genmod data = d desc;
	where r = 0;
	model v = a / d = b link = log;
	estimate "RR" a 1 / exp;
	ods select modelinfo estimates;
	title2 "R = 1, V but no W";
proc genmod data = d desc;
	where r = 1;
	model y = a w / d = b link = log;
	estimate "RR" a 1 / exp;
	ods select modelinfo estimates;
	title2 "R = 0, W but no V";
run;

* Save data;
data _null_;
	set d;
	file "&dir.Satoex20jan26.dat";
	put a 1 w 3 r 5 y 7 v 9;

* Step 3, run each analyses from paper...;

* Cohorts fused with estimating functions;
proc iml;
	use d;
		read all var {a} into a;
		read all var {y} into y;
		read all var {r} into r;
		read all var {w} into w;
		read all var {v} into v;
	close c;
	n = nrow(y);
	idw = loc(w = .); * find missing W and replace with 0s;
	w[idw] = 0;
	idv = loc(v = .); * find missing V and replace with 0s;
	v[idv] = 0;
	q = 26; * number of parameters;

	* Estimating Functions, derivatives of objective functions;
	start ef(beta) global(a, y, w, v, r, n);
		f1 = (1 - r) # v # (y - beta[1]); * se;
		f2 = (1 - r) # (1 - v) # ((1 - y) - beta[2]); * sp;

		* IPW w/RG;
		pi = 1 / (1 + exp(-(beta[3] + w # beta[4]))); * nuisance model;
		f3 = r # (a - pi); * intercept;
		f4 = r # (a - pi) # w; * effect of w;
		f5 = r # (1 - a) # (y - beta[5]) / (1 - pi); * untreated IPW risk;
		f6 = r # a # (y - beta[6]) / pi; * treated IPW risk;
		f7 = j(n, 1, beta[7] # (beta[1] + beta[2] - 1) - (beta[5] + beta[2] - 1) ); * untreated RG;
		f8 = j(n, 1, beta[8] # (beta[1] + beta[2] - 1) - (beta[6] + beta[2] - 1) ); * treated RG;
		f9 = j(n, 1, beta[8] - beta[7] - beta[9] ); * ipw rd;

		* gcomp;
		p = 1 / (1 + exp(-(beta[10] + beta[11] * a + beta[12] * w + beta[13] * (a#w) )));	
		f10 = r # (y - p);
		f11 = r # (y - p) # a;
		f12 = r # (y - p) # w;
		f13 = r # (y - p) # (a#w);
		p0 = 1 / (1 + exp(-(beta[10] + beta[11] * 0 + beta[12] * w + beta[13] * (0#w) )));
		p1 = 1 / (1 + exp(-(beta[10] + beta[11] * 1 + beta[12] * w + beta[13] * (1#w) )));
		f14 = r # (p0 - beta[14]); * untreated gcomp risk;
		f15 = r # (p1 - beta[15]); * treated gcomp risk;
		f16 = j(n, 1, (beta[15] - beta[14]) - beta[16] ); * rd by gcomp;

		* RG on gcomp;
		f17 = j(n, 1, beta[17] # (beta[1] + beta[2] - 1) - (beta[14] + beta[2] - 1) ); * untreated gcomp RG;
		f18 = j(n, 1, beta[18] # (beta[1] + beta[2] - 1) - (beta[15] + beta[2] - 1) ); * treated gcomp RG;
		f19 = j(n, 1, (beta[18] - beta[17]) - beta[19] ); * rd by gcomp w/RG;

		* AIPW;
		f20 = r # (((1 - a) # y / (1 - pi) + (a - pi) / (1 - pi) # p0 ) - beta[20]); * untreated aipw risk;
		f21 = r # ((a # y / pi - (a - pi) / pi # p1 ) - beta[21]); * treated aipw risk;
		f22 = j(n, 1, (beta[21] - beta[20]) - beta[22] ); * rd by gcomp;

		* RG on AIPW;
		f23 = j(n, 1, beta[23] # (beta[1] + beta[2] - 1) - (beta[20] + beta[2] - 1) ); * untreated AIPW RG;
		f24 = j(n, 1, beta[24] # (beta[1] + beta[2] - 1) - (beta[21] + beta[2] - 1) ); * treated AIPW RG;
		f25 = j(n, 1, (beta[24] - beta[23]) - beta[25] ); * rd by AIPW w/RG;
		*f26 = j(n, 1, beta[24] / beta[23] - beta[26] ); * rr by AIPW w/RG;
		f26 = j(n, 1, log(beta[24]) - log(beta[23]) - log(beta[26]));

		f = f1 || f2 || f3 || f4 || f5 || f6 || f7 || f8 || f9
			|| f10 || f11 || f12 || f13 || f14 || f15 || f16 || f17 || f18
			|| f19 || f20 || f21 || f22 || f23 || f24 || f25 || f26; * horizontal stack;
		return(f);
	finish ef;
	start eequat(beta);	* Concatinant estimating function into single estimating equation;
		ef = ef(beta);
		return(ef[+,]); * Return row sums, 1 by q vector;
	finish eequat;

	* Newton, find roots as bhat;
	beta = j(1, q, .1); * Initial parameter values;	
	optn = q || 0; * q roots/parameters, 1 print some;
	tc = j(1, 12, .); * Missing values are set to defaults;
	tc[1] = 500; * max iterations, default 200;
	tc[2] = 1000; * max function calls, defaul 500;
	tc[6] = 1e-12; * convergence, default 1e-5;
	call nlplm(rc, bhat, "eequat", beta, optn,, tc); *nlplm is Levenberg-Marquardt method, allows multiple roots; 

	*Bread;
	par = j(1, 3, .); *par is a length 3 vector of details, missing sets to defaults;
	par[1] = q; *tell FD we have q parameters;
	call nlpfdd(func, bread, hess, "eequat", bhat, par); *eval derivative at bhat;
	bread = - bread / n; *negative derivative, averaged;

	* Meat;
	errors = ef(bhat); * Value of estimating functions at beta hat, n by q matrix;
	outerprod = errors` * errors; * Outerproduct of residuals, note transpose is flipped, to give q by q matrix;
	meat = outerprod / n; * Average over n;

	*Sandwich;
	sandwich = ( inv(bread) * meat * inv(bread)` ) / n;
	beta = bhat`;
	stderr = sqrt(vecdiag(sandwich));
	lo = beta - 1.96 * stderr;
	hi = beta + 1.96 * stderr;

	var = {sens, spec, ps_int, ps_w, a0_ipw, a1_ipw, a0_ipwrg, a1_ipwrg,
		rd_ipwrg, om_int, om_a, om_w, om_aw, a0_g, a1_g, rd_g, a0_grg, a1_grg, rd_grg,
		a0_aipw, a1_aipw, rd_aipw, a0_aipwrg, a1_aipwrg, rd_aipwrg, rr_aipwrg};
	title2 "Fused IPW Estimator";
	print var beta stderr lo hi;
quit;
run;
