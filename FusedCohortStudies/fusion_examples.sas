/****************************************************************************************************************
Fusion 5 sim 27nov24.sas;

Stephen R Cole (2025/01/13)
****************************************************************************************************************/

options pagesize = 60 linesize = 100 nodate pageno = 1;
dm "log; clear; out; clear;";
%LET dir = C:\Users\colesr\Dropbox\Cole\;
ods listing gpath = "&dir.\" image_dpi = 600;

/*
data a;
	call streaminit(3);
	do i = 1 to 1000000;
		w = rand("bernoulli", .5);
		if 			w = 0 then a = rand("bernoulli", .3);
			else if w = 1 then a = rand("bernoulli", .6);
		if 			w = 0 and a = 0 then v = rand("bernoulli", .2);
			else if w = 0 and a = 1 then v = rand("bernoulli", .1);
			else if w = 1 and a = 0 then v = rand("bernoulli", .3);
			else if w = 1 and a = 1 then v = rand("bernoulli", .2);
		if 			v = 1 then y = rand("bernoulli", .9);
			else if v = 0 then y = rand("bernoulli", .25);
		output;
	end;
proc freq data = a;
	tables w a v y;
	tables w * a * y * v / list;
	title "Population structure, appoximated by 1M units";
run;
*/

data a;
	input w a v y n;
	cards;
0    0    0    0      209
0    0    0    1       70
0    0    1    0        7
0    0    1    1       63
0    1    0    0      102
0    1    0    1       34
0    1    1    0        2
0    1    1    1       14
1    0    0    0      104
1    0    0    1       35
1    0    1    0        6
1    0    1    1       55
1    1    0    0      179
1    1    0    1       60
1    1    1    0        6
1    1    1    1       54
;

proc sort data = a; 
	by w a y v n;
proc print data = a noobs; sum n;
	var w a y v n;
run;

data a;
	set a;
	retain id 0;
	do i = 1 to n; id = id + 1; output; end;
	drop n i;
run;

* Full data;
proc logistic data = a desc noprint;
	model a = w;
	output out = den (keep = id den) p = den;
data b;
	merge a den;
	by id;
	if a then weight = 1 / den;
		else weight = 1 / (1 - den);
	label den = ;
proc genmod data = b desc;
	class id;
	weight weight;
	model v = a / d = b link = id;
	repeated subject = id / type = ind;
	ods select modelinfo geeemppest;
	title2 "Full data";

* Make two equal-sized subcohorts;
data c;
	set b (in = inb) b;
	r = inb;
proc freq data = c;
	table r * w * a * v * y / list;
data c;
	set c;
	if r = 1 then v = .;
		else if r = 0 then w = .;
run;
proc freq data = c;
	table r * w * a * v * y / list missing;
	title2 "Observed data";run;

* Study 1;
proc genmod data = c desc;
	where r = 1;
	class id;
	weight weight;
	model y = a / d = b link = id;
	repeated subject = id / type = ind;
	ods select modelinfo geeemppest;
	ods output geeemppest = study1 (keep = parm estimate stderr);
	title2 "Study 1";

* Study 2;
proc genmod data = c desc;
	where r = 0;
	model v = a / d = b link = id;
	ods select modelinfo parameterestimates;
	ods output parameterestimates = study2 (keep = parameter estimate stderr);
	title2 "Study 2";

* Pooled-Crude;
proc genmod data = c desc;
	model y = a / d = b link = id;
	ods select modelinfo parameterestimates;
	title2 "Pooled Studies (crude)";
run;

* Meta-analysis;
data study1;
	set study1;
	where parm = "a";
	b1 = estimate;
	se1 = stderr;
	keep b1 se1;
data study2;
	set study2;
	where parameter = "a";
	b2 = estimate;
	se2 = stderr;
	keep b2 se2;
data meta;
	merge study1 study2;
	v1 = se1**2;
	v2 = se2**2;
	ivwv = 1 / (1/v1 + 1/v2);
	se = sqrt(ivwv);
	b = ivwv * ( 1/v1 * b1 + 1/v2 * b2);
	lo = b - 1.96 * se;
	hi = b + 1.96 * se;
proc print data = meta noobs;
	var b se lo hi b1 se1 b2 se2;
	title2 "Meta analysis";
run;

* Cohorts fused with estimating functions;
proc iml;
	use c;
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
	q = 25; * number of parameters;

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

		f = f1 || f2 || f3 || f4 || f5 || f6 || f7 || f8 || f9
			|| f10 || f11 || f12 || f13 || f14 || f15 || f16 || f17 || f18
			|| f19 || f20 || f21 || f22 || f23 || f24 || f25; * horizontal stack;
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
		a0_aipw, a1_aipw, rd_aipw, a0_aipwrg, a1_aipwrg, rd_aipwrg};
	title2 "Fused IPW Estimator";
	print var beta stderr lo hi;
quit;
run;
