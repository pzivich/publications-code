/********************************************************************************************
Sensitivity Analyses for Means or Proportions with Missing Data

Steve Cole (2022/11/17)
*********************************************************************************************/

/*** SETUP ***/
options nocenter pagesize = 60 linesize = 100 nodate pageno = 1;
dm "log; clear; out; clear;";
%LET dir = C:\file\path\;
ods listing gpath = "&dir.\" image_dpi = 600;


/*** Loading data ***/
*Read manipulated Lau data;
data a;
	infile "&dir.\lau_wihs.dat";
	input id 1-4 
          black 6 
          age 8-9 
          cd4 11-14    /*true CD4 */
          cd41 16-19   /*alpha=0*/
          cd42 21-24   /*alpha=0.1*/
          cd43 26-29   /*alpha=-0.1*/
          cd44 31-34;  /*conditional on W*/
run;

* Format read data;
data a;
	set a;
	* indicator for missing;
	if cd41 > . then s1 = 1; else s1 = 0;
	if cd42 > . then s2 = 1; else s2 = 0;
	if cd43 > . then s3 = 1; else s3 = 0;
	if cd44 > . then s4 = 1; else s4 = 0;

	* binary CD4 processing;
	if cd4 < 200 then lowcd4 = 1; else lowcd4 = 0;
	if cd41 = . then lowcd41 = .; 
		else if cd41 >= 200 then lowcd41 = 0; else lowcd41 = 1;
	if cd42 = . then lowcd42 = .; 
		else if cd42 >= 200 then lowcd42 = 0; else lowcd42 = 1;
	if cd43 = . then lowcd43 = .; 
		else if cd43 >= 200 then lowcd43 = 0; else lowcd43 = 1;
	if cd44 = . then lowcd44 = .; 
		else if cd44 >= 200 then lowcd44 = 0; else lowcd44 = 1;

* checking format;
proc print data = a (obs = 20) noobs;
	title1 "Sensitivity Analyses, Lau example";
proc means data = a n mean std stderr min max sum fw=6 maxdec=3;
run;

*Boxplots;
data box;
	set a;
	do x = 1 to 5;
		if x = 1 then y = cd4;           * Full data;
			else if x = 2 then y = cd42; * alpha = 0;
			else if x = 3 then y = cd41; * alpha = -.01; * swapped these to match other figures;
			else if x = 4 then y = cd43; * alpha = .01;
			else if x = 5 then y = cd44; * conditional on W;
		output;
	end;
ods graphics / reset imagename = "Boxplots 5dec22" border = off imagefmt = png height = 4in width = 4in;
proc sgplot data = box noautolegend noborder; 
	title;
	xaxis label = "Data Set" values = (1 to 5);
	yaxis label = "CD4 count, cells/ml" values=(0 to 2000 by 200) offsetmin=.01 offsetmax=0;
	scatter x = x y = y / jitter transparency = .9
		markerattrs = (color = black symbol = circlefilled size = 3);
	vbox y / category = x transparency = 0 boxwidth = .2 nooutliers fillattrs = (color = gray) 
		lineattrs = (pattern = solid) whiskerattrs = (pattern = solid);
run;

/*** Computing nonparametric bounds ***/
data lower;
	set a;
	if s1 = 0 then cd41 = 0;
	if s2 = 0 then cd42 = 0;
	if s3 = 0 then cd43 = 0;
	if s4 = 0 then cd44 = 0;
	if cd41 < 200 then lowcd41 = 1; else lowcd41 = 0;
	if cd42 < 200 then lowcd42 = 1; else lowcd42 = 0;
	if cd43 < 200 then lowcd43 = 1; else lowcd43 = 0;
	if cd44 < 200 then lowcd44 = 1; else lowcd44 = 0;
proc means data = lower;
	var cd4 cd41-cd44 lowcd4 lowcd41-lowcd44;
	title1 "Sensitivity Analyses, Lau example";
	title2 "Lower bound, missing = 0";

data upper;
	set a;
	if s1 = 0 then cd41 = 1933;
	if s2 = 0 then cd42 = 1933;
	if s3 = 0 then cd43 = 893; * Observed max here is 893;
	if s4 = 0 then cd44 = 1933;
	if cd41 < 200 then lowcd41 = 1; else lowcd41 = 0;
	if cd42 < 200 then lowcd42 = 1; else lowcd42 = 0;
	if cd43 < 200 then lowcd43 = 1; else lowcd43 = 0;
	if cd44 < 200 then lowcd44 = 1; else lowcd44 = 0;
proc means data = upper;
	var cd4 cd41-cd44 lowcd4 lowcd41-lowcd44;
	title1 "Sensitivity Analyses, Lau example";
	title2 "Upper bound, missing = 1933 (893 for cd43)";
run;

/****************************************
Example 1: mean CD4
****************************************/
%let start1 = 0; %let start2 = 0; 

/*M-estimator for given alpha*/
%macro sa(data, y, s, alpha);
proc iml;
	*read in data;
	use &data.; 
		read all var {&y.} into y;
		read all var {&s.} into s;
	close &data.;
	n = nrow(y);

	* set missing to zero for estimation;
	do i = 1 to n; if s[i] = 0 then y[i] = 0; 
    end;

	* estimating functions;
	start f(beta) global(y, s, n, q);
		f1 = j(1, 1, 0);
		f2 = j(1, 1, 0);
		nu = (1 / (1 + exp( -(beta[1] + q))));
		do i = 1 to n;
			f1 = f1 + s[i] / nu[i] - 1;
			f2 = f2 +
				s[i] # y[i] / ( 1 / (1 + exp( -(beta[1] + q[i])))) - beta[2];
		end;
		f = f1 // f2;
		return (f);
  	finish f;

	* root-finding (estimate parameters);
	r = 2;                      *# parms;
	alpha = &alpha. / 1000;     * scaling alpha (since SAS goes by 1 unit);
	q = alpha # y;              * prod;
	beta = j(1, 2, 0);          * Initial values;
	beta = &start1. || &start2.;
	optn = r || 0;              * 2 roots/parameters, 1 print some;
	tc = j(1, 12, .);           * Missing values are set to defaults;
	tc[6] = 1e-8;               * error tolerance for completion;
	call nlplm(rc, bhat, "f", beta, optn,, tc);

	*Compute bread matrix;
	par = j(1, 3, .);                               * par is a length 3 vector of details, missing sets to defaults;
	par[1] = r;                                     * tell FD we have 2 parameters;
	call nlpfdd(func, bread, hess, "f", bhat, par); * eval derivative at bhat;
	bread = - bread / n;                            * negative derivative, averaged;

	*Compute meat matrix;
	meat = j(r, r, 0);
	ef1 = j(1, 1, 0);
	ef2 = j(1, 1, 0);
	ef = j(r, 1, 0);
	do i = 1 to n;
		ef1 = s[i] / (1 / (1 + exp( -(bhat[1] + q[i])))) - 1;
		ef2 = s[i] # y[i] / ( 1 / (1 + exp( -(bhat[1] + q[i])))) - bhat[2];
		ef = ef1 // ef2;
		meat = meat + ef * ef`;
	end;
	meat = meat / n;

	* Compute sandwich matrix;
	sandwich = ( inv(bread) * meat * inv(bread)` ) / n;
	b = bhat`;
	se = sqrt(vecdiag(sandwich));

    title1 "Lau 2009 WIHS data";
	title2 "Sensitivity analysis, example 1";
	print alpha,, bread meat sandwich,, b se;
	gamma = b[2];
	ase = se[2];
	start1 = b[1]; start2 = b[2];
	create out var {alpha gamma ase start1 start2};
		append;
	close out; 
quit;
run;

data _null_;
	set out;
	call symput("start1",start1);
	call symput("start2",start2);
run;

proc append base = plot data = out force;
run;
%mend sa;

* wrapper loop for M-estimator over range of alpha_hat range;
%macro loop(data, y, s, lo, hi);
	data plot;
		alpha = .;
		gamma = .;
		ase = .;
		start1 = .; start2 = .;
	%do alpha = &lo. %to &hi.;
		%sa(&data., &y., &s., &alpha.);
	%end;
%mend loop;

*Call wrapper loop of M-estimator for example 1 for alpha=0;
%loop(a, cd41, s1, -20, 50);

*Plot for example 1 for alpha=0;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 1 alpha_0" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.05 to .05 by .01) offsetmin = 0 offsetmax = 0;
	yaxis label = "E(Y)" values=(0 to 1000 by 200) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline 305.4 773.7 / axis = y lineattrs = (pattern = 3 color = black);
	refline 0 / axis = x lineattrs = (pattern = 1 color = black);
	inset ("Panel" = "B") / position = topright noborder textattrs = (size = 10 color = black);
run;

*Call wrapper loop of M-estimator for example 1 for alpha=0.01;
%loop(a, cd42, s2, -20, 50);

*Plot for example 1 for alpha = .01;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 1 alpha_01" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.05 to .05 by .01) offsetmin = 0 offsetmax = 0;
	yaxis label = "E(Y)" values=(0 to 1000 by 200) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline 351.7 821.6 / axis = y lineattrs = (pattern = 3 color = black);
	refline .01 / axis = x lineattrs = (pattern = 1 color = black);
	inset ("Panel" = "C") / position = topright noborder textattrs = (size = 10 color = black);
run;

*Call wrapper loop of M-estimator for example 1 for alpha=-0.01;
%loop(a, cd43, s3, -50, 50);

*Plot example 1 for alpha = -.01;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 1 alpha_neg01" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.05 to .05 by .01) offsetmin = 0 offsetmax = 0;
	yaxis label = "E(Y)" values=(0 to 1000 by 200) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline 224.1 441.2 / axis = y lineattrs = (pattern = 3 color = black);
	refline -.01 / axis = x lineattrs = (pattern = 1 color = black);
	inset ("Panel" = "A") / position = topright noborder textattrs = (size = 10 color = black);
run;


/****************************************
Example 2: proportion of low CD4
****************************************/

/*M-estimator for given alpha*/
%macro sa(data, y, s, alpha);
proc iml;
	* Loading data;
	use &data.; 
		read all var {&y.} into y;
		read all var {&s.} into s;
	close &data.;
	n = nrow(y);
	
	* Estimating functions;
    do i = 1 to n; if s[i] = 0 then y[i] = 0; end; *set missing to 0 for estimation;
	start f(beta) global(y, s, n, q);
		f1 = j(1, 1, 0);
		f2 = j(1, 1, 0);
		nu = (1 / (1 + exp( -(beta[1] + q))));
		do i = 1 to n;
			f1 = f1 + s[i] / nu[i] - 1;
			f2 = f2 +  /*note difference with Y here*/
				s[i] # (y[i] < 200) / ( 1 / (1 + exp( -(beta[1] + q[i])))) - beta[2];
		end;
		f = f1 // f2;
		return (f);
  	finish f;

	* Root-finding (estiamte parameters);
	r = 2;                  * # parms;
	alpha = &alpha. / 1000; * scaling alpha(since SAS goes by 1 unit);
	q = alpha # y;          * prod;
	beta = j(1, 2, 0);      * Initial values;
	optn = r || 0;          * 2 roots/parameters, 1 print some;
	tc = j(1, 12, .);       * Missing values are set to defaults;
	tc[6] = 1e-8;           * error tolerance for completion;
	call nlplm(rc, bhat, "f", beta, optn,, tc);

	* Compute bread matrix;
	par = j(1, 3, .);                               * par is a length 3 vector of details, missing sets to defaults;
	par[1] = r;                                     * tell FD we have 2 parameters;
	call nlpfdd(func, bread, hess, "f", bhat, par); * eval derivative at bhat;
	bread = - bread / n;                            * negative derivative, averaged;

	* Compute meat matrix;
	meat = j(r, r, 0);
	ef1 = j(1, 1, 0);
	ef2 = j(1, 1, 0);
	ef = j(r, 1, 0);
	do i = 1 to n;
		ef1 = s[i] / (1 / (1 + exp( -(bhat[1] + q[i])))) - 1;
		ef2 = s[i] # (y[i] < 200) / ( 1 / (1 + exp( -(bhat[1] + q[i])))) - bhat[2];
		ef = ef1 // ef2;
		meat = meat + ef * ef`;
	end;
	meat = meat / n;

	* Compute sandwich matrix;
	sandwich = ( inv(bread) * meat * inv(bread)` ) / n;
	b = bhat`;
	se = sqrt(vecdiag(sandwich));

	title1 "Lau 2009 WIHS data";
	title2 "Sensitivity analysis, example 2";
	print alpha,, bread meat sandwich,, b se;
	gamma = b[2];
	ase = se[2];
	create out var {alpha gamma ase};
		append;
	close out; 
quit;
run;

proc append base = plot data = out force;
run;
%mend sa;

* wrapper loop for M-estimator over range of alpha_hat range;
%macro loop(data, y, s, lo, hi);
	data plot;
		alpha = .;
		gamma = .;
		ase = .;
	%do alpha = &lo. %to &hi.;
		%sa(&data., &y., &s., &alpha.);
	%end;
%mend loop;

*Call wrapper loop of M-estimator for example 2 for alpha=0;
%loop(a, cd41, s1, -20, 50);

*Plot example 2 for alpha = 0;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 2 alpha_0" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.02 to .05 by .01) offsetmin = 0 offsetmax = 0;
	yaxis label = "F(200)" values=(0 to .6 by .1) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline .173 .415 / axis = y lineattrs = (pattern = 3 color = black);
	refline 0 / axis = x lineattrs = (pattern = 1 color = black);
	inset ("Panel" = "B") / position = topright noborder textattrs = (size = 10 color = black);
run;

*Call wrapper loop of M-estimator for example 2 for alpha=0.01;
%loop(a, cd42, s2, -20, 50);

*Plot example 2 for alpha = .01;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 2 alpha_01" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.02 to .05 by .01) offsetmin = 0 offsetmax = 0;
	yaxis label = "F(200)" values=(0 to .6 by .1) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline .076 .319 / axis = y lineattrs = (pattern = 3 color = black);
	refline .01 / axis = x lineattrs = (pattern = 1 color = black);
	inset ("Panel" = "C") / position = topright noborder textattrs = (size = 10 color = black);
run;

*Call wrapper loop of M-estimator for example 2 for alpha=-0.01;
%loop(a, cd43, s3, -20, 50);

*Plot example 2 for alpha=-.01;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 2 alpha_neg01" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.02 to .05 by .01) offsetmin = 0 offsetmax = 0;
	yaxis label = "F(200)" values=(0 to .6 by .1) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline .225 .468 / axis = y lineattrs = (pattern = 3 color = black);
	refline -.01 / axis = x lineattrs = (pattern = 1 color = black);
	inset ("Panel" = "A") / position = topright noborder textattrs = (size = 10 color = black);
run;


/****************************************
Example 3: covariates in nuisance model
****************************************/
%let start1 = 0; %let start2 = 0; %let start3 = 0; %let start4 = 0;

/*** Inverse Probability Weighting ***/
data a2;
	set a;
	* Processing age into categories;
	age30 = 1;
	if . < age <= 30 then do; agegp = 1; age30 = 0; end;
		else if 30 < age <= 35 then agegp = 2;
		else if 35 < age <= 40 then agegp = 3;
		else if 40 < age <= 45 then agegp = 4;
		else if 45 < age then agegp = 5;
proc freq data = a2;
	tables agegp * age30; 

* Fitting the IPW nuisance model with observables;
proc logistic data = a2 desc;
	*class agegp / param = ref;
	model s4 = age30 black;
	output out = ipw p = pi;
	title2 "IPW weight model";
data ipw;
	set ipw;
	if s4 = 1 then ipw = 1 / pi;
		else ipw = 0;
	label pi =;
proc means data = ipw;
	var s4 pi ipw;
	title2 "IPW weights";
proc means data = ipw n mean std stderr min max;
	var cd44 lowcd44;
	weight ipw;
	title2 "IPW complete case estimator";

* Unweighted analysis;
proc genmod data = ipw;
	model lowcd44 = ;
	weight ipw;
	ods select modelinfo parameterestimates;
run;

* IPW analysis;
proc genmod data = ipw;
	class id;
	model lowcd44 = ;
	weight ipw;
	repeated subject = id / type = ind;
	ods select modelinfo geeemppest;
run;

/*M-estimator for given alpha*/
%macro sa(alpha);
proc iml;
	* Loading data;
	use a2;
		read all var {cd44} into y;
		read all var {s4} into s;
		read all var {age30 black} into w;
	close a2;
	n = nrow(y);
	do i = 1 to n; if s[i] = 0 then y[i] = 0; 
	end; *set missing to 0 for estimation;
	x = j(n, 1, 1) || w; * design matrix;
	
	* estimating functions;
	start f(beta) global(y, s, w, x, n, q, nu);
		f1 = j(3, 1, 0);
		f2 = j(1, 1, 0);
		nu = 1 / (1 + exp(-(x * beta[1:3] + q)));
		do i = 1 to n;	
			/*note use of the design matrix here*/
			f1 = f1 + x[i,]` * (s[i] / nu[i] - 1);
			f2 = f2 +
				s[i] # (y[i] < 200) / nu[i] - beta[4];
		end;
		f = f1 // f2;
		return (f);
  	finish f;

	* root-finding (parameter estimation);
	r = 4;                                                * # parms;
	alpha = &alpha. / 1000;                               * rescaling alpha (since SAS only allows ints);
	q = alpha # y;                                        * prod;
	beta = &start1. || &start2. || &start3. || &start4.;  * starting values;
	optn = r || 0;                                        * r roots/parameters, 1 print some;
	tc = j(1, 12, .);                                     * Missing values are set to defaults;
	tc[6] = 1e-8;                                         * error tolerance for completion;
	call nlplm(rc, bhat, "f", beta, optn,, tc);

	* Compute bread matrix;
	par = j(1, 3, .);                               * par is a length 3 vector of details, missing sets to defaults;
	par[1] = r;                                     * tell FD we have r parameters;
	call nlpfdd(func, bread, hess, "f", bhat, par); * eval derivative at bhat;
	bread = - bread / n;                            * negative derivative, averaged;

	* Compute meat matrix;
	meat = j(r, r, 0);
	ef1 = j(1, 1, 0);
	ef2 = j(1, 1, 0);
	ef = j(r, 1, 0);
	do i = 1 to n;
		ef1 = x[i,]` * (s[i] / nu[i] - 1);
		ef2 = s[i] # (y[i] < 200) / nu[i] - bhat[4];
		ef = ef1 // ef2;
		meat = meat + ef * ef`;
	end;
	meat = meat / n;

	* Compute sandwich matrix;
	sandwich = ( inv(bread) * meat * inv(bread)` ) / n;
	b = bhat`;
	se = sqrt(vecdiag(sandwich));

	title1 "Lau 2009 WIHS data";
	title2 "Sensitivity analysis, example 3";
	print alpha,, bread meat sandwich,, b se;
	* dynamically update starting values (faster computation);
	if abs(b[1]) < 30 then start1 = b[1]; else start1 = 0;
	if abs(b[2]) < 30 then start2 = b[2]; else start2 = 0;
	if abs(b[3]) < 30 then start3 = b[3]; else start3 = 0;
	if abs(b[4]) < 30 then start4 = b[4]; else start4 = 0;
	gamma = b[4];
	ase = se[4];
	create out var {alpha gamma ase start1 start2 start3 start4};
		append;
	close out; 
quit;
run;

data _null_;
	set out;
	call symput("start1", start1);
	call symput("start2", start2);
	call symput("start3", start3);
	call symput("start4", start4);
run;

proc append base = plot data = out force;
run;
%mend sa;

* wrapper loop for M-estimator over range of alpha_hat range;
%macro loop(lo, hi);
	data plot;
		alpha = .;
		gamma = .;
		ase = .;
		start1 = .; start2 = .; start3 = .; start4 = .;
	%do alpha = &lo. %to &hi.;
		%sa(&alpha.);
	%end;
%mend loop;

*Call wrapper loop of M-estimator for example 3;
%loop(-20, 120);

*Plot example 3;
data plot;
	set plot;
	lo = gamma - 1.96 * ase;
	hi = gamma + 1.96 * ase;
ods graphics / reset imagename = "SA ex 3" border = off imagefmt = png height = 3in width = 5in;
proc sgplot data = plot noautolegend noborder; 
	title;
	xaxis label = "(*ESC*){unicode alpha}" labelattrs = (style = italic) values=(-.02 to .12 by .02) offsetmin = 0 offsetmax = 0;
	yaxis label = "F(200)" values=(0 to .6 by .1) offsetmin = 0 offsetmax = 0;
	series x = alpha y = gamma / lineattrs = (color = red pattern = 1 thickness = 1);
	band x = alpha lower = lo upper = hi / transparency = .5 fillattrs = (color = pink);
	refline .047 .410 / axis = y lineattrs = (pattern = 3 color = black);
	refline .01 / axis = x lineattrs = (pattern = 1 color = black);
run;

/*END*/
