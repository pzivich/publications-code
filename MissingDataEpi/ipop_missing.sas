/***************************************************************************
* Missing Outcome Data in Epidemiologic Studies
*		SAS code to recreate the results
*
* Steve Cole (2022/06/23)
***************************************************************************/

options center pagesize = 60 linesize = 100 nodate pageno = 1;
dm "log; clear; out; clear;";

%let b = 500; * # bootstrap samples;

*IPOP full data;
data full;
	input short p17 ptb n;
	no17p = 1 - p17;
	cards;
0 0 0 209
0 0 1  13
0 1 0 200
0 1 1  15
1 0 0 154
1 0 1  23
1 1 0 165
1 1 1  21
;
proc print data = full noobs;
	var short no17p ptb n;
	title1 "IPOP Missing Data";
	title2 "Full Data";
data full;
	set full;
	retain i 0;
	do j = 1 to n; 
		i = i + 1;
		output; 
	end;
	drop j n;
proc freq data = full;
	tables short * no17p no17p * ptb / nocol;
proc genmod data = full desc;
	model ptb = no17p / d = b link = log;
	estimate "lnRR" no17p 1 / exp;
	ods select modelinfo estimates;
	title2 "Full Data Risk Ratio";
run;

*About 25% outcomes MCAR;
data mcar;
	input short p17 ptb n;
	no17p = 1 - p17;
	cards;
0 0 0 150
0 0 1 11
0 0 . 54
0 1 0 157
0 1 1 10
0 1 . 55
1 0 0 124
1 0 1 16
1 0 . 46
1 1 0 116
1 1 1 17
1 1 . 44
;
proc print data = mcar noobs;
	var short no17p ptb n;
	title2 "MCAR";
data mcar;
	set mcar;
	retain i 0;
	do j = 1 to n; 
		if ptb = . then missing = 1;
			else missing = 0;
		i = i + 1;
		output; 
	end;
	drop j n;
proc freq data = mcar;
	tables no17p * ptb / nocol;
proc genmod data = mcar desc;
	model ptb = no17p / d = b link = log;
	estimate "lnRR" no17p 1 / exp;
	ods select modelinfo estimates;
	title2 "MCAR Observed Data Risk Ratio";
run;

*Gcomp;
data mcar2;
	set mcar;
	if no17p then ptb1 = ptb;
		else ptb1 = .;
	if ^no17p then ptb0 = ptb;
		else ptb0 = .;
proc surveyselect data = mcar2 out = mcar2b(drop = numberhits rename = (replicate = j) ) seed = 127 
	method = urs samprate = 1 outhits rep = &b. noprint; 
data mcar2b;
	set mcar2b mcar2 (in = in0);
	if in0 then j = 0;
proc sort data = mcar2b; 
	by j;
run; options nonotes; run;
proc logistic data = mcar2b desc noprint;
	by j;
	model ptb0 = short;
	output out = mcar3 p = m0; * probability of outcome;
proc logistic data = mcar3 desc noprint;
	by j;
	model ptb1 = short;
	output out = mcar4 p = m1; * probability of outcome;
run; options notes; run;
data mcar5;
	set mcar4;
	by j;
	retain y0 y1 0;
	if first.j then do; y0 = 0; y1 = 0; end;
	ptbm = max(ptb, 0); *set missing to 0 for below;
	y0 = y0 + m0;
	y1 = y1 + m1;
	if last.j then do;
		y0 = y0 / 800;
		y1 = y1 / 800;
		rd = y1 - y0;
		rr = y1 / y0;
		lnrr = log(rr);
		output;
	end;
	keep j y0 y1 rd rr lnrr;
proc print data = mcar5 noobs;
	where j = 0;
	title2 "MCAR Gcomp";
proc means data = mcar5 std noprint;
	where j > 0;
	var rd lnrr;
	output out = mcar6;
proc print data = mcar6 noobs;
	where _stat_ = "STD";
	var rd lnrr;
	title2 "MCAR Bootstrap SEs";
run;

*IPW;
data mcar;
	set mcar;
	if ptb = . then s = 0;
		else s = 1;
proc logistic data = mcar desc noprint;
	model s = no17p short no17p * short;
	output out = ipw p = pi;
data ipw;
	set ipw;
	if s then weight = 1 / pi;
		else weight = 0;
proc means data = ipw;
	var s weight;
proc means data = ipw;
	where s = 1;
	var s weight; run;
proc genmod data = ipw desc;
	where ptb > .;
	class i;
	model ptb = no17p / d = b link = log;
	weight weight;
	estimate "lnRR" no17p 1 / exp;
	repeated subject = i / type = ind;
	ods select modelinfo estimates geeemppest;
	title2 "MCAR IPW Risk Ratio";
run;

*MI;
proc mi data = mcar seed = 3 nimpute = 50 out = mi;
	class ptb;
	fcs nbiter = 10 logistic(ptb = short no17p short*no17p);
	var short no17p ptb;
run; ods select none; run;
proc genmod data = mi desc;
	by _imputation_;
	model ptb = no17p / d = b link = log covb;
	ods output parameterestimates = parms parminfo = info covb = covb;
run; ods select all; run;
proc mianalyze parms = parms covb = covb parminfo = info;
	modeleffects Intercept no17p;
	title2 "MCAR MI Risk Ratio";
run;


*MAR positive;
data marp;
	input short p17 ptb n;
	no17p = 1 - p17;
	cards;
0 0 0 100
0 0 1 8
0 0 . 107
0 1 0 209
0 1 1 13
1 0 0 165
1 0 1 21
1 1 0 77
1 1 1 12
1 1 . 88
;
proc print data = marp noobs;
	var short no17p ptb n;
	title2 "MAR positive";
data marp;
	set marp;
	retain i 0;
	do j = 1 to n; 
		if ptb = . then missing = 1;
			else missing = 0;
		i = i + 1;
		output; 
	end;
	drop j n;
proc freq data = marp;
	tables no17p * ptb / nocol;
proc genmod data = marp desc;
	model ptb = no17p / d = b link = log;
	estimate "lnRR" no17p 1 / exp;
	ods select modelinfo estimates;
	title2 "MAR positive, Observed Data Risk Ratio";
run;

*Gcomp;
data marp2;
	set marp;
	if no17p then ptb1 = ptb;
		else ptb1 = .;
	if ^no17p then ptb0 = ptb;
		else ptb0 = .;
proc surveyselect data = marp2 out = marp2b(drop = numberhits rename = (replicate = j) ) seed = 127 
	method = urs samprate = 1 outhits rep = &b. noprint; 
data marp2b;
	set marp2b marp2 (in = in0);
	if in0 then j = 0;
proc sort data = marp2b; 
	by j;
run; options nonotes; run;
proc logistic data = marp2b desc noprint;
	by j;
	model ptb0 = short;
	output out = marp3 p = m0; * probability of outcome;
proc logistic data = marp3 desc noprint;
	by j;
	model ptb1 = short;
	output out = marp4 p = m1; * probability of outcome;
run; options notes; run;
data marp5;
	set marp4;
	by j;
	retain y0 y1 0;
	if first.j then do; y0 = 0; y1 = 0; end;
	ptbm = max(ptb, 0); *set missing to 0 for below;
	y0 = y0 + m0;
	y1 = y1 + m1;
	if last.j then do;
		y0 = y0 / 800;
		y1 = y1 / 800;
		rd = y1 - y0;
		rr = y1 / y0;
		lnrr = log(rr);
		output;
	end;
	keep j y0 y1 rd rr lnrr;
proc print data = marp5 noobs;
	where j = 0;
	title2 "MAR positive, g comp";
proc means data = marp5 std noprint;
	where j > 0;
	var rd lnrr;
	output out = marp6;
proc print data = marp6 noobs;
	where _stat_ = "STD";
	var rd lnrr;
	title2 "MAR positive, bootstrap SEs";
run;

*IPW;
data marp;
	set marp;
	if ptb = . then s = 0;
		else s = 1;
proc logistic data = marp desc noprint;
	model s = no17p short no17p * short;
	output out = ipw p = pi;
data ipw;
	set ipw;
	if s then weight = 1 / pi;
		else weight = 0;
proc means data = ipw;
	var s weight;
proc means data = ipw;
	where s = 1;
	var s weight; run;
proc genmod data = ipw desc;
	where ptb > .;
	class i;
	model ptb = no17p / d = b link = log;
	weight weight;
	estimate "lnRR" no17p 1 / exp;
	repeated subject = i / type = ind;
	ods select modelinfo estimates geeemppest;
	title2 "MAR positive, IPW Risk Ratio";
run;

*MI;
proc mi data = marp seed = 3 nimpute = 50 out = mi;
	class ptb;
	fcs nbiter = 10 logistic(ptb = short no17p short*no17p);
	var short no17p ptb;
run; ods select none; run;
proc genmod data = mi desc;
	by _imputation_;
	model ptb = no17p / d = b link = log covb;
	ods output parameterestimates = parms parminfo = info covb = covb;
run; ods select all; run;
proc mianalyze parms = parms covb = covb parminfo = info;
	modeleffects Intercept no17p;
	title2 "MAR positive, MI Risk Ratio";
run;


*MAR negative;
data marn;
	input short p17 ptb n;
	no17p = 1 - p17;
	cards;
0 0 0 200
0 0 1  15
0 1 0 209
0 1 1  13
1 0 0 165
1 0 1  21
1 1 . 177
;
proc print data = marn noobs;
	var short no17p ptb n;
	title2 "MAR negative";
data marn;
	set marn;
	retain i 0;
	do j = 1 to n; 
		if ptb = . then missing = 1;
			else missing = 0;
		i = i + 1;
		output; 
	end;
	drop j n;
proc freq data = marn;
	tables no17p * ptb / nocol;
proc genmod data = marn desc;
	model ptb = no17p / d = b link = log;
	estimate "lnRR" no17p 1 / exp;
	ods select modelinfo estimates;
	title2 "MAR negative, Observed Data Risk Ratio";
run;

*Gcomp;
data marn2;
	set marn;
	if no17p then ptb1 = ptb;
		else ptb1 = .;
	if ^no17p then ptb0 = ptb;
		else ptb0 = .;
proc surveyselect data = marn2 out = marn2b(drop = numberhits rename = (replicate = j) ) seed = 127 
	method = urs samprate = 1 outhits rep = &b. noprint; 
data marn2b;
	set marn2b marn2 (in = in0);
	if in0 then j = 0;
proc sort data = marn2b; 
	by j;
run; options nonotes; run;
proc logistic data = marn2b desc noprint;
	by j;
	model ptb0 = short;
	output out = marn3 p = m0; * probability of outcome;
proc logistic data = marn3 desc noprint;
	by j;
	model ptb1 = short;
	output out = marn4 p = m1; * probability of outcome;
run; options notes; run;
data marn5;
	set marn4;
	by j;
	retain y0 y1 0;
	if first.j then do; y0 = 0; y1 = 0; end;
	ptbm = max(ptb, 0); *set missing to 0 for below;
	y0 = y0 + m0;
	y1 = y1 + m1;
	if last.j then do;
		y0 = y0 / 800;
		y1 = y1 / 800;
		rd = y1 - y0;
		rr = y1 / y0;
		lnrr = log(rr);
		output;
	end;
	keep j y0 y1 rd rr lnrr;
proc print data = marn5 noobs;
	where j = 0;
	title2 "MAR negative, g comp";
proc means data = marn5 std noprint;
	where j > 0;
	var rd lnrr;
	output out = marn6;
proc print data = marn6 noobs;
	where _stat_ = "STD";
	var rd lnrr;
	title2 "MAR negative, bootstrap SEs";
run;

*IPW;
data marn;
	set marn;
	if ptb = . then s = 0;
		else s = 1;
proc logistic data = marn desc noprint;
	model s = no17p short no17p * short;
	output out = ipw p = pi;
data ipw;
	set ipw;
	if s then weight = 1 / pi;
		else weight = 0;
proc means data = ipw;
	var s weight;
proc means data = ipw;
	where s = 1;
	var s weight; run;
proc genmod data = ipw desc;
	where ptb > .;
	class i;
	model ptb = no17p / d = b link = log;
	weight weight;
	estimate "lnRR" no17p 1 / exp;
	repeated subject = i / type = ind;
	ods select modelinfo estimates geeemppest;
	title2 "MAR negative, IPW Risk Ratio";
run;

*MI;
proc mi data = marn seed = 3 nimpute = 50 out = mi;
	class ptb;
	fcs nbiter = 10 logistic(ptb = short no17p short*no17p);
	var short no17p ptb;
run; ods select none; run;
proc genmod data = mi desc;
	by _imputation_;
	model ptb = no17p / d = b link = log covb;
	ods output parameterestimates = parms parminfo = info covb = covb;
run; ods select all; run;
proc mianalyze parms = parms covb = covb parminfo = info;
	modeleffects Intercept no17p;
	title2 "MAR negative, MI Risk Ratio";
run;


*MNAR;
data mnar;
	input short p17 ptb n;
	no17p = 1 - p17;
	cards;
0 0 0  85
0 0 1  15
0 0 . 115
0 1 0 209
0 1 1   7
0 1 .   6
1 0 0 165
1 0 1  21
1 1 0  69
1 1 1  12
1 1 .  96
;
proc print data = mnar noobs;
	var short no17p ptb n;
	title2 "MNAR";
data mnar;
	set mnar;
	retain i 0;
	do j = 1 to n; 
		if ptb = . then missing = 1;
			else missing = 0;
		i = i + 1;
		output; 
	end;
	drop j n;
proc freq data = mnar;
	tables no17p * ptb / nocol;
proc genmod data = mnar desc;
	model ptb = no17p / d = b link = log;
	estimate "lnRR" no17p 1 / exp;
	ods select modelinfo estimates;
	title2 "MNAR Observed Data Risk Ratio";
run;

*Gcomp;
data mnar2;
	set mnar;
	if no17p then ptb1 = ptb;
		else ptb1 = .;
	if ^no17p then ptb0 = ptb;
		else ptb0 = .;
proc surveyselect data = mnar2 out = mnar2b(drop = numberhits rename = (replicate = j) ) seed = 127 
	method = urs samprate = 1 outhits rep = &b. noprint; 
data mnar2b;
	set mnar2b mnar2 (in = in0);
	if in0 then j = 0;
proc sort data = mnar2b; 
	by j;
run; options nonotes; run;
proc logistic data = mnar2b desc noprint;
	by j;
	model ptb0 = short;
	output out = mnar3 p = m0; * probability of outcome;
proc logistic data = mnar3 desc noprint;
	by j;
	model ptb1 = short;
	output out = mnar4 p = m1; * probability of outcome;
run; options notes; run;
data mnar5;
	set mnar4;
	by j;
	retain y0 y1 0;
	if first.j then do; y0 = 0; y1 = 0; end;
	ptbm = max(ptb, 0); *set missing to 0 for below;
	y0 = y0 + m0;
	y1 = y1 + m1;
	if last.j then do;
		y0 = y0 / 800;
		y1 = y1 / 800;
		rd = y1 - y0;
		rr = y1 / y0;
		lnrr = log(rr);
		output;
	end;
	keep j y0 y1 rd rr lnrr;
proc print data = mnar5 noobs;
	where j = 0;
	title2 "MNAR g comp";
proc means data = mnar5 std noprint;
	where j > 0;
	var rd lnrr;
	output out = mnar6;
proc print data = mnar6 noobs;
	where _stat_ = "STD";
	var rd lnrr;
	title2 "MNAR bootstrap SEs";

*IPW;
data mnar;
	set mnar;
	if ptb = . then s = 0;
		else s = 1;
proc logistic data = mnar desc noprint;
	model s = no17p short no17p * short;
	output out = ipw p = pi;
data ipw;
	set ipw;
	if s then weight = 1 / pi;
		else weight = 0;
proc means data = ipw;
	var s weight;
proc means data = ipw;
	where s = 1;
	var s weight; run;
proc genmod data = ipw desc;
	where ptb > .;
	class i;
	model ptb = no17p / d = b link = log;
	weight weight;
	estimate "lnRR" no17p 1 / exp;
	repeated subject = i / type = ind;
	ods select modelinfo estimates geeemppest;
	title2 "MNAR, IPW Risk Ratio";
run;

*MI;
proc mi data = mnar seed = 3 nimpute = 50 out = mi;
	class ptb;
	fcs nbiter = 10 logistic(ptb = short no17p short*no17p);
	var short no17p ptb;
run; ods select none; run;
proc genmod data = mi desc;
	by _imputation_;
	model ptb = no17p / d = b link = log covb;
	ods output parameterestimates = parms parminfo = info covb = covb;
run; ods select all; run;
proc mianalyze parms = parms covb = covb parminfo = info;
	modeleffects Intercept no17p;
	title2 "MNAR, MI Risk Ratio";
run;

run;
quit;
run;
