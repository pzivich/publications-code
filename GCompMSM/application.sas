/**********************************************************************************************************
Practical Implementation of g-computation for marginal structural models

Paul Zivich (2023/11/29)
**********************************************************************************************************/

/*** Reading in formatted data ***/
PROC IMPORT OUT=dat 
            DATAFILE="C:\Users\zivic\Documents\Research\#PZivich\GCompMSM\actg.csv"
            DBMS=CSV REPLACE;
	GETNAMES=YES;
    DATAROW=2; 
RUN;
DATA dat;
	SET dat;
	IF karnof = 90 THEN karnof_90 = 1;
		ELSE karnof_90 = 0;
	IF karnof = 100 THEN karnof_100 = 1;
		ELSE karnof_100 = 0;
RUN;


/*** G-computation estimator of MSM as an M-estimator ***/

PROC IML;                            /*All steps are completed in PROC IML*/
	/***********************************************
	Read data into IML */
	use dat;								/*Open input data from above*/
		read all var {cd4_20wk} into y;		/*Read in each column needed as its own vector*/
		read all var {treat} into a;
		read all var {male} into v;
		read all var {idu} into w1;
		read all var {white} into w2;
		read all var {karnof_90} into w3;
		read all var {karnof_100} into w4;
		read all var {agec} into w5;
		read all var {age_rs1} into w6;
		read all var {age_rs2} into w7;
		read all var {age_rs3} into w8;
		read all var {cd4c_0wk} into w9;
		read all var {cd4_rs1} into w10;
		read all var {cd4_rs2} into w11;
		read all var {cd4_rs3} into w12;

	close dat;
	n = nrow(y);                        	/*Save number of observations in data */

	/***********************************************
	Defining estimating equation */
	q = 24;									/*Save number parameters to be estimated*/

	/*Start to define estimating function */
	START efunc(beta) global(y, a, v, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12);	   	 
		/*Processing predicted values from models*/
		m1 = beta[1] + beta[2]*v + beta[3]*1 + beta[4]*1*v;  *Design matrix M(1);
		m0 = beta[1] + beta[2]*v + beta[3]*0 + beta[4]*0*v;  *Design matrix M(0);
		*Predictions under natural course;
		yhat = beta[5] + beta[6]*a + beta[7]*v + beta[8]*a#v 
                       + beta[9]*w1 + beta[10]*w2 
					   + beta[11]*w3 + beta[12]*w4 
					   + beta[13]*w5 + beta[14]*w6 + beta[15]*w7 + beta[16]*w8
					   + beta[17]*w9 + beta[18]*w10 + beta[19]*w11 + beta[20]*w12
					   + beta[21]*w9#v + beta[22]*w10#v + beta[23]*w11#v + beta[24]*w12#v;
		*Predictions under setting A=1;
		y1hat = beta[5] + beta[6]*1 + beta[7]*v + beta[8]*1*v 
                       + beta[9]*w1 + beta[10]*w2 
					   + beta[11]*w3 + beta[12]*w4 
					   + beta[13]*w5 + beta[14]*w6 + beta[15]*w7 + beta[16]*w8
					   + beta[17]*w9 + beta[18]*w10 + beta[19]*w11 + beta[20]*w12
					   + beta[21]*w9#v + beta[22]*w10#v + beta[23]*w11#v + beta[24]*w12#v;
		*Predictions under setting A=0;
		y0hat = beta[5] + beta[6]*0 + beta[7]*v + beta[8]*0*v 
                       + beta[9]*w1 + beta[10]*w2 
					   + beta[11]*w3 + beta[12]*w4 
					   + beta[13]*w5 + beta[14]*w6 + beta[15]*w7 + beta[16]*w8
					   + beta[17]*w9 + beta[18]*w10 + beta[19]*w11 + beta[20]*w12
					   + beta[21]*w9#v + beta[22]*w10#v + beta[23]*w11#v + beta[24]*w12#v;
		
		/*Estimating functions of marginal structural model*/
		ef_1 = (y1hat - m1) + (y0hat - m0); 		/*EF for \beta_0*/
		ef_2 = (y1hat - m1)#v + (y0hat - m0)#v; 	/*EF for \beta_1*/
		ef_3 = (y1hat - m1) + (y0hat - m0)*0; 	 	/*EF for \beta_2*/
		ef_4 = (y1hat - m1)#v + (y0hat - m0)#v*0;	/*EF for \beta_3*/
		/*Estimating functions of outcome model*/
		ef_5 = (y - yhat);							/*EF for intercept*/
		ef_6 = (y - yhat)#a;						/*EF for ART*/
		ef_7 = (y - yhat)#v;						/*EF for male*/
		ef_8 = (y - yhat)#a#v;						/*EF for ART*male*/
		ef_9 = (y - yhat)#w1;						/*EF for idu*/
		ef_10 = (y - yhat)#w2;						/*EF for white*/
		ef_11 = (y - yhat)#w3;						/*EF for karnof_90*/
		ef_12 = (y - yhat)#w4;						/*EF for karnof_100*/
		ef_13 = (y - yhat)#w5;						/*EF for age*/
		ef_14 = (y - yhat)#w6;						/*EF for age_rs1*/
		ef_15 = (y - yhat)#w7;						/*EF for age_rs2*/
		ef_16 = (y - yhat)#w8;						/*EF for age_rs3*/
		ef_17 = (y - yhat)#w9;						/*EF for cd4*/
		ef_18 = (y - yhat)#w10;						/*EF for cd4_rs1*/
		ef_19 = (y - yhat)#w11;						/*EF for cd4_rs2*/
		ef_20 = (y - yhat)#w12;						/*EF for cd4_rs3*/
		ef_21 = (y - yhat)#w9#v;					/*EF for male*cd4*/
		ef_22 = (y - yhat)#w10#v;					/*EF for male*cd4_rs1*/
		ef_23 = (y - yhat)#w11#v;					/*EF for male*cd4_rs2*/
		ef_24 = (y - yhat)#w12#v;					/*EF for male*cd4_rs3*/

		/*Stacking estimating functions together*/
		ef_mat = ef_1||ef_2||ef_3||ef_4
					 ||ef_5||ef_6||ef_7||ef_8||ef_9||ef_10
					 ||ef_11||ef_12||ef_13||ef_14||ef_15
					 ||ef_16||ef_17||ef_18||ef_19||ef_20
                     ||ef_21||ef_22||ef_23||ef_24;
		RETURN(ef_mat);                         	/*Return n by q matrix for estimating functions*/
	FINISH efunc;                       			/*End definition of estimating equation*/

	START eequat(beta);					 			/*Start to define estimating equation (single argument)*/ 
		ef = efunc(beta);							/*Compute estimating functions*/
		RETURN(ef[+,]);                  			/*Return column sums, 1 by q vector)*/
	FINISH eequat;                       			/*End definition of estimating equation*/

	/***********************************************
	Root-finding */
	initial = {0.,0.,0.,0.,
			   0.,0.,0.,0.,0.,
			   0.,0.,0.,0.,0.,
			   0.,0.,0.,0.,0.,
			   0.,0.,0.,0.,0.};        		* Initial parameter values;
	optn = q || 1;                      	* Set options for nlplm, (8 - requests 8 roots,1 - printing summary output);
	tc = j(1, 12, .);                   	* Create vector for Termination Criteria options, set all to default using .;
	tc[6] = 1e-9;                       	* Replace 6th option in tc to change default tolerance;
	CALL nlplm(rc,                      	/*Use the Levenberg-Marquardt root-finding method*/
			   beta_hat,                	/*... name of output parameters that give the root*/
			   "eequat",                	/*... function to find the roots of*/
			   initial,                 	/*... starting values for root-finding*/
               optn, ,                  	/*... optional arguments for root-finding procedure*/
               tc);                     	/*... update convergence tolerance*/

	/***********************************************
	Baking the bread (approximate derivative) */
	par = q||.||.;                   		* Set options for nlpfdd, (3 - 3 parameters, . = default);
	CALL nlpfdd(func,                   	/*Derivative approximation function*/
                deriv,                  	/*... name of output matrix that gives the derivative*/
                na,                     	/*... name of output matrix that gives the 2nd derivative - we do not need this*/
                "eequat",               	/*... function to approximate the derivative of*/
                beta_hat,               	/*... point where to find derivative*/
                par);                   	/*... details for derivative approximation*/ 
	bread = - (deriv) / n;             	 	* Negative derivative, averaged;

	/***********************************************
	Cooking the filling (matrix algebra) */
	residuals = efunc(beta_hat);		    * Value of estimating functions at beta hat (n by q matrix);
	outerprod = residuals` * residuals;     * Outerproduct of residuals (note transpose is flipped from slides);
	filling = outerprod / n; 				* Divide by n for filling;

	/***********************************************
	Assembling the sandwich (matrix algebra) */
	sandwich = ( inv(bread) * filling * inv(bread)` ) / n;

	/***********************************************
	Formatting results for output */
	vars = {"beta_0","beta_1","beta_2","beta_3",
			"eta_0","eta_1","eta_2","eta_3","eta_4","eta_5",
			"eta_6","eta_7","eta_8","eta_9","eta_10",
			"eta_11","eta_12","eta_13","eta_14","eta_15",
			"eta_16","eta_17","eta_18","eta_19"}; 
	est = beta_hat`;                    			/*Point estimates*/
	se = sqrt(vecdiag(sandwich));       			/*Extract corresponding SE for each parameter*/
	lcl = est - 1.96*se; 							/*Calculated lcl*/
	ucl = est + 1.96*se;							/*Calculate ucl*/
	PRINT vars est se lcl ucl;     					/*Print information to the Results Viewer*/

	CREATE ests_mest VAR {vars est se lcl ucl};   	/*Create an output data set called `out`*/
		APPEND;                         		  	/*... that includes the parameter estimates, variance, and SE*/
	CLOSE ests_mest;                          		/*Close the output*/
	QUIT;                                   
RUN;


/*** IPW estimator of MSM as an M-estimator ***/

PROC IML;                            /*All steps are completed in PROC IML*/
	/***********************************************
	Read data into IML */
	use dat;								/*Open input data from above*/
		read all var {cd4_20wk} into y;		/*Read in each column needed as its own vector*/
		read all var {treat} into a;
		read all var {male} into v;
		read all var {idu} into w1;
		read all var {white} into w2;
		read all var {karnof_90} into w3;
		read all var {karnof_100} into w4;
		read all var {agec} into w5;
		read all var {age_rs1} into w6;
		read all var {age_rs2} into w7;
		read all var {age_rs3} into w8;
		read all var {cd4c_0wk} into w9;
		read all var {cd4_rs1} into w10;
		read all var {cd4_rs2} into w11;
		read all var {cd4_rs3} into w12;

	close dat;
	n = nrow(y);                        	/*Save number of observations in data */

	/***********************************************
	Defining estimating equation */
	q = 18;									/*Save number parameters to be estimated*/

	/*Start to define estimating function */
	START efunc(beta) global(y, a, v, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12);
		yhat = beta[1] + beta[2]*v + beta[3]*a + beta[4]*a#v;  *Design matrix M;
		*Predictions of propensity score;
		loahat = beta[5] + beta[6]*v 
						 + beta[7]*w1 
					     + beta[8]*w2 
					     + beta[9]*w3 + beta[10]*w4 
					     + beta[11]*w5 + beta[12]*w6 + beta[13]*w7 + beta[14]*w8
					     + beta[15]*w9 + beta[16]*w10 + beta[17]*w11 + beta[18]*w12;
		ahat = exp(loahat) / (1+exp(loahat));
		if ahat > 0.99 then ahat = 0.99;
		if ahat < 0.01 then ahat = 0.01;
		ipw = a / ahat + (1-a)/(1-ahat);
		
		/*Estimating functions of marginal structural model*/
		ef_1 = ipw#(y - yhat); 						/*EF for \beta_0*/
		ef_2 = ipw#(y - yhat)#v;					/*EF for \beta_1*/
		ef_3 = ipw#(y - yhat)#a;  					/*EF for \beta_2*/
		ef_4 = ipw#(y - yhat)#a#v;					/*EF for \beta_3*/
		/*Estimating functions of outcome model*/
		ef_5 = (a - ahat);							/*EF for intercept*/
		ef_6 = (a - ahat)#v;						/*EF for male*/
		ef_7 = (a - ahat)#w1;						/*EF for idu*/
		ef_8 = (a - ahat)#w2;						/*EF for white*/
		ef_9 = (a - ahat)#w3;						/*EF for karnof_90*/
		ef_10 = (a - ahat)#w4;						/*EF for karnof_100*/
		ef_11 = (a - ahat)#w5;						/*EF for age*/
		ef_12 = (a - ahat)#w6;						/*EF for age_rs1*/
		ef_13 = (a - ahat)#w7;						/*EF for age_rs2*/
		ef_14 = (a - ahat)#w8;						/*EF for age_rs3*/
		ef_15 = (a - ahat)#w9;						/*EF for cd4*/
		ef_16 = (a - ahat)#w10;						/*EF for cd4_rs1*/
		ef_17 = (a - ahat)#w11;						/*EF for cd4_rs2*/
		ef_18 = (a - ahat)#w12;						/*EF for cd4_rs3*/

		/*Stacking estimating functions together*/
		ef_mat = ef_1||ef_2||ef_3||ef_4
					 ||ef_5||ef_6||ef_7||ef_8||ef_9||ef_10
					 ||ef_11||ef_12||ef_13||ef_14||ef_15
					 ||ef_16||ef_17||ef_18;
		RETURN(ef_mat);                         	/*Return n by q matrix for estimating functions*/
	FINISH efunc;                       			/*End definition of estimating equation*/

	START eequat(beta);					 			/*Start to define estimating equation (single argument)*/ 
		ef = efunc(beta);							/*Compute estimating functions*/
		RETURN(ef[+,]);                  			/*Return column sums, 1 by q vector)*/
	FINISH eequat;                       			/*End definition of estimating equation*/

	/***********************************************
	Root-finding */
	initial = {0.,0.,0.,0.,
			   0.,0.,0.,0.,0.,
			   0.,0.,0.,0.,0.,
			   0.,0.,0.,0.};        		* Initial parameter values;
	optn = q || 1;                      	* Set options for nlplm, (8 - requests 8 roots,1 - printing summary output);
	tc = j(1, 12, .);                   	* Create vector for Termination Criteria options, set all to default using .;
	tc[6] = 1e-9;                       	* Replace 6th option in tc to change default tolerance;
	CALL nlplm(rc,                      	/*Use the Levenberg-Marquardt root-finding method*/
			   beta_hat,                	/*... name of output parameters that give the root*/
			   "eequat",                	/*... function to find the roots of*/
			   initial,                 	/*... starting values for root-finding*/
               optn, ,                  	/*... optional arguments for root-finding procedure*/
               tc);                     	/*... update convergence tolerance*/

	/***********************************************
	Baking the bread (approximate derivative) */
	par = q||.||.;                   		* Set options for nlpfdd, (3 - 3 parameters, . = default);
	CALL nlpfdd(func,                   	/*Derivative approximation function*/
                deriv,                  	/*... name of output matrix that gives the derivative*/
                na,                     	/*... name of output matrix that gives the 2nd derivative - we do not need this*/
                "eequat",               	/*... function to approximate the derivative of*/
                beta_hat,               	/*... point where to find derivative*/
                par);                   	/*... details for derivative approximation*/ 
	bread = - (deriv) / n;             	 	* Negative derivative, averaged;

	/***********************************************
	Cooking the filling (matrix algebra) */
	residuals = efunc(beta_hat);		    * Value of estimating functions at beta hat (n by q matrix);
	outerprod = residuals` * residuals;     * Outerproduct of residuals (note transpose is flipped from slides);
	filling = outerprod / n; 				* Divide by n for filling;

	/***********************************************
	Assembling the sandwich (matrix algebra) */
	sandwich = ( inv(bread) * filling * inv(bread)` ) / n;

	/***********************************************
	Formatting results for output */
	vars = {"beta_0","beta_1","beta_2","beta_3",
			"eta_0","eta_1","eta_2","eta_3","eta_4","eta_5",
			"eta_6","eta_7","eta_8","eta_9","eta_10",
			"eta_11","eta_12","eta_13"}; 
	est = beta_hat`;                    			/*Point estimates*/
	se = sqrt(vecdiag(sandwich));       			/*Extract corresponding SE for each parameter*/
	lcl = est - 1.96*se; 							/*Calculated lcl*/
	ucl = est + 1.96*se;							/*Calculate ucl*/
	PRINT vars est se lcl ucl;     					/*Print information to the Results Viewer*/

	CREATE ests_mest VAR {vars est se lcl ucl};   	/*Create an output data set called `out`*/
		APPEND;                         		  	/*... that includes the parameter estimates, variance, and SE*/
	CLOSE ests_mest;                          		/*Close the output*/
	QUIT;                                   
RUN;

/*END*/
