/**********************************************************************************************************
Bespoke Instrumental Variable via two-stage regression as an M-estimator

Paul Zivich (2023/11/22)
**********************************************************************************************************/

* Reading in formatted data;
PROC IMPORT OUT=dat 
            DATAFILE="C:\Users\zivic\Documents\Research\#PZivich\LetterRichardson\processed.csv" 
            DBMS=CSV REPLACE;
	GETNAMES=YES;
    DATAROW=2; 
RUN;


* M-estimator via PROC IML;
PROC IML;                            /*All steps are completed in PROC IML*/
	/***********************************************
	Read data into IML */
	use dat;							/*Open input data from above*/
		read all var {r} into r;		/*Read in each column needed as its own vector*/
		read all var {a} into a;
		read all var {y} into y;
		read all var {l1} into l1;
		read all var {l2} into l2;
	close dat;
	n = nrow(r);                        /*Save number of observations in data */

	/***********************************************
	Defining estimating equation */
	q = 8;											/*Save number parameters to be estimated*/

	START efunc(beta) global(r, a, y, l1, l2);	   	/*Start to define estimating function */ 
		/*Processing predicted values from models*/
		yhat = beta[3] + beta[4]*l1 + beta[5]*l2;  	/*hatY = E[Y | L, R=0]*/
		ahat = beta[6] + beta[7]*l1 + beta[8]*l2;  	/*hatA = Pr(A=1 | L, R=1)*/
		cde = beta[1] + beta[2]*ahat;              	/*Parameters of interest*/
		ytilde = y - yhat;                         	/*tildeY = Y - hatY*/
		
		/*Estimating functions*/
		ef_1 = r#(ytilde - cde); 					/*EF for \beta_0*/
		ef_2 = r#(ytilde - cde)#ahat;				/*EF for \beta_1*/
		ef_3 = (1-r)#(y - yhat);					/*EF for intercept of hatY*/
		ef_4 = (1-r)#(y - yhat)#l1;					/*EF for L1 term of hatY*/
		ef_5 = (1-r)#(y - yhat)#l2;					/*EF for L2 term of hatY*/
		ef_6 = r#(a - ahat);						/*EF for intercept of hatA*/
		ef_7 = r#(a - ahat)#l1;						/*EF for L1 term of hatA*/
		ef_8 = r#(a - ahat)#l2;						/*EF for L2 term of hat A*/

		/*Stacking estimating functions together*/
		ef_mat = ef_1||ef_2||ef_3||ef_4||ef_5||ef_6||ef_7||ef_8;
		RETURN(ef_mat);                         	/*Return n by q matrix for estimating functions*/
	FINISH efunc;                       			/*End definition of estimating equation*/

	START eequat(beta);					 			/*Start to define estimating equation (single argument)*/ 
		ef = efunc(beta);							/*Compute estimating functions*/
		RETURN(ef[+,]);                  			/*Return column sums, 1 by q vector)*/
	FINISH eequat;                       			/*End definition of estimating equation*/

	/***********************************************
	Root-finding */
	initial = {0.,0.,0.,0.,0.,0.,0.,0.};        * Initial parameter values;
	optn = q || 1;                      * Set options for nlplm, (8 - requests 8 roots,1 - printing summary output);
	tc = j(1, 12, .);                   * Create vector for Termination Criteria options, set all to default using .;
	tc[6] = 1e-9;                       * Replace 6th option in tc to change default tolerance;
	CALL nlplm(rc,                      /*Use the Levenberg-Marquardt root-finding method*/
			   beta_hat,                /*... name of output parameters that give the root*/
			   "eequat",                /*... function to find the roots of*/
			   initial,                 /*... starting values for root-finding*/
               optn, ,                  /*... optional arguments for root-finding procedure*/
               tc);                     /*... update convergence tolerance*/

	/***********************************************
	Baking the bread (approximate derivative) */
	par = q||.||.;                   	* Set options for nlpfdd, (3 - 3 parameters, . = default);
	CALL nlpfdd(func,                   /*Derivative approximation function*/
                deriv,                  /*... name of output matrix that gives the derivative*/
                na,                     /*... name of output matrix that gives the 2nd derivative - we do not need this*/
                "eequat",               /*... function to approximate the derivative of*/
                beta_hat,               /*... point where to find derivative*/
                par);                   /*... details for derivative approximation*/ 
	bread = - (deriv) / n;              * Negative derivative, averaged;

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
	vars = {"beta_0","beta_1","Y|R=0","R|L1,R=0","R|L2,R=0","A|R=1","A|L1,R=1","A|L2,R=1"}; 
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
