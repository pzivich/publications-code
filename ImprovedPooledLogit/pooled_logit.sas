PROC IMPORT DATAFILE='/path/to/data/lau.csv' dbms=CSV OUT=lau REPLACE;
RUN;

PROC PRINT DATA=lau;
RUN;

DATA lau_ind; 
    SET lau; 
    t_days = CEIL(365.25 * t);
    DO i=1 TO t_days;
        days = i;
        IF days = t_days AND eventtype = 2 THEN event = 1;
        ELSE event = 0;
        OUTPUT;
    END;
RUN;

/* Get Unique event times  */
PROC SQL;
    CREATE TABLE unique_event_times AS 
    SELECT DISTINCT days 
    FROM lau_ind 
    WHERE event = 1
    ORDER BY days;
QUIT;

PROC SORT data=lau_ind;
BY days;

/* Use Conditional OUTPUT to create our 4 datasets to compare.  */
DATA lau_1y_improved lau_1y lau_5y_improved lau_5y; 
    MERGE lau_ind unique_event_times(IN=etime);
    BY days;
    
    IF days <= 365 THEN OUTPUT lau_1y;
    IF days <= 5 * 365 THEN OUTPUT lau_5y;
    IF etime AND days <= 365 THEN OUTPUT lau_1y_improved;
    IF etime AND days <= 5 * 365 THEN OUTPUT lau_5y_improved;
RUN;

PROC CONTENTS DATA=lau_1y;
RUN;


/* Run Pooled Logistic Regression - test case on 1y */
PROC GENMOD DATA=lau_1y_improved DESC;
    CLASS event days black(REF='0') baseidu(REF='0');
    MODEL event = days black baseidu ageatfda cd4nadir / LINK=LOGIT DIST=BINOMIAL;
RUN;

PROC GENMOD DATA=lau_1y DESC;
    CLASS event days black(REF='0') baseidu(REF='0');
    MODEL event = days black baseidu ageatfda cd4nadir / LINK=LOGIT DIST=BINOMIAL;
RUN;

/* Now, can create macro to time logistic regression */
%MACRO time_pooled(ds, i=100); 
    ODS SELECT NONE;
    %DO iter=1 %TO &i.; 
        ODS OUTPUT ParameterEstimates=param_estimates;
        %let starttime = %SYSFUNC(DATETIME());
        PROC GENMOD DATA=&ds. DESC;
            CLASS event days black(REF='0') baseidu(REF='0');
            MODEL event = days black baseidu ageatfda cd4nadir / LINK=LOGIT DIST=BINOMIAL;
        RUN;
        %let endtime = %SYSFUNC(DATETIME());
        
        DATA t&iter.;
            SET param_estimates;
            IF parameter <> 'DAYS';
            iter = &iter;
            starttime = &starttime;
            endtime = &endtime; 
        RUN;
        
        %if %sysfunc(exist(time_&ds.)) %then %do;
            PROC APPEND BASE=time_&ds. NEW=t&iter.;
        %end;
        %else %do;
            DATA time_&ds.;
                SET t&iter.;
            RUN;
        %end;
        
    %END;
    ODS SELECT ALL;
%MEND;

/* Run the benchmark for pooled logistic regression */
/* Note that times do not include ODS OUTPUT, which is possibly the largest bottleneck */
%time_pooled(lau_1y, i=100);
%time_pooled(lau_1y_improved, i=100);
%time_pooled(lau_5y, i=100);
%time_pooled(lau_5y_improved, i=100);
