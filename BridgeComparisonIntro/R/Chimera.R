###############################################################################
# Chimera.R is an R file containing the fusion inverse probability weighting
#   estimator for the bridged treatment comparison with survival data.
#
# Paul Zivich (2022/6/11)
###############################################################################

library(rlang)
library(dplyr)
library(survival)
library(ggplot2)


#' Bridging inverse probability weighting (IPW) estimator.
#'
#' Note
#' ----
#'  This estimator expects survival data in the form of one row for each unique 
#'  individual.
#' 
#' The bridging IPW estimator consists of three sets of nuisance functions: the 
#' treatment model, the censoring model, and the sampling model.
#'
#' @param data Data set containing all the necessary variables
#' @param treatment Column name for the treatment of interest
#' @param sample Column name for location variable
#' @param outcome Column name for the outcome variable
#' @param censor Column name for the censoring variable
#' @param time Column name for the time variable
#' @param sample_model R formula for the sampling / location model
#' @param treatment_model R formula for the treatment model
#' @param censor_model R formula for the censoring model
#' @param verbose Whether to display the nuisance model parameters (default is TRUE)
#' @param diagnostic Whether to display the diagnostic twister plot (default is FALSE)
#' @param permutation Whether to run the permutation test (default is FALSE)
#' @param permutation_n Number of iterations to run for the permutation test
#' @param censor_shift Value to shift censored observations by to break ties. Default is 1e-5
#' @return data.frame of point estimates, variance, and 95% confidence intervals
survival.fusion.ipw <- function(data, treatment, sample, outcome, censor, time, 
                                sample_model, treatment_model, censor_model,
                                verbose=TRUE, diagnostic=FALSE, permutation=FALSE,
                                permutation_n=1000, censor_shift=1e-5){
    ### Step 0: Data Prep ###
    d = data %>% arrange(across(c(sample, treatment, time)))
    n_local = sum(d$study)                       # Number of observations in local
    d_distal = nrow(d) - n_local                 # Number of observations in distal
    unique_times = sort(c(0,                     # Extracting all unique event times
                          unique(d[d[, outcome]==1, time]),
                          max(d[, time])))
    # Breaking all censoring ties manually 
    d[, time] = d[, time] + ifelse(d$censor == 1, censor_shift, 0)
    
    ### Step 1: Estimating Nuisance Models ###
    # Sampling model
    d$pr_local = nuisance_location(data=d,
                                   model=sample_model, 
                                   verbose=verbose)
    # Treatment model
    d$pr_treat = nuisance_treatment(data=d,
                                    model=treatment_model, 
                                    location=sample, 
                                    treatment=treatment, 
                                    verbose=verbose)
    # Censoring model
    d$pr_ucens = c(nuisance_censor(data=d[d[, sample]==0, ],
                                   model=censor_model, verbose=verbose),
                   nuisance_censor(data=d[d[, sample]==1, ],
                                   model=censor_model, verbose=verbose)
    )

    # Splitting data by study
    ds0 = d[d[, sample]==0, ]
    ds1 = d[d[, sample]==1, ]

    # Calculating intermediary variables needed for estimator
    fuse_weight = ifelse(d[, sample] == 1, 1, (1-d$pr_local) / (d$pr_local))
    baseline_weight = d$pr_treat * fuse_weight
    timed_weight = d$pr_ucens
    fuse_ipw = sum((1-d[, sample]) * 1/fuse_weight)
    
    # Storage of results
    psi = c()
    psi_se = c()
    diag_psi = c()
    diag_se = c()
    diag_p1 = c()
    diag_p0 = c()
    
    # Estimating the risk difference at each time
    for (tau in unique_times){
        # Pr(Y^{a=2} | S=1)
        numerator = (d[, sample]                          # I(S=1)
                     * ifelse(d[, treatment] == 2, 1, 0)  # I(A=2)
                     * ifelse(d[, time] <= tau, 1, 0)     # I(T<=t)
                     * d[, outcome])                      # Y
        pr_local_r2_i = numerator / (baseline_weight * timed_weight)
        pr_local_r2 = sum(pr_local_r2_i) / n_local
        
        # Pr(Y^{a=1} | S=1)
        numerator = (d[, sample]                          # I(S=1)
                     * ifelse(d[, treatment] == 1, 1, 0)  # I(A=1)
                     * ifelse(d[, time] <= tau, 1, 0)     # I(T<=t)
                     * d[, outcome])                      # Y
        pr_local_r1_i = numerator / (baseline_weight * timed_weight)
        pr_local_r1 = sum(pr_local_r1_i) / n_local

                # Pr(Y^{a=1} | S=0)
        numerator = ((1 - d[, sample])                    # I(S=0)
                     * ifelse(d[, treatment] == 1, 1, 0)  # I(A=1)
                     * ifelse(d[, time] <= tau, 1, 0)     # I(T<=t)
                     * d[, outcome])                      # Y
        pr_fusion_r1_i = numerator / (baseline_weight * timed_weight)
        pr_fusion_r1 = sum(pr_fusion_r1_i) / fuse_ipw
        
        # Pr(Y^{a=0} | S=0)
        numerator = ((1 - d[, sample])                    # I(S=0)
                     * ifelse(d[, treatment] == 0, 1, 0)  # I(A=0)
                     * ifelse(d[, time] <= tau, 1, 0)     # I(T<=t)
                     * d[, outcome])                      # Y
        pr_fusion_r0_i = numerator / (baseline_weight * timed_weight)
        pr_fusion_r0 = sum(pr_fusion_r0_i) / fuse_ipw
        
        # Estimation!
        psi_tau = (pr_local_r2 - pr_local_r1) + (pr_fusion_r1 - pr_fusion_r0)
        var = sum(((pr_local_r2_i - pr_local_r1_i) + 
                       (pr_fusion_r1_i - pr_fusion_r0_i)
                   - psi_tau)**2)  / (n_local**2)
        se_tau = sqrt(var)
        psi = c(psi, psi_tau)
        psi_se = c(psi_se, se_tau)

        # Estimation for Diagnostics
        if (diagnostic){
            diag_psi_tau = pr_local_r1 - pr_fusion_r1
            diag_psi = c(diag_psi, diag_psi_tau)
            diag_var = sum((pr_local_r1_i - pr_fusion_r1_i - 
                                diag_psi_tau)**2) / n_local**2
            diag_se = c(diag_se, sqrt(diag_var))
        }
        if (permutation){
            diag_p1 = c(diag_p1, pr_local_r1)
            diag_p0 = c(diag_p0, pr_fusion_r1)
        }
    }

    # Packing up results to send back to user
    results = data.frame(t=unique_times,
                         rd=psi, 
                         rd_se=psi_se,
                         rd_lcl = psi - 1.96*psi_se,
                         rd_ucl = psi + 1.96*psi_se)
    
    # Running the diagnostic procedures if requested
    if (diagnostic){
        # Diagnostic twister plot
        diagnose = data.frame(t=unique_times,
                              rd=diag_psi,
                              rd_lcl = diag_psi - 1.96*diag_se,
                              rd_ucl = diag_psi + 1.96*diag_se)
        # Creating plot with function from below
        p = twister_plot(diagnose,
                         xvar=rd,
                         lcl=rd_lcl, ucl=rd_ucl,
                         yvar=t,
                         xlab="Difference in Shared", ylab="Time",
                         reference_line=0.0)
        # Displaying plot in console
        print(p)
    }
    
    # Permutation diagnostic procedure
    if (permutation){
        # Calculating the observed area between the step functions
        observed_area = area_between_steps(x=unique_times, 
                                           y0=diag_p1, 
                                           y1=diag_p0)
        # Data prep for permutation iterations
        pd = data.frame(d)                                     # Create copy
        pd$base_weight = 1 / baseline_weight                   # Saving weight
        pd$full_weight = 1 / (baseline_weight * timed_weight)  # Saving weight
        pd = pd %>% filter(pd[, treatment] == 1)               # Only treated
        
        # Estimating the area under permutations 
        perm_areas = sapply(1:permutation_n, 
                            permute_iteration, 
                            data=pd, time=time, 
                            location=sample, outcome=outcome, 
                            treatment=treatment)
        pvalue = mean(ifelse(perm_areas > observed_area, 1, 0))
        
        # Displaying permutation results        
        message("====================================================")
        message("Permutation Test")
        message("====================================================")
        message(paste("Observed area: ", toString(observed_area)))
        message(paste("No. Permutations: ", toString(permutation_n)))
        message("----------------------------------------------------")
        message(paste("P-value: ", toString(pvalue)))
        message("====================================================")
    }
    
    # Returning the results
    return(results)    
}


#' Function used to estimate the nuisance model for sampling / location
#' 
#' @param data Data set containing all the necessary variables
#' @param model R formula for the selection model
#' @param verbose Whether to display the nuisance model parameters
#' @return estimated probability for the specific location
nuisance_location <- function(data, model, verbose){
    # Location model: Pr(S|W)
    nuisance_sampling <- glm(model, 
                             data=data, 
                             family=binomial())
    if (verbose){
        message("=================================================================")
        message("Sampling Model")
        print(summary(nuisance_sampling))
        message("=================================================================")
    }
    prob_sampling <- predict(nuisance_sampling, type="response")
    return(prob_sampling)
}


#' Function used to estimate the nuisance model for treatment
#' 
#' @param data Data set containing all the necessary variables
#' @param model R formula for the selection model
#' @param location Column name for location variable
#' @param treatment Column name for the treatment of interest
#' @param verbose Whether to display the nuisance model parameters
#' @return estimated probability for received treatment
nuisance_treatment <- function(data, model, location, treatment, verbose){
    ds0 = data %>% filter(data[, location]==0)
    ds1 = data %>% filter(data[, location]==1)
    ds1[,treatment] = ds1[,treatment] - 1
    
    # Assigned treatment: Pr(A=a,s)
    nuisance_treatment_s0 <- glm(model, 
                                 data=ds0, 
                                 family=binomial())
    if (verbose){
        message("=================================================================")
        message("Treatment Model : S=0")
        print(summary(nuisance_treatment_s0))
        message("=================================================================")
    }
    
    nuisance_treatment_s1 <- glm(model, 
                                 data=ds1, 
                                 family=binomial())
    if (verbose){
        message("=================================================================")
        message("Treatment Model : S=1")
        print(summary(nuisance_treatment_s1))
        message("=================================================================")
    }
    # stacking together
    pr_t_s0 = predict(nuisance_treatment_s0, type="response")
    pr_t_s1 = predict(nuisance_treatment_s1, type="response")
    
    prob_treatment = c(ifelse(ds0[, treatment] == 1, pr_t_s0, 1-pr_t_s0), 
                       ifelse(ds1[, treatment] == 1, pr_t_s1, 1-pr_t_s1))
    
    return(prob_treatment)
}


#' Function used to estimate the nuisance model for censoring
#' 
#' @param data Data set containing all the necessary variables
#' @param model R formula for the selection model
#' @param verbose Whether to display the nuisance model parameters
#' @return estimated probability of remaining uncensored till last time
nuisance_censor <- function(data, model, verbose){
    # Fixing some scoping issues of the formulas (only looks local)
    environment(model) <- list2env(list(data=data))
    
    # Censoring Model: Pr(C>t|V,A,S)
    censor_model <- coxph(model, 
                          data=data, 
                          method="breslow"
                          )
    if (verbose){
        message("=================================================================")
        message("Censoring Model")
        print(summary(censor_model))
        message("=================================================================")
    }
    prob_uncensor = predict(censor_model, type="survival") 
    return(prob_uncensor)
}


#### Step Ribbon ####
StatStepribbon <- ggproto("StatStepribbon",
                          Stat,
                          compute_group=function(., data, scales, direction = "hv",
                                                 yvars = c( "ymin", "ymax" ), ...)
                          {
                              direction <- match.arg( direction, c( "hv", "vh" ) )
                              data <- as.data.frame( data )[ order( data$x ), ]
                              n <- nrow( data )
                              
                              if ( direction == "vh" ) {
                                  xs <- rep( 1:n, each = 2 )[ -2 * n ]
                                  ys <- c( 1, rep( 2:n, each = 2 ) )
                              } else {
                                  ys <- rep( 1:n, each = 2 )[ -2 * n ]
                                  xs <- c( 1, rep( 2:n, each = 2))
                              }
                              
                              data.frame(
                                  x = data$x[ xs ]
                                  , data[ ys, yvars, drop=FALSE ]
                                  , data[ xs, setdiff( names( data ), c( "x", yvars ) ), drop=FALSE ]
                              )
                          },
                          required_aes=c( "x", "ymin", "ymax" ),
                          default_geom=GeomRibbon,
                          default_aes=aes( x=..x.., ymin = ..y.., ymax=Inf )
)

stat_stepribbon = function( mapping=NULL, data=NULL, geom="ribbon",
                            position="identity") {
    layer(stat=StatStepribbon, mapping=mapping, data=data, geom=geom, position=position )
}

#### Twister Plot####

#' Twister Plot
#'
#' @param dat A \code{data.frame} with the risk difference, upper and lower confidence limits, and times
#' @param xvar The variable name for the risk difference. Defaults to RD.
#' @param lcl  The variable name for the lower confidence limit of the risk difference. Defaults to RD_LCL.
#' @param ucl  The variable name for the upper confidence limit of the risk difference. Defaults to RD_UCL.
#' @param yvar The variable name for time. Defaults to "t".
#' @param xlab The x-axis label. Defaults to "Risk Difference".
#' @param ylab The y-axis label. Defaults to "Days".
#' @param treat_labs A vector containing the names of the treatment groups. Defaults to c("Treat", "Control")
#'
#' @return a \code{ggplot} object
twister_plot <- function(dat,
                         xvar = RD,
                         lcl = RD_LCL,
                         ucl = RD_UCL,
                         yvar = t,
                         xlab = "Risk Difference",
                         ylab = "Days",
                         reference_line = 0.0,
                         log_scale = FALSE){
    
    base_breaks <- function(n = 10){
        function(x) {
            axisTicks(log10(range(x, na.rm = TRUE)), log = TRUE, n = n)
        }
    }
    
    `%>%` <- magrittr::`%>%`
    pull <- dplyr::pull
    
    t_lim <- max(dat %>% pull({{yvar}}))
    if (log_scale) {
        x_lim <- max(abs(log(dat %>% pull({{lcl}}))), 
                     abs(log(dat %>% pull({{ucl}}))))
        y_scale = scale_y_continuous(limits = c(exp(-x_lim), exp(x_lim)), 
                                     trans="log", 
                                     breaks=base_breaks())
        text_loc = c(-x_lim/2, x_lim/2)
    } else {
        x_lim <- max(abs(dat %>% pull({{lcl}})), abs(dat %>% pull({{ucl}})))
        y_scale = scale_y_continuous(limits = c(-x_lim, x_lim))
        text_loc = c(-x_lim/2, x_lim/2)
    }
    
    p <- ggplot(data = dat, aes(x = {{yvar}}, y = {{xvar}})) + 
        geom_step() +
        geom_ribbon(
            aes(ymin = {{lcl}}, ymax = {{ucl}}),
            stat = "stepribbon",
            alpha = 0.2,
            direction = "hv"
        ) +
        geom_hline(yintercept = reference_line, linetype = "dotted") +
        y_scale + 
        scale_x_continuous(limits = c(0, t_lim), expand = c(0, 0)) +
        coord_flip(clip = "off") +
        theme(axis.line = element_line(colour = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.background = element_blank(),
              panel.border = element_rect(colour = "black", fill = NA),
              plot.margin = unit(c(2, 1, 1, 1), "lines")) +
        xlab(ylab) +
        ylab(xlab)
    p
}

#' Function to calculate the area between two step functions. Used in the 
#' diagnostic procedure.
#' @param x X-values for the step functions
#' @param y0 Y-values for the first step function at each unique X
#' @param y1 Y-values for the second step function at each unique X
area_between_steps = function(x, y0, y1, signed=FALSE){
    # Calculating difference is x-steps
    x_measures = lead(x) - x
    # Calculating difference between y's
    y_measures = y0 - y1

    # Optional logic for the signed difference
    if (!signed){
        y_measures = abs(y_measures)
    }
    
    # Return the calculated area
    return(sum(x_measures * y_measures, na.rm=T))
}


#' Function for a single iteration of the permutation procedure
permute_iteration = function(x, data, time, location, outcome, treatment){
    # Permute the data set
    d = data.frame(data)
    d[, location] = sample(data[,location])
    
    # Setting up storage    
    distal_r1 = c()
    local_r1 = c()
    unique_t = sort(c(0, unique(d[d[, outcome]==1, time])))
    
    # Estimating the risk difference at each time
    for (tau in unique_t){
        # Pr(Y^{a=1} | S=0)
        numerator = ((1 - d[, location])                 # I(W=d)
                     * ifelse(d[, treatment] == 1, 1, 0)  # I(A=1)
                     * ifelse(d[, time] <= tau, 1, 0)    # I(T<=t)
                     * d[, outcome])                     # Y
        pr_fusion_r1_i = numerator * d$full_weight
        pr_fusion_r1 = sum(pr_fusion_r1_i) / sum((1 - d[, location]) * d$base_weight)
        distal_r1 = c(distal_r1, pr_fusion_r1)
        
        # Pr(Y^{a=1} | S=1)
        numerator = (d[, location]                       # I(W=l)
                     * ifelse(d[, treatment] == 1, 1, 0)  # I(A=1)
                     * ifelse(d[, time] <= tau, 1, 0)    # I(T<=t)
                     * d[, outcome])                     # Y
        pr_local_r1_i = numerator * d$full_weight
        pr_local_r1 = sum(pr_local_r1_i) / sum(d[, location] * d$base_weight)
        local_r1 = c(local_r1, pr_local_r1)
    }
    
    val = area_between_steps(x=unique_t,
                             y0=local_r1,
                             y1=distal_r1)
    return(val)
}