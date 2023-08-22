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
library(zoo)


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
#' @param sample Column name for sample variable
#' @param outcome Column name for the outcome variable
#' @param censor Column name for the censoring variable
#' @param time Column name for the time variable
#' @param sample_model R formula for the sampling / sample model
#' @param treatment_model R formula for the treatment model
#' @param censor_model R formula for the censoring model
#' @param verbose Whether to display the nuisance model parameters (default is TRUE)
#' @param diagnostic_plot Whether to display the diagnostic twister plot (default is FALSE)
#' @param diagnostic_test Whether to run the diagnostic test based on the integrated risk difference (IRD) (default is FALSE)
#' @param bootstrap_n Number of iterations to run for the bootstrap variance
#' @param censor_shift Value to shift censored observations by to break ties. Default is 1e-4
#' @return data.frame of point estimates, variance, and 95% confidence intervals
survival.fusion.ipw <- function(data, treatment, sample, outcome, censor, time, 
                                sample_model, treatment_model, censor_model,
                                verbose=TRUE, diagnostic_plot=FALSE, 
                                diagnostic_test=FALSE, bootstrap_n=1000,
                                censor_shift=1e-4){
    ### Computing point estimates ###
    estimates = survival.bridge.point(data=data, 
                                      treatment=treatment, 
                                      outcome=outcome, 
                                      time=time, 
                                      sample=sample, 
                                      sample_model=sample_model, 
                                      treatment_model=treatment_model, 
                                      censor_model=censor_model,
                                      censor_shift=censor_shift, 
                                      verbose=verbose,
                                      resample=F)
    point_est = estimates[[1]]
    results = point_est

    ### Computing variance estimates via bootstrap ###
    var_ests = sapply(1:bootstrap_n, 
                      survival.bridge.point, 
                      data=data, 
                      treatment=treatment, 
                      outcome=outcome, 
                      time=time, 
                      sample=sample, 
                      sample_model=sample_model, 
                      treatment_model=treatment_model, 
                      censor_model=censor_model,
                      censor_shift=censor_shift, 
                      verbose=F,
                      resample=T)

    ### Variance Estimation ###
    var_est = data.frame()
    for (col in c("rd", "rd_diag")){
        point_est$constant = 1                               # R is a pain to merge...
        align_var = merge(point_est[, c("t", "constant")],   # so merging to get t's
                          var_ests[[1]], by="t", all=T)      # for each variable
        align_var = na.locf(align_var)
        var_est_c = matrix(align_var[[col]])                     # Making into matrix
        for (i in 2:bootstrap_n){                                # Appending rest to matrix
            align_var = merge(point_est[, c("t", "constant")],   # so merging to get t's
                              var_ests[[i]], by="t", all=T)      # for each variable
            align_var = na.locf(align_var)
            var_est_c = cbind(var_est_c, 
                              matrix(align_var[[col]]))
        }
        results[, paste(col, ".se", sep="")] = apply(var_est_c, 1, var)^0.5
    }
    
    # Calculating confidence intervals
    results$rd_lcl = results$rd - 1.96*results$rd.se
    results$rd_ucl = results$rd + 1.96*results$rd.se
    results$rd_diag_lcl = results$rd_diag - 1.96*results$rd_diag.se
    results$rd_diag_ucl = results$rd_diag + 1.96*results$rd_diag.se
    
    # Running the diagnostic procedures if requested
    if (diagnostic_plot){
        # Creating plot with function from below
        p = twister_plot(results,
                         xvar=rd_diag,
                         lcl=rd_diag_lcl, 
                         ucl=rd_diag_ucl,
                         yvar=t,
                         xlab="Difference in Shared", 
                         ylab="Time",
                         reference_line=0.0)
        # Displaying plot in console
        print(p)
    }
    
    # Diagnostic test procedure
    if (diagnostic_test){
        # Calculating the observed area between the step functions
        observed_area = area_between_steps(x=results$t, 
                                           rd=results$rd_diag,
                                           signed=T)

        # Estimating the area under resamples and re-estimates
        # perm_areas = sapply(1:bootstrap_n, 
        #                    permute_iteration, 
        #                    data=perm_ind, 
        #                    time=time, 
        #                    sample=sample, 
        #                    outcome=outcome, 
        #                    treatment=treatment)
        # pvalue = mean(ifelse(perm_areas > observed_area, 1, 0))
        var_area_ests = sapply(1:bootstrap_n, 
                          diagnostic_test_iteration, 
                          data=data, 
                          treatment=treatment, 
                          outcome=outcome, 
                          time=time, 
                          sample=sample, 
                          sample_model=sample_model, 
                          treatment_model=treatment_model, 
                          censor_model=censor_model,
                          censor_shift=censor_shift, 
                          verbose=F,
                          resample=T)
        se_area = var(var_area_ests)^0.5
        zscore = observed_area / se_area
        area_lcl = observed_area - 1.96*se_area
        area_ucl = observed_area + 1.96*se_area
        pvalue = pnorm(zscore, mean=0, sd=1, lower.tail=F)*2
        
        # Displaying diagnostic test based on the IRD results        
        message("====================================================")
        message("Diagnostic Test")
        message("====================================================")
        message(paste("No. Bootstraps: ", toString(bootstrap_n)))
        message("----------------------------------------------------")
        message(paste("Area:    ", toString(round(observed_area, 3))))
        message(paste("95% CI:  ", toString(round(area_lcl, 3)), 
                      toString(round(area_ucl, 3))))
        message(paste("P-value: ", toString(round(pvalue, 3))))
        message("====================================================")
    }
    
    # Returning the results
    return(results)    
}


#' Bridging inverse probability weighting (IPW) point estimator.
#'
#' Note
#' ----
#'  This is an internal function called by the survival.fusion.ipw estimator 
#'  for both the point estimates and variance estimates
#'  
#' @param data Data set containing all the necessary variables
#' @param treatment Column name for the treatment of interest
#' @param sample Column name for sample variable
#' @param outcome Column name for the outcome variable
#' @param censor Column name for the censoring variable
#' @param time Column name for the time variable
#' @param sample_model R formula for the sampling / sample model
#' @param treatment_model R formula for the treatment model
#' @param censor_model R formula for the censoring model
#' @param verbose Whether to display the nuisance model parameters (default is TRUE)
#' @param diagnostic Whether to display the diagnostic twister plot (default is FALSE)
#' @param bootstrap_n Number of iterations to run for the bootstrap variance
#' @param censor_shift Value to shift censored observations by to break ties. Default is 1e-4
#' @return data.frame of point estimates, variance, and 95% confidence intervals
survival.bridge.point <- function(data, treatment, outcome, time, sample, 
                                  sample_model, treatment_model, censor_model,
                                  censor_shift, verbose, resample, ...){
    ### Step 0: Data Prep ###
    if (resample){
        dat = data[sample(nrow(data), replace=TRUE),]
    }
    else {
        dat = data
    }
    d = dat %>% arrange(across(c(sample, treatment, time)))
    n_local = sum(d$study)                       # Number of observations in local
    d_distal = nrow(d) - n_local                 # Number of observations in distal
    unique_times = sort(c(0,                     # Extracting all unique event times
                          unique(d[d[, outcome]==1, time]),
                          max(d[, time])))
    # Breaking all censoring ties manually 
    d[, time] = d[, time] + ifelse(d$censor == 1, censor_shift, 0)

    ### Step 1: Estimating Nuisance Models ###
    # Sampling model
    d$pr_local = nuisance_sample(data=d,
                                 model=sample_model, 
                                 verbose=verbose)
    # Treatment model
    d$pr_treat = nuisance_treatment(data=d,
                                    model=treatment_model, 
                                    sample=sample, 
                                    treatment=treatment, 
                                    verbose=verbose)
    # Censoring model
    d$pr_ucens = nuisance_censor(data=d,
                                 model=censor_model, 
                                 verbose=verbose)
    # d$pr_ucens = c(nuisance_censor(data=d[d[, sample]==0, ],
    #                               model=censor_model, verbose=verbose),
    #               nuisance_censor(data=d[d[, sample]==1, ],
    #                               model=censor_model, verbose=verbose)

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
    diag_psi = c()
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
        psi = c(psi, psi_tau)
        
        # Estimation for Diagnostics
        diag_psi_tau = pr_local_r1 - pr_fusion_r1
        diag_psi = c(diag_psi, diag_psi_tau)
        diag_p1 = c(diag_p1, pr_local_r1)
        diag_p0 = c(diag_p0, pr_fusion_r1)
    }
    # Processing outputs
    result = data.frame(t=unique_times,
                        rd=psi,
                        rd_diag=diag_psi)
    return(list(result))
}



#' Function used to estimate the nuisance model for sampling / sample
#' 
#' @param data Data set containing all the necessary variables
#' @param model R formula for the selection model
#' @param verbose Whether to display the nuisance model parameters
#' @return estimated probability for the specific sample
nuisance_sample <- function(data, model, verbose){
    # sample model: Pr(S|W)
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
#' @param sample Column name for sample variable
#' @param treatment Column name for the treatment of interest
#' @param verbose Whether to display the nuisance model parameters
#' @return estimated probability for received treatment
nuisance_treatment <- function(data, model, sample, treatment, verbose){
    ds0 = data %>% filter(data[, sample]==0)
    ds1 = data %>% filter(data[, sample]==1)
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
area_between_steps = function(x, rd, signed=TRUE){
    # Calculating difference is x-steps
    x_measures = lead(x) - x

    # Optional logic for the signed difference
    if (!signed){
      y_measures = abs(rd)
    } else {
      y_measures = rd
    }
    
    # Return the calculated area
    return(sum(x_measures * y_measures, na.rm=T))
}


#' Function for a single iteration of the diagnostic IRD procedure
diagnostic_test_iteration = function(data, treatment, outcome, time, sample, 
                                     sample_model, treatment_model, censor_model,
                                     censor_shift, verbose, resample, ...){
    estimates = survival.bridge.point(data=data, 
                                      treatment=treatment, 
                                      outcome=outcome, 
                                      time=time, 
                                      sample=sample, 
                                      sample_model=sample_model, 
                                      treatment_model=treatment_model, 
                                      censor_model=censor_model,
                                      censor_shift=censor_shift, 
                                      verbose=F,
                                      resample=T)
    results = estimates[[1]]
    area = area_between_steps(x=results$t, 
                              rd=results$rd_diag,
                              signed=T)
    return(area)
}
