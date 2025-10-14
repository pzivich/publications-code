#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file runs the parametric model analysis reported in Appendix 2.
#
# Paul Zivich (2025/10/14)
#####################################################################################################################

library(plyr)
library(geex)
library(resample)
library(dplyr)


##########################################################
##### Setting up data
##########################################################

nhanes <- read.csv("nhanes.csv")
nhanes$height = nhanes$height / 2.54
nhanes <- nhanes[!is.na(nhanes$height) & !is.na(nhanes$weight), ]
nhanes$agelt8 <- ifelse(nhanes$age<8, 1, 0)


##########################################################
##### Extrapolation, mean SPB
##########################################################

#estimate the variance using M-estimation (geex package)
estfun_extrap <- function(data, models){
    W <- data$sample_weight
    M <- data$miss
    Y <- data$sbp
    
    #fit outcome model (by missingness variable, so we can later limit to complete case)
    Xmat <- grab_design_matrix(data=data, rhs_formula=grab_fixed_formula(models$out))
    out_scores <- grab_psiFUN(models$out, data)
    out_pos <- 1:ncol(Xmat)
    
    function(theta){
        p <- length(theta)
        #get predicted values from model for everyone
        Y.imp <- Xmat %*% theta[out_pos]
        
        #estimating equations
        c(W*out_scores(theta[out_pos])*(1-M),
          W*(Y.imp-theta[p]))
    }
}

#function to call M-estimator
geex_extrap <- function(data, out_formula){
    out_model  <- glm(out_formula, data=data)
    models <- list(out=out_model)
    
    geex_results_extrap <- m_estimate(
        estFUN = estfun_extrap,
        data = data,
        root_control = setup_root_control(start=c(coef(out_model), 100)),
        outer_args = list(models = models))
    return(geex_results_extrap)
}

#to avoid missing values in dataset, set all missing values to 999 (these will get replaced with imputed values during estimation)
nhanes.hold <- nhanes
nhanes.hold$sbp <- ifelse(nhanes$miss==1, 999, nhanes$sbp)

#run the M-estimation code to get standard error
extrap.res <- geex_extrap(nhanes.hold, sbp ~ female*(age + age_sp1 + age_sp2 
                                                     + height + h_sp1 + h_sp2 
                                                     + weight + w_sp1 + w_sp2))

#format output and compute CIs
extrap.est <- extrap.res@estimates[length(extrap.res@estimates)]
extrap.se <- as.numeric(sqrt(extrap.res@vcov[length(extrap.res@estimates),length(extrap.res@estimates)]))
extrap.est.all <- as.data.frame(cbind(extrap.est,extrap.se))
names(extrap.est.all) <- c('est','se')
extrap.est.all$LCL <- as.numeric(extrap.est.all$est) - 1.96*as.numeric(extrap.est.all$se)
extrap.est.all$UCL <- as.numeric(extrap.est.all$est) + 1.96*as.numeric(extrap.est.all$se)
extrap.est.all$type <- "Extrapolated"
extrap.est.all


##########################################################
##### Synthesis, mean SPB
##########################################################

### bring in data for mathematical model for children 2-7
height.params <- read.csv("height_params.csv")
sbp_params <- read.csv("sbp_params.csv")
sbp_params <- within(sbp_params, rm("code"))

#first, add height cutoffs to the dataset
nhanes$gender <- ifelse(nhanes$female==1, 'f', 'm')
nhanes2 <- join(nhanes, height.params, by=c("gender", "age"))
nhanes2$height <- nhanes2$height / 2.54
nhanes2$height_cat <- ifelse(nhanes2$height<nhanes2$c1,
                             1,
                             ifelse(nhanes2$height<nhanes2$c2,
                                    2,
                                    ifelse(nhanes2$height<nhanes2$c3,
                                           3,
                                           ifelse(nhanes2$height<nhanes2$c4,
                                                  4,
                                                  ifelse(nhanes2$height<nhanes2$c5,
                                                         5,
                                                         ifelse(nhanes2$height<nhanes2$c6,
                                                                6, 7))))))

#add on median and p90 sbps based on age, gender, and height category
nhanes3 <- join(nhanes2, sbp_params, by=c("gender", "age", "height_cat"))
nhanes3$norm_mean <- nhanes3$median
nhanes3$norm_sd <- (nhanes3$p90 - nhanes3$norm_mean) / qnorm(0.9)

### bootstrap function, from which we will derive point estimate and 95% CIs
bootstrap_syn <- function(B, bootdata){
    if(B == 0) return(0)
    if(B>0){
        boot.est <- matrix(NaN, nrow=B, ncol=1)
        datbi <- samp.bootstrap(nrow(bootdata), B)
        for(i in 1:B){
            dati <- bootdata[datbi[,i],]

            # split data into positive and nonpositive
            positive<-dati[dati$agelt8==0, ]
            npositive<-dati[dati$agelt8==1, ]
            
            # Fit outcome model for positive region, use it to impute those missing in positive region
            out.model  <- glm(sbp ~ female*(age + age_sp1 + age_sp2 
                                            + height + h_sp1 + h_sp2
                                            + weight + w_sp1 + w_sp2),
                              data=positive,
                              weights=positive$sample_weight)
            positive$sbp_hat <- predict(out.model, positive, type="response")

            #get a random draw from mathematical model for nonpositive region
            npositive$sbp_hat <- rnorm(nrow(npositive), 
                                       mean=npositive$norm_mean, 
                                       sd=npositive$norm_sd)
            
            #combine
            allimputed <- rbind(positive, npositive)
            
            #store weighted mean
            boot.est [i] <- sum(allimputed$sbp_hat * allimputed$sample_weight) / sum(allimputed$sample_weight)
        }
        return(boot.est)
    }
}

#run the function with 10000 bootstraps
syn.boots <- bootstrap_syn(10000, nhanes3)

#format output and compute CIs
syn.est <- median(syn.boots)
syn.LCL <- (quantile(syn.boots, probs=0.025, na.rm=FALSE))
syn.UCL <- (quantile(syn.boots, probs=0.975, na.rm=FALSE))
syn.all <- as.data.frame(cbind(syn.est, syn.LCL, syn.UCL))
syn.all$type <- "Synthesis"
syn.all$se <- NA
names(syn.all) <- c('est', 'LCL', 'UCL', 'type', 'se')

# Output 
#
#       est      se      LCL      UCL         type
#  100.7531 1.55831 97.69881 103.8074 Extrapolated
# 
#      est      LCL     UCL      type se
# 100.4801 99.94589 101.0217 Synthesis NA
