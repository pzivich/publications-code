#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file runs the bounds reported in the main paper.
#
# Paul Zivich (2025/10/14)
#####################################################################################################################

library(plyr)
library(geex)
library(resample)
library(dplyr)


###############################################
# Setting up data

setwd("C:/Users/zivic/Documents/open-source/publications-code/SynthesisMNTE")
nhanes <- read.csv("data/nhanes.csv")
nhanes$agelt8 <- ifelse(nhanes$age<8, 1, 0)

nhanes$age8 <- ifelse(nhanes$age == 8, 1, 0)
nhanes$age9 <- ifelse(nhanes$age == 9, 1, 0)
nhanes$age10 <- ifelse(nhanes$age == 10, 1, 0)
nhanes$age11 <- ifelse(nhanes$age == 11, 1, 0)
nhanes$age12 <- ifelse(nhanes$age == 12, 1, 0)
nhanes$age13 <- ifelse(nhanes$age == 13, 1, 0)
nhanes$age14 <- ifelse(nhanes$age == 14, 1, 0)
nhanes$age15 <- ifelse(nhanes$age == 15, 1, 0)
nhanes$age16 <- ifelse(nhanes$age == 16, 1, 0)
nhanes$age17 <- ifelse(nhanes$age == 17, 1, 0)


###############################################
# Synthesis bounds

#to avoid missing values in dataset, set all missing values to 999
nhanes.hold <- nhanes
nhanes.hold$sbp <- ifelse(nhanes$miss==1, 999, nhanes$sbp)


#estimate the variance using M-estimation (geex package)
estfun_extrap <- function(data, models){
  W <- data$sample_weight
  M <- data$miss
  Y <- data$sbp

  #fit outcome model (by missingness variable, so we can later limit to complete case)
  Xmat <- grab_design_matrix(data=data, rhs_formula=grab_fixed_formula(models$out))
  out_scores <- grab_psiFUN(models$out, data)
  out_pos <- 1:ncol(Xmat)

  #     yhat = np.where(x_star == 1, yhat, mu_nonpos)           # Setting SBP for those in nonpositive region

  function(theta){
      p <- length(theta)
      #get predicted values from model for everyone
      Y.imp <- Xmat %*% theta[out_pos]
      Y.imp <- ifelse(data$agelt8 == 1, boundary, Y.imp)
      
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

### Lower bound ###
boundary <- 70
bound.res <- geex_extrap(nhanes.hold, sbp ~ age8 + age9 + age10 + age11 + age12
                         + age13 + age14 + age15 + age16 + age17 - 1)

#format output and compute CIs
bound.est <- bound.res@estimates[length(bound.res@estimates)]
bound.se <- as.numeric(sqrt(bound.res@vcov[length(bound.res@estimates),length(bound.res@estimates)]))
bound <- as.data.frame(cbind(bound.est,bound.se))
names(bound) <- c('est','se')
bound$LCL <- as.numeric(bound$est) - 1.96*as.numeric(bound$se)
bound$type <- "Lower"
bound

### Upper bound ###
boundary <- 120
bound.res <- geex_extrap(nhanes.hold, sbp ~ age8 + age9 + age10 + age11 + age12
                         + age13 + age14 + age15 + age16 + age17 - 1)

#format output and compute CIs
bound.est <- bound.res@estimates[length(bound.res@estimates)]
bound.se <- as.numeric(sqrt(bound.res@vcov[length(bound.res@estimates),length(bound.res@estimates)]))
bound <- as.data.frame(cbind(bound.est,bound.se))
names(bound) <- c('est','se')
bound$UCL <- as.numeric(bound$est) + 1.96*as.numeric(bound$se)
bound$type <- "Upper"
bound


# Output
#       est        se      LCL  type
#  92.70003 0.4292731 91.85866 Lower
# 
#     est        se     UCL  type
#  109.93 0.2739797 110.467 Upper
