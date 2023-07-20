####################################################################################################################
# Empirical sandwich variance estimator for iterated conditional expectation g-computation
#   Illustrative application with Add Health
#
# Rachael Ross
####################################################################################################################

library("tidyverse")
library("numDeriv")
library("rootSolve")
library("geex")

home <- Sys.getenv("HOME")
path <- paste0(home, "/03 UNC/03 Research Project Materials/24 Paul Z/ICE M-est/")
setwd(path)

##################################
# Load data 

d <- read.csv("addhealth_processed.csv")
n <- nrow(d)
d$intercept <- rep(1,n)

##################################
# Creating design matrices

# Observed Outcome: Hypertension (I or II) measured at Wave IV
y <- d$htn_w4

# Setting action pattern for intervention
da <- d %>%
  mutate(cigarette_w1 = 0,
         cigarette_w3 = 0,
         cigarette_w1w3 = 0)

# Observed Confounders Wave III
cols1 <- c('intercept',
        'cigarette_w1', 'cigarette_w3', 'cigarette_w1w3',
        'age_13', 'age_14', 'age_15', 'age_16', 'age_18', 'age_19',
        'height_w3', 'height_w3_sp1', 'height_w3_sp2', 'height_w3_sp3',
        'weight_w3', 'weight_w3_sp1', 'weight_w3_sp2', 'weight_w3_sp3',
        'gender_w3', 'ethnic_w3',
        'race_w3_1', 'race_w3_2', 'race_w3_3',
        'educ_w3_0', 'educ_w3_1', 'educ_w3_3',
        'exercise_w3_0', 'exercise_w3_1', 'exercise_w3_2',
        'alcohol_w3_1', 'alcohol_w3_2', 'alcohol_w3_3', 'alcohol_w3_4',
        'srh_w3_1', 'srh_w3_2', 'srh_w3_3', 'srh_w3_4',
        'hbp_w3', 'hins_w3', 'tried_cigarette')

length(cols1)
X1 <- as.matrix(d[,cols1])
X1a <- as.matrix(da[,cols1])
follow1 <- (1-d$cigarette_w1)*(1-d$cigarette_w3)

# Observed Confounders Wave I
cols0 = c('intercept',
        'cigarette_w1',
        'age_13', 'age_14', 'age_15', 'age_16', 'age_18', 'age_19',
        'height_w1', 'height_w1_sp1', 'height_w1_sp2', 'height_w1_sp3',
        'weight_w1', 'weight_w1_sp1', 'weight_w1_sp2', 'weight_w1_sp3',
        'gender_w1', 'ethnic_w1',
        'race_w1_1', 'race_w1_2', 'race_w1_3',
        'educ_w1_7', 'educ_w1_8', 'educ_w1_9', 'educ_w1_11', 'educ_w1_12',
        'exercise_w1_0', 'exercise_w1_1', 'exercise_w1_2',
        'alcohol_w1_1', 'alcohol_w1_2', 'alcohol_w1_3', 'alcohol_w1_4',
        'srh_w1_1', 'srh_w1_2', 'srh_w1_3', 'srh_w1_4',
        'tried_cigarette')

length(cols0)
X0 <- as.matrix(d[,cols0])
X0a <- as.matrix(da[,cols0])
follow0 <- (1-d$cigarette_w1)*(1-d$cigarette_w3)

##################################
# M-estimation functions for unstratified ICE

# Estimating function for logistic model
ef_logit <- function(beta,X,y){
  ef <- as.vector(y - plogis(X %*% beta))*X
  return(ef)
}

# Estimating function for unstratified ice for specific plan
ef_ice_unstr <- function(theta,X,Xa){
  
  # Wave III
  ef_wave3 <- ef_logit(theta[1:40],X=X[[1]],y=y)
  
  # Predicted outcome at Wave III
  ytilde_3 <- plogis(Xa[[1]] %*% theta[1:40])
  
  # Wave I
  ef_wave1 <- ef_logit(theta[41:78],X=X[[2]],y=ytilde_3)
  
  # Predicted outcome at Wave I
  ytilde_1 <- plogis(Xa[[2]] %*% theta[41:78])
  
  # Mean 
  ef_risk <- ytilde_1 - theta[79]
  
  return(cbind(ef_wave3,ef_wave1,ef_risk))
}

# Estimating function for stack of plans (ban,nc) with risk difference
ef_stacked_ice_unstr <- function(theta){
  theta_set1 <- theta[1:79]
  theta_set2 <- theta[80:(80+78)]
  diff <- theta[(80+78+1)]
  
  ef_ban <- ef_ice_unstr(theta_set1,X=list(X1,X0),Xa=list(X1a,X0a))
  ef_nc <- ef_ice_unstr(theta_set2,X=list(X1,X0),Xa=list(X1,X0))
  ef_rd <- rep(theta_set1[79] - theta_set2[79], n) - diff
  
  stack <- cbind(ef_ban,ef_nc,ef_rd)
  stack <-  replace(stack, is.na(stack), 0) 
  return(stack)
}

# Estimating equation (column sums of estimating function)
estequation <- function(theta){ 
  ee <- colSums(ef_stacked_ice_unstr(theta))
  return(ee)
}

##################################
# M-estimation implementation

# Root-finding
rootfinding <- rootSolve::multiroot(f = estequation,    # Function to find root(s) of
                             start = c(-2,rep(0,78),    # Starting values for root-finding procedure
                                       -2,rep(0,79)))           
ests <- rootfinding$root

# Baking the bread (approximate derivative)
deriv <- numDeriv::jacobian(func = estequation,   # Function to find derivative of
                            x = ests)             # Array of values to compute derivative at (root of estimating equation)
bread <- -1*deriv / n

# Cooking the filling (matrix algebra)
outerprod <- t(ef_stacked_ice_unstr(ests)) %*% ef_stacked_ice_unstr(ests) # Outer product of the residuals
filling <- outerprod/n 

# Assembling the sandwich (matrix algebra)
sandwich <- solve(bread) %*% filling %*% t(solve(bread))
se <- sqrt(diag(sandwich / n))

# Results for parameters of interest
results <- tibble(
  label = c("Risk under ban","Risk under natural course","Difference"),
  ests = ests[c(79,(80+78),159)],
  se = se[c(79,(80+78),159)],
  lcl = ests - 1.96*se,
  ucl = ests + 1.96*se
)

results



