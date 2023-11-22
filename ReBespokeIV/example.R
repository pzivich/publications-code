###############################################################################
# Bespoke Instrumental Variable via two-stage regression as an M-estimator
#
# Paul Zivich (2023/11/22)
###############################################################################

library(geex)

# Reading in Data
setwd("C:/Users/zivic/Documents/Research/#PZivich/LetterRichardson/")
d <- read.csv("processed.csv")


# The root-finding algorithm has trouble with generic starting values,
#   so we will instead 'pre-wash' the nuisance model fits and use those
#   as starting values. 

ef_nui <- function(data){
    r <- data$R
    a <- data$A
    y <- data$Y
    l1 <- data$L1
    l2 <- data$L2
    function(theta){
        yhat <- theta[1] + theta[2]*l1 + theta[3]*l2
        ahat <- theta[4] + theta[5]*l1 + theta[6]*l2

        c((1-r)*(y - yhat),
          (1-r)*(y - yhat)*l1,
          (1-r)*(y - yhat)*l2,
          r*(a - ahat),
          r*(a - ahat)*l1,
          r*(a - ahat)*l2)
    }
}

inits <- c(0, 0, 0, 0, 0, 0)
mestr <- m_estimate(estFUN=ef_nui, data=d,
                    root_control=setup_root_control(start=inits))
inits_wash <- roots(mestr)


# Solving the overall estimating equations with Geex

ef_overall <- function(data){
    r <- data$R
    a <- data$A
    y <- data$Y
    l1 <- data$L1
    l2 <- data$L2
    function(theta){
        yhat <- theta[3] + theta[4]*l1 + theta[5]*l2
        ahat <- theta[6] + theta[7]*l1 + theta[8]*l2
        cde <- theta[1] + theta[2]*ahat
        ytilde <- y - yhat
        
        c(r*(ytilde - cde),
          r*(ytilde - cde)*ahat,
          (1-r)*(y - yhat),
          (1-r)*(y - yhat)*l1,
          (1-r)*(y - yhat)*l2,
          r*(a - ahat),
          r*(a - ahat)*l1,
          r*(a - ahat)*l2)
    }
}

inits <- c(0, 0, inits_wash)
mestr <- geex::m_estimate(estFUN=ef_overall, data=d,
                          root_control=setup_root_control(start=inits))
theta <- roots(mestr)               # Point estimates
se <- sqrt(diag(vcov(mestr)))       # Variance estimates

# Processing results to display
beta0 = theta[1]
beta0_se = se[1]
beta0_ci = c(beta0 - 1.96*beta0_se, beta0 + 1.96*beta0_se)
beta1 = theta[2]
beta1_se = se[2]
beta1_ci = c(beta1 - 1.96*beta1_se, beta1 + 1.96*beta1_se)

print(c(beta0, beta0_ci))
print(c(beta1, beta1_ci))

