###################################################################################################################
# Estimating Equation Essentials -- R
#   R code to replicate the example described in the text
#
# Paul Zivich (last edit: 2026/07/29)
###################################################################################################################

############################################
# Loading Libraries

library("tidyverse")
library("numDeriv")
library("rootSolve")
library("geex")

############################################
# Data

dat <- tibble(ystar=c(0, 1, 0, 1),
              r=c(1, 1, 0, 0),
              n=c(120,80,15,85)) %>%
  uncount(n)
n <- nrow(dat)     # Number of observations

############################################
# By-Hand Calculation

mu_star <- mean(dat$ystar[dat$r == 1])
alpha <- mean(dat$ystar[dat$r == 0])
mu = mu_star / alpha
print("By-Hand")
print(paste("Naive Proportion:    ", round(mu_star,3)))
print(paste("Sensitivity:         ", round(alpha,3)))
print(paste("Corrected Proportion:", round(mu,3)))

############################################
# Defining estimating equation

############################################
# Defining estimating equation

estimating_function <- function(theta){
  mu_tilde <- theta[1]
  alpha_tilde <- theta[2]

  # Parameter-specific estimating functions
  ef_mean = dat$r * (dat$ystar - mu_tilde*alpha_tilde)
  ef_sens = (1-dat$r) * (dat$ystar - alpha_tilde)

  # Stacking the estimating functions together into vectors
  return(cbind(ef_mean, ef_sens))
} # Function outputs an n by p matrix

estimating_equation <- function(theta){
  estf = estimating_function(theta)       # Return estimating function (n by p)
  este = colSums(estf)                    # Sum over n contributions to estimating functions (p vector)
  return(este)
}

############################################
# Root-finding

proc <- rootSolve::multiroot(f = estimating_equation,     # Function to find root(s) of
                             start = c(0.5,0.5))          # Starting values for root-finding procedure
theta_root <- proc$root                                   # Vector of theta hat

############################################
# Baking the bread (approximate derivative)

deriv <- numDeriv::jacobian(func = estimating_equation,   # Function to find derivative of
                            x = theta_root)               # Array of values to compute derivative at (root of estimating equation)
bread <- -1*deriv / n                                     # Definition of the bread matrix

############################################
# Cooking the filling (matrix algebra)

outerprod <- t(estimating_function(theta_root)) %*% estimating_function(theta_root) # Outer product of the residuals
filling <- outerprod/n

############################################
# Assembling the sandwich (matrix algebra)

sandwich <- solve(bread) %*% filling %*% t(solve(bread))
se <- sqrt(diag(sandwich / n))

ests_mest <-as.data.frame(cbind("theta"=theta_root,
                                "se"=se)) %>%
  mutate(lcl = theta - 1.96*se,
         ucl = theta + 1.96*se)
intlist <- c("mu", "alpha")
row.names(ests_mest) <- intlist

print("Correct Proportion -- EE")
print(round(ests_mest,3))

####################################################################################################################
# Using geex instead of by-hand

geex_ef <- function(data){               # Function of estimating functions (to be used in geex::m_estimate)
  ystar <- data$ystar
  r <- data$r
  function(theta){
      mu_tilde <- theta[1]
      alpha_tilde <- theta[2]

      # Parameter-specific estimating functions
      ef_mean = r * (ystar - mu_tilde*alpha_tilde)
      ef_sens = (1-r) * (ystar - alpha_tilde)

      # Stacking the estimating functions together into vectors
      c(ef_mean, ef_sens)
      }
}

mestr <- geex::m_estimate(estFUN = geex_ef,                      # Function of estimating functions
                          data = dat,                            # Data to be used (must be data frame)
                          root_control = setup_root_control(start = c(0.5, 0.5)))   # Set starting values

theta_geex <- roots(mestr)               # Extract roots
se_geex <- sqrt(diag(vcov(mestr)))       # Extract finite sample variance and take sqrt to get se

ests_geex <-as.data.frame(cbind("theta"=theta_geex,
                                "se"=se_geex)) %>%
  mutate(lcl = theta - 1.96*se,
         ucl = theta + 1.96*se)
row.names(ests_geex) <- intlist

print("Corrected Proportion -- geex")
print(round(ests_geex,3))

# END
