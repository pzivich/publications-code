###################################################################################################################
# Confidence Regions for Multiple Outcomes, Effect Modifiers, and Other Multiple Comparisons
#
# Jess Edwards (with formatting modifications from Paul Zivich)
###################################################################################################################

# set up workspace ----
library(tidyverse)
library(rootSolve)

# read in data ----

dat <- read.csv("data/actg.csv")
a <- dat$treat
y1 <- dat$cd4_20wk
y2 <- dat$cd8_20wk

# helper functions ----

#' m-estimator engine
#' @param ef an estimating function
#' @param init initial parameter values
mestimator <- function(ef, init, ...){
  require(rootSolve)
  require(tidyverse)

  # get estimated coefficients
  ef_colsums <- function(...) {
    return(colSums(ef(...)))
  }
  fit <- multiroot(ef_colsums, start = init,
                   rtol = 1e-6, atol = 1e-8, ctol = 1e-8, ...  )
  betahat <- fit$root

  # bread
  efhat <- ef(betahat, ...)
  n <- nrow(efhat)
  pd <- gradient(ef_colsums, betahat, ...)  # derivative of estimating function at betahat
  pd <- as.matrix(pd)
  bread <- -pd/n

  # meat1
  meat1 <- (t(efhat) %*% efhat)/n

  # sandwich
  sandwich1 <- (solve(bread) %*% meat1 %*% t(solve(bread)))/n
  se_sandwich1 <- sqrt(diag(sandwich1))

  results <- list(data.frame(estimates = betahat, var = se_sandwich1), covariance = sandwich1)
  return(results)
}

## estimating functions ----

ef1 <- function(theta, a_, y1_, y2_){
  psi <- c(theta[1], theta[2])
  mu1 <- theta[3]
  mu0 <- theta[4]
  om1 <- theta[5]
  om0 <- theta[6]

  #efs
  ef_psi1 <- mu1 - mu0 - theta[1]
  ef_psi2 <- om1 - om0 - theta[2]
  ef_mu1 <- a_ * (y1_ - mu1)
  ef_mu0 <- (1 - a_) * (y1_ - mu0)
  ef_om1 <- a_ * (y2_ - om1)
  ef_om0 <- (1 - a_) * (y2_ - om0)

  stacked <- cbind(ef_psi1, ef_psi2, ef_mu1, ef_mu0, ef_om1, ef_om0)
  return(stacked)
}


ef2 <- function(theta, covs, y_){
  beta <- theta
  x <- as.matrix(cbind(1, covs))
  ef <- (x) * as.vector(y_ - (x %*% beta))
  return(ef)
}


ef3 <- function(theta, covs, y_){
  beta <- (theta)
  x <- as.matrix(cbind(1, covs))
  ef <- (x) * as.vector(y_ - (x %*% beta))
  return(ef)
}

## helper function for conf.bands ----
#' a function to compute confidence bands
#' @param theta parameter estimates
#' @param covariance covariance matrix
#' @param alpha significance level
#' @param method method for producing bands (bonferroni or supt)
#' @param n_draws number of draws for manual simulation in supt method; if set to 0 defaults to mvtnorm approach
conf.bands <- function(theta, covariance, alpha = 0.05, method = "supt",  n_draws = 0){
  stderr <- sqrt(diag(covariance))

  # get new critical value
  k <- length(theta)
  if(method == "bonferroni"){
    crit_val <- qnorm(1 - alpha / (2 * k))
  }
  if(method == "supt" & n_draws > 0){
    mvn <- matrix(MASS::mvrnorm(n_draws, rep(0, k), covariance), ncol = k)
    scaled_mvn <- t(abs(t(mvn)/(stderr)))
    ts <- apply(scaled_mvn, 1, FUN = "max")
    crit_val <- quantile(ts, probs = (1 - alpha))
  }
  if(method == "supt" & n_draws == 0){
    crit_val <- mvtnorm::qmvnorm(1 - alpha,
                                 tail = "both.tails",
                                 # corr = covariance/(stderr %o% stderr), #equivalent to below
                                 corr = cov2cor(covariance),
                                 keepAttr = FALSE,
                                 ptol = .0001, maxiter = 1e4)$quantile
  }
  print(paste0("Critical value is ", crit_val))

  # compute bands
  bands <- matrix(c(theta - crit_val * stderr,
                    theta + crit_val * stderr),
                  byrow = F, ncol = 2, nrow = nrow(covariance))

  return((bands))
}


####################################
# example 1 ----

init_theta <- c(0, 0, .5, .5, .5, .5)
fullm <- mestimator(ef, init = init_theta, a_ = a, y1_ = y1, y2_ = y2)
covar <- fullm$covariance[1:2, 1:2]
psi <- fullm[[1]]$estimates[1:2]

## bonferroni ----
bonf.bands <- conf.bands(psi, covar, method = "bonferroni")


## sup-t with simulation----
supt.bands <- conf.bands(psi, covar, method = "supt", n_draws = 200000)

## sup-t with mvtnorm ----
supt.bands2 <- conf.bands(psi, covar, method = "supt", n_draws = 0)


####################################
# example 2  ----

covs <- dat[, c("treat", "male")] %>%
  mutate(treatmale = treat * male)

fullm <- mestimator(ef2, init = c(0, 0, 0, 0), covs = covs, y_ = y1)

# output covariance
covar <- fullm$covariance

# get point estimates
psi <- fullm[[1]]$estimates

## bonferroni ----
bonf.bands <- conf.bands(psi, covar, method = "bonferroni")

## sup-t with simulation----
supt.bands <- conf.bands(psi, covar, method = "supt", n_draws = 100000)

## sup-t with mvtnorm ----
supt.bands2 <- conf.bands(psi, covar, method = "supt", n_draws = 0)


####################################
# example 3 ----

covs <- dat[,c("treat", "cd4c_0wk", "cd4_rs1", "cd4_rs2", "cd4_rs3")] %>%
  mutate(treatcd4 = treat*(cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3))

fullm <- mestimator(ef2, init = c(0, 0, 0, 0, 0, 0, 0), covs = covs, y_ = y1)
covar <- fullm$covariance
covariance <- covar
psi <- fullm[[1]]$estimates

## get preds (50) ---
dp <- read.csv("data/actg_predict_g50.csv")
xp_raw <- dp[,c("treat", "cd4c_0wk", "cd4_rs1", "cd4_rs2", "cd4_rs3")] %>%
  mutate(treatcd4 = treat*(cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3))
xvals <- dp[,"cd4_0wk"]
xp <- as.matrix(cbind(1, xp_raw))
yhat <- t(as.vector(psi) %*% t(xp))
cov_p <- (xp) %*% covariance %*% t(xp)


## bonferroni ----
bonf.bands50 <- conf.bands(yhat, cov_p, method = "bonferroni")

## sup-t with simulation----
supt.bands50 <- conf.bands(yhat, cov_p, method = "supt", n_draws = 200000)

## sup-t with mvtnorm ----
supt.bands50_v2 <- conf.bands(yhat, cov_p, method = "supt", n_draws = 0)

## plot 50 ----
g50 <- data.frame(xvals = xvals, yhat = yhat, lclb = bonf.bands50[,1], uclb = bonf.bands50[,2],
                  lcls = supt.bands50[,1], ucls = supt.bands50[,2])


pl <- ggplot() +
  geom_ribbon(aes(x = xvals, ymin = lclb, ymax = uclb), data = g50,
              fill = "salmon", alpha = 0.5)  +
  geom_ribbon(aes(x = xvals, ymin = lcls, ymax = ucls), data = g50,
              fill = "blue", alpha = 0.5) +

  geom_line(aes(x = xvals, y = yhat), data = g50) +
  ylim(100, 800) +
  theme_bw()

pl
ggsave(pl, file = "confbands/output/pl.png", width = 7, height = 7, units = "cm")


## get preds (1000) ---
dp <- read.csv("data/actg_predict_g1000.csv")
xp_raw <- dp[,c("treat", "cd4c_0wk", "cd4_rs1", "cd4_rs2", "cd4_rs3")] %>%
  mutate(treatcd4 = treat*(cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3))
xvals <- dp[,"cd4_0wk"]
xp <- as.matrix(cbind(1, xp_raw))
yhat <- t(as.vector(psi) %*% t(xp))
cov_p <- (xp) %*% covariance %*% t(xp)


## bonferroni ----
bonf.bands1000 <- conf.bands(yhat, cov_p, method = "bonferroni")

## sup-t with simulation----
supt.bands1000 <- conf.bands(yhat, cov_p, method = "supt", n_draws = 20000)

## sup-t with mvtnorm ----
supt.bands1000_v2 <- conf.bands(yhat, cov_p, method = "supt", n_draws = 0)

### plot 1000 ----
g1000 <- data.frame(xvals = xvals, yhat = yhat, lclb = bonf.bands1000[,1], uclb = bonf.bands1000[,2],
                  lcls = supt.bands1000[,1], ucls = supt.bands1000[,2])


pl1000 <- ggplot() +
  geom_ribbon(aes(x = xvals, ymin = lclb, ymax = uclb), data = g1000,
              fill = "salmon", alpha = 0.5)  +
  geom_ribbon(aes(x = xvals, ymin = lcls, ymax = ucls), data = g1000,
              fill = "blue", alpha = 0.5) +

  geom_line(aes(x = xvals, y = yhat), data = g1000) +
  ylim(100, 800) +
  theme_bw()

pl1000
ggsave(pl1000, file = "confbands/output/pl1000.png", width = 7, height = 7, units = "cm")
