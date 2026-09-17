###############################################################################################################
# This script contains the M-estimator engine
###############################################################################################################

#' m-estimator engine
#' @param ef an estimating function
#' @param init initial parameter values

mestimator <- function(ef, init, ...){
  
  require(rootSolve)
  require(tidyverse)
  
  # get estimated coefficients
  ef_rowsums <- function(...) {
    return(rowSums(ef(...)))
  }
  
  fit <- multiroot(ef_rowsums, start = init,
                   rtol = 1e-6, atol = 1e-8, ctol = 1e-8, ...  )
  betahat <- fit$root
  
  # bread
  efhat <- ef(betahat, ...)
  n <- ncol(efhat)
  # derivative of estimating function at betahat
  pd <- gradient(ef_rowsums, betahat, ...) 
  pd <- as.matrix(pd)
  bread <- -pd/n
  
  # meat1
  meat1 <- (efhat %*% t(efhat))/n
  
  # sandwich
  sandwich1 <- ((solve(bread)) %*% meat1 %*% t(solve(bread)))/n
  se_sandwich1 <- sqrt(diag(sandwich1))
  
  results <- list(data.frame(estimates = betahat, se = se_sandwich1), covariance = sandwich1)
  return(results)
  
}


#' m-estimator engine with stacked cols instead of rows
#' @param ef an estimating function
#' @param init initial parameter values

# mestimator <- function(ef, init, ...){
  
#   require(rootSolve)
#   require(tidyverse)
  
#   # get estimated coefficients
#   ef_colsums <- function(x, ...) {
#     return(colSums(ef(x, ...)))
#   }
# #   fit <- optim(init, function(x) sum(colSums(ef(x, ...))^2), 
# #              method = "Nelder-Mead", control = list(maxit = 10000))
# #   betahat <- fit$par
#   fit <- multiroot(ef_colsums, start = init,
#                    rtol = 1e-6, atol = 1e-6, ctol = 1e-6, ...  )
#   betahat <- fit$root
  
#   # bread
#   efhat <- ef(betahat, ...)
#   n <- nrow(efhat)
#   # derivative of estimating function at betahat
#   pd <- gradient(ef_colsums, betahat, ...) 
#   pd <- as.matrix(pd)
#   bread <- -pd/n
  
#   # meat1
#   meat1 <- (t(efhat) %*% (efhat))/n
  
#   # sandwich
#   sandwich1 <- ((solve(bread)) %*% meat1 %*% t(solve(bread)))/n
#   se_sandwich1 <- sqrt(diag(sandwich1))
  
#   results <- list(data.frame(estimates = betahat, se = se_sandwich1), 
#   							 covariance = sandwich1, 
# 							 meat = meat1, 
# 							 bread = bread)
#   return(results)
  
# }


# old m-estimator function to test

# mestimator <- function(ef, init){
  
#   require(rootSolve)
#   require(tidyverse)
  
#   # get estimated coefficients
#   ef_colsums <- function(x) {
#     return(colSums(ef(x)))
#   }
# #   fit <- multiroot(ef_colsums, start = init, maxiter = 10000,
# #                    rtol = 1e-6, atol = 1e-8, ctol = 1e-8  )
# #   betahat <- fit$root
#   fit <- optim(init, function(x) sum(colSums(ef(x))^2), 
#               method = "Nelder-Mead", control = list(maxit = 100000))
#   betahat <- fit$par
#   # bread
#   efhat <- ef(betahat)
#   n <- nrow(efhat)
#   # derivative of estimating function at betahat
#   pd <- gradient(function(x) colSums(ef(x)), betahat) 
#   pd <- as.matrix(pd)
#   bread <- -pd/n
  
#   # meat1
#   meat1 <- (t(efhat) %*% efhat)/n
  
#   # sandwich
#   sandwich1 <- (solve(bread) %*% meat1 %*% t(solve(bread)))/n
#   se_sandwich1 <- sqrt(diag(sandwich1))
  
#   results <- list(data.frame(estimates = betahat, se = se_sandwich1), 
#   							 covariance = sandwich1)
#   return(results)
  
# }
