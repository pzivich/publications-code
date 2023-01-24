#####################################################################################################################
# Introducing Proximal Causal Inference for Epidemiologists
#
#   R code for the described simulations
#
# (2022/11/15)
#####################################################################################################################

library(dplyr)
library(rootSolve)
library(tidyverse)

require(rootSolve)
require(tidyverse)

set.seed(7777777)
k <- 4000
n <- 500
true <- -1


# generate data ----
gen <- function(seed, scenario){
  set.seed(seed)
  x <-  rnorm(n)                               # Generate age
  u <- x +rnorm(n)                             # Generate immune function
  w <- u + rnorm(n)                            # Generate treatment proxy
  z <- u + rnorm(n)                            # Generate outcome proxy
  
  # Generate action
  a1 <- rbinom(n, 1, plogis(z + x))            # Action in scenario 1
  a2 <- rbinom(n, 1, plogis(z + u + x))        # Action in scenario 2
  a3 <- rbinom(n, 1, plogis(z + u + x))        # Action in scenario 3
  
  # Generate potential outcome under no action
  y01 <- w + x + rnorm(n)                      # Potential outcome in scenario 1
  y02 <- w + u + x + rnorm(n)                  # Potential outcome in scenario 2
  y03 <- w + u + x + z + rnorm(n)              # Potential outcome in scenario 3
  y11 <- y01 + true                            # Potential outcome in scenario 1
  y12 <- y02 + true                            # Potential outcome in scenario 2
  y13 <- y03 + true                            # Potential outcome in scenario 3

  # Causal consistency
  y1 <- a1*y11 + (1-a1)*y01                    # Outcome in scenario 1
  y2 <- a2*y12 + (1-a2)*y02                    # Outcome in scenario 2
  y3 <- a3*y13 + (1-a3)*y03                    # Outcome in scenario 3

  # Generating output data frames
  sc1 <- data.frame(a=a1, y=y1, x, w, z)       # Data scenario 1
  sc2 <- data.frame(a=a2, y=y2, x, w, z)       # Data scenario 2
  sc3 <- data.frame(a=a3, y=y3, x, w, z)       # Data scenario 3
  if(scenario == 1) ret.dat <- sc1
  if(scenario == 2) ret.dat <- sc2
  if(scenario == 3) ret.dat <- sc3
  return(ret.dat)
}

#' standard g-computation

gcomp <- function(dat, covs){
  mod <- lm(as.formula(paste0("y ~ a +", paste(covs, collapse = "+"))),
            data = dat)
  py0 <- predict(mod, newdata = dat %>% mutate(a = 0), type = "response")
  py1 <- predict(mod, newdata = dat %>% mutate(a = 1), type = "response")
  diff <- mean(py1 - py0)
  return(diff)
}

# data analysis functions ----

#' simple g-computation in linear setting

gcomp_lm <- function(dat, covs){
  mod <- summary(lm(as.formula(paste0("y ~ a +", paste(covs, collapse = "+"))), 
            data = dat))
  diff <- coef(mod)[2, 1]
  se <- coef(mod)[2, 2]
  return(c(diff, se))
}

#' Proximal causal learning with M-estimation
#' 
pcl_m2 <- function(theta, mydata, out = "ef"){
  # read in variables
  a <- mydata$a
  x <- mydata$x
  w <- mydata$w
  z <- mydata$z
  y <- mydata$y
  # model for what
  what <- theta[1] + theta[2] * a + theta[3] * x + theta[4] * z 

  # matrix of covars in model for what
  wmat <- as.matrix(cbind(rep(1, nrow(mydata)), a, x, z), ncol = 4, 
                    nrow = nrow(mydata))

  # estimating function for what
  ef1 <- (t(wmat)  %*% (w - what))            # gives px1 vector (summed over units)
  ef1_b <- wmat * (w - what)                  # gives nxp matrix (rows are units)
  
  # model for yhat
  yhat <- theta[5] + theta[6]*a + theta[7]*x + theta[8] * what  

  # matrix of covars in model for yhat
  ymat <- as.matrix(cbind(rep(1, nrow(mydata)), a, x, what), ncol = 4, 
                    nrow = nrow(mydata))

  #estimating function for yhat
  ef2 <- t(ymat) %*% (y - yhat)               # px1 vector, summed over units
  ef2_b <- (ymat) * (y - yhat)                # nxp matrix (rows are units)
  f <- c(ef1, ef2)                            # p-dimensional vector
  if(out == "full") return(cbind(ef1_b, ef2_b)) else return(f) # full used in var calculation only
}

# call m-estimator for PCL
mpcl_mest <- function(dat){
  theta_start <- rep(.05, 8)
  results <- mestimator(pcl_m2, init = theta_start, dat, mydata = dat)
  return(results)
}

##' M-estimation engine
mestimator <- function(ef, init, dat, ...){
  # get estimated coefficients
  fit <- multiroot(ef, start = init, out = "ef",
                   rtol = 1e-9, atol = 1e-12, ctol = 1e-12, ...  )
  betahat <- fit$root
  n <- nrow(dat)

  # bread
  # derivative of estimating function at betahat
  pd <- gradient(ef, betahat, out = "ef", ...) 
  pd <- as.matrix(pd)
  bread <- -pd/n
  se_mod <- sqrt(abs(diag(-solve(pd))))
  
  # meat1
  efhat <- ef(betahat, out = "full", ...)
  meat <- list()
  meat1 <- (t(efhat) %*% efhat)/n
  
  # sandwich
  sandwich1 <- (solve(bread) %*% meat1 %*% t(solve(bread)))/n
  se_sandwich1 <- sqrt(diag(sandwich1))
  
  results <- data.frame(betahat, se_sandwich1, se_mod)
  return(results)
}


# run sims ----

wrap <- function(i, sc){
  sdat <- gen(i, sc)
  gmin <- gcomp_lm(sdat, covs = c("x", "w"))
  gstan <- gcomp_lm(sdat, covs = c("x", "w", "z"))
  rmpcl <- unname(mpcl_mest(sdat)[6, c(1:2)])
  return(unlist(c(gmin, gstan, rmpcl)))
}

results1 <- matrix(NA, nrow = k, ncol = 6)
results2 <- matrix(NA, nrow = k, ncol = 6)
results3 <- matrix(NA, nrow = k, ncol = 6)
for(i in 1:(k)){
  results1[i,] <- wrap(i, 1)
  results2[i,] <- wrap(i, 2)
  results3[i,] <- wrap(i, 3)
}


# summarize sims ----

allres <- data.frame(rbind(results1, results2, results3), 
                     scenario = rep(c(1, 2, 3), each = k), 
                     sim = rep(c(1:k), 3))
colnames(allres) <- c("gmin_est", "gmin_se", "gstan_est", 
                      "gstan_se", "pcl_est", "pcl_se", "sc", "sim")

ests <- allres[,c(1, 3, 5, 7)]
colnames(ests) <- c("gmin", "gstan", "pcl", "sc")

resultstab <- ests %>% 
  group_by(sc) %>% 
  summarize_all(list(bias = function(x){mean(x) - true}, 
                     ese = sd)) 

ses <- allres[,c(2, 4, 6, 7)]
colnames(ses) <- c("gmin", "gstan", "pcl", "sc")

asetab <- ses %>% 
  group_by(sc) %>% 
  summarize_all(list(ase = mean))

covtab <- allres %>% 
  pivot_longer(cols = -c(sc, sim), names_to = c("method", "metric"), 
               values_to = "ests", names_sep = "_") %>% 
  pivot_wider(id_cols = c(sim, sc, method), names_from = metric, values_from = ests) %>% 
  mutate(cov = ifelse((est - 1.96*se)<true & true <(est + 1.96*se), 1, 0)) %>% 
  group_by(sc, method) %>% 
  summarize(coverage = mean(cov))

restab <- merge(resultstab, asetab, by = "sc") %>% 
  pivot_longer(cols = -c(sc), names_to = c("method", "metric"), 
               values_to = "value", 
               names_sep = "_") %>% 
  pivot_wider(id_cols = c(sc, method), names_from = metric, values_from = "value") %>% 
  mutate(ser = ase/ese, 
         rmse = sqrt(bias^2+ese^2))

restab2 <- merge(restab, covtab, by = c("sc", "method")) 

restab2

# Results: 2022/11/15
# sc method  bias   ese   ase  ser  rmse coverage
# 1    pcl -0.002 0.119 0.119 0.99 0.119  0.96
# 1   gmin -0.003 0.109 0.109 1.00 0.109  0.95
# 1  gstan -0.002 0.119 0.118 1.00 0.119  0.95
# 2    pcl -0.013 0.208 0.209 1.01 0.208  0.95
# 2   gmin  0.650 0.140 0.140 1.02 0.664  0.00
# 2  gstan  0.258 0.148 0.148 1.00 0.297  0.60
# 3    pcl -0.911 0.560 0.555 0.99 1.069  0.67
# 3   gmin  1.957 0.208 0.216 1.04 1.968  0.00
# 3  gstan  0.262 0.149 0.147 0.99 0.301  0.57

# END OF SCRIPT
