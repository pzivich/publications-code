###############################################################################################################
# A Simplified Strategy for Handling Incidental Clustering
#   Paul N Zivich, Jessie K Edwards, Bonnie E Shook-Sa, Stephen R Cole
#
# The following script recreates the application results presented in the manuscript.
###############################################################################################################

### set up environment ----
library(dplyr)
setwd("/path/...")
source("mestimation_engine.r")

### read in data ----
dat <- read.csv("cbihs.csv")
head(dat)

### set up vectors ----
y <- as.matrix(with(dat, ifelse(is.na(hiv_test), -999, hiv_test)))
r <- as.matrix(with(dat, ifelse(y < 0, 0, 1)))
g <- as.vector(dat$venue_id)
W <- model.matrix(~ female + C(educ) + C(country) + age +  sexual_debut + ever_test, data = dat)


### helper functions ----

getci <- function(myest, mese){
    lcl <- unname(myest - 1.96 * mese)
    ucl <- unname(myest + 1.96 * mese)
    return(c(lcl, ucl))
}

aggregate_clusters <- function(est_funcs, group){
	id_vector <- as.vector(group)
	unique_ids <- unique(id_vector)
	cluster_matrix <- t(outer(unique_ids, id_vector, "=="))
	return(est_funcs %*% cluster_matrix)
}


############################################
# naive prevalence ----

## no clustering ----

# estimating function
ef_naive <- function(theta){
    return(t(as.matrix(r * (y - theta))))
}

# set starting values
init_theta <- c(.5)
# compute point estimate and se
naive <- mestimator(ef_naive, init = init_theta)[[1]]
# get 95% CI
naive_ci <- getci(naive[1], naive[2])
paste0(naive[1], " (95% CI:", naive_ci[[1]], ", ", naive_ci[[2]], ")")

## naive with clustering ----

# estimating functions
ef_ncluster <- function(theta, cluster){
    return(aggregate_clusters(ef_naive(theta), cluster))
}

# compute point estimate and se
naive_cluster <- mestimator(ef = ef_ncluster, init = init_theta, cluster = g)[[1]]
clust_ci <- getci(naive_cluster[1], naive_cluster[2])
paste0(naive_cluster[1], " (95% CI:", clust_ci[[1]], ", ", clust_ci[[2]], ")")


############################################
# accounting for missing data ----

############################################
## AIPW ----

# estimating function
ef_aipw <- function(theta){
	mu <- theta[1]
	alpha <- as.matrix(theta[2 : (ncol(W) + 1)])
	beta <- as.matrix(theta[(ncol(W) + 2) : (ncol(W) * 2 + 1)])

	# missingness model 
	pi <- plogis(W %*% alpha)
	ef_pi <- t(W * as.vector((r - pi)))
	ipmw <- r / pi

	# outcome model
	y_hat <- plogis(W %*% beta)
	ef_out <- t(W * as.vector(ipmw * (y - y_hat)))

	# estimator
	ef_mu  <- t(y_hat - mu)

	# stack
	return(rbind(ef_mu , ef_pi, ef_out))
}

### aipw with no no clustering ----

# starting values
theta_init <- c(.04, rep(0, ncol(W)), rep(0, ncol(W)))

# compute point estimate and se
aipw <- mestimator(ef_aipw, init = theta_init)[[1]]
aipw_ci <- getci(aipw[1,1], aipw[1,2])
paste0(aipw[1,1], " (95% CI:", aipw_ci[[1]], ", ", aipw_ci[[2]], ")")

### aipw with clustering ----

ef_aipwcluster <- function(theta, cluster){
    return(aggregate_clusters(ef_aipw(theta), cluster))
}

aipw_cluster <- mestimator(ef = ef_aipwcluster, init = theta_init, cluster = g)[[1]]
aipw_ci2 <- getci(aipw_cluster[1,1], aipw_cluster[1,2])
paste0(aipw_cluster[1,1], " (95% CI:", aipw_ci2[[1]], ", ", aipw_ci2[[2]], ")")

############################################
## IPW ----

ef_ipw <- function(theta){
	mu <- theta[1]
	alpha <- as.matrix(theta[2 : (ncol(W) + 1)])

	# missingness model 
	pi <- plogis(W %*% alpha)
	ef_pi <- t(W * as.vector((r - pi)))
	ipmw <- r / pi

	# estimator
	ef_mu  <- t(ipmw * (y - mu))

	# stack
	return(rbind(ef_mu, ef_pi))
}


### IPW with no clustering ----
theta_init <- c(.5, rep(0, ncol(W)))

ipw <- mestimator(ef_ipw, init = theta_init)[[1]]
ipw_ci <- getci(ipw[1,1], ipw[1,2])
paste0(ipw[1,1], " (95% CI:", ipw_ci[[1]], ", ", ipw_ci[[2]], ")")

### ipw with clustering ----

ef_ipwcluster <- function(theta, cluster){
  return(aggregate_clusters(ef_ipw(theta), cluster))
}

ipw_cluster <- mestimator(ef = ef_ipwcluster, init = theta_init, cluster = g)[[1]]
ipw_ci2 <- getci(ipw_cluster[1,1], ipw_cluster[1,2])
paste0(ipw_cluster[1,1], " (95% CI:", ipw_ci2[[1]], ", ", ipw_ci2[[2]], ")")

############################################
## g-comp ----

ef_gcomp <- function(theta){
	mu <- theta[1]
	beta <- as.matrix(theta[2 : (ncol(W) + 1)])

	# outcome model
	y_hat <- plogis(W %*% beta)
	ef_out <- t(W * as.vector(r * (y - y_hat)))
	
	# estimator
	ef_mu  <- t(y_hat - mu)

	# stack
	return(rbind(ef_mu, ef_out))
}

# starting values

theta_init <- c(.04, rep(0, ncol(W)))

### gcomp with no clustering ----

gcomp <- mestimator(ef_gcomp, init = theta_init)[[1]]
gcomp_ci <- getci(gcomp[1,1], gcomp[1,2])
paste0(gcomp[1,1], " (95% CI:", gcomp_ci[[1]], ", ", gcomp_ci[[2]], ")")

### gcomp with clustering ----

ef_gcompcluster <- function(theta, cluster){
    return(aggregate_clusters(ef_gcomp(theta), cluster))
}

gcomp_cluster <- mestimator(ef = ef_gcompcluster, init = theta_init, cluster = g)[[1]]
gcomp_ci2 <- getci(gcomp_cluster[1,1], gcomp_cluster[1,2])
paste0(gcomp_cluster[1,1], " (95% CI:", gcomp_ci2[[1]], ", ", gcomp_ci2[[2]], ")")


############################################
# beautify results ----

results <- data.frame(labs = c("naive", "naive_cluster", 
                               "aipw", "aipw_cluster", 
                               "ipw", "ipw_cluster", 
                               "gcomp", "gcomp_cluster"), 
                      ests = c(naive[1,1], naive_cluster[1,1], 
                               aipw[1,1], aipw_cluster[1,1], 
                               ipw[1,1], ipw_cluster[1,1], 
                               gcomp[1,1], gcomp_cluster[1,1]), 
                      se = c(naive[1,2], naive_cluster[1,2], 
                             aipw[1,2], aipw_cluster[1,2], 
                             ipw[1,2], ipw_cluster[1,2], 
                             gcomp[1,2], gcomp_cluster[1,2]))  
cis <- data.frame(rbind(naive_ci, clust_ci, 
                              aipw_ci, aipw_ci2,
                              ipw_ci, ipw_ci2, 
                              gcomp_ci, gcomp_ci2)) %>% 
  rename(lcl = X1, ucl = X2)

res <- bind_cols(results, cis)
res
