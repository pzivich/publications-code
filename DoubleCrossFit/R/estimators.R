# Newey and Robins DCDR

# Function that does DCDR using input data and super learner with a given seed
DCDR_Single <- function(data, exposure, outcome, covarsT, covarsO, learners, control){

  #Split sample
  splits <- sample(rep(1:3, diff(floor(nrow(data) * c(0, 1/3, 2/3, 3/3)))))
  
  data <- data %>% mutate(s=splits)
  
  #Create nested dataset
  dat_nested <- data %>%
    group_by(s) %>%
    nest()
  
  #P-score model
  
  pi_fitter <- function(df){
    SuperLearner(Y=as.matrix(df[, exposure]), X=df[, covarsT], family=binomial(), SL.library=learners, cvControl=control)
  }
  
  dat_nested <- dat_nested %>% 
    mutate(pi_fit=map(data, pi_fitter))
  
  #Calc p-scores using each split
  data <- data %>%
    mutate(pi1 = predict(dat_nested$pi_fit[[1]], newdata = data[, covarsT])$pred,
           pi2 = predict(dat_nested$pi_fit[[2]], newdata = data[, covarsT])$pred,
           pi3 = predict(dat_nested$pi_fit[[3]], newdata = data[, covarsT])$pred)
  
  #Outcome model
  mu_fitter <- function(df){
    SuperLearner(Y=as.matrix(df[, outcome]), X=df[, c(exposure, covarsO)], family=binomial(), SL.library=learners, cvControl=control)
  }
  
  dat_nested <- dat_nested %>%
    mutate(mu_fit=map(data, mu_fitter))
  
  #Calc mu using each split
  dat1 <- data %>%
    mutate(statin=1)
  
  dat0 <- data %>%
    mutate(statin=0)
  
  data <- data %>%
    mutate(mu1_1 = predict(dat_nested$mu_fit[[1]], newdata = dat1[, c(exposure, covarsO)])$pred,
           mu1_2 = predict(dat_nested$mu_fit[[2]], newdata = dat1[, c(exposure, covarsO)])$pred,
           mu1_3 = predict(dat_nested$mu_fit[[3]], newdata = dat1[, c(exposure, covarsO)])$pred,
           mu0_1 = predict(dat_nested$mu_fit[[1]], newdata = dat0[, c(exposure, covarsO)])$pred,
           mu0_2 = predict(dat_nested$mu_fit[[2]], newdata = dat0[, c(exposure, covarsO)])$pred,
           mu0_3 = predict(dat_nested$mu_fit[[3]], newdata = dat0[, c(exposure, covarsO)])$pred,
           pi1 = ifelse(pi1<0.025, 0.025, ifelse(pi1>.975,.975, pi1)),
           pi2 = ifelse(pi2<0.025, 0.025, ifelse(pi2>.975,.975, pi2)),
           pi3 = ifelse(pi3<0.025, 0.025, ifelse(pi3>.975,.975, pi3)),
           ipw1 = statin/pi1+(1-statin)/(1-pi1),
           ipw2 = statin/pi2+(1-statin)/(1-pi2),
           ipw3 = statin/pi3+(1-statin)/(1-pi3)
           )
  
  data <- data %>%
    mutate(a1_1 = statin*ipw2*(Y-mu1_3)+mu1_3,
           a1_2 = statin*ipw3*(Y-mu1_1)+mu1_1,
           a1_3 = statin*ipw1*(Y-mu1_2)+mu1_2,
           a0_1 = (1-statin)*ipw2*(Y-mu0_3)+mu0_3,
           a0_2 = (1-statin)*ipw3*(Y-mu0_1)+mu0_1,
           a0_3 = (1-statin)*ipw1*(Y-mu0_2)+mu0_2
           )
  
  #Get split-specific estimates
  r1_1 = mean(filter(data, s==1)$a1_1)
  r1_2 = mean(filter(data, s==2)$a1_2)
  r1_3 = mean(filter(data, s==3)$a1_3)
  r0_1 = mean(filter(data, s==1)$a0_1)
  r0_2 = mean(filter(data, s==2)$a0_2)
  r0_3 = mean(filter(data, s==3)$a0_3)
  rd_1 = r1_1 - r0_1
  rd_2 = r1_2 - r0_2
  rd_3 = r1_3 - r0_3
  
  #Get influence functions
  data <- data %>%
    mutate(
      if1_1 = a1_1 - r1_1,
      if1_2 = a1_2 - r1_2,
      if1_3 = a1_3 - r1_3,
      if0_1 = a0_1 - r0_1,
      if0_2 = a0_2 - r0_2,
      if0_3 = a0_3 - r0_3,
      ifd_1 = a1_1 - a0_1 - rd_1,
      ifd_2 = a1_2 - a0_2 - rd_2,
      ifd_3 = a1_3 - a0_3 - rd_3
    )
  
  #Results
  r1 <- (r1_1 + r1_2 + r1_3) / 3
  r0 <- (r0_1 + r0_2 + r0_3) / 3
  rd <- (rd_1 + rd_2 + rd_3) / 3
  v1 <- (var(filter(data,s==1)$if1_1) + var(filter(data,s==2)$if1_2) + var(filter(data,s==3)$if1_3)) / (3*nrow(data))
  v0 <- (var(filter(data,s==1)$if0_1) + var(filter(data,s==2)$if0_2) + var(filter(data,s==3)$if0_3)) / (3*nrow(data))
  vd <- (var(filter(data,s==1)$ifd_1) + var(filter(data,s==2)$ifd_2) + var(filter(data,s==3)$ifd_3)) / (3*nrow(data))

  results <-tibble(r1, r0, rd, v1, v0, vd)
  return(results)
}

# Function that does DCDR over multiple sample splits and combines results
DCDR_Multiple <-function(data, exposure, outcome, covarsT, covarsO, learners, control, num_cf){
  
  #Initialize results
  runs <- tibble(r1=double(), r0=double(), rd=double(), v1=double(), v0=double(), vd=double())
  
  #Run on num_cf splits
  for(cf in 1:num_cf){
    runs <- bind_rows(runs, DCDR_Single(data, exposure, outcome, covarsT, covarsO, learners, control))
  }
  #Medians of splits
  medians <- sapply(runs, median)
  
  #Corrected variance terms
  runs <- runs %>%
    mutate(mv1 = v1 + (r1-medians[1])^2,
           mv0 = v0 + (r0-medians[2])^2,
           mvd = vd + (rd-medians[3])^2)
  
  results <- sapply(runs, median)
  names(results) <- c("r1","r0","rd","v1","v0","vd","mv1","mv0","mvd")  
  return(bind_rows(results))
    
}

# Function that does DCTMLE using input data and super learner with a given seed
DCTMLE_Single <- function(data, exposure, outcome, covarsT, covarsO, learners, control){
  
  #Split sample
  splits <- sample(rep(1:3, diff(floor(nrow(data) * c(0, 1/3, 2/3, 3/3)))))
  
  data <- data %>% mutate(s=splits)
  
  #Create nested dataset
  dat_nested <- data %>%
    group_by(s) %>%
    nest()
  
  #P-score model
  
  pi_fitter <- function(df){
    SuperLearner(Y=as.matrix(df[, exposure]), X=df[, covarsT], family=binomial(), SL.library=learners, cvControl=control)
  }
  
  dat_nested <- dat_nested %>% 
    mutate(pi_fit=map(data, pi_fitter))
  
  #Calc p-scores using each split
  data <- data %>%
    mutate(pi1 = predict(dat_nested$pi_fit[[1]], newdata = data[, covarsT])$pred,
           pi2 = predict(dat_nested$pi_fit[[2]], newdata = data[, covarsT])$pred,
           pi3 = predict(dat_nested$pi_fit[[3]], newdata = data[, covarsT])$pred)
  
  #Outcome model
  mu_fitter <- function(df){
    SuperLearner(Y=as.matrix(df[, outcome]), X=df[, c(exposure, covarsO)], family=binomial(), SL.library=learners, cvControl=control)
  }
  
  dat_nested <- dat_nested %>%
    mutate(mu_fit=map(data, mu_fitter))
  
  #Calc mu using each split
  dat1 <- data %>%
    mutate(statin=1)
  
  dat0 <- data %>%
    mutate(statin=0)
  
  data <- data %>%
    mutate(mu_1 = predict(dat_nested$mu_fit[[1]], newdata = data[, c(exposure, covarsO)])$pred,
           mu_2 = predict(dat_nested$mu_fit[[2]], newdata = data[, c(exposure, covarsO)])$pred,
           mu_3 = predict(dat_nested$mu_fit[[3]], newdata = data[, c(exposure, covarsO)])$pred,
           mu1_1 = predict(dat_nested$mu_fit[[1]], newdata = dat1[, c(exposure, covarsO)])$pred,
           mu1_2 = predict(dat_nested$mu_fit[[2]], newdata = dat1[, c(exposure, covarsO)])$pred,
           mu1_3 = predict(dat_nested$mu_fit[[3]], newdata = dat1[, c(exposure, covarsO)])$pred,
           mu0_1 = predict(dat_nested$mu_fit[[1]], newdata = dat0[, c(exposure, covarsO)])$pred,
           mu0_2 = predict(dat_nested$mu_fit[[2]], newdata = dat0[, c(exposure, covarsO)])$pred,
           mu0_3 = predict(dat_nested$mu_fit[[3]], newdata = dat0[, c(exposure, covarsO)])$pred,
           pi1 = ifelse(pi1<0.025, 0.025, ifelse(pi1>.975,.975, pi1)),
           pi2 = ifelse(pi2<0.025, 0.025, ifelse(pi2>.975,.975, pi2)),
           pi3 = ifelse(pi3<0.025, 0.025, ifelse(pi3>.975,.975, pi3)),
           H1_1 = statin/pi1,
           H1_2 = statin/pi2,
           H1_3 = statin/pi3,
           H0_1 = (1-statin)/(1-pi1),
           H0_2 = (1-statin)/(1-pi2),
           H0_3 = (1-statin)/(1-pi3)
    )
  
  data <- data %>%
    mutate(mu_1 = ifelse(mu_1 == 0, 1e-17, ifelse(mu_1==1, 1-1e-17, mu_1)),
           mu_2 = ifelse(mu_2 == 0, 1e-17, ifelse(mu_2==1, 1-1e-17, mu_2)),
           mu_3 = ifelse(mu_3 == 0, 1e-17, ifelse(mu_3==1, 1-1e-17, mu_3)),
           mu1_1 = ifelse(mu1_1 == 0, 1e-17, ifelse(mu1_1==1, 1-1e-17, mu1_1)),
           mu1_2 = ifelse(mu1_2 == 0, 1e-17, ifelse(mu1_2==1, 1-1e-17, mu1_2)),
           mu1_3 = ifelse(mu1_3 == 0, 1e-17, ifelse(mu1_3==1, 1-1e-17, mu1_3)),
           mu0_1 = ifelse(mu0_1 == 0, 1e-17, ifelse(mu0_1==1, 1-1e-17, mu0_1)),
           mu0_2 = ifelse(mu0_2 == 0, 1e-17, ifelse(mu0_2==1, 1-1e-17, mu0_2)),
           mu0_3 = ifelse(mu0_3 == 0, 1e-17, ifelse(mu0_3==1, 1-1e-17, mu0_3)))

  
  epsilon_1 <- coef(glm(Y ~ -1 + H0_2 + H1_2 + offset(qlogis(mu_3)), data = data %>% filter(s==1), family = binomial))
  epsilon_2 <- coef(glm(Y ~ -1 + H0_3 + H1_3 + offset(qlogis(mu_1)), data = data %>% filter(s==2), family = binomial))
  epsilon_3 <- coef(glm(Y ~ -1 + H0_1 + H1_1 + offset(qlogis(mu_2)), data = data %>% filter(s==3), family = binomial))
  
  
  data = data %>% mutate(
    mu0_1_1 = plogis(qlogis(mu0_3) + epsilon_1[1] / (1 - pi2)),
    mu0_1_2 = plogis(qlogis(mu0_1) + epsilon_2[1] / (1 - pi3)),
    mu0_1_3 = plogis(qlogis(mu0_2) + epsilon_3[1] / (1 - pi1)),
    mu1_1_1 = plogis(qlogis(mu1_3) + epsilon_1[2] / (pi2)),
    mu1_1_2 = plogis(qlogis(mu1_1) + epsilon_2[2] / (pi3)),
    mu1_1_3 = plogis(qlogis(mu1_2) + epsilon_3[2] / (pi1)))
  
  #Get split-specific estimates
  r1_1 = mean(filter(data, s==1)$mu1_1_1)
  r1_2 = mean(filter(data, s==2)$mu1_1_2)
  r1_3 = mean(filter(data, s==3)$mu1_1_3)
  r0_1 = mean(filter(data, s==1)$mu0_1_1)
  r0_2 = mean(filter(data, s==2)$mu0_1_2)
  r0_3 = mean(filter(data, s==3)$mu0_1_3)
  rd_1 = r1_1 - r0_1
  rd_2 = r1_2 - r0_2
  rd_3 = r1_3 - r0_3

  #Get influence functions
  data <- data %>%
    mutate(
      if1_1 = statin/pi2*(Y-mu1_1_1) + mu1_1_1 - r1_1,
      if1_2 = statin/pi3*(Y-mu1_1_2) + mu1_1_2 - r1_2,
      if1_3 = statin/pi1*(Y-mu1_1_3) + mu1_1_3 - r1_3,
      if0_1 = (1-statin)/(1-pi2)*(Y-mu0_1_1) + mu0_1_1 - r0_1,
      if0_2 = (1-statin)/(1-pi3)*(Y-mu0_1_2) + mu0_1_2 - r0_2,
      if0_3 = (1-statin)/(1-pi1)*(Y-mu0_1_3) + mu0_1_3 - r0_3,
      ifd_1 = if1_1 - if0_1,
      ifd_2 = if1_2 - if0_2,
      ifd_3 = if1_3 - if0_3
    )
  
  #Results
  r1 <- (r1_1 + r1_2 + r1_3) / 3
  r0 <- (r0_1 + r0_2 + r0_3) / 3
  rd <- (rd_1 + rd_2 + rd_3) / 3
  v1 <- (var(filter(data,s==1)$if1_1) + var(filter(data,s==2)$if1_2) + var(filter(data,s==3)$if1_3)) / (3*nrow(data))
  v0 <- (var(filter(data,s==1)$if0_1) + var(filter(data,s==2)$if0_2) + var(filter(data,s==3)$if0_3)) / (3*nrow(data))
  vd <- (var(filter(data,s==1)$ifd_1) + var(filter(data,s==2)$ifd_2) + var(filter(data,s==3)$ifd_3)) / (3*nrow(data))
  
  
  results <-tibble(r1, r0, rd, v1, v0, vd)
  return(results)
}

# Function that does DCTMLE over multiple sample splits and combines results
DCTMLE_Multiple <-function(data, exposure, outcome, covarsT, covarsO, learners, control, num_cf){
  
  #Initialize results
  runs <- tibble(r1=double(), r0=double(), rd=double(), v1=double(), v0=double(), vd=double())
  
  #Run on num_cf splits
  for(cf in 1:num_cf){
    runs <- bind_rows(runs, DCTMLE_Single(data, exposure, outcome, covarsT, covarsO, learners, control))
  }
  #Medians of splits
  medians <- sapply(runs, median)
  
  #Corrected variance terms
  runs <- runs %>%
    mutate(mv1 = v1 + (r1-medians[1])^2,
           mv0 = v0 + (r0-medians[2])^2,
           mvd = vd + (rd-medians[3])^2)
  
  results <- sapply(runs, median)
  names(results) <- c("r1","r0","rd","v1","v0","vd","mv1","mv0","mvd")  
  return(bind_rows(results))
  
}

# IPW Estimator
IPW <- function(data, exposure, outcome, covars, learners, control){

  pi_fit <- SuperLearner(Y=as.matrix(data[, exposure]), X=data[, covars], family=binomial(), SL.library=learners, cvControl = control)
  
  data <- data %>%
    mutate(pi = predict(pi_fit, newdata = data[,covars])$pred)
    
  data <- data %>%
    mutate(pi = ifelse(pi<.025, .025, ifelse(pi>.975, .975, pi)),
           ipw = statin/pi+(1-statin)/(1-pi))
  
  data <- data %>% 
    mutate(ip1 = statin*Y*ipw,
           ip0 = (1-statin)*Y*ipw)
  
  r1 <- mean(data$ip1)
  r0 <- mean(data$ip0)
  rd <- r1 - r0
  
  data <- data %>%
    mutate(if1 = ip1 - r1,
           if0 = ip0 - r0,
           ifd = ip1 - ip0 - rd)
  
  v1 <- var(data$if1)/nrow(data)
  v0 <- var(data$if0)/nrow(data)
  vd <- var(data$ifd)/nrow(data)
  
  results <- c(r1, r0, rd, v1, v0, vd)
  names(results) <- c("r1","r0","rd","v1","v0","vd")

  return(bind_rows(results))
}



# Gcomp Estimator
gcomp <- function(data, exposure, outcome, covars, learners, control){
  
  mu_fit <- SuperLearner(Y=as.matrix(data[, outcome]), X=data[, c(exposure, covars)], family=binomial(), SL.library=learners, cvControl=control)
  
  dat1 <- data %>%
    mutate(statin=1)
  
  dat0 <- data %>%
    mutate(statin=0)
  
  data <- data %>%
    mutate(mu1 = predict(mu_fit, newdata = dat1[, c(exposure, covars)])$pred,
           mu0 = predict(mu_fit, newdata = dat0[, c(exposure, covars)])$pred
    )
  
  r1 <- mean(data$mu1)
  r0 <- mean(data$mu0)
  rd <- r1 - r0
  
  results <- c(r1,r0,rd)
  names(results) <- c("r1", "r0", "rd")
  return(bind_rows(results))
  
}

# AIPW Estimator
AIPW <- function(data, exposure, outcome, covarsT, covarsO, learners, control){
  
  pi_fit <- SuperLearner(Y=as.matrix(data[, exposure]), X=data[, covarsT], family=binomial(), SL.library=learners, cvControl=control)
  mu_fit <- SuperLearner(Y=as.matrix(data[, outcome]), X=data[, c(exposure, covarsO)], family=binomial(), SL.library=learners, cvControl=control)
  
  dat1 <- data %>%
    mutate(statin=1)
  
  dat0 <- data %>%
    mutate(statin=0)
  
  data <- data %>%
    mutate(mu1 = predict(mu_fit, newdata = dat1[, c(exposure, covarsO)])$pred,
           mu0 = predict(mu_fit, newdata = dat0[, c(exposure, covarsO)])$pred,
           pi = predict(pi_fit, newdata = data[,covarsT])$pred)
  
  data <- data %>%
    mutate(pi = ifelse(pi<.025, .025, ifelse(pi>.975,.975,pi)),
           ipw = statin/pi + (1-statin)/(1-pi),
           a1 = statin*ipw*(Y-mu1)+mu1,
           a0 = (1-statin)*ipw*(Y-mu0)+mu0)
  
  r1 <- mean(data$a1)
  r0 <- mean(data$a0)
  rd <- r1 - r0
  
  data <- data %>%
    mutate(if1 = a1-r1,
           if0 = a0 -r0,
           ifd = a1 - a0 - rd)
  
  v1 <- var(data$if1)/nrow(data)
  v0 <- var(data$if0)/nrow(data)
  vd <- var(data$ifd)/nrow(data)
  
  results <- c(r1, r0, rd, v1, v0, vd)
  names(results) <- c("r1","r0","rd","v1","v0","vd")
  return(bind_rows(results))
}

# TMLE Estimator
TMLE <- function(data, exposure, outcome, covarsT, covarsO, learners, control){
  
  pi_fit <- SuperLearner(Y=as.matrix(data[, exposure]), X=data[, covarsT], family=binomial(), SL.library=learners, cvControl=control)
  mu_fit <- SuperLearner(Y=as.matrix(data[, outcome]), X=data[, c(exposure, covarsO)], family=binomial(), SL.library=learners, cvControl=control)
  
  dat1 <- data %>%
    mutate(statin=1)
  
  dat0 <- data %>%
    mutate(statin=0)
  
  data <- data %>%
    mutate(mu = predict(mu_fit, newdata = data[, c(exposure, covarsO)])$pred,
           mu1 = predict(mu_fit, newdata = dat1[, c(exposure, covarsO)])$pred,
           mu0 = predict(mu_fit, newdata = dat0[, c(exposure, covarsO)])$pred,
           pi = predict(pi_fit, newdata = data[,covarsT])$pred)

  
  data <- data %>%
    mutate(pi = ifelse(pi<.025, .025, ifelse(pi>.975,.975,pi)),
           H1 = statin/pi,
           H0 = (1-statin)/(1-pi))
  
  data <- data %>%
    mutate(mu = ifelse(mu == 0, 1e-17, ifelse(mu==1, 1-1e-17, mu)),
           mu1 = ifelse(mu1 == 0, 1e-17, ifelse(mu1==1, 1-1e-17, mu1)),
           mu0 = ifelse(mu0 == 0, 1e-17, ifelse(mu0==1, 1-1e-17, mu0)))
  
  epsilon <- coef(glm(Y ~ -1 + H0 + H1 + offset(qlogis(mu)), data = data, family = binomial))
  
  data = data %>% mutate(
    mu0_1 = plogis(qlogis(mu0) + epsilon[1] / (1 - pi)),
    mu1_1 = plogis(qlogis(mu1) + epsilon[2] / pi))
  
  
  r1 <- mean(data$mu1_1)
  r0 <- mean(data$mu0_1)
  rd <- r1 - r0
  
  data <- data %>%
    mutate(if1 = statin/pi*(Y-mu1_1) + mu1_1 - r1,
           if0 = (1-statin)/(1-pi)*(Y-mu0_1) + mu0_1 - r0,
           ifd = if1 - if0)
  
  v1 <- var(data$if1)/nrow(data)
  v0 <- var(data$if0)/nrow(data)
  vd <- var(data$ifd)/nrow(data)
  
  results <- c(r1, r0, rd, v1, v0, vd)
  names(results) <- c("r1","r0","rd","v1","v0","vd")
  return(bind_rows(results))
}