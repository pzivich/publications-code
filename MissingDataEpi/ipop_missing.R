###############################################################################
# Missing Outcome Data in Epidemiologic Studies
#		R code to recreate the results
#
# Rachael K Ross (2022/06/24)
###############################################################################

dat <- tibble(cerv4cm=c(0,0,0,0,1,1,1,1),
              #p17=c(0,0,1,1,0,0,1,1), 
              p17=c(1,1,0,0,1,1,0,0),
              y=c(1,0,1,0,1,0,1,0),
              n=c(15,215-15,13,222-13,21,186-21,23,177-23),
              na=c(11,161-11,10,167-10,16,140-16,17,133-17),
              nb=c(8,108-8,13,222-13,21,186-21,12,89-12),
              nc=c(15,215-15,13,222-13,21,186-21,0,0),
              nd=c(15,100-15,7,216-7,21,186-21,12,81-12))
dat
dat %>%
  group_by(y) %>%
  summarise(n=sum(n),na=sum(na),nb=sum(nb),nc=sum(nc),nd=sum(nd))
dat %>%
  summarise(n=sum(n),na=sum(na),nb=sum(nb),nc=sum(nc),nd=sum(nd))

#### crude

crude <- function(var){
  longdat <- uncount(dat, weights = var)
  mod <- glm(y ~ p17, family=binomial(link="log"), data=longdat)
  tibble(rr = exp(coef(mod)[[2]]),
         lcl = tidy(mod, conf.int=T, exp = T)[[2,"conf.low"]],
         ucl = tidy(mod, conf.int=T, exp = T)[[2,"conf.high"]],
         se = coef(summary(mod))[[2,2]])
         
}

crude(dat$n)
crude(dat$na)
crude(dat$nb)
crude(dat$nc)
crude(dat$nd)


#### adjusted: g-comp with 500 bootstrap

gcomp <- function(data){

  # conditional outcome model in exposed
  moda1 <- glm(y ~ cerv4cm, family=binomial, data=data[data$p17==1,])
  ra1 <- data %>%
    mutate(out = predict(moda1, ., type="response")) %>%
    summarise(risk=mean(out)) %>%
    pull(risk)
  
  # conditional outcome model in unexposed
  moda0 <- glm(y ~ cerv4cm, family=binomial, data=data[data$p17==0,])
  ra0 <- data %>%
    mutate(out = predict(moda0, ., type="response")) %>%
    summarise(risk=mean(out)) %>%
    pull(risk)
  
  tibble(rr=ra1/ra0,lnrr=log(ra1/ra0))
}

# Point estimates
# gcomp(uncount(dat, weights = dat$na))
# gcomp(uncount(dat, weights = dat$nb))
# gcomp(uncount(dat, weights = dat$nc))
# gcomp(uncount(dat, weights = dat$nd))


# Bootstrap
myboot <- function(var){
  longdat = uncount(dat, weights = var)
  
  N = nrow(longdat)
  nreps=500
  df_boot = tibble(rep = 1:nreps, 
                   bs = replicate(nreps, longdat[sample(1:N, N, replace=TRUE),], simplify = FALSE),
                   results = map(bs, gcomp))
  results <- as_tibble(unnest(df_boot,results))
  tibble(rr = gcomp(longdat)[[1]],
         se = sd(results$lnrr),
         lcl = exp(log(rr) - 1.96*se),
         ucl = exp(log(rr) + 1.96*se))
}

myboot(dat$na)
myboot(dat$nb)
myboot(dat$nc)
myboot(dat$nd)


