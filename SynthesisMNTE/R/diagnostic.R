# Program: MNE_replication.R
# Developed by: Bonnie Shook-Sa
# Last Updated: 12.17.24

library(plyr)
library(geex)
library(resample)
library(dplyr)

###### Bring in NHANES data and add indicator for those 2-7
nhanes <- read.csv("nhanes.csv")
nhanes$agelt8 <- ifelse(nhanes$age<8, 1, 0)

### bring in data for mathematical model for children 2-7
height.params <- read.csv("height_params2.csv")
sbp_params <- read.csv("sbp_params2.csv")
sbp_params <- within(sbp_params, rm("code"))

#first, add height cutoffs to the dataset
nhanes$gender <- ifelse(nhanes$female==1, 'f', 'm')
nhanes2 <- join(nhanes, height.params, by=c("gender", "age"))
nhanes2$height <- nhanes2$height / 2.54
nhanes2$height_cat <- ifelse(nhanes2$height<nhanes2$c1,
                             1,
                             ifelse(nhanes2$height<nhanes2$c2,
                                    2,
                                    ifelse(nhanes2$height<nhanes2$c3,
                                           3,
                                           ifelse(nhanes2$height<nhanes2$c4,
                                                  4,
                                                  ifelse(nhanes2$height<nhanes2$c5,
                                                         5,
                                                         ifelse(nhanes2$height<nhanes2$c6,
                                                                6,7))))))

#add on median and p90 sbps based on age, gender, and height category
nhanes3 <- join(nhanes2, sbp_params, by=c("gender", "age", "height_cat"))
nhanes3$norm_mean <- nhanes3$median
nhanes3$norm_sd <- (nhanes3$p90 - nhanes3$norm_mean) / qnorm(0.9)

##########################################################
##### Diagnostic
##########################################################

#First compare the mean SBP for those 8-17 from NHANES with mathematical model predictions for each age
nhanes4 <- nhanes3[nhanes3$agelt8==0,]

mean.sbp<-nhanes4 %>%
    group_by(age) %>%
              mutate(wtd_mean_sbp=weighted.mean(sbp, sample_weight, na.rm=TRUE))
mean.sbp.age<-unique(mean.sbp[c("age", "wtd_mean_sbp")])

math.sbp<-nhanes4 %>%
  group_by(age) %>%
  mutate(math_mean_sbp=weighted.mean(norm_mean, sample_weight, na.rm=TRUE))
math.sbp.age <- unique(math.sbp[c("age", "math_mean_sbp")])

compare.means <- merge(mean.sbp.age,math.sbp.age,by=c("age"))

#Then, use the Monte Carlo procedure

bootstrap_diagnostic <- function(B, bootdata){
  if(B == 0) return(0)
  if(B>0){
    boot.est <- matrix(NaN, nrow=nrow(compare.means) , ncol=B)
    datbi<-samp.bootstrap(nrow(bootdata), B)
    for(i in 1:B){
      dati <- bootdata[datbi[,i],]
      positive <- dati #here we are only working with the positive region

      # Fit outcome model for positive region, use it to impute those missing in positive region
      out.model  <- glm(sbp ~ miss*as.factor(age), data=dati, weights=sample_weight)
      positive.miss <- subset(positive, (is.na(positive[,'sbp'])))
      positive.miss$miss <- 0
      positive.miss$sbp <- predict(out.model, positive.miss, type="response")
      positive.nmiss <- subset(positive, (!is.na(positive[,'sbp'])))
      positive.all <- rbind(positive.nmiss, positive.miss)

      #get a random draw from mathematical model for positive region
      positive.all$sbp.draw <- rnorm(nrow(positive.all), mean=positive.all$norm_mean, sd=positive.all$norm_sd)

      #calculate weighted means (by age) for statistical and math models
      mean.sbp.boot <- positive.all %>%
        group_by(age) %>%
        mutate(wtd_mean_sbp=weighted.mean(sbp, sample_weight))
      mean.sbp.age.boot <- unique(mean.sbp.boot[c("age", "wtd_mean_sbp")])

      math.sbp.boot <- positive.all %>%
        group_by(age) %>%
        mutate(math_mean_sbp=weighted.mean(sbp.draw, sample_weight))
      math.sbp.age.boot <- unique(math.sbp.boot[c("age", "math_mean_sbp")])

      #store difference
      compare.means.boot <- merge(mean.sbp.age.boot,math.sbp.age.boot,by=c("age"))
      compare.means.boot$dif <- compare.means.boot$wtd_mean_sbp-compare.means.boot$math_mean_sbp

      boot.est [,i] <- compare.means.boot$dif
    }
    return(boot.est)
  }
}

#run the function with 10000 bootstraps
syn.diagnostic <- bootstrap_diagnostic(10000, nhanes4)

#format output and compute CIs
syn.diagnostic.est <- apply(syn.diagnostic, 1, median, na.rm=F)
syn.diagnostic.LCL <- apply(syn.diagnostic, 1, quantile, probs=0.025, na.rm=F)
syn.diagnostic.UCL <- apply(syn.diagnostic, 1, quantile, probs=0.975, na.rm=F)
syn.diagnostic.all <- as.data.frame(cbind(syn.diagnostic.est, syn.diagnostic.LCL, syn.diagnostic.UCL))
syn.diagnostic.all$age <- seq(1:10)+7
names(syn.diagnostic.all) <- c('diff', 'LCL', 'UCL', 'age')
write.csv(syn.diagnostic.all,"diagnostic_MC_diffs.csv")
