#####################################################################################################################
# Accounting for Missing Data in Public Health Research Using a Synthesis of Statistical and Mathematical Models
#   This file runs the AIPW analysis reported in Appendix 3.
#
# Paul Zivich (2025/10/14)
#####################################################################################################################

library(plyr)
library(geex)
library(resample)
library(dplyr)

##########################################################
##### Setting up data
##########################################################

setwd("C:/Users/zivic/Documents/open-source/publications-code/SynthesisMNTE")
nhanes <- read.csv("data/nhanes.csv")

nhanes$height = nhanes$height / 2.54
nhanes <- nhanes[!is.na(nhanes$height) & !is.na(nhanes$weight), ]
nhanes$agelt8 <- ifelse(nhanes$age<8, 1, 0)


##########################################################
##### Extrapolation, mean SPB
##########################################################

# Leaving off of the Appendix (since not focus of paper)

##########################################################
##### Synthesis, mean SPB
##########################################################

### bring in data for mathematical model for children 2-7
height.params <- read.csv("height_params.csv")
sbp_params <- read.csv("sbp_params.csv")
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
                                                                6, 7))))))

#add on median and p90 sbps based on age, gender, and height category
nhanes3 <- join(nhanes2, sbp_params, by=c("gender", "age", "height_cat"))
nhanes3$norm_mean <- nhanes3$median
nhanes3$norm_sd <- (nhanes3$p90 - nhanes3$norm_mean) / qnorm(0.9)

### bootstrap function, from which we will derive point estimate and 95% CIs
bootstrap_syn <- function(B, bootdata){
    if(B == 0) return(0)
    if(B>0){
        boot.est <- matrix(NaN, nrow=B, ncol=1)
        datbi <- samp.bootstrap(nrow(bootdata), B)
        for(i in 1:B){
            dati <- bootdata[datbi[,i],]

            # split data into positive and nonpositive
            positive<-dati[dati$agelt8==0, ]
            npositive<-dati[dati$agelt8==1, ]

            # Computing IPMW
            ps.model <- glm(miss ~ female + age + age_sp1 + age_sp2
                                  + height + h_sp1 + h_sp2
                                  + weight + w_sp1 + w_sp2,
                            data=positive, family = "binomial")
            pr_obs <- 1 - predict(ps.model, positive, type="response")
            ipw <- ifelse(is.na(positive$sbp), 0, 1 / pr_obs)
            full_weight <- ipw * positive$sample_weight

            # Fit outcome model for positive region, use it to impute those missing in positive region
            out.model <- glm(sbp ~ female*(age + age_sp1 + age_sp2
                                           + height + h_sp1 + h_sp2
                                           + weight + w_sp1 + w_sp2),
                             data=positive,
                             weights=full_weight)
            positive$sbp_hat <- predict(out.model, positive, type="response")

            #get a random draw from mathematical model for nonpositive region
            npositive$sbp_hat <- rnorm(nrow(npositive),
                                       mean=npositive$norm_mean,
                                       sd=npositive$norm_sd)

            #combine
            allimputed <- rbind(positive, npositive)

            #store weighted mean
            boot.est [i] <- sum(allimputed$sbp_hat * allimputed$sample_weight) / sum(allimputed$sample_weight)
        }
        return(boot.est)
    }
}

#run the function with 10000 bootstraps
syn.boots <- bootstrap_syn(10000, nhanes3)

#format output and compute CIs
syn.est <- median(syn.boots)
syn.LCL <- (quantile(syn.boots, probs=0.025, na.rm=FALSE))
syn.UCL <- (quantile(syn.boots, probs=0.975, na.rm=FALSE))
syn.all <- as.data.frame(cbind(syn.est, syn.LCL, syn.UCL))
syn.all$type <- "Synthesis"
syn.all$se <- NA
names(syn.all) <- c('est', 'LCL', 'UCL', 'type', 'se')
syn.all

# Output
# 
#      est      LCL      UCL      type se
# 100.4806 99.93952 101.0032 Synthesis NA


