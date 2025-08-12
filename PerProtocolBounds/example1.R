# Per-protocol Bounds #
# Example 1 #

# set up environment ----
library(tidyverse)
source <- "path/to/file"

# read in data ----
ipop <- read.table(paste0(source, "/ipop.dat"))
names(ipop) <- c("pid", "cervix", "p17", "preterm", "adhere")

# IPOP ----

## ITT ----
mod1 <- glm(preterm ~ p17, data = ipop, family = "binomial"(link = "identity"))
rd <- coefficients((mod1))[2]
rd_ci <- round(confint(mod1)[2,], 2)
print(paste0("RD: ", round(rd, 2), "; 95% CI: ", rd_ci[1], ", ", rd_ci[2]))

mod2 <- glm(preterm ~ p17, data = ipop, family = "binomial"(link = "log"))
rr <- exp(coefficients((mod2))[2])
rr_ci <- round(exp(confint(mod2)[2,]), 2)
print(paste0("RR: ", round(rr, 2), "; 95% CI: ", rr_ci[1], ", ", rr_ci[2]))

## Simple PP ----
ipop_pp <- ipop %>% filter(adhere == 1)
mod1 <- glm(preterm ~ p17, data = ipop_pp, family = "binomial"(link = "identity"))
rd <- coefficients((mod1))[2]
rd_ci <- round(confint(mod1)[2,], 2)
print(paste0("RD: ", round(rd, 2), "; 95% CI: ", rd_ci[1], ", ", rd_ci[2]))

mod2 <- glm(preterm ~ p17, data = ipop_pp, family = "binomial"(link = "log"))
rr <- exp(coefficients((mod2))[2])
rr_ci <- round(exp(confint(mod2)[2,]), 2)
print(paste0("RR: ", round(rr, 2), "; 95% CI: ", rr_ci[1], ", ", rr_ci[2]))

## Bounds ----
ipop_bounds <- ipop %>% 
  mutate(preterm_up = ifelse(adhere == 1, 
                             preterm, 
                             ifelse(p17 == 1, 1, 0)), 
         preterm_low = ifelse(adhere == 1, 
                              preterm, 
                              ifelse(p17 == 1, 0, 1)))
mod1_up <- glm(preterm_up ~ p17, data = ipop_bounds, family = "binomial"(link = "identity"))
rd <- coefficients((mod1_up))[2]
rd_ci <- round(confint(mod1_up)[2,], 2)
print(paste0("Upper Bound for RD: ", round(rd, 2), "; 95% CI: ", rd_ci[1], ", ", rd_ci[2]))

mod1_low <- glm(preterm_low ~ p17, data = ipop_bounds, family = "binomial"(link = "identity"))
rd <- coefficients((mod1_low))[2]
rd_ci <- round(confint(mod1_low)[2,], 2)
print(paste0("Lower Bound for RD: ", round(rd, 2), "; 95% CI: ", rd_ci[1], ", ", rd_ci[2]))


mod2_up <- glm(preterm_up ~ p17, data = ipop_bounds, family = "binomial"(link = "log"))
rr <- exp(coefficients((mod2_up))[2])
rr_ci <- round(exp(confint(mod2_up)[2,]), 2)
print(paste0("Upper Bound for RR: ", round(rr, 2), "; 95% CI: ", rr_ci[1], ", ", rr_ci[2]))

mod2_low <- glm(preterm_low ~ p17, data = ipop_bounds, family = "binomial"(link = "log"))
rr <- exp(coefficients((mod2_low))[2])
rr_ci <- round(exp(confint(mod2_low)[2,]), 2)
print(paste0("Lower Bound for RR: ", round(rr, 2), "; 95% CI: ", rr_ci[1], ", ", rr_ci[2]))

