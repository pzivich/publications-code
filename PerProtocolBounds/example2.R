# Per-protocol Bounds #
# Example 2 #

# set up environment ----
library(tidyverse)
source <- "path/to/file"

# read in data ----

actg <- read.table(paste0(source, "/actg320.dat"))
names(actg) <- c("id", "male", "black", "hispanic", "idu", "art", "d", "drop", "r", "age", "karnof", "days", "cd4",
                 "stop", "t", "delta")


# analysis ----
tau <- 365.25*1

get_rdrr <- function(time, ind, tx, lab = "ITT"){
  rdat <- data.frame(time = time, ind = ind, tx = tx)
  kms <- summary(survfit(Surv(time, ind) ~ tx, data = rdat))
  
  kmdat <- data.frame(time = kms$time, 
                      r = 1 - kms$surv, 
                      tx = kms$strata, 
                      se = kms$std.err)
  
  rdrr <- kmdat %>% 
    group_by(tx) %>% 
    filter(time <= tau) %>% 
    summarize(risk = last(r), se = last(se)) %>% 
    ungroup() %>% 
    mutate(tx = substr(tx, 4, 4)) %>% 
    pivot_wider(names_from = tx, values_from = c(risk, se)) %>% 
    mutate(rd = risk_1 - risk_0, 
           rr = risk_1 / risk_0, 
           serd = sqrt(se_1^2 + se_0 ^ 2), 
           selnrr = sqrt((1/risk_1)^2 * se_1^2 + (1/risk_0)^2 * se_0^2)) %>% 
    slice(n()) %>% 
    mutate(rdlcl = rd -1.96 * serd, 
           rducl = rd + 1.96 * serd, 
           rrlcl = exp(log(rr) - 1.96 * selnrr), 
           rrucl = exp(log(rr) + 1.96 * selnrr))
  
  print(paste0(lab, " RD: ", round(rdrr$rd, 2), "; 95% CI: ", round(rdrr$rdlcl, 2), ", ", round(rdrr$rducl, 2)))
  print(paste0(lab, " RR: ", round(rdrr$rr, 2), "; 95% CI: ", round(rdrr$rrlcl, 2), ", ", round(rdrr$rrucl, 2)))
  
}

## ITT ----
get_rdrr(actg$t, actg$delta, actg$art)

## Bounds ----

actg_bounds <- actg %>% 
  mutate(stop_time = as.numeric(stop), 
         adhere = as.numeric(is.na(stop_time) | stop_time > pmin(t, tau)), 
         t_low = case_when(adhere == 1  ~ t, 
                           adhere == 0 & art == 1 ~ tau, 
                           adhere == 0 & art == 0 ~ stop_time), 
         t_up  = case_when(adhere == 1  ~ t, 
                           adhere == 0 & art == 1 ~ stop_time, 
                           adhere == 0 & art == 0 ~ tau), 
         delta_low  = case_when(adhere == 1  ~ delta, 
                                adhere == 0 & art == 1 ~ 0, 
                                adhere == 0 & art == 0 ~ 1), 
         delta_up   = case_when(adhere == 1  ~ delta, 
                                adhere == 0 & art == 1 ~ 1, 
                                adhere == 0 & art == 0 ~ 0))

get_rdrr(actg_bounds$t_low, actg_bounds$delta_low, actg_bounds$art, lab = "Lower")
get_rdrr(actg_bounds$t_up, actg_bounds$delta_up, actg_bounds$art, lab = "Upper")


