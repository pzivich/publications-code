# APPROACH 1: naive Aalen-Johansen Estimator ####
# NOTE: here we're using the 'survival' package to fit the naive (unadjusted)
# Aalen-Johansen estimator to a traditional "wide" dataset using the original 
# timescale (t.days). The 'mstate' option is used to implement the AJ estimator.
aj = survival::survfit(
  survival::Surv(time = wide$t.days, event = wide$d, type = "mstate") ~ 1
)

# A summary of the fit AJ estimator will provide survival and risk of each
# event type, for each time that an event occurs.
ajf = summary(aj)

# Saving the AJ results.
# Setting column names for estimated probabilities (ajf$pstate). The first 
# column is the estimated survival and each column after is the risk of a
# given outcome type.
colnames(ajf$pstate) = c("s",paste0("r",1:J)) 
# binding event times and estimated probabilities as results_aj.
results_aj = dplyr::tibble(t.days = ajf$time) |>
  cbind(dplyr::as_tibble(ajf$pstate))
head(results_aj)
tail(results_aj)
