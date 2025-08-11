# APPROACH 2: Multiple Pooled Logistic Regression g-computation ####
# > Step 1. Elongating data ####
# _ Step 1a. Creating rows for the maximum number of discretized time units ####
# The uncount function will create 1:k rows (depending on the specified timescale)
# for each individual.
long = tidyr::uncount(
  data = wide,
  weights = 
    if(timescale=="days"){
      max(t.days)
    } else if(timescale=="weeks"){
      max(t.weeks)
    } else if(timescale=="months"){
      max(t.months)
    },
  .id = "t_out"
) 

# _ Step 1b. Coding indicators ####
long = long |> 
  dplyr::mutate(
    # Setting the individual's discretized event time based on the specified timescale
    t.discrete = dplyr::case_when(
      timescale=="days" ~ t.days,
      timescale=="weeks" ~ t.weeks,
      timescale=="months" ~ t.months
    ),
    # event indicator for each interval: 0 until an event, j for event, NA after event
    event = dplyr::case_when(
      t.discrete > t_out ~ 0,
      t.discrete == t_out ~ d,
      T ~ NA
    )
  )

# _ OPTIONAL Step 1c. Filtering to times where events occur ####
# Here we're removing rows in which no events occur. This saves processing time,
# as the function wouldn't increment at these times, however it only removes rows
# when the timescale is "days" in the present example. This step can be commented
# out if you like to fit the model including nonevent intervals.
# Identifying event times for specified timescale
if(timescale=="days"){
  eventtimes = dplyr::pull(unique(wide[wide$d!=0,"t.days"]))
} else if(timescale=="weeks"){
  eventtimes = dplyr::pull(unique(wide[wide$d!=0,"t.weeks"]))
} else if(timescale=="months"){
  eventtimes = dplyr::pull(unique(wide[wide$d!=0,"t.months"]))
}
# Filtering to only include rows with eventtimes on the timescale of interest
long = long[long$t_out %in% eventtimes,] 

# > Step 2. Fit multiple pooled logit models ####
# NOTE: we will fit J pooled logit models (in this example J=2).
# _ Model 1: J=1 'censoring' J=2 ####
# NOTE: Modeling event as 1 vs. 0/2. Here we restrict to rows without J=2 to 
# maintain risk sets. 
multiple1 = glm(event==1 ~ I(as.factor(t_out)), data = long[long$event!=2,], family = binomial(link = "logit"))
# _ Model 2: J=2 'censoring' J=1 ####
# NOTE: Modeling event as 2 vs. 0/1.
multiple2 = glm(event==2 ~ I(as.factor(t_out)), data = long, family = binomial(link = "logit"))

# > Step 3. Generate conditional discrete-time hazards and event free probabilities ####
# Here we're estimating "conditional discrete hazards (cause-specific 
# for each cause of death/competing event conditioned for event and hazard of 
# competing event) for each subject in each time interval" Young et al. (2020)
long$h1 = predict(object = multiple1, newdata = long, type = "response")
long$h2 = predict(object = multiple2, newdata = long, type = "response")
long$p0 = (1-long$h1)*(1-long$h2)

# > Step 4a. Event-free survival (s) up to interval k, for each i ####
# Note: Here we calculate individual i's event-free survival up to k
# as the cumulative product of p0. p0 was calculated as the joint probability
# of the complement of each conditional discrete-time hazard.
long = long |>
  dplyr::group_by(pid) |>
  dplyr::mutate(
    s = cumprod(p0)
  ) |>
  dplyr::ungroup()

# > Step 4b. Individual-level cumulative outcome (r) for each event type ####
long = long |>
  dplyr::group_by(pid) |>
  dplyr::mutate(
    r1 = cumsum(h1*(1-h2)*dplyr::lag(s, default = 1)),
    r2 = cumsum(h2*dplyr::lag(s, default = 1))
  ) |>
  dplyr::ungroup()
long.multi = long

# > Step 5. Calculate mean cumulative outcome at each interval k ####
# NOTE: Mean of s and h1-hJ at t_out=k will produce event-free survival
# estimates and each event's risk estimate.
results_multiple = long |>
  dplyr::group_by(t_out) |>
  dplyr::summarise(s = mean(s), r1 = mean(r1), r2 = mean(r2)) |>
  # appending suffix indicating approach to survival and risk estimates
  dplyr::rename_at(dplyr::vars(matches("^(s|r)")), ~ paste0(.,"_multiple"))
head(results_multiple)
tail(results_multiple)
