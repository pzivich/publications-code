# APPROACH 3: Pooled Multinomial Logistic Regression g-computation ####
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

# > Step 2: Fit Pooled multinomial logit model ####
# NOTE: Model will be fit to discretized time (t_out) with some user-specified
# functional form. Here we are using disjoint indicator variables. Model
# can also include relevant covariates (W), treatment (A), and interactions.
# See applied example (code block 2) for a more complex model.
# NOTE: model is fit using 'vglm' package from 'VGAM' library.
multinomial = VGAM::vglm(event ~ I(as.factor(t_out)), data = long, family = "multinomial")

# > Step 3. Predicted discrete-time hazards (i,k) ####
# NOTE: predictions using our fit model and 'VGAM' predict function which
# will produce a dataframe with J+1 cols (1st col  is event-free)
h = VGAM::predict(object = multinomial, newdata = long[long$t_out %in% eventtimes,], type = "response")
# renaming each column to its respective h (h0 to hJ)
colnames(h) = paste0("h",0:J)
# merging back to the dataset
long = long |> cbind(dplyr::as_tibble(h)) # merging back to dataset

# > Step 4a. Event-free survival (S) up to interval k, for each i ####
# Note: Here we calculate individual i's event-free survival up to k
# as the cumulative product of h0 (k-specific probability of no event).
# The group_by statement restricts the cumulative product to each pid.
long = long |>
  dplyr::group_by(pid) |> 
  dplyr::mutate(s = cumprod(h0)) |> 
  dplyr::ungroup() 

# > Step 4b. Individual-level cumulative outcome (mu) for each event type ####
# Note: the cumulative outcome of event type j is the cumulative sum of
# the probability of the event at interval k * the probability of survival
# through k-1 (lagged s). The group_by is used as described above.
long = long |>
  dplyr::group_by(pid) |>
  dplyr::mutate(
    cumsum(
      dplyr::across(paste0("h",1:J), .names = "mu{col}") * 
        dplyr::lag(s, default = 1)
    )
  )|>
  dplyr::ungroup()
# Renaming cumulative outcome variables to muj
names(long) = sub('^muh', 'mu', names(long))
long.multinomial=long

# > Step 5. Calculate mean cumulative outcome at each interval k ####
# NOTE: Mean of s and h1-hJ at t_out=k will construct event-free survival
# function and each event's cumulative incidence function.
results_multinomial = long |>
  dplyr::group_by(t_out) |>
  # Will calculate the event free survival and cumulative incidence of each event type 1 to J
  dplyr::summarise(dplyr::across(all_of(c("s",paste0("mu",1:J))),list(mean),.names="{col}")) 
names(results_multinomial) = sub('^mu', 'r', names(results_multinomial)) # mu to r
results_multinomial = results_multinomial |>
  # appending suffix indicating the multinomial model to the results
  dplyr::rename_at(dplyr::vars(matches("^(s|r)")), ~ paste0(.,"_multinomial"))
head(results_multinomial)
tail(results_multinomial)
