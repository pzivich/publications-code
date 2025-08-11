# > Creating a skeleton tibble to align event times ####
# NOTE: We're doing this to align the AJ results which are on the
# daily timescale with parametric g-computation results that can be
# on the daily, weekly, or monthly timescale, depending on the specified
# option. This ensures the correct estimates are being compared at each
# point in time. 
# _ Daily timescale ####
time.days = dplyr::tibble(t.days = 0:730)
# _ Aligning weeks to days ####
# Note: the one-week risk shouldn't apply before day seven. This code ensures
# that weekly risks will only increment at the last day of the time interval.
# This is necessary to ensure the daily risks are compared to the correct weekly
# risk.
time.weeks = dplyr::tibble(t.days = 0:730,
                    t.weeks = ceiling(t.days/7)) |>
  dplyr::summarise(t.days = max(t.days), .by = "t.weeks") 
# _ Aligning months to days ####
# Note: same as above, but comparing monthly risks to daily risks.
time.months = dplyr::tibble(t.days = 0:730,
                     t.months = ceiling(t.days/30.44)) |>
  dplyr::summarise(t.days = max(t.days), .by = "t.months") 
# _ Merging aligned times ####
times = time.days |>
  dplyr::left_join(time.weeks) |>
  dplyr::left_join(time.months) |>
  # Setting first row equal to zero and then down-filling weeks/months.
  dplyr::mutate(t.weeks = ifelse(t.days==0,0,t.weeks),
                t.months = ifelse(t.days==0,0,t.months)) |>
  tidyr::fill(everything(), .direction = "down") |>
  # Using the timescale specified in 01_comparing approaches to determine t_out
  # which will be used to merge in the parametric g-computation results.
  dplyr::mutate(t_out = dplyr::case_when(
    timescale=="days" ~ t.days,
    timescale=="weeks" ~ t.weeks,
    timescale=="months" ~ t.months
  ))
# > Merging the results to the aligned times ####
results = times |>
  # AJ is left-joined on days as it is applied to the original timescale
  dplyr::left_join(results_aj, by = "t.days") |>
  # Multiple pooled logit g-comp. results are merged to specified discretized timescale
  dplyr::left_join(results_multiple, by = "t_out") |>
  # Pooled multinomial logit g-comp. results are merged to specified discretized timescale
  dplyr::left_join(results_multinomial, by = "t_out") |>
  # Setting survival estimates to 1 and risk esimtates to 0 at t=0
  dplyr::mutate(across(.cols = starts_with("s"), ~ ifelse(t.days==0,1,.)),
         across(.cols = starts_with("r"), ~ ifelse(t.days==0,0,.))) |>
  # Down-filling results so differences can be taken
  tidyr::fill(everything(), .direction = "down") 





