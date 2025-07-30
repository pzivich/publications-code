# Specifically readr=2.1.4, dplyr=1.1.1, tidyr=1.3.0,
if (!require("tidyverse")) {
  install.packages('tidyverse')
}
library(tidyverse)

if (!require("microbenchmark")) {
  install.packages('microbenchmark')
}
library(microbenchmark)


# Read in dataset - then subset
df = read_csv("lau.csv") |>
  mutate(
    t_days = ceiling(365.25 * t),
    event_date = t_days,
  ) |>
  uncount(t_days, .id="days") |>
  mutate(
    event_indicator = ((event_date - days) == 0) & (eventtype == 2)
    )

# Tidyverse-friendly way to find all unique event times, as a vector
unique_event_times = df |> filter(event_indicator) |> pull(days) |> unique()


# Create the 1y dataset - filter by days, and optionally filter by being at a unique event time
df_1y = df |> filter(days <= 365.25)
df_1y_improved = df |>
  filter(days <= 365.25) |>
  filter(days %in% unique_event_times)

# Create the 5y dataset - same way as 1y dataset
df_5y = df |> filter(days <= 5 * 365.25)
df_5y_improved = df |>
  filter(days <= 5 * 365.25) |>
  filter(days %in% unique_event_times)


# Improved implementation
model_1y_improved = glm(
  event_indicator ~ factor(days) + ageatfda + BASEIDU + black + cd4nadir,
  data = df_1y_improved,
  family=binomial(link="logit")
)

# Benchmark of 1-year improved implementation with 100 trials
mb_1y_improved = microbenchmark(
  glm(
    event_indicator ~ factor(days) + ageatfda + BASEIDU + black + cd4nadir,
    data = df_1y_improved,
    family=binomial(link="logit")
  ),
  times=100
)

# Benchamrk of 1-year standard implementation with 100 trials
mb_1y = microbenchmark(
  glm(
    event_indicator ~ factor(days) + ageatfda + BASEIDU + black + cd4nadir,
    data = df_1y,
    family=binomial(link="logit")
  ),
  times=100
)

# Benchmark of 5-year improved implementation with 100 trials
mb_5y_improved = microbenchmark(
  glm(
    event_indicator ~ factor(days) + ageatfda + BASEIDU + black + cd4nadir,
    data = df_5y_improved,
    family=binomial(link="logit")
  ),
  times=100
)

# Benchmark of 5-year standard implementation with 100 trials. Takes a LONG time.
mb_5y = microbenchmark(
  glm(
    event_indicator ~ factor(days) + ageatfda + BASEIDU + black + cd4nadir,
    data = df_5y,
    family=binomial(link="logit")
  ),
  times=100
)





