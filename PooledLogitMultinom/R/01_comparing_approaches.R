# > Setting up the example dataset ####
# Note: This applied example uses data from Lau, Cole, and Gange (2009; doi: 
# 10.1093/aje/kwp107). Time (t) is measured in years from baseline and was 
# limited to a two-year follow-up period (i.e. tau=2) for the purposes of this 
# illustrative example. Individuals with event times greater than two years were
# administratively censored at the end of the follow-up period (i.e. t.star=2, 
# eventtype.star=0).
# _ Setting two-year follow-up period ####
tau = 2 
# _ Importing and cleaning data ####
wide = readr::read_csv("analysis/03_clean_data/multinomial_logit/lau2009.csv") |>
  janitor::clean_names() |>
  dplyr::mutate(
    # Administratively censoring individuals with t > tau (i.e. t.star=tau,
    # and eventtype.star = 0)
    t.star = ifelse(t<=tau,t,tau), # observed event time will be t or tau (2)
    eventtype.star = ifelse(t<=tau,eventtype,0),
    # Discretized time variables (days/weeks/months)
    t.days = ceiling(t.star*(365.25)),
    t.weeks = ceiling(t.star*(365.25)/7),
    t.months = ceiling(t.star*(365.25)/30.44))|>
  dplyr::rename(pid = id, d = eventtype.star)
# _ Indicating the number of event types (assumes 0-J coding scheme) ####
J = length(unique(wide$d))-1
# _ Specifying timescale for parametric g-computation ####
timescale = "weeks"

# > APPROACH 1: naïve Aalen-Johansen estimator ####
source("analysis/01_code/09_multinomial_logit_g_formula/R/01a_naive_AJ.R")

# > APPROACH 2: Multiple pooled logit g-computation ####
source("analysis/01_code/09_multinomial_logit_g_formula/R/01b_multiple_pooled_logit.R") 
# Timing - Months: 0.566 sec; Weeks: 35.966 sec; Days: 746.011 sec.

# > APPROACH 3: Pooled multinomial logit g-computation ####
source("analysis/01_code/09_multinomial_logit_g_formula/R/01c_pooled_multinomial_logit.R") 
# Timing - Months: 2.631 sec; Weeks: 163.724 sec; Days: 3481.876 sec.

# > Merging Results ####
source("analysis/01_code/09_multinomial_logit_g_formula/R/01d_merge_results.R") 

# _ Table 1. Comparing AJ and different parametric g-formula approaches ####
table1 = dplyr::tibble(
  survival = rbind(max(abs(results$s - results$s_multiple)), max(abs(results$s - results$s_multinomial)), max(abs(results$s_multiple - results$s_multinomial))),
  risk_haart = rbind(max(abs(results$r1 - results$r1_multiple)), max(abs(results$r1 - results$r1_multinomial)), max(abs(results$r1_multiple - results$r1_multinomial))),
  risk_ad = rbind(max(abs(results$r2 - results$r2_multiple)), max(abs(results$r2 - results$r2_multinomial)), max(abs(results$r2_multiple - results$r2_multinomial)))
)
table1

# _ Figure 1. Cumulative incidence curves by event type and approach ####
library(ggplot2)
library(patchwork)
figure1a = ggplot() + 
  geom_step(aes(x = t.days, y = r1, col = "Naïve AJ", linetype = "Naïve AJ"), data = results) +
  geom_step(aes(x = t.days, y = r1_multinomial, col = "Multinomial g-computation", linetype = "Multinomial g-computation"), data = results) +
  geom_step(aes(x = t.days, y = r1_multiple, col = "Multiple logit g-computation", linetype = "Multiple logit g-computation"), data = results) +
  scale_linetype_manual(name="Estimator:",breaks=c("Naïve AJ","Multiple logit g-computation","Multinomial g-computation"), values = c("solid","dashed","dotted")) +
  scale_color_manual(name="Estimator:",breaks=c("Naïve AJ","Multiple logit g-computation","Multinomial g-computation"),values = c("black","grey25","grey50")) +  scale_y_continuous("Risk",limits = c(0,0.4)) +
  scale_x_continuous("Weeks", limits = c(0,730), breaks = c(0,182.5,365,547.5,730), labels = c(0,26,52,78,104)) +
  scale_y_continuous("Risk",limits = c(0,0.35)) +
  ggtitle("A. HAART Initiation") +
  theme(legend.position = "bottom",legend.box.spacing = unit(0, "pt")) +
  theme_classic()
figure1b = ggplot() + 
  geom_step(aes(x = t.days, y = r2, col = "Naïve AJ", linetype = "Naïve AJ"), data = results) +
  geom_step(aes(x = t.days, y = r2_multinomial, col = "Multinomial g-computation", linetype = "Multinomial g-computation"), data = results) +
  geom_step(aes(x = t.days, y = r2_multiple, col = "Multiple logit g-computation", linetype = "Multiple logit g-computation"), data = results) +
  scale_linetype_manual(name="Estimator:",breaks=c("Naïve AJ","Multiple logit g-computation","Multinomial g-computation"), values = c("solid","dashed","dotted")) +
  scale_color_manual(name="Estimator:",breaks=c("Naïve AJ","Multiple logit g-computation","Multinomial g-computation"),values = c("black","grey25","grey50")) +
  scale_y_continuous("",limits = c(0,0.35)) +
  scale_x_continuous("Weeks", limits = c(0,730), breaks = c(0,182.5,365,547.5,730), labels = c(0,26,52,78,104)) +
  ggtitle("B. AIDS/Death") +
  theme(legend.position = "bottom",legend.box.spacing = unit(0, "pt")) +
  theme_classic()
combined = (figure1a + figure1b & theme(legend.position = "bottom",panel.grid.major = element_line(color="grey90"))) +
  plot_layout(guides = "collect")
combined 

# ggsave("analysis/04_results/multinomial_logit/figure1.png", plot = combined, device = "png",
#        width = 17, height = 8.5, units = "cm", dpi = 600)



