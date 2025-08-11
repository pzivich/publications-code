# > Reading in dataset ####
wide = readr::read_csv("analysis/03_clean_data/multinomial_logit/lau2009.csv") |>
  janitor::clean_names() |>
  dplyr::mutate(t_old = t,
                t = ceiling(t_old*(365.25/7))) |>
  dplyr::rename(pid = id, d = eventtype)
# _ Specify follow-up period (τ) ####
tau=104

# > Step 1. Elongating data ####
d.long = tidyr::uncount(
  data = wide,
  weights = max(t),
  .id = "t_out"
) |> 
  # event indicator for each interval: 0 until an event, j for event, NA after event
  dplyr::mutate(
    event = dplyr::case_when(
      t > t_out ~ 0,
      t == t_out ~ d,
      T ~ NA
    )
  )
# _ Determine number of event types (J) ####
# This assumes 0-J coding scheme yielding 1-J event types
# and 0 coded as no event. We subtract 2 to account for 0 and NA
J = length(unique(d.long$event))-2
# _ Filter to max follow-up (will only impact lau) ####
d.long = d.long |> 
  dplyr::filter(t_out<=tau)

# > Step 1. Fit pmlm ####
# pmnlr = VGAM::vglm(
#   event ~ baseidu*(splines::bs(ageatfda) + splines::bs(cd4nadir) + black + I(as.factor(t_out))),
#   data = d.long,
#   family = "multinomial"
# ) ~ 698.7 seconds (11.6 minutes)
# saveRDS(pmnlr, "pmnlr.rds")
pmnlr <- readRDS("analysis/03_clean_data/multinomial_logit/pmnlr.rds")

# > Step 3-5 with helper function ####
# _ Steps 3-5 function ####
pmnlr_g_comp <- function(fit, dl, trt_level){
  a <- dplyr::mutate(dl, baseidu = trt_level) # Set treatment level
  # > Step 3. Predicted discrete-time hazards (i,k)
  pi = VGAM::predict(object = fit, newdata = a, type = "response")
  colnames(pi) = paste0("pi",0:J)
  a = a |> cbind(dplyr::as_tibble(pi)) # merging back to dataset
  # > Step 4a. Event-free survival (s) up to interval k, for each i
  a = a |>
    dplyr::group_by(pid) |> 
    dplyr::mutate(s = cumprod(pi0)) |> 
    dplyr::ungroup() 
  # > Step 4b. Individual-level cumulative outcome (h) for each event type
  a = a |>
    dplyr::group_by(pid) |>
    dplyr::mutate(
      cumsum(
        dplyr::across(paste0("pi",1:J), .names = "h{col}") * 
          dplyr::lag(s, default = 1)
      )
    )|>
    dplyr::ungroup()
  names(a) = sub('^hpi', 'h', names(a))
  # > Step 5. Calculate mean cumulative outcome at each interval k
  results = a |>
    dplyr::group_by(t_out) |>
    dplyr::summarise(dplyr::across(all_of(c("s",paste0("h",1:J))),list(mean),.names="{col}"))
  names(results) = sub('^h', 'r', names(results))
  return(results)
}

# _ Steps 3-5: All had history of injection drug use at baseline (IDU) ####
results_idu = pmnlr_g_comp(fit = pmnlr, dl = d.long, trt_level = 1)

# _ Steps 3-5: None had history of injection drug use at baseline (non-IDU) ####
results_nidu = pmnlr_g_comp(fit = pmnlr, dl = d.long, trt_level = 0)

# > Results ####
results = results_idu |> 
  dplyr::left_join(results_nidu, by = "t_out", suffix = c(".idu",".non_idu")) |>
  dplyr::mutate(
    rd.1 = r1.idu - r1.non_idu,
    rd.2 = r2.idu - r2.non_idu
  )
results |>
  dplyr::filter(t_out==26|t_out==52|t_out==104)

# _ Cumulative incidence curves
library(ggplot2)
library(patchwork)
figure2a = ggplot() + 
  geom_step(aes(x = t_out, y = r1.idu, linetype = "IDU", color = "HAART Initiation"), data = results) +
  geom_step(aes(x = t_out, y = r1.non_idu, linetype = "non-IDU", color = "HAART Initiation"), data = results) +
  scale_linetype_manual(name="Scenario:",breaks=c("IDU","non-IDU"), values = c("solid","dashed")) +
  scale_color_manual(breaks="HAART Initiation",values="black") +
  scale_y_continuous("Risk",limits = c(0,0.35)) +
  scale_x_continuous("Weeks", limits = c(0,104), breaks = c(0,26,52,78,104)) +
  ggtitle("A. HAART Initiation") +
  guides(color="none") +
  theme_classic() +
  theme(legend.position = "bottom",legend.box.spacing = unit(0, "pt"))
figure2b = ggplot() + 
  geom_step(aes(x = t_out, y = r2.idu, linetype = "IDU", color = "AIDS/Death"), data = results) +
  geom_step(aes(x = t_out, y = r2.non_idu, linetype = "non-IDU", color = "AIDS/Death"), data = results) +
  scale_linetype_manual(name="Scenario:",breaks=c("IDU","non-IDU"), values = c("solid","dashed")) +
  scale_color_manual(breaks="AIDS/Death",values="grey75") +
  scale_y_continuous("",limits = c(0,0.35)) +
  scale_x_continuous("Weeks", limits = c(0,104), breaks = c(0,26,52,78,104)) +
  ggtitle("B. AIDS/Death") +
  guides(color="none") + 
  theme_classic() +
  theme(legend.position = "bottom",legend.box.spacing = unit(0, "pt")) 
figure2c = ggplot() + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "darkgrey") +
  geom_step(aes(x = t_out, y = rd.1, color = "HAART Initiation"), data = results) +
  geom_step(aes(x = t_out, y = rd.2, color = "AIDS/Death"), data = results) +
  scale_color_manual(name="Event Type:",breaks=c("HAART Initiation","AIDS/Death"), values = c("black","grey75")) +
  scale_x_continuous("Weeks", limits = c(0,104), breaks = c(0,26,52,78,104)) +
  scale_y_continuous("Risk Difference",limits = c(-0.15,0.15), breaks = c(-0.15,-0.05,0.05,0.15)) +
  coord_flip() +
  ggtitle("C. Difference") +
  theme_classic() +
  theme(legend.position = "bottom",legend.box.spacing = unit(0, "pt"))
combined = (figure2a + figure2b & theme(legend.position = "bottom")) +
  plot_layout(guides = "collect")
figure2 = combined  + figure2c + plot_layout(ncol = 3, widths = c(1.5,1.5,1)) &
  theme(legend.position = "bottom",panel.grid.major = element_line(color="grey90"))
figure2 

# ggsave("analysis/04_results/multinomial_logit/figure2.png", plot = figure2, device = "png",
#1        width = 17, height = 8.5, units = "cm", dpi = 600)
