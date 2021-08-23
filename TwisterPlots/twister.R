###############################################################################
# Twister Plots
#
# Alex Breskin (2021/5/14), Paul Zivich (2021/6/28)
##############################################################################

library(ggplot2)
library(rlang)

#### Step Ribbon ####
StatStepribbon <- ggproto("StatStepribbon",
                          Stat,
                          compute_group=function(., data, scales, direction = "hv",
                                                 yvars = c( "ymin", "ymax" ), ...)
                          {
                                  direction <- match.arg( direction, c( "hv", "vh" ) )
                                  data <- as.data.frame( data )[ order( data$x ), ]
                                  n <- nrow( data )
                                  
                                  if ( direction == "vh" ) {
                                          xs <- rep( 1:n, each = 2 )[ -2 * n ]
                                          ys <- c( 1, rep( 2:n, each = 2 ) )
                                  } else {
                                          ys <- rep( 1:n, each = 2 )[ -2 * n ]
                                          xs <- c( 1, rep( 2:n, each = 2))
                                  }
                                  
                                  data.frame(
                                          x = data$x[ xs ]
                                          , data[ ys, yvars, drop=FALSE ]
                                          , data[ xs, setdiff( names( data ), c( "x", yvars ) ), drop=FALSE ]
                                  )
                          },
                          required_aes=c( "x", "ymin", "ymax" ),
                          default_geom=GeomRibbon,
                          default_aes=aes( x=..x.., ymin = ..y.., ymax=Inf )
)

stat_stepribbon = function( mapping=NULL, data=NULL, geom="ribbon",
                            position="identity") {
        layer(stat=StatStepribbon, mapping=mapping, data=data, geom=geom, position=position )
}


#### Twister Plot####

#' Twister Plot
#'
#' @param dat A \code{data.frame} with the risk difference, upper and lower confidence limits, and times
#' @param xvar The variable name for the risk difference. Defaults to RD.
#' @param lcl  The variable name for the lower confidence limit of the risk difference. Defaults to RD_LCL.
#' @param ucl  The variable name for the upper confidence limit of the risk difference. Defaults to RD_UCL.
#' @param yvar The variable name for time. Defaults to "t".
#' @param xlab The x-axis label. Defaults to "Risk Difference".
#' @param ylab The y-axis label. Defaults to "Days".
#' @param treat_labs A vector containing the names of the treatment groups. Defaults to c("Treat", "Control")
#'
#' @return a \code{ggplot} object
#' @export
#'
#' @examples
#' twister_plot(dat, treat_labs = c("Vaccine", "Placebo"))
#' 
twister_plot <- function(dat,
                         xvar = RD,
                         lcl = RD_LCL,
                         ucl = RD_UCL,
                         yvar = t,
                         xlab = "Risk Difference",
                         ylab = "Days",
                         reference_line = 0.0,
                         log_scale = FALSE,
                         treat_labs = c("Vaccine", "Control")){
        
        base_breaks <- function(n = 10){
                function(x) {
                        axisTicks(log10(range(x, na.rm = TRUE)), log = TRUE, n = n)
                }
        }
        
        `%>%` <- magrittr::`%>%`
        pull <- dplyr::pull
        
        t_lim <- max(dat %>% pull({{yvar}}))
        if (log_scale) {
                x_lim <- max(abs(log(dat %>% pull({{lcl}}))), 
                              abs(log(dat %>% pull({{ucl}}))))
                y_scale = scale_y_continuous(limits = c(exp(-x_lim), exp(x_lim)), 
                                             trans="log", 
                                             breaks=base_breaks())
                text_loc = c(-x_lim/2, x_lim/2)
        } else {
                x_lim <- max(abs(dat %>% pull({{lcl}})), abs(dat %>% pull({{ucl}})))
                y_scale = scale_y_continuous(limits = c(-x_lim, x_lim))
                text_loc = c(-x_lim/2, x_lim/2)
        }
        
        p <- ggplot(data = dat, aes(x = {{yvar}}, y = {{xvar}})) + 
                geom_step() +
                geom_ribbon(
                        aes(ymin = {{lcl}}, ymax = {{ucl}}),
                        stat = "stepribbon",
                        alpha = 0.2,
                        direction = "hv"
                ) +
                geom_hline(yintercept = reference_line, linetype = "dotted") +
                y_scale + 
                scale_x_continuous(limits = c(0, t_lim), expand = c(0, 0)) +
                coord_flip(clip = "off") +
                theme(axis.line = element_line(colour = "black"),
                      panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank(),
                      panel.background = element_blank(),
                      panel.border = element_rect(colour = "black", fill = NA),
                      plot.margin = unit(c(2, 1, 1, 1), "lines")) +
                geom_text(data = head(dat, 1), 
                          label = sprintf("Favors %s", treat_labs[1]), 
                          x = t_lim+ 5, y = text_loc[1]) +
                geom_text(data = head(dat, 1), 
                          label = sprintf("Favors %s", treat_labs[2]), 
                          x = t_lim+ 5, y = text_loc[2]) +
                xlab(ylab) +
                ylab(xlab)
        p
}

#### Example Twister Plot ####

# Reading in data
setwd("/file/path/to/data/")
data <- read.csv("data_twister.csv")

# Risk Difference Twister Plot
twister_plot(dat = data,                          
             xvar = RD,
             lcl = RD_LCL,
             ucl = RD_UCL,
             yvar = t,
             treat_labs = c("Vaccine", "Control"))

# Risk Ratio Twister Plot
twister_plot(dat = data,                          
             xvar = RR,
             lcl = RR_LCL,
             ucl = RR_UCL,
             yvar = t,
             xlab = "Risk Ratio",
             reference_line = 1.0,
             log_scale = T,
             treat_labs = c("Vaccine", "Control"))
