###########################################################################
# Practical Implementation of g-computation for marginal structural models
#
# Developed by: Bonnie Shook-Sa (05.09.24)
###########################################################################

library(geex)

##############################
####### loading data #########
##############################
actg <- as.data.frame(read.csv("actg.csv"))

##############################
##### derive variables #######
##############################
# table(actg$karnof)
actg$karnof90<-ifelse(actg$karnof==90,1,0)
actg$karnof100<-ifelse(actg$karnof==100,1,0)

###########################################################
################Estimation- IPW ###########################
###########################################################

#propensity model
prop.model <- glm(treat ~ male + idu + white + karnof90 + karnof100 + agec + age_rs1 + age_rs2 + age_rs3 + cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3, family = binomial(), data = actg)
actg$PS<-predict(prop.model,actg, type="response")
actg$IPTW<-ifelse(actg$treat==1,actg$PS^(-1),(1-actg$PS)^(-1))

#fit IPW MSM
MSM.IPTW <- glm(cd4_20wk ~ treat + male + treat*male, weights=IPTW, data = actg)

# specify the estimating functions
estfun_IPW <- function(data,models){   
  A<-data$treat
  V<-data$male
  Y<-data$cd4_20wk
  
  #propensity model
  Xe <- grab_design_matrix(data = data, rhs_formula = grab_fixed_formula(models$e))
  e_pos <- 1:ncol(Xe)
  e_scores <- grab_psiFUN(models$e, data)
  
  #MSM design matrix
  Xmsm <- grab_design_matrix(data = data, rhs_formula = grab_fixed_formula(models$m))

  function(theta){
    e <- plogis(Xe %*% theta[e_pos])                # estimated propensity score
    W<-A*e^(-1)+(1-A)*(1-e)^(-1)                    # estimated IPTW
    mu <- Xmsm %*% theta[(ncol(Xe)+1):(ncol(Xe)+4)] # estimated M*beta
 
    c(e_scores(theta[e_pos]), # scores from propensity model
      c(W*(Y-mu),             # scores from MSM
        W*A*(Y-mu),
        W*V*(Y-mu),
        W*A*V*(Y-mu)))
  }
}

#estimate the variance using geex
geex_IPW <- function(data, propensity_formula, MSM_formula, coef.MSM){
  e_model  <- glm(propensity_formula, data = data, family =binomial)
  msm_model  <- glm(MSM_formula, data = data)
  models <- list(e = e_model, m=msm_model)
  
  geex_resultsIPW <-m_estimate(
    estFUN = estfun_IPW, 
    data   = data, 
    root_control = setup_root_control(start = c(coef(e_model),coef.MSM)),
    outer_args = list(models = models))
  return(geex_resultsIPW)
}

#format output and compute CIs
IPW<-geex_IPW(actg, treat ~ male + idu + white + karnof90 + karnof100 + agec + age_rs1 + age_rs2 + age_rs3 + cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3, cd4_20wk ~ treat + male + treat*male, coef(MSM.IPTW))

MSM.parms<-c("int","treat","male","treat*male")
IPW.MSM.ests<-IPW@estimates[(length(IPW@estimates)-3):length(IPW@estimates)]
se.all<-sqrt(diag(IPW@vcov))
IPW.MSM.se<-se.all[(length(IPW@estimates)-3):length(IPW@estimates)]
IPW.ests<-as.data.frame(cbind(as.vector(MSM.parms),as.vector(IPW.MSM.ests),as.vector(IPW.MSM.se)))
colnames(IPW.ests)<-c('parameter', 'est','se')
IPW.ests$LCL<-as.numeric(IPW.ests$est)-1.96*as.numeric(IPW.ests$se)
IPW.ests$UCL<-as.numeric(IPW.ests$est)+1.96*as.numeric(IPW.ests$se)

write.csv(IPW.ests,"IPW.ests.csv")


##########################################################
################Estimation- GF ###########################
##########################################################

# specify the estimating functions
estfun_GF <- function(data,models){   
  V<-data$male

  # score equations for outcome model
  Xm <- grab_design_matrix(data = data, rhs_formula = grab_fixed_formula(models$m))
  m_scores <- grab_psiFUN(models$m, data)
  m_pos <- 1:ncol(Xm)
  
  # create design matrices for all treated and all untreated
  data0 <- data
  data0$treat <- rep(0,nrow(data0))
  Xm0 <- grab_design_matrix(data = data0,rhs_formula = grab_fixed_formula(models$m))
  
  data1 <- data
  data1$treat <- rep(1,nrow(data1))
  Xm1 <- grab_design_matrix(data = data1,rhs_formula = grab_fixed_formula(models$m))
  
  # design matrices for MSM
  Xmsm0 <- grab_design_matrix(data = data0,rhs_formula = grab_fixed_formula(models$msm))
  Xmsm1 <- grab_design_matrix(data = data1,rhs_formula = grab_fixed_formula(models$msm))

  function(theta){
    # estimated potential outcomes
    Y0hat <- Xm0 %*% theta[m_pos]
    Y1hat <- Xm1 %*% theta[m_pos]
    
    # Ma*betas
    mu0 <- Xmsm0 %*% theta[(ncol(Xm)+1):(ncol(Xm)+ncol(Xmsm0))]
    mu1 <- Xmsm1 %*% theta[(ncol(Xm)+1):(ncol(Xm)+ncol(Xmsm0))]
    
    c(m_scores(theta[m_pos]),        # score equations from outcome model
      c((Y1hat-mu1)+(Y0hat-mu0),     # score equations from MSM, intercept
        (Y1hat-mu1),                 # score equations from MSM, A
        V*(Y1hat-mu1)+V*(Y0hat-mu0), # score equations from MSM, V
        V*(Y1hat-mu1)))              # score equations from MSM, A*V
  }
}

#estimate the variance using geex
geex_GF <- function(data, outcome_formula, MSM_formula, coef.MSM){
  
  m_model  <- glm(outcome_formula, data = data)
  msm_model  <- glm(MSM_formula, data = data)
  models <- list(m = m_model, msm=msm_model)
  
  geex_resultsGF <-m_estimate(
    estFUN = estfun_GF, 
    data   = data, 
    root_control = setup_root_control(start = c(coef(m_model),coef.MSM)),
    outer_args = list(models = models))
  return(geex_resultsGF)
}

#format output and compute CIs
GF <- geex_GF(actg, cd4_20wk ~ treat + male + treat*male + idu + white + karnof90 + karnof100 + agec + age_rs1 + age_rs2 + age_rs3 + cd4c_0wk + cd4_rs1 + cd4_rs2 + cd4_rs3 + male*cd4c_0wk + male*cd4_rs1 + male*cd4_rs2 + male*cd4_rs3, cd4_20wk ~ treat + male + treat*male, c(0,0,0,0))

GF.MSM.ests <- GF@estimates[(length(GF@estimates)-3):length(GF@estimates)]
se.all.GF <- sqrt(diag(GF@vcov))
GF.MSM.se <- se.all.GF[(length(GF@estimates)-3):length(GF@estimates)]
GF.ests <- as.data.frame(cbind(as.vector(MSM.parms),as.vector(GF.MSM.ests),as.vector(GF.MSM.se)))
colnames(GF.ests)<-c('parameter', 'est','se')
GF.ests$LCL <- as.numeric(GF.ests$est)-1.96*as.numeric(GF.ests$se)
GF.ests$UCL <- as.numeric(GF.ests$est)+1.96*as.numeric(GF.ests$se)

write.csv(GF.ests,"GF.ests.csv")
