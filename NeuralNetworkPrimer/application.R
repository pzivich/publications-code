################################################################################
# A Primer on Neural Networks
# 
# Paul Zivich (2023/11/30)
################################################################################

library(nloptr)

################################################################################
# Reading in data

setwd('C:/Users/zivic/Documents/open-source/publications-code/NeuralNetworkPrimer')
d <- read.csv("nhanes.csv")

outcome = d$hdl
predictors = d[, c('age', 'female', 'weight', 'height')]


################################################################################
# Defining loss, activation, objective functions

mse <- function(y, yhat){
    # Mean squared error loss function
    return(sum((y - yhat)^2))
}

identity <- function(x){
    # Identity activation function
    return(x)
}

leaky_relu <- function(x, scale=0.01){
    # Leaky rectified linear unit
    xstar = ifelse(x < 0, scale*x, x)
    return(xstar)
}

objective <- function(params, nn, loss_function, y){
    # Generic objective function for neural networks
    yhat = nn(params)
    return(loss_function(y, yhat))
}

# Maximization algorithm arguments
nm_control <- list(maxeval=100000, xtol_rel=1e-9)

################################################################################
# One-layer neural network

nn_one_layer <- function(params){
    # Parameters for neural network
    intercept = params[1]
    coefficients = params[2:length(params)]

    # Output layer
    ws = as.matrix(predictors) %*% as.matrix(coefficients) + intercept
    yhat = identity(ws)

    # Returning the generated predictions from the neural network
    return(yhat)
}

# Training the neural network
starting_vals = c(0, 0, 0, 0, 0)                  # Starting values for optim
nn1_fit = nloptr::neldermead(starting_vals,       # Nelder-Mead algorithm
                             objective,           # ... with objective
                             nn=nn_one_layer,     # ... for neural net
                             loss_function=mse,   # ... with loss function
                             y=outcome,           # ... observed outcome
                             control=nm_control)  # ... specifications
nn1_fit$par                                       # Neural network parameters

# Loss function results
objective(nn1_fit$par, nn_two_layer, mse, outcome)

# Comparing to linear regression
flm = lm(hdl ~ age + female + weight + height, d)
flm$coefficients                                  # GLM parameters


################################################################################
# Two-layer neural network

nn_two_layer <- function(params){
    # Parameters for neural network
    intercept_h1_n1 = params[1]
    coeff_h1_n1 = params[2:5]
    intercept_h1_n2 = params[6]
    coeff_h1_n2 = params[7:10]
    intercept_out = params[11]
    coef_out = params[12:length(params)]
    
    # Hidden layer
    hn1 = as.matrix(predictors) %*% as.matrix(coeff_h1_n1) + intercept_h1_n1
    hn1 = leaky_relu(hn1, scale=0.01)
    hn2 = as.matrix(predictors) %*% as.matrix(coeff_h1_n2) + intercept_h1_n2
    hn2 = leaky_relu(hn2, scale=0.01)
    
    # Output layer
    ws = intercept_out + coef_out[1]*hn1 + coef_out[2]*hn2
    yhat = identity(ws)
    
    # Returning the generated predictions from the neural network
    return(yhat)
}

# Starting values from Python random draw
starting_vals = c(-4.85766562, -0.68064447, 4.78745109, 1.66116221, 2.39312546, 
                  3.37231983, 4.87319149, -3.08506736, -3.66220176, -3.75072766, 
                  -4.34286364, 2.32722006, 0.71739167)
nn2_fit = nloptr::neldermead(starting_vals,        # Nelder-Mead algorithm
                             objective,            # ... with objective
                             nn=nn_two_layer,      # ... for neural net
                             loss_function=mse,    # ... with loss function
                             y=outcome,            # ... observed outcome
                             control=nm_control)   #... specifications

# Loss function results
objective(nn2_fit$par, nn_two_layer, mse, outcome)  # 1049877

# Starting values from Python random draw
starting_vals = c(-0.12558097, 1.67103887, 2.79717236, -2.09890536, 
                  -2.81267949, -0.79644393, 3.23312734, 1.61378279, 
                  2.25609064, -3.16291761,  2.77545355, -2.84309898,
                  2.33620732)
nn2_fit = nloptr::neldermead(starting_vals,       # Nelder-Mead algorithm
                             objective,           # ... with objective
                             nn=nn_two_layer,     # ... for neural net
                             loss_function=mse,   # ... with loss function
                             y=outcome,           # ... observed outcome
                             control=nm_control)  # ... specifications

# Loss function results
objective(nn2_fit$par, nn_two_layer, mse, outcome)  # 1001958

################################################################################
# Three-layer neural network

nn_three_layer <- function(params){
    # Parameters for neural network
    intercept_h1_n1 = params[1]
    coeff_h1_n1 = params[2:5]
    intercept_h1_n2 = params[6]
    coeff_h1_n2 = params[7:8]
    intercept_h1_n3 = params[9]
    coeff_h1_n3 = params[10:11]
    intercept_h2_n1 = params[12]
    coeff_h2_n1 = params[13:14]
    intercept_h2_n2 = params[16]
    coeff_h2_n2 = params[16:17]
    intercetp_out = params[18]
    coef_out = params[19:length(params)]
    
    # Hidden layer
    h1n1 = as.matrix(predictors) %*% as.matrix(coeff_h1_n1) + intercept_h1_n1
    h1n1 = leaky_relu(h1n1, scale=0.01)
    h1n2 = as.matrix(predictors[,2:3]) %*% as.matrix(coeff_h1_n2) + intercept_h1_n2
    h1n2 = leaky_relu(h1n2, scale=0.01)
    h1n3 = as.matrix(predictors[,3:4]) %*% as.matrix(coeff_h1_n3) + intercept_h1_n3
    h1n3 = leaky_relu(h1n3, scale=0.01)
    
    # Hidden layer
    h2n1 = intercept_h2_n1 + coeff_h2_n1[1]*h1n1 + coeff_h2_n1[2]*h1n2
    h2n1 = leaky_relu(h2n1, scale=0.01)
    h2n2 = intercept_h2_n2 + coeff_h2_n2[1]*h1n2 + coeff_h2_n2[2]*h1n3
    h2n2 = leaky_relu(h2n2, scale=0.01)
    
    # Output layer
    ws = intercetp_out + coef_out[1]*h2n1 + coef_out[2]*h2n2
    yhat = identity(ws)
    
    # Returning the generated predictions from the neural network
    return(yhat)
}

# Starting values from Python random draw
starting_vals = c(-4.85766562, -0.68064447, 4.78745109, 1.66116221, 2.39312546,
                  3.37231983, 4.87319149, -3.08506736, -3.66220176, -3.75072766, 
                  -4.34286364, 2.32722006, 0.71739167, -2.07179524, -4.87551265, 
                  -0.76882816, 3.13232629, 3.46530322, 1.1681697, -2.12807223)
nn3_fit = nloptr::neldermead(starting_vals,       # Nelder-Mead algorithm
                             objective,           # ... with objective
                             nn=nn_three_layer,   # ... for neural net
                             loss_function=mse,   # ... with loss function
                             y=outcome,           # ... observed outcome
                             control=nm_control)  # ... specifications

# Loss function results
objective(nn3_fit$par, nn_three_layer, mse, outcome)
