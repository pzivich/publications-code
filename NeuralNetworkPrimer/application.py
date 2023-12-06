#######################################################################################################################
# A Primer on Neural Networks
#
# Paul Zivich (2023/11/30)
#######################################################################################################################

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_glm

from activations import identity, leaky_relu


#######################################################################################################################
# Reading in data

d = pd.read_csv("nhanes.csv")
d['intercept'] = 1

outcome = np.asarray(d['hdl'])
predictors = np.asarray(d[['age', 'female', 'weight', 'height']])

#######################################################################################################################
# Defining loss, objective functions


def mse(y, yhat):
    return np.sum((y - yhat)**2)


def objective(params, nn, loss_function, y):
    yhat = nn(params)
    return loss_function(y, yhat)


#######################################################################################################################
# One-layer neural network

def nn_one_layer(params):
    # Parameters for neural network
    intercept, coefficients = params[0], params[1:]

    # Output layer
    ws = np.dot(predictors, coefficients) + intercept   # Weighted sum of inputs and predictors
    yhat = identity(ws)                                 # Activation function (identity)

    # Returning the generated predictions from the neural network
    return yhat


# Training the neural network
starting_vals = np.array([0, 0, 0, 0, 0])             # Starting values for the optimization
opt = minimize(objective,                             # Objective (loss) function to minimize
               x0=starting_vals,                      # ... with provided starting values
               args=(nn_one_layer, mse, outcome),     # ... extra arguments passed to `objective`
               method='nelder-mead',                  # ... using the Nelder-Mead algorithm
               tol=1e-9,                              # ... setting maximum error tolerance till convergence
               options={"maxiter": 100000})           # ... increasing iterations to large number
# print(opt.message)                                  # Check that neural network found a minimum
est_params_nn1 = opt.x                                # Return estimated parameters for neural network
yhat_nn1 = nn_one_layer(est_params_nn1)               # Generating predictions for observations from neural network
print("Loss:", mse(outcome, yhat_nn1))
print("Neural Network Parameters")
print(est_params_nn1)

# Comparing neural network parameters to GLM
def psi(theta):
    return ee_glm(theta=theta, X=X, y=outcome,
                  distribution='normal',
                  link='identity')


X = d[['intercept', 'age', 'female', 'weight', 'height']]
estr = MEstimator(psi, init=starting_vals)            # Setup M-estimator
estr.estimate()                                       # Estimate parameters via root-finding
est_params_glm = estr.theta                           # Return estimated parameters by GLM
print("GLM Parameters")
print(est_params_glm)


#######################################################################################################################
# Two-layer neural network


def nn_two_layer(params):
    # Parameters for neural network
    intercept_h1_n1, coef_h1_n1 = params[0], params[1:5]
    intercept_h1_n2, coef_h1_n2 = params[5], params[6:10]
    intercept_out, coef_out = params[10], params[10:]

    # Hidden layer
    hn1 = np.dot(predictors, coef_h1_n1) + intercept_h1_n1  # Weighted sum of inputs and predictors for hidden node 1
    hn1 = leaky_relu(hn1, scale=0.01)                       # Activation function for hidden node 1
    hn2 = np.dot(predictors, coef_h1_n2) + intercept_h1_n2  # Weighted sum of inputs and predictors for hidden node 2
    hn2 = leaky_relu(hn2, scale=0.01)                       # Activation function for hidden node 2

    # Output layer
    ws = intercept_out + coef_out[0]*hn1 + coef_out[1]*hn2  # Weighted sum of inputs and predictors for output node
    output = identity(ws)                                   # Activation function for output node (identity)

    # Returning the generated predictions from the neural network
    return output


# Training the neural network
rng = np.random.RandomState(30032023)                 # Random state for pseudo-random number generator
starting_vals = rng.uniform(-5, 5, size=13)           # Starting values for optimization (random uniform over -5 to 5)
opt = minimize(objective,                             # Objective (loss) function to minimize
               x0=starting_vals,                      # ... with provided starting values
               args=(nn_two_layer, mse, outcome),     # ... extra arguments passed to `objective`
               method='nelder-mead',                  # ... using the Nelder-Mead algorithm
               tol=1e-9,                              # ... setting maximum error tolerance till convergence
               options={"maxiter": 100000})           # ... increasing iterations to large number

# print(opt.message)                                  # Check that neural network found a minimum
est_params_nn2 = opt.x                                # Return estimated parameters for neural network
yhat_nn2 = nn_two_layer(est_params_nn2)               # Generating predictions for observations from neural network

# Training the neural network, but with different starting values
rng = np.random.RandomState(30041993)                 # Random state for pseudo-random number generator
starting_vals = rng.uniform(-5, 5, size=13)           # Starting values for optimization (random uniform over -5 to 5)
opt = minimize(objective,                             # Objective (loss) function to minimize
               x0=starting_vals,                      # ... with provided starting values
               args=(nn_two_layer, mse, outcome),     # ... extra arguments passed to `objective`
               method='nelder-mead',                  # ... using the Nelder-Mead algorithm
               tol=1e-9,                              # ... setting maximum error tolerance till convergence
               options={"maxiter": 100000})           # ... increasing iterations to large number
# print(opt.message)                                  # Check that neural network found a minimum
print("Loss:", mse(outcome, yhat_nn2))
print("Loss:", mse(outcome, nn_two_layer(opt.x)))


#######################################################################################################################
# Three-layer neural network


def nn_three_layer(params):
    # Parameters for neural network
    intercept_h1_n1, coef_h1_n1 = params[0], params[1:5]
    intercept_h1_n2, coef_h1_n2 = params[5], params[6:8]
    intercept_h1_n3, coef_h1_n3 = params[8], params[9:11]
    intercept_h2_n1, coef_h2_n1 = params[11], params[12:14]
    intercept_h2_n2, coef_h2_n2 = params[14], params[15:17]
    intercept_out, coef_out = params[17], params[18:]

    # Hidden layer
    h1n1 = np.dot(predictors, coef_h1_n1) + intercept_h1_n1          # Weighted sum for hidden node 1
    h1n1 = leaky_relu(h1n1, scale=0.01)                              # Activation function for hidden node 1
    h1n2 = np.dot(predictors[:, 1:3], coef_h1_n2) + intercept_h1_n2  # Weighted sum for hidden node 2
    h1n2 = leaky_relu(h1n2, scale=0.01)                              # Activation function for hidden node 2
    h1n3 = np.dot(predictors[:, 2:], coef_h1_n3) + intercept_h1_n3   # Weighted sum for hidden node 3
    h1n3 = leaky_relu(h1n3, scale=0.01)                              # Activation function for hidden node 3

    # Hidden layer
    h2n1 = intercept_h2_n1 + coef_h2_n1[0]*h1n1 + coef_h2_n1[1]*h1n2  # Weighted sum for hidden node 1
    h2n1 = leaky_relu(h2n1, scale=0.01)                               # Activation function for hidden node 1
    h2n2 = intercept_h2_n2 + coef_h2_n2[0]*h1n2 + coef_h2_n2[1]*h1n3  # Weighted sum for hidden node 2
    h2n2 = leaky_relu(h2n2, scale=0.01)                               # Activation function for hidden node 2

    # Output layer
    ws = intercept_out + coef_out[0]*h2n1 + coef_out[1]*h2n2  # Weighted sum of inputs and predictors for output node
    output = identity(ws)                                     # Activation function for output node (identity)

    # Returning the generated predictions from the neural network
    return output


# Training the neural network
rng = np.random.RandomState(30032023)                 # Random state for pseudo-random number generator
starting_vals = rng.uniform(-5, 5, size=20)           # Starting values for optimization (random uniform over -5 to 5)
opt = minimize(objective,                             # Objective (loss) function to minimize
               x0=starting_vals,                      # ... with provided starting values
               args=(nn_three_layer, mse, outcome),   # ... extra arguments passed to `objective`
               method='nelder-mead',                  # ... using the Nelder-Mead algorithm
               tol=1e-9,                              # ... setting maximum error tolerance till convergence
               options={"maxiter": 100000})           # ... increasing iterations to large number
# print(opt.message)                                  # Check that neural network found a minimum
est_params_nn3 = opt.x                                # Return estimated parameters for neural network
yhat_nn3 = nn_three_layer(est_params_nn3)             # Generating predictions for observations from neural network
print("Loss:", mse(outcome, yhat_nn3))
