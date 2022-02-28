#####################################################################################################
#
# Python code for example in
# Re: Using numerical methods to design simulations: revisiting the balancing intercept
# Paul Zivich and Rachael Ross
#
# Objective: solve for intercepts in data generating models to achieve desired marginal distribution
#
#####################################################################################################

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import root, newton

np.random.seed(777743)

# Setup the baseline data
n = 10000000
d = pd.DataFrame()
d['X'] = np.random.normal(size=n)
d['C'] = 1
print("E[X]:     ", np.mean(d['X']))


########################################
# Solving for balancing intercept for A

# where model is Pr(A | X) = logit(\alpha_0 + alpha_coefs[0]*X)
desired_margin_a = 0.45        # Desired margin
alpha_coefs = [0.25]           # Model coefficients (besides intercept)
W = np.asarray(d[['C', 'X']])  # Covariates to include


def generate_pr_a(intercepts):
    """Function to calculate the probability of A given an intercept
    """
    alpha = np.asarray([intercepts[0]] + alpha_coefs)  # Takes intercept and puts together with specified coefs

    # Calculating the probability of A given the coefficients
    logit_pr_a = np.dot(W, alpha)            # log-odds of A
    prob_a = 1 / (1 + np.exp(-logit_pr_a))   # converting to probability of A
    return prob_a                            # Function returns array / vector of probabilities


def objective_function_a(intercepts):
    """Objective function to use with a root-finding algorithm to solve for the intercept that provides the desired
    marginal probability of A
    """
    prob_a = generate_pr_a(intercepts=intercepts)               # Calculate probability of A for given intercept
    marginal_pr_a = np.mean(prob_a)                             # Calculate the marginal probability of A
    difference_from_desired = marginal_pr_a - desired_margin_a  # Calculate difference between current and desired marg
    return difference_from_desired                              # Return the current difference for the intercept


# Root-finding procedure for Pr(A)
root_a = newton(objective_function_a,       # The objective function
                x0=np.asarray([0.]),        # Initial starting values for procedure
                tol=1e-12, maxiter=1000)     # Arguments for root-finding algorithm

# Examining results
print("alpha_0:  ", root_a)
print("Pr(A=1):  ", np.mean(generate_pr_a(root_a)))

########################################
# Solving for balancing intercept for M

# where model is Pr(M=1 | X) / Pr(M=0 | X) = ln(\beta_10 + beta_coefs[0][0]*A + beta_coefs[0][1]*X)
#                Pr(M=2 | X) / Pr(M=0 | X) = ln(\beta_20 + beta_coefs[1][0]*A + beta_coefs[1][1]*X)
desired_margin_m = np.array([0.5, 0.35, 0.15])        # Desired margins
beta_coefs = [[1.2, -0.15],                           # Coefficients for M=1 vs. M=0 besides intercept
              [0.65, -0.07]]                          # Coefficients for M=2 vs. M=0 besides intercept
d['A'] = np.random.binomial(n=1,                      # Generating values of A from model
                            p=generate_pr_a(root_a),  # Using previously numer. approx. of intercept
                            size=d.shape[0])          # size is number of obs
V = np.asarray(d[['C', 'A', 'X']])                    # Covariates to include in model


def generate_pr_m(intercepts):
    """Function to calculate the probability of M for each possible value of M given intercepts
    """
    beta_10 = np.asarray([intercepts[0]] + beta_coefs[0])  # Takes intercept and puts together with M=1 specified coefs
    beta_20 = np.asarray([intercepts[1]] + beta_coefs[1])  # Takes intercept and puts together with M=2 specified coefs

    # Calculating denominator for probability model
    denom = 1 + np.exp(np.dot(V, beta_10)) + np.exp(np.dot(V, beta_20))

    # Calculating probability of M for each category via multinomial logit model
    prob_m = np.array([1 / denom,                             # Probability of M=0
                       np.exp(np.dot(V, beta_10)) / denom,    # Probability of M=1
                       np.exp(np.dot(V, beta_20)) / denom],   # Probability of M=2
                      )

    # Extra step to check if probability sums to 1 for each individual
    if not np.all(np.sum(prob_m, axis=0).round(7) == 1.):  # (rounding to avoid approximation errors)
        warnings.warn("Some Pr didn't sum to 1... :(",     # Warn user if fails to sum to 1 for any individual
                      UserWarning)

    return prob_m                                          # Function returns 2D array / vector of probabilities


def objective_function_m(intercepts):
    """Objective function to use with a root-finding algorithm to solve for the intercept that provides the desired
    marginal probabilities of M
    """
    prob_m = generate_pr_m(intercepts=intercepts)                # Calculate probability of A for given intercept
    marginal_pr_m = np.mean(prob_m, axis=1)                      # Calculate the marginal probability of M across types
    difference_from_desired = marginal_pr_m - desired_margin_m   # Calculate difference between current and desired marg
    return difference_from_desired[1:]                           # Return the current difference for all BUT M=0


opt_m = root(objective_function_m,      # The objective function
             x0=np.asarray([0., 0.]),   # Initial starting values for procedure (need 2 intercepts here!)
             method='lm', tol=1e-12)     # Arguments for root-finding algorithm

# Examining results
print("beta_0:   ", opt_m.x)
print("Pr(M):    ", np.mean(generate_pr_m(opt_m.x), axis=1))

########################################
# Solving for balancing intercept for Y

# where the model is Y = \gamma_0 + gamma_coefs[0]*A + gamma_coefs[1]*(M=1) + gamma_coefs[2]*(M=2)
#                        + gamma_coefs[3]*X + Normal(0, 3)
desired_margin_y = 10.                     # Desired margin
gamma_coefs = [-1.55, 0.25, 0.45, 0.25]    # Coefficients for Y model besides intercept


def random_multinomial(a, p):
    """Quick function to generate random values from input multinomial probabilities
    """
    s = p.cumsum(axis=0)
    r = np.random.rand(p.shape[1])
    k = (s < r).sum(axis=0)
    return np.asarray(a)[k]


d['M'] = random_multinomial(a=[0, 1, 2],                # Generating values of M from model
                            p=generate_pr_m(opt_m.x))   # Using previously numer. approx. of intercept
d['M1'] = np.where(d['M'] == 1, 1, 0)                   # Creating indicator variables (for ease)
d['M2'] = np.where(d['M'] == 2, 1, 0)                   # Creating indicator variables (for ease)
Z = np.asarray(d[['C', 'A', 'M1', 'M2', 'X']])          # Covariates to include in model
error = np.random.normal(scale=3, size=d.shape[0])      # How error terms are simulated


def generate_y(intercepts):
    """Function to calculate the values of Y given an intercept
    """
    gamma = np.asarray([intercepts[0]] + gamma_coefs)    # Takes intercept and puts together with specified coefs

    # Calculating Y values given the coefficients
    y = np.dot(Z, gamma)  # notice that we ignore the error term here (since safely ignorable for approx. intercepts)
    return y              # Function returns array / vector of Y values


def objective_function_y(intercepts):
    """Objective function to use with a root-finding algorithm to solve for the intercept that provides the desired
    marginal probability of A
    """
    val_y = generate_y(intercepts=intercepts)                   # Calculate probability of A for given intercept
    marginal_mu_y = np.mean(val_y)                              # Calculate the marginal mean of Y
    difference_from_desired = marginal_mu_y - desired_margin_y  # Calculate difference between current and desired marg
    return difference_from_desired                              # Return the current difference for the intercept


# Root-finding procedure for Pr(A)
root_y = newton(objective_function_y,       # The objective function
                x0=np.asarray([0.]),        # Initial starting values for procedure
                tol=1e-12, maxiter=1000)     # Arguments for root-finding algorithm

# Examining results
print("gamma_0:  ", root_y)
print("E[Y]:     ", np.mean(generate_y(root_y) + error))
