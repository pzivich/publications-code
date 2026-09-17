######################################################################################################################
# Estimating Equation Essentials -- Python
#   Python code to replicate the example described in the text
#
######################################################################################################################

##############################################################
# Loading dependencies

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import root
from scipy.optimize import approx_fprime
import delicatessen as deli
from delicatessen import MEstimator

print("versions")
print('NumPy:        ', np.__version__)
print('Pandas:       ', pd.__version__)
print('SciPy:        ', sp.__version__)
print('Delicatessen: ', deli.__version__, '\n')

##############################################################
# Loading data from table format

d1 = pd.DataFrame()
d1['Y'] = [0, ]*120 + [1, ]*80
d1['R'] = 1

d0 = pd.DataFrame()
d0['Y'] = [0, ]*15 + [1, ]*85
d0['R'] = 0
d0['V'] = 1

# Formatting data for the estimating functions
d = pd.concat([d1, d0], ignore_index=True)    # Stacking data from sources
r = np.asarray(d['R'])                        # Indicator for study
y = np.asarray(d['Y'])                        # Mismeasured version of Y
n = d.shape[0]                                # Number of observations

############################################
# By-hand Calculation

mu_star = np.mean(d.loc[d['R'] == 1, 'Y*'])    # Manually calculate the naive proportion
alpha = np.mean(d.loc[d['R'] == 0, 'Y*'])      # Manually calculate the sensitivity
mu = mu_star / alpha                           # Manually calculate the corrected proportion
print("By-Hand")
print("Naive Proportion:    ", mu_star)
print("Sensitivity:         ", alpha)
print("Corrected Proportion:", mu)

############################################
# Defining estimating equation

# Pulling out needed variables from data set
ystar = np.asarray(d['Y*'])
r = np.asarray(d['R'])

def estimating_function(theta):
    mu_tilde, alpha_tilde = theta

    # Parameter-specific estimating functions
    ef_mean = r * (ystar - alpha_tilde*mu_tilde)
    ef_sens = (1-r) * (ystar - alpha_tilde)

    # Stacking the estimating functions together into vectors
    return np.vstack([ef_mean, ef_sens])


def estimating_equation(theta):
    # Summing together the estimating functions into the estimating equation
    estf = np.asarray(estimating_function(theta))  # Return estimating function
    return np.sum(estf, axis=1)                    # Sum over all estimating functions


############################################
# Root-finding

proc = root(estimating_equation,          # Function to find root(s) of
            x0=np.array([0.5, 0.5]),      # ... starting values for root-finding procedure
            method='lm')                  # ... algorithm to use (Levenberg-Marquardt here)
theta_root = proc["x"]                    # Extract the parameter estimates from root-finder object

############################################
# Baking the bread (approximate derivative)

deriv = approx_fprime(xk=proc["x"],            # Array of values to compute derivative at (root of estimating equation)
                      f=estimating_equation,   # ... function to find derivative of
                      epsilon=1e-9)            # ... distance of points for numerical approximation (should be small)
bread = -1*deriv / n

############################################
# Cooking the filling (matrix algebra)

filling = np.dot(estimating_function(theta_root), estimating_function(theta_root).T)
filling = filling / n

############################################
# Assembling the sandwich (matrix algebra)

bread_inv = np.linalg.inv(bread)
sandwich = np.dot(np.dot(bread_inv, filling), bread_inv.T) / n
se = np.sqrt(np.diag(sandwich))

print("Correct Proportion -- EE")
print("Proportion:", theta_root[0])
print("SE:        ", np.round(se[0], 5))
print("95% CI:    ", np.round([theta_root[0] - 1.96*se[0],
                               theta_root[0] + 1.96*se[0]],
                              3))

####################################################################################################################
# Using delicatessen instead of by-hand calculations

mestr = MEstimator(estimating_function, init=[0.5, 0.5])
mestr.estimate(solver='lm')

print("Corrected Proportion -- delicatessen")
print("theta: ", mestr.theta)
print("95% CI:", mestr.confidence_intervals())

# END
