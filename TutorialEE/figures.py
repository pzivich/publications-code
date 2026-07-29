######################################################################################################################
# Estimating Equation Essentials -- Python
#   Python code to generate the figures
#
# Paul Zivich (last edit: 2026/07/29)
######################################################################################################################

##############################################################
# Loading dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
from delicatessen import MEstimator
from delicatessen.derivative import approx_differentiation

##############################################################
# Loading data from table format

d1 = pd.DataFrame()
d1['Y'] = [0, ]*120 + [1, ]*80
d1['R'] = 1

d0 = pd.DataFrame()
d0['Y'] = [0, ]*15 + [1, ]*85
d0['R'] = 0
d0['V'] = 1

d = pd.concat([d1, d0], ignore_index=True)
r = np.asarray(d['R'])
y = np.asarray(d['Y'])


##############################################################
# Defining the estimating function

def psi(theta):
    mu, alpha = theta

    ef_mean = r * (y - alpha*mu)
    ef_sens = (1-r) * (y - alpha)

    return np.vstack([ef_mean, ef_sens])


##############################################################
# Figure 1: grid of values for root-finding

# Solving for grid of all combinations
resolution = 21
x = np.linspace(0., 1., resolution)
output = []
for v1 in reversed(x):
    output_row = []
    for v2 in x:
        efuncs = psi([v1, v2])
        es = np.sum(efuncs, axis=1)
        output_row.append(es[0]**2 + es[1]**2)
    output.append(output_row)

# Finding maximum error and the best point in the grid
error_max = np.max(output)
point = np.unravel_index(np.argmin(output), (resolution, resolution))

# Generating the corresponding heat-map
heatmap = heatmap(np.asarray(output) / error_max, vmin=0, vmax=1.0, cmap='inferno_r', linewidths=0.1, linecolor='gray')
plt.plot([point[1]+.5, ], [point[0]+.5, ], 'o', color='darkred')
plt.text(point[1]+.75, point[0]+0.1, r'$\hat{\theta}$', color='darkred', fontdict={"size": 16})

# secant root-finder path (found using print statements inside later psi() function call)
plt.plot([10.5, ], [10.5, ], 's', color='k')
plt.annotate("", xytext=(10.5, 10.5), xy=(10.5+7, 10.5-3), arrowprops=dict(arrowstyle="->"))
plt.annotate("", xytext=(17.5, 7.5), xy=(17.5+0, 7.5+2), arrowprops=dict(arrowstyle="->"))
plt.annotate("", xytext=(17.5, 9.5), xy=(17.5+0, 9.5+2), arrowprops=dict(arrowstyle="->"))
plt.plot([18.5, ], [2.5, ], 's', color='k')
plt.annotate("", xytext=(18.5, 2.5), xy=(18.5-1, 2.5+5), arrowprops=dict(arrowstyle="->"))

# Making plot look nice with labels and structure
plt.xlabel(r"Sensitivity, $\tilde{\alpha}$")
plt.ylabel(r"Mean, $\tilde{\mu}$")
plt.yticks([0.5, 5.5, 10.5, 15.5, 20.5], ["1", "0.75", "0.50", "0.25", "0"])
plt.xticks([0.5, 5.5, 10.5, 15.5, 20.5], ["0", "0.25", "0.50", "0.75", "1"])
plt.tight_layout()
plt.savefig("figure1_solver.png", format='png', dpi=300)
plt.close()


##############################################################
# Figure 2: slopes near theta

# Solving estimating equations with secant method
estr = MEstimator(psi, init=[0.5, 0.5])
estr.estimate(solver='newton')
estr.print_results(decimals=3)
mu_hat, alpha_hat = estr.theta

def eequation(theta):
    # Manually computing the estimating equation
    #   negative here is to make it the bread
    return -np.mean(psi(theta), axis=1)

# Computing the derivatives at theta
dpsi = approx_differentiation(estr.theta, eequation)
rho = np.mean(d['R'])

###############################
# Figure 2a: deriv mu
plt.subplot(221)
x = np.linspace(0.46, 0.48, 100)
z = []
for v in x:
    # print(v, alpha_hat)
    efuncs = psi([v, alpha_hat])
    es = np.sum(efuncs, axis=1)
    z.append(es[0])

plt.plot(x, z)
plt.axhline(linestyle=':', color='gray')
plt.ylim([-1, 1])
plt.yticks([-1, 0, 1])
plt.xlim(0.46, 0.48)
plt.xticks([0.46, 0.47, 0.48], [0.46, r"$\hat{\mu}$", 0.48])
plt.ylabel(r"$e_1$")

###############################
# Figure 2b: deriv alpha
plt.subplot(222)
x = np.linspace(0.80, 0.88, 100)
z = []
for v in x:
    # print(v, alpha_hat)
    efuncs = psi([mu_hat, v])
    es = np.sum(efuncs, axis=1)
    z.append(es[0])

plt.plot(x, z)
plt.axhline(linestyle=':', color='gray')
plt.ylim([-1, 1])
plt.yticks([-1, 0, 1])
plt.xlim(0.84, 0.86)
plt.xticks([0.84, 0.85, 0.86])
plt.xticks([0.84, 0.85, 0.86], [0.84, r"$\hat{\alpha}$", 0.86])

###############################
# Figure 2c: deriv mu
plt.subplot(223)
x = np.linspace(0.46, 0.48, 100)
z = []
for v in x:
    # print(v, alpha_hat)
    efuncs = psi([v, alpha_hat])
    es = np.sum(efuncs, axis=1)
    z.append(es[1])

plt.plot(x, z)
plt.axhline(linestyle=':', color='gray')
plt.ylim([-1, 1])
plt.yticks([-1, 0, 1])
plt.xlim(0.46, 0.48)
plt.xticks([0.46, 0.47, 0.48], [0.46, r"$\hat{\mu}$", 0.48])
plt.ylabel(r"$e_2$")

###############################
# Figure 2d: deriv alpha
plt.subplot(224)
x = np.linspace(0.84, 0.86, 100)
z = []
for v in x:
    efuncs = psi([mu_hat, v])
    es = np.sum(efuncs, axis=1)
    z.append(es[1])

plt.plot(x, z)
plt.axhline(linestyle=':', color='gray')
plt.ylim([-1, 1])
plt.yticks([-1, 0, 1])
plt.xlim(0.84, 0.86)
plt.xticks([0.84, 0.85, 0.86], [0.84, r"$\hat{\alpha}$", 0.86])
plt.tight_layout()
plt.savefig("figure2.png", format='png', dpi=300)
plt.close()
