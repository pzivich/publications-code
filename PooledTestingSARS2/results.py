import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib import colorbar
from matplotlib import rc

from test_efficiency import overall_diagnostic_calculator, optimal_poolsize

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=True)


def generate_plot(prevalence_symptoms, ax, vmax=4., color='jet'):
    """Wrapper function that goes through a range of values to generate a heatmap of the corresponding efficiencies
    comparing a non-differentiated to a differentiated pooling strategy.

    Parameters
    ----------
    prevalence_symptoms : float
        Overall prevalence of symptoms
    ax : matplotlib axes object
        matplotlib axes on which to put the plot (useful for the multi-panel plotting functionality)
    vmax : int, float, optional
        Max value for the efficiency scale
    color : str
        Colormap to use for the heatmap. Default is jet.

    Returns
    -------
    None
    """
    # Tell function the global parameters (so doesn't look locally for them)
    global dimensiony, dimensionx, prevalences_asymp, prevalences_symp, sensitivity, specificity

    # Blank matrix to store results
    abl = np.empty((dimensiony, dimensionx))      # Generate empty matrix
    abl[:] = np.nan                               # set as nan
    rel = np.empty((dimensiony, dimensionx))      # Generate empty matrix
    rel[:] = np.nan                               # set as nan
    res = np.empty((dimensiony, dimensionx))      # Generate empty matrix
    res[:] = np.nan                               # set as nan

    # Calculate the efficiency for each cell
    for j in range(len(prevalences_asymp)):     # Go through the indices of prevalence among asymptomatic

        # Efficiency among asymptomatic
        asymp_eff = optimal_poolsize(max_pool_size=25,                  # Calculate efficiency
                                     mad_value=0.20,                    # ... with pre-defined max pool size and MAD
                                     sensitivity=sensitivity[1],        # ... at the corresponding sensitivity
                                     specificity=specificity[1],        # ... and corresponding specificity
                                     prevalence=prevalences_asymp[j])   # ... for current asymptomatic prevalence

        for k in range(len(prevalences_symp)):  # Go through the indices of prevalence among symptomatic
            if prevalences_asymp[j] > prevalences_symp[k]:              # When asymptomatic prev is greater than symp
                abl[j, k] = np.nan                                      # ... set all efficiency diffs as missing
            else:                                                       # Otherwise continue with calculations

                # Efficiency among symptomatic
                symp_eff = optimal_poolsize(max_pool_size=25,                # Calculate the efficiency
                                            mad_value=0.20,                  # ... for given max pool and MAD
                                            sensitivity=sensitivity[0],      # ... at the corresponding sensitivity
                                            specificity=specificity[0],      # ... and corresponding specificity
                                            prevalence=prevalences_symp[k])  # ... for current symptomatic prevalence

                # Efficiency for non-differentiated testing
                overall_prevalence = (prevalences_symp[k]*prevalence_symptoms +     # Calculate overall prevalence of S
                                      prevalences_asymp[j]*(1-prevalence_symptoms))
                # Calculate overall sensitivity and specificity using a function
                overall_sensitivity, overall_specificity = overall_diagnostic_calculator(sensitivity=sensitivity,
                                                                                         specificity=specificity,
                                                                                         pr_s=prevalence_symptoms,
                                                                                         pr_d_s=prevalences_symp[k],
                                                                                         pr_d_ns=prevalences_asymp[j])
                # Calculate the overall efficiency for the non-differentiated pooled testing
                overall_eff = optimal_poolsize(max_pool_size=25,                   # Efficiency with same max pool size
                                               mad_value=0.20,                     # ... and MAD value
                                               sensitivity=overall_sensitivity,    # ... with overall sensitivity
                                               specificity=overall_specificity,    # ... and overall specificity
                                               prevalence=overall_prevalence)      # ... and overall prevalence

                # Calculating measures that capture differences in efficiency
                diff_eff = asymp_eff[1] * (1 - prevalence_symptoms) + symp_eff[1] * prevalence_symptoms
                if (overall_eff[1] - diff_eff) >= 0:               # If efficiency is better, save as that value
                    abl[j, k] = (overall_eff[1] - diff_eff)
                else:                                              # Otherwise, set as NaN to 'signify' in the heatmap
                    abl[j, k] = np.nan

    # Plotting the heatmap of the results
    res = sbn.heatmap(np.flipud(abl),            # Heatmap from seaborn and flipping the orientation of the results
                      ax=ax,                     # ... plotted on a specific ax
                      vmax=vmax, vmin=0,         # ... with a specified max and min of zero
                      cmap=color,                # ... with a colormap
                      xticklabels=False,         # ... and hiding x ticks
                      yticklabels=False,         # ... hiding y ticks
                      cbar=False)                # ... and suppress colorbar for manual addition later

    # Some plot editing
    for _, spine in res.spines.items():          # Going through the splines in the result
        spine.set_visible(True)                  # ... and setting them to be visible
        spine.set_linewidth(1.05)                # ... with a width of 1.05


###################################################################################################################
# Figures 1-4
###################################################################################################################

# Setting overall parameters for the dimensions and the range of sensitivity / specificity values
dimensiony = 25
dimensionx = 3*dimensiony
prevalences_asymp = np.linspace(0.01, 0.20, dimensiony)
prevalences_symp = np.linspace(0.01, 0.60, dimensionx)

################
# Figure 1
fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(22, 33)
f1 = fig.add_subplot(gs[:10, :30])
f2 = fig.add_subplot(gs[12:, :30])
f3 = fig.add_subplot(gs[:, 32:])

sensitivity = [0.75, 0.50]
specificity = [0.95, 0.95]
generate_plot(prevalence_symptoms=0.01, ax=f1)
f1.text(-9, -1, "A)", size=14)
f1.set_yticks([0, dimensiony/2, dimensiony])
f1.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f1.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f1.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f1.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")

sensitivity = [0.85, 0.60]
specificity = [0.99, 0.99]
generate_plot(prevalence_symptoms=0.01, ax=f2)
f2.text(-9, -1, "B)", size=14)
f2.set_yticks([0, dimensiony/2, dimensiony])
f2.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f2.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f2.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f2.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")
f2.set_xlabel(r"Probability of Infection "
              "\n"
              r"Given Symptoms/Exposure")

cbar = colorbar.ColorbarBase(f3, cmap=plt.get_cmap("gray"),
                             orientation="vertical")
cbar.set_ticks([0, 1/4, 2/4, 3/4, 1])
cbar.set_ticklabels([0, 1, 2, 3, 4])
cbar.ax.yaxis.set_label_position("left")
cbar.ax.set_ylabel(r"Efficiency Gain")
plt.show()


################
# Figure 2
fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(22, 33)
f1 = fig.add_subplot(gs[:10, :30])
f2 = fig.add_subplot(gs[12:, :30])
f3 = fig.add_subplot(gs[:, 32:])

sensitivity = [0.75, 0.50]
specificity = [0.95, 0.95]
generate_plot(prevalence_symptoms=0.1, ax=f1)
f1.text(-9, -1, "A)", size=14)
f1.set_yticks([0, dimensiony/2, dimensiony])
f1.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f1.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f1.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f1.set_ylabel(r"Probability of infection"
              "\n"
              r"given no symptoms/exposure")

sensitivity = [0.85, 0.60]
specificity = [0.99, 0.99]
generate_plot(prevalence_symptoms=0.1, ax=f2)
f2.text(-9, -1, "B)", size=14)
f2.set_yticks([0, dimensiony/2, dimensiony])
f2.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f2.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f2.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f2.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")
f2.set_xlabel(r"Probability of Infection "
              "\n"
              r"Given Symptoms/Exposure")

cbar = colorbar.ColorbarBase(f3, cmap=plt.get_cmap("gray"),
                             orientation="vertical")
cbar.set_ticks([0, 1/4, 2/4, 3/4, 1])
cbar.set_ticklabels([0, 1, 2, 3, 4])
cbar.ax.yaxis.set_label_position("left")
cbar.ax.set_ylabel(r"Efficiency Gain")
plt.show()

################
# Figure 3
fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(22, 33)
f1 = fig.add_subplot(gs[:10, :30])
f2 = fig.add_subplot(gs[12:, :30])
f3 = fig.add_subplot(gs[:, 32:])

sensitivity = [0.75, 0.50]
specificity = [0.95, 0.95]
generate_plot(prevalence_symptoms=0.5, ax=f1)
f1.text(-9, -1, "A)", size=14)
f1.set_yticks([0, dimensiony/2, dimensiony])
f1.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f1.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f1.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f1.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")

sensitivity = [0.85, 0.60]
specificity = [0.99, 0.99]
generate_plot(prevalence_symptoms=0.5, ax=f2)
f2.text(-9, -1, "B)", size=14)
f2.set_yticks([0, dimensiony/2, dimensiony])
f2.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f2.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f2.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f2.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")
f2.set_xlabel(r"Probability of Infection "
              "\n"
              r"Given Symptoms/Exposure")

cbar = colorbar.ColorbarBase(f3, cmap=plt.get_cmap("gray"),
                             orientation="vertical")
cbar.set_ticks([0, 1/4, 2/4, 3/4, 1])
cbar.set_ticklabels([0, 1, 2, 3, 4])
cbar.ax.yaxis.set_label_position("left")
cbar.ax.set_ylabel(r"Efficiency Gain")
plt.show()

################
# Figure 4
fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(22, 33)
f1 = fig.add_subplot(gs[:10, :30])
f2 = fig.add_subplot(gs[12:, :30])
f3 = fig.add_subplot(gs[:, 32:])

sensitivity = [0.75, 0.50]
specificity = [0.95, 0.95]
generate_plot(prevalence_symptoms=0.75, ax=f1)
f1.text(-9, -1, "A)", size=14)
f1.set_yticks([0, dimensiony/2, dimensiony])
f1.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f1.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f1.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f1.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")

sensitivity = [0.85, 0.60]
specificity = [0.99, 0.99]
generate_plot(prevalence_symptoms=0.75, ax=f2)
f2.text(-9, -1, "B)", size=14)
f2.set_yticks([0, dimensiony/2, dimensiony])
f2.set_yticklabels(["0.2", "0.1", "0.0"], fontsize=10)
f2.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f2.set_xticklabels(["0.00", "0.15", "0.30", "0.45", "0.60"], fontsize=10)
f2.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")
f2.set_xlabel(r"Probability of Infection "
              "\n"
              r"Given Symptoms/Exposure")

cbar = colorbar.ColorbarBase(f3, cmap=plt.get_cmap("gray"),
                             orientation="vertical")
cbar.set_ticks([0, 1/4, 2/4, 3/4, 1])
cbar.set_ticklabels([0, 1, 2, 3, 4])
cbar.ax.yaxis.set_label_position("left")
cbar.ax.set_ylabel(r"Efficiency Gain")
plt.show()

###################################################################################################################
# Figure 5: university setting
###################################################################################################################

prevalences_asymp = np.linspace(0.009, 0.05, dimensiony)
prevalences_symp = np.linspace(0.05, 0.20, dimensionx)

fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(22, 33)
f1 = fig.add_subplot(gs[:10, :30])
f2 = fig.add_subplot(gs[12:, :30])
f3 = fig.add_subplot(gs[:, 32:])

sensitivity = [0.75, 0.50]
specificity = [0.95, 0.95]
generate_plot(prevalence_symptoms=0.15, ax=f1)
f1.text(-9, -1, "A)", size=14)
f1.set_yticks([0, dimensiony])
f1.set_yticklabels(["0.05", "0.0"], fontsize=10)
f1.set_xticks([0, dimensionx/4, dimensionx/2, dimensionx*0.75, dimensionx])
f1.set_xticklabels(["0.00", "0.05", "0.10", "0.15", "0.20"], fontsize=10)
f1.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")

sensitivity = [0.85, 0.60]
specificity = [0.99, 0.99]
generate_plot(prevalence_symptoms=0.15, ax=f2)
f2.text(-9, -1, "B)", size=14)
f2.set_yticks([0, dimensiony])
f2.set_yticklabels(["0.05", "0.0"], fontsize=10)
f2.set_xticks([0, dimensionx/2, dimensionx])
f2.set_xticklabels(["0.00", "0.05", "0.10"], fontsize=10)
f2.set_ylabel(r"Probability of Infection"
              "\n"
              r"Given No Symptoms/Exposure")
f2.set_xlabel(r"Probability of Infection "
              "\n"
              r"Given Symptoms/Exposure")

cbar = colorbar.ColorbarBase(f3, cmap=plt.get_cmap("gray"),
                             orientation="vertical")
cbar.set_ticks([0, 1/4, 2/4, 3/4, 1])
cbar.set_ticklabels([0, 1, 2, 3, 4])
cbar.ax.yaxis.set_label_position("left")
cbar.ax.set_ylabel(r"Efficiency Gain")
plt.show()
