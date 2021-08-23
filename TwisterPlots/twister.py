########################################################################################################################
# Twister Plots
#
# Paul Zivich (2021/5/19)
########################################################################################################################

# Importing required dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##########################
# Twister Plot Function
def twister_plot(data, xvar, lcl, ucl, yvar, xlab="Risk Difference", ylab="Days", log_scale=False, reference_line=0.0,
                 treat_labs=("Treatment", "Placebo"), treat_labs_top=True, treat_labs_spacing="\t\t\t"):
    """Function to generate a twister plot from input data. Returns matplotlib axes which can have xlims and ylims
    set to the desired levels.

    Parameters
    ----------
    data : pandas DataFrame
        Pandas dataframe with the risk difference, upper and lower confidence limits, and times
    xvar : str
        The variable/column name for the risk difference.
    lcl : str
        The variable/column name for the lower confidence limit of the risk difference.
    ucl : str
        The variable name for the upper confidence limit of the risk difference.
    yvar : str
        The variable name for time.
    xlab : str, optional
        The x-axis label. Defaults to "Risk Difference".
    ylab : str, optional
        The y-axis label. Defaults to "Days".
    treat_labs : list, set, optional
        List of strings containing the names of the treatment groups. Only the first two elements are used in the
        labels. Defaults to 'Favors Treatment' and 'Favors Placebo'.
    treat_labs_top : bool, optional
        Whether to place the `treat_labs` at the top (True) or bottom (False). Defaults to True.
    treat_labs_spacing : str, optional
        Spacing to use between the treatment group names.

    Returns
    -------
    Matplotlib axes object

    Examples
    --------

    Example of plotting functionality

    >>> ax = twister_plot(data, xvar="RD", lcl="RD_LCL", ucl="RD_UCL", yvar="t")
    >>> ax.legend(loc='lower right')  # Added legend to the lower right corner of the plot
    >>> plt.tight_layout()  # Sets spacing of the border of the plot
    >>> plt.show()  # displays the generated image

    """
    max_t = data[yvar].max()  # Extract max y value for the plot

    # Initializing plot
    fig, ax = plt.subplots(figsize=(5, 7))  # fig_size is width by height
    ax.vlines(reference_line, 0, max_t,
              colors='gray',  # Sets color to gray for the reference line
              linestyles='--',  # Sets the reference line as dashed
              label=None)  # drawing dashed reference line at RD=0

    # Step function for Risk Difference
    ax.step(data[xvar],  # Risk Difference column
            data[yvar].shift(-1).ffill(),  # time column (shift is to make sure steps occur at correct t
            # label="RD",  # Sets the label in the legend
            color='k',  # Sets the color of the line (k=black)
            where='post')
    # Shaded step function for Risk Difference confidence intervals
    ax.fill_betweenx(data[yvar],  # time column (no shift needed here)
                     data[ucl],  # upper confidence limit
                     data[lcl],  # lower confidence limit
                     label="95% CI",  # Sets the label in the legend
                     color='k',  # Sets the color of the shaded region (k=black)
                     alpha=0.2,  # Sets the transparency of the shaded region
                     step='post')

    ax2 = ax.twiny()  # Duplicate the x-axis to create a separate label
    # "test \t test".expandtabs()
    ax2.set_xlabel("Favors " + treat_labs[0] + treat_labs_spacing.expandtabs() +  # Manually create some custom spacing
                   "Favors " + treat_labs[1],  # Top x-axes label for 'favors'
                   fontdict={"size": 10})
    ax2.set_xticks([])  # Removes top x-axes tick marks
    # Option to add the 'favors' label below the first x-axes label
    if not treat_labs_top:
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 36))

    ax.set_ylim([0, max_t])  # Sets the min and max of the y-axis
    ax.set_ylabel(ylab)  # Sets the y-label
    if log_scale:
        ax.set_xscale("log")
        xlimit = np.max([np.abs(np.log(data[lcl])),
                         np.abs(np.log(data[ucl]))])  # Extract the x-limits to use
        spacing = xlimit*2 / 20  # Sets a spacing factor. 20 seems to work well enough
        ax.set_xlim([np.exp(-xlimit - spacing), np.exp(xlimit + spacing)])  # Sets the min and max of the x-axis
    else:
        xlimit = np.max([np.abs(data[lcl]), np.abs(data[ucl])])  # Extract the x-limits to use
        spacing = xlimit*2 / 20  # Sets a spacing factor. 20 seems to work well enough
        ax.set_xlim([-xlimit-spacing, xlimit+spacing])  # Sets the min and max of the x-axis

    ax.set_xlabel(xlab,  # Sets the x-axis main label (bottom label)
                  fontdict={"size": 11,  # "weight": "bold"
                            })
    return ax


##########################
# Setup data
# Reading in data
data = pd.read_csv("data_twister.csv")  # .csv read in and managed using pandas
# data.info()  # checking that it read in correctly

##########################
# Example: Difference
ax = twister_plot(data, xvar="RD", lcl="RD_LCL", ucl="RD_UCL", yvar="t",
                  treat_labs=["Vaccine", "Placebo"])

# Formatting the axes and labels
ax.legend(loc='lower right')  # Added legend to the lower right corner of the plot
ax.set_yticks([i for i in range(0, 113, 7)])  # Sets the y-axes tick marks

plt.tight_layout()  # Sets spacing of the border of the plot
# plt.savefig("twister_plot_python.png", format='png', dpi=600)  # Saves the generated figure as .png
plt.show()  # displays the generated image

##########################
# Example: Ratio
ax = twister_plot(data, xvar="RR", lcl="RR_LCL", ucl="RR_UCL", yvar="t",
                  reference_line=1.0, log_scale=True, treat_labs=["Vaccine", "Placebo"])

# Formatting the axes and labels
ax.legend(loc='lower right')  # Added legend to the lower right corner of the plot
ax.set_yticks([i for i in range(0, 113, 7)])  # Sets the y-axes tick marks
ax.set_xticks([0.1, 0.25, 1, 5, 10])  # Sets the x-axes tick marks
ax.set_xticklabels(["0.10", "0.25", "1", "5", "10"])

plt.tight_layout()  # Sets spacing of the border of the plot
# plt.savefig("twister_plot_python.png", format='png', dpi=600)  # Saves the generated figure as .png
plt.show()  # displays the generated image
