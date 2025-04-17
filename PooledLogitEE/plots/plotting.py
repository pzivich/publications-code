######################################################################################################################
# Plotting helper functions
#
# Paul Zivich (Last update: 2025/4/16)
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


def twister_plot(data, point, lcl, ucl, time, color='k', reference_line=0.0, log_scale=False,
                 favors=True, favors_label=("Treatment", "Placebo"), favors_spacing="\t\t\t", step=True, ax=None):
    """Function to generate a twister plot for the risk difference or risk ratio from survival or longitundinal
    analysis results. Twister plots are an alternative to the dynamic risk plots.

    Parameters
    ----------
    data : pandas DataFrame
        Pandas dataframe with the risk difference, upper and lower confidence limits, and times
    point : str
        Column name for the measure.
    lcl : str
        Column name for the lower confidence limit.
    ucl : str
        Column name for the upper confidence limit.
    time : str
        Column variable name for time.
    color : str, optional
        Color to draw the line for the point estimates and color of the shaded region for the confidence intervals.
        Default is 'k'.
    reference_line : int, float, optional
        Value to draw the reference line at. Default is 0.
    log_scale : bool, optional
        Whether to plot on the log-scale. Default is False.
    favors : bool, optional
        Whether to draw the secondary labels indicating favors A versus B on a minor x-axis. Default is True.
    favors_label : list, set, optional
        List of strings containing the names of the treatment groups. Only the first two elements are used in the
        labels. Defaults to 'Treatment' and 'Placebo'.
    favors_spacing : str, optional
        Spacing to use between the ``favors_label``. Default is "\t\t\t".
    step : bool, optional
        Whether to plot the measure using a step function (True) or not (False). Default is True.
    ax : matplotlib axes object, None, optional
        Argument to use an input ax defined outside of the function. Default is None, which generates a new matplotlib
        axes.

    Returns
    -------
    Matplotlib axes object

    Examples
    --------
    Example of plotting functionality

    >>> import numpy as np
    >>> import pandas as pd
    >>> from condatanser.plots import twister_plot

    Generating a twister plot

    >>> ax = twister_plot(data, xvar="RD", lcl="RD_LCL", ucl="RD_UCL", yvar="t")
    >>> ax.legend(loc='lower right')
    >>> plt.tight_layout()
    >>> plt.show()

    References
    ----------
    Zivich PN, Cole SR, & Breskin A. (2021). Twister Plots for Time-to-Event Studies.
    *American Journal of Epidemiology*, 190(12), 2730-2731.
    """
    # Initializing plot if none is provided
    if ax is None:
        ax = plt.gca()

    # Extracting the max time value to set the limits of the y-axis
    max_t = data[time].max()

    # Drawing the reference line for the null value (default is zero)
    ax.axvline(reference_line,                           # Reference line from 0 to max(time) at reference
               color='gray',                             # ... sets color to gray for the reference line
               linestyle='--',                           # ... sets the reference line as dashed
               label=None)                               # ... with no label when legend is called

    # Step function for measure
    if step:                                             # Handling step function parameters
        ax.step(data[point],                             # Draw step function at point estimates
                data[time].shift(-1).ffill(),            # ... with shifted time column (ensure steps occur at t)
                label=None,                              # ... drop line from appearing in legend
                color=color,                             # ... sets the color of the line as requested
                where='post')                            # ... steps are all post
        step_fill = 'post'                               # Argument for fill_betweenx
    else:
        ax.plot(data[point],                             # Draw step function at point estimates
                data[time],                              # ... with shifted time column (ensure steps occur at t)
                label=None,                              # ... drop line from appearing in legend
                color=color),                            # ... sets the color of the line as requested
        step_fill = None                                 # Argument for fill_betweenx

    # Shaded step function for confidence intervals
    ax.fill_betweenx(data[time],                         # Shade in confidence interval across time
                     data[ucl],                          # ... from upper confidence limit
                     data[lcl],                          # ... to lower confidence limit
                     label="95% CI",                     # ... with a label in the legend
                     color=color,                        # ... as the requested color
                     alpha=0.2,                          # ... with some transparency for ease
                     step=step_fill)                     # ... then apply steps or not given step argument

    # Code to put the secondary x-axis for favors A versus favors B as done in the letter
    if favors:                                              # Adding favors labels if requested
        ax2 = ax.twiny()                                    # ... duplicate the x-axis to create a separate label
        ax2.set_xlabel("Favors " + favors_label[0] +        # ... favors the left
                       favors_spacing.expandtabs() +        # ... manually create some custom tab spacing
                       "Favors " + favors_label[1])         # ... favors the right
        ax2.set_xticks([])                                  # ... removes x-axes tick marks for duplicate x-axis
        ax2.xaxis.set_ticks_position('bottom')              # ... setting the ticks on the bottom
        ax2.xaxis.set_label_position('bottom')              # ... setting the position on the bottom
        ax2.spines['bottom'].set_position(('outward', 36))  # ... moving favors label to prevent overlap

    # Manipulate the x-axis and y-axis scales
    ax.set_ylim([0, max_t])                                 # Sets the min and max of the y-axis
    ax.set_ylabel("Time")                                   # Sets the y-label as time (generic)
    ax.set_xlabel("Measure")                                # Generic x-axis label for the measure (easy to update)
    if log_scale:                                           # If adding a log-scale for the risk ratio
        ax.set_xscale("log", base=np.e)                     # ... set as log x-scale
        xlimit = np.max([np.abs(np.log(data[lcl])),         # ... extract log scale limits to use
                         np.abs(np.log(data[ucl]))])        # ... with absolute value to keep symmetric
        spacing = xlimit*2 / 20                             # ... arbitrary spacing factor (20 seems to work)
        ax.set_xlim([np.exp(-xlimit - spacing),             # ... set the x limits with spacing factor
                     np.exp(xlimit + spacing)])
    else:                                                   # Otherwise use the regular linear scale
        xlimit = np.max([np.abs(data[lcl]),                 # ... extract the linear scale limits to use
                         np.abs(data[ucl])])                # ... with absolute value to keep symmetric
        spacing = xlimit*2 / 20                             # ... arbitrary spacing factor (20 seems to work)
        ax.set_xlim([-xlimit-spacing, xlimit+spacing])      # ... sets the min and max of the x-axis

    # Return the generated axis object
    return ax
