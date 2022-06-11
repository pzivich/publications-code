import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def bound_probability(v, bounds):
    """Background function to bound the predicted probabilities (copy of ``zepid.calc.probability_bounds``)
    """
    v = np.asarray(v)
    if type(bounds) is float:  # Symmetric Bounding
        if bounds < 0 or bounds > 1:
            raise ValueError('Bound value must be between (0, 1)')
        v[v < bounds] = bounds
        v[v > 1 - bounds] = 1 - bounds

    elif type(bounds) is str:  # Catching string inputs
        raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')

    elif type(bounds) is int:  # Catching string inputs
        raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')

    else:  # Asymmetric bounds
        if bounds[0] > bounds[1]:
            raise ValueError('Bound thresholds must be listed in ascending order')
        if len(bounds) > 2:
            warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                          'specified bounds are used by the bound statement. So only ' +
                          str(bounds[0:2]) + ' will be used', UserWarning)
        if type(bounds[0]) is str or type(bounds[1]) is str:
            raise ValueError('Bounds must be floats between (0, 1)')
        if (bounds[0] < 0 or bounds[1] > 1) or (bounds[0] < 0 or bounds[1] > 1):
            raise ValueError('Both bound values must be between (0, 1)')
        v[v < bounds[0]] = bounds[0]
        v[v > bounds[1]] = bounds[1]

    return v


def twister_plot(data, xvar, lcl, ucl, yvar, treat_labs=None, treat_labs_top=True, figsize=(5, 7)):
    """Twister plot generator. Used for presentation of diagnostics and results for the fusion estimator

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
    treat_labs : str, optional
        String containing the names of the treatment groups. Defaults to 'Favors Vaccine' and 'Favors Placebo'.
    treat_labs_top : bool, optional
        Whether to place the `treat_labs` at the top (True) or bottom (False). Defaults to True.
    figsize : set, list, optional
        Size of figure to generate

    Returns
    -------
    Matplotlib axes object
    """
    max_t = data[yvar].max() + 1  # Extract max y value for the plot

    # Initializing plot
    fig, ax = plt.subplots(figsize=figsize)  # fig_size is width by height
    ax.vlines(0, 0, max_t,
              colors='gray',                 # Sets color to gray for the reference line
              linestyles='--',               # Sets the reference line as dashed
              label=None)                    # drawing dashed reference line at RD=0

    # Step function for Risk Difference
    ax.step(data[xvar],                      # Risk Difference column
            data[yvar].shift(-1).ffill(),    # time column (shift is to make sure steps occur at correct t
            # label="RD",                    # Sets the label in the legend
            color='k',                       # Sets the color of the line (k=black)
            where='post')
    # Shaded step function for Risk Difference confidence intervals
    ax.fill_betweenx(data[yvar],             # time column (no shift needed here)
                     data[ucl],              # upper confidence limit
                     data[lcl],              # lower confidence limit
                     label="95% CI",         # Sets the label in the legend
                     color='k',              # Sets the color of the shaded region (k=black)
                     alpha=0.2,              # Sets the transparency of the shaded region
                     step='post')

    ax2 = ax.twiny()  # Duplicate the x-axis to create a separate label
    ax2.set_xlabel(treat_labs,               # Top x-axes label for 'favors'
                   fontdict={"size": 10})
    ax2.set_xticks([])                       # Removes top x-axes tick marks

    # Option to add the 'favors' label below the first x-axes label
    if not treat_labs_top:
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 36))

    ax.set_ylim([0, max_t])

    return ax


def area_under_steps(data, time, prob):
    """Calculates the area under a step-function given in a dataframe format. The `time` column corresponds to the
    x-axis and `prob` corresponds to the y-axis. The approach used here assumes that steps are 'post' (as matplotlib's
    step function where='post' operates). The simple algorithm works by calculating the area of the rectangles of the
    step function.

    Note
    ----
    The x-axis is assumed to be measured against zero (i.e., `prob` - 0)

    Parameters
    ----------
    data : dataframe
        Pandas data frame consisting of a time (x-axis) and prob (y-axis) columns to calculate the area. Note that the
        function sorts by time (to make sure the correct area is being calculated).
    time : str
        x-axis column to calculate the area.
    prob : str
        y-axis column to calculate the area. Relative to zero

    Examples
    --------

    >>> area_under_steps(data, time="x", prob="y")
    """
    return np.sum((data[time].shift(-1) - data[time]) * data[prob])


def area_between_steps(data, time, prob1, prob2, signed=False):
    """Calculates the area between two step functions The `time` column corresponds to the x-axis and `prob1` and
    `prob2` correspond to the y-axis values. The approach used here assumes that steps are 'post' (as matplotlib's
    step function where='post' operates). The simple algorithm works by calculating the area of the rectangles between
    the two step functions. First, distance measures of the x-axis are calculated. Then y-axis distance between the
    two functions are calculated. The y-axis can either correspond to the signed-area (allows for negative area) or the
    geometric-area (positive only). Rectangle areas are then summed together

    Note
    ----
    Signed area can be negative and corresponds to prob1 - prob2.

    Parameters
    ----------
    data : dataframe
        Pandas data frame consisting of a time (x-axis) and prob (y-axis) columns to calculate the area. Note that the
        function sorts by time (to make sure the correct area is being calculated).
    time : str
        x-axis column to calculate the area.
    prob1 : str
        y-axis column to calculate the area for the first function.
    prob1 : str
        y-axis column to calculate the area for the second function.
    signed : bool, optional
        Whether to calculate the signed-area. The signed area can be negative, whereas the geometric-area is always
        postive between the two curves (i.e., absolute(prob1 - prob2). Default is False, which corresponds to the
        geometric-area.
    """
    # Calculating distances in x and y directions
    x_measures = data[time].shift(-1) - data[time]
    y_measures = data[prob1] - data[prob2]

    # Absolute value of y-distances if geometric-area
    if not signed:
        y_measures = np.abs(y_measures)

    # Calculating rectangle areas then summing together
    return np.sum(x_measures * y_measures)


class BreslowEstimator:
    """Breslow estimator for the various metrics of survival for use with the Cox Proportional Hazard Model. Basically,
    uses a non-parametric approach to estimate the baseline hazard, then predicts from the estimated Cox model.

    Parameters
    ----------
    data : DataFrame
        Data set including all variables
    time : str
        Column label for time variable
    delta : DataFrame
        Column label for event indicator variable
    verbose : bool, optional
        Whether to display the CoxPHFitter results
    """
    def __init__(self, data, time, delta, verbose=False):
        self.data = data
        self.time = time
        self.delta = delta

        self._verbose_ = verbose
        self._covariate_matrix_ = None
        self._cox_model_ = None

        self.fit_coxph = None
        self.coef = None
        self.coef_i = None
        self.event_times = None
        self.h_t = None

    def cox_model(self, model, ties="breslow", strata=None):
        """Estimate the Cox Model, and generate all the necessary coefficients for the Breslow estimator.

        Parameters
        ----------
        model : str
            Model in patsy format for variables of interest
        ties : str, optional
            How ties are handled. By default, ties are handled using the Breslow method. Efron method can also be
            requested by 'efron'.
        strata : None, str, optional
            Variable name to stratify by, if a stratified Cox model is requested. Default is None, which fits an
            unstratified Cox model
        """
        if strata is None:
            self._stratification_ = '_no_strata__specificied_'
            self.data[self._stratification_] = 1
        else:
            self._stratification_ = strata

        # If no model, set coefficient and coef_i to zero. Therefore, each person only counts as 1
        if model == "" or model.isspace():
            self.coef = np.array([0])
            self.coef_i = np.zeros([1, self.data.shape[0]])
            self._covariate_matrix_ = pd.DataFrame(np.zeros([self.data.shape[0], 1]))

        # If a model is provided
        else:
            # Process data for CoxPH (patsy '-1' magic causes issues when used with 'C()' patsy magic)
            cmat = patsy.dmatrix(model, self.data, return_type='dataframe')
            self._covariate_matrix_ = cmat.loc[:, cmat.columns != 'Intercept'].copy()

            # Estimating CoxPH Model
            mod = smf.phreg(self.time + " ~ " + model,
                            self.data,
                            strata=strata,
                            status=self.data[self.delta].values,
                            ties=ties)
            self._cox_model_ = mod.fit(method='newton', tol=1e-7)
            if self._verbose_:
                print(self._cox_model_.summary())

            self.coef = self._cox_model_.params
            self.coef_i = np.dot(np.asarray([self.coef]), np.asarray(self._covariate_matrix_).T)

    def predict_cumulative_hazard(self):
        """Predict the survival function, H_i(T_i), for each individuals last observed time. Note that this returns the
        predicted cumulative hazard at T_i for each individual, unlike the
        predict_marginal_cumulative_hazard() function.

        Returns
        -------
        numpy.array of probabilities of the event of interest at the final time

        """
        times = np.asarray(self.data[self.time])  # event times for each i
        strat = sorted(np.unique(self.data[self._stratification_]))

        # Calculating Breslow baseline hazard (vectorized)
        x = np.asarray(self._covariate_matrix_)  # covariates for all i
        r = np.where(np.subtract.outer(times, times) >= 0, 1, 0)  # indicator when still part of n for t
        v = np.where(np.subtract.outer(times, times) <= 0, 1, 0)  # indicator when no longer part of n for t
        y = np.asarray(self.data[self.delta])  # vector of event indicators
        s = np.asarray(self.data[self._stratification_])  # vector of strata

        Ht = np.zeros(times.shape)
        for value in strat:
            subset = np.where(s == value, 1, 0)
            subset_matrix = np.where(np.add.outer(subset, subset) > 1, 1, 0)
            ys = y * subset
            rs = r * subset_matrix
            vs = v * subset_matrix
            dens = np.dot(np.exp(np.dot(x, self.coef)).T, rs)
            # print(dens)
            H0 = np.dot((ys / np.where(dens == 0, 1e-12, dens)).T, vs)
            Ht = Ht + H0*np.exp(np.dot(x, self.coef))

        return -Ht

    def predict_survival(self):
        """Predict the survival function, S_i(T_i), for each individuals last observed time. Note that this returns the
        predicted survival at T_i for each individual, unlike the predict_marginal_survival() function.

        Returns
        -------
        numpy.array of probabilities of the event of interest at the final time

        """
        # Call predict_cumulative_hazard then transform accordingly
        return np.exp(self.predict_cumulative_hazard())

    def predict_risk(self):
        """Predict the risk function, F_i(T_i), for each individuals observation time. Note that this returns the
        predicted risk at T_i for each individual, unlike the predict_marginal_risk() function.

        Returns
        -------
        numpy.array of probabilities of the event of interest at the final time
        """
        return 1 - np.exp(self.predict_cumulative_hazard())

    def predict_hazard_matrix(self):
        """Predicts the hazard at each time for each individual. Returns a matrix of the hazard. This function is used
        to calculate the population standardized survival/risk/cumulative hazard curves.

        Note: it is also possible to extract hazards at the last observed time (or other transformations). But this
        uses a for-loop, which is slower (10-20x) than the vectorized version used in the predict_*() functions. If
        the predict_marginal_*() can be sped up by vectorization, then the same update will be made

        """
        # Calculating Breslow baseline hazard (for loop)
        # This method is slower than the vector approach used in predict_risk()

        self.event_times = sorted(np.unique(self.data[self.time]))  # pulling out all unique event times
        stratum = sorted(np.unique(self.data[self._stratification_]))
        # All unique strata in the event columns
        events = self.data.groupby([self._stratification_, self.time]
                                   )[self.delta].sum()  # event counts for each unique t and each strata
        events = events.reindex(pd.MultiIndex.from_product([stratum, self.event_times])
                                ).fillna(0)  # Fill all unique combinations
        events = events.rename_axis([self._stratification_, self.time]).reset_index()  # reseting index for processing

        # Getting Baseline hazards
        h0i = []  # empty list to store all different strata baseline hazards
        for v in stratum:  # go through each unique strata
            h0 = np.array([np.nan] * len(self.event_times))  # create empty vector for baseline hazard for strata
            for i in range(len(self.event_times)):  # go through each unique event time
                num = float(events.loc[(events[self.time] == self.event_times[i]) &
                                       (events[self._stratification_] == v), self.delta])  # number of events
                den = np.sum(np.exp(np.dot(
                    np.asarray(self._covariate_matrix_.loc[(self.data[self.time] >= self.event_times[i]) &
                                                           (self.data[self._stratification_] == v)]),
                    self.coef)))  # number in strata and time for the denominator
                h0[i] = num / np.where(den == 0, 1e-12, den)  # prevent the denominator from being zero...
            h0i.append(h0)  # adding the completed baseline hazards for strata level v

        # Multiplying the correct baseline hazards with the correct persons by strata V
        ht_matrix = np.zeros([len(self.event_times), self.data.shape[0]])  # creating empty 'complete' hazard matrix
        for i in range(len(stratum)):  # Go through the index of each strata (same as the index of the strata h0)
            h_t = np.dot(np.asarray([h0i[i]]).T,  # baseline hazard for specific strata
                         np.exp(self.coef_i))  # times the individual coefficients
            update_ht = np.multiply(h_t,  # baseline hazard times individual coefficient
                                    np.where(self.data[self._stratification_] == stratum[i], 1, 0))  # 'correct' strata
            ht_matrix = ht_matrix + update_ht  # add to complete hazard matrix (since adding to zero, will be hazards)

        return ht_matrix

    def predict_marginal_cumulative_hazard(self):
        """Predicts the marginal risk function, F(t), for each observation time. Note that this marginalizes over the
        population, unlike the predict_risk() function.

        """
        # Transforming hazard -> cumulative hazard in the entire matrix
        chaz = self.predict_hazard_matrix().cumsum(axis=0)

        d = pd.DataFrame()
        d['time'] = self.event_times
        d = d.set_index('time')
        d['survival'] = np.mean(chaz, axis=1)

        # Creating a time zero
        dz = pd.DataFrame({"risk": 0.0}, index=[0])
        dz.index.name = "time"
        return dz.append(d)

    def predict_marginal_survival(self):
        """Predicts the marginal risk function, F(t), for each observation time. Note that this marginalizes over the
        population, unlike the predict_risk() function.

        """
        # Transforming hazard -> cumulative hazard -> survival in the entire matrix
        surv = np.exp(-self.predict_hazard_matrix().cumsum(axis=0))

        d = pd.DataFrame()
        d['time'] = self.event_times
        d = d.set_index('time')
        d['survival'] = np.mean(surv, axis=1)

        # Creating a time zero
        dz = pd.DataFrame({"risk": 0.0}, index=[0])
        dz.index.name = "time"
        return dz.append(d)

    def predict_marginal_risk(self):
        """Predicts the marginal risk function, F(t), for each observation time. Note that this marginalizes over the
        population, unlike the predict_risk() function.

        """
        # Transforming hazard -> cumulative hazard -> survival -> risk in the entire matrix
        risk = 1 - np.exp(-self.predict_hazard_matrix().cumsum(axis=0))

        d = pd.DataFrame()
        d['time'] = self.event_times
        d = d.set_index('time')
        # Averaging together all F_i(t) for each t
        d['risk'] = np.mean(risk, axis=1)

        # Creating a time zero
        dz = pd.DataFrame({"risk": 0.0}, index=[0])
        dz.index.name = "time"
        return dz.append(d)
