import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from multiprocessing import Pool
from scipy.stats import norm
from delicatessen.utilities import spline


class PooledLogitGComputation:
    """G-computation for time-to-event data using the standard pooled logistic regression implementation.

    Parameters
    ----------
    data : DataFrame
        Data set including all variables
    exposure : str
        Column label for the exposure variable
    time : str
        Column label for time variable
    delta : DataFrame
        Column label for event indicator variable
    resolution : float, int, None, optional
        Optional parameter to set the resolution of time in the long data set conversion. Default (`None`) divides the
        observed time distribution into 100 uniform intervals. The resolution can be specified as well with either a
        float or int. For example, if resolution is set to 7, then each row in the long data frame will correspond to 7
        units of time.
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    verbose : bool, optional
        Whether to print intermediate model results to the console. Default is False, which runs quietly.
    """
    def __init__(self, data, exposure, time, delta, resolution=None, alpha=0.05, verbose=False):
        self.exposure = exposure     # Exposure variable (to flip for the parameters)
        self.time = time             # Time variable
        self.delta = delta           # Delta (event indicator) variable, with T=T^* -> 1, else 0
        self.idvar = "__id_var__"    # Create a unique ID variable (used in extension to long data frame)

        # Set resolution of the time scale (in how much a row corresponds to)
        if resolution is None:                                # If None, time consists of 100 unique intervals
            self.resolution = np.max(data[self.time]) / 100
        else:                                                 # Otherwise, use the set resolution provided by user
            if not isinstance(resolution, (int, float)):      # Error if not integer or float
                raise ValueError("The `resolution` scale must be an integer or float")
            self.resolution = resolution

        # Add an identifier variable!
        self.data = data.copy()                                     # Copy data (so I don't overwrite)
        self.data[self.idvar] = list(range(self.data.shape[0]))     # Create the ID variable

        # Convert data to long data set
        self.long_data = self._convert_to_long_data_()
        # print(self.long_data.dropna(subset=[self.delta]).shape)

        # Checking that new data has EXACTLY the same number of events
        if np.sum(self.data[delta]) != np.sum(self.long_data[delta]):
            raise ValueError("Something went wrong in the long data set conversion! There was "
                             + str(np.sum(self.data[delta])) + " events in the input data and the long data set has "
                             + str(int(np.sum(self.long_data[delta]))) + " events.")

        # Additional storage
        self._verbose_ = verbose             # Whether to run the verbose options
        self.alpha = alpha                   # Saving alpha level for confidence intervals
        self._model_specification_ = None    # Saving the nuisance model specification
        self.fit_pooled_logit = None         # Storage for the nuisance model
        self._tspline_called_ = False        # Whether time splines were created
        self._tspline_term_ = None           # term for time splines
        self._tspline_rstd_ = None           # restricted time splines
        self._tspline_nknots_ = None         # number of knots in time splines
        self._tspline_knots_ = None          # knot locations of time splines

    def create_time_splines(self, knots, term=3, restricted=True):
        """Function to generate spline terms for the time column.

        Note
        ----
        While the use of restricted cubic splines is recommended in Hernan's "The hazards of hazard ratios", it is
        likely better practice to use an indicator term for time.

        Parameters
        ----------
        knots : list
            Location of specified knots in a list. To specify the location of knots, put desired numbers for knots
            into a list. Be sure that the length of the list is the same as the specified number of knots. Default is
            None, so that the function will automatically determine knot locations without user specification
        term : integer, float, optional
            High order term for the spline terms. To calculate a quadratic spline change to 2, cubic spline
            change to 3, etc. Default is 3, i.e. a cubic spline.
        restricted : bool, optional
            Whether to return a restricted spline. Note that the restricted spline returns one less column than the
            number of knots. An unrestricted spline returns the same number of columns as the number of knots. Default
            is True, providing a restricted spline.
        """
        n_knots = len(knots)
        # Determining number of columns to create if restricted or unrestricted splines
        if restricted:      # Restricted has one less than the number of knots
            spline_cols = [self.time + "_spline" + str(i+1) for i in range(n_knots - 1)]
        else:               # Unrestricted has the same as the number of knots
            spline_cols = [self.time + "_spline" + str(i+1) for i in range(n_knots)]

        # Create new spline columns and add to data set
        self.long_data[spline_cols] = spline(variable=self.long_data[self.time],
                                             knots=knots,
                                             power=term,
                                             restricted=restricted,
                                             normalized=False)

        # Saving time spline data (for later variance_bootstrap calls)
        self._tspline_called_ = True         # Set presence of time splines as True
        self._tspline_term_ = term           # term for time splines
        self._tspline_rstd_ = restricted     # restricted time splines
        self._tspline_nknots_ = n_knots      # number of knots in time splines
        self._tspline_knots_ = knots         # knot locations of time splines

        # Returning some details if verbose is specified
        if self._verbose_:
            print("Created spline columns; " + str(spline_cols))         # Write out column names
            print(self.long_data[[self.time, ] + spline_cols].head(10))  # Provide head of columns

    def outcome_model(self, model):
        """Estimate the pooled logistic outcome model. Uses a logistic regression model to estimate the specified model
        and the long data set.

        Parameters
        ----------
        model : str
            Independent variables to predict the event indicator for each unit of time. Example) 'var1 + var2 + var3'.
            The exposure must be included in the model to allow for any non-null effects
        """
        # Check if exposure is included in the model
        if self.exposure not in model:
            warnings.warn("The exposure variable doesn't look like its included in the specified model", UserWarning)

        # Restrict data to only observed times
        data_to_fit = self.long_data.loc[self.long_data[self.delta].notna()].copy()

        # Estimate the GLM
        f = sm.families.family.Binomial()                              # Specify logistic family GLM
        self.fit_pooled_logit = smf.glm(self.delta + " ~ " + model,    # Nuisance model form
                                        data=data_to_fit,              # Long dataframe restricted to valid obs
                                        family=f).fit()                # Logistic GLM
        self._model_specification_ = model

        # Returning model fit details if verbose if specified
        if self._verbose_:
            print(self.fit_pooled_logit.summary())

    def estimate(self, bs_iterations=200, seed=None, n_cpus=1):
        """Estimate the risk function via g-computation under the specified policy.

        Parameters
        ----------
        bs_iterations : int, optional
            Number of bootstraps to resample to estimate the variance. Default is 200 resamples.
        seed : int, None, optional
            Random seed for generation of the bootstrapped samples.
        n_cpus : int, optional
            Number of CPU tasks to use to run the bootstrap procedure. Since bootstrapping is embarrassingly parallel,
            can use multiple cores to speed up task. Default is 1 (which is slowest).
        """
        # Running point estimation procedure (detailed in point_estimate)
        psi = self.point_estimate()                      # Parameter of interest
        # Running variance estimation procedure (detailed in variance_bootstrap)
        var_psi = self.variance_bootstrap(bs_iterations=bs_iterations,      # Number of bootstrapped samples
                                          seed=seed,                        # Seed for consistent bootstraps
                                          n_cpus=n_cpus)                    # Number of CPU in multiprocessing.Pool

        # Calculating everything
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)  # Z-value for the corresponding alpha
        columns = psi.columns                                  # Columns in the results data (for building output)
        fdat = pd.DataFrame()                                  # Empty data frame for results data
        for c in columns:                                      # Generating for each column
            fdat[c] = psi[c]                                        # Copy point estimate to results
            fdat["Var_" + c] = var_psi["Var_" + c]                  # Copy variance estimate to results
            fdat["LCL_" + c] = fdat[c] - zalpha * np.sqrt(fdat["Var_" + c])  # Calculate lower CI
            fdat["UCL_" + c] = fdat[c] + zalpha * np.sqrt(fdat["Var_" + c])  # Calculate upper CI

        return fdat  # Return the results in a nice data frame

    def point_estimate(self):
        """Point estimate for the risk function via g-computation under the specified policy.
        """
        if self.fit_pooled_logit is None:                                            # Breslow has been estimated
            raise ValueError("PooledLogitGComputation.outcome_model() "
                             "must be specified before .estimate()")

        init_value = 0
        psi1 = self._estimate_one_sample_(policy=1, parameter='risk')
        psi0 = self._estimate_one_sample_(policy=0, parameter='risk')
        d = pd.DataFrame()                                        # Creating blank data frame
        d[self.time] = psi1.index                                  # Creating column of times
        d = d.set_index(self.time)                                # Setting time as the index
        d["R1"] = psi1                           # Adding risk estimates
        d["R0"] = psi0                                 # Add Psi referent
        d["RD"] = d["R1"] - d["R0"]
        dz = pd.DataFrame({"R1": init_value,                          # Creating a time zero
                           "R0": init_value,
                           "RD": 0,
                           }, index=[0])
        dz.index.name = self.time                                    # Sets time as index for t=0 row
        return dz.append(d)                                          # appends zero data then returns

    def variance_bootstrap(self, bs_iterations=200, seed=None, n_cpus=1):
        """Bootstrap-based estimator of the variance for the parameter(s) of interest. The input data set is resampled
        with replacement, then the estimation procedure is repeated with this new data, and this whole process is
        repeated for a number of iterations. The estimated variance is then defined as the variance of the estimates
        from the resampled observations.

        Parameters
        ----------
        bs_iterations : int, optional
            Number of bootstraps to resample to estimate the variance. Default is 200 resamples.
        seed : int, None, optional
            Random seed for generation of the bootstrapped samples.
        n_cpus : int, optional
            Number of CPU tasks to use to run the bootstrap procedure. Since bootstrapping is embarrassingly parallel,
            can use multiple cores to speed up task. Default is 1 (which is slowest).
        """
        # Creating list of params to feed forward into the bootstrapping procedure
        n = self.data.shape[0]                  # Number of observations in the data set
        rng = np.random.default_rng(seed)       # Setting the seed for bootstraps

        params = [[self.data,                                                         # resampled data
                   rng.choice(n, size=n, replace=True),                               # Indices to pull
                   self.exposure, self.time, self.delta, self.resolution, False,      # class object inits
                   self._tspline_called_, self._tspline_term_, self._tspline_rstd_,   # time splines
                   self._tspline_nknots_, self._tspline_knots_,                       # ...rest of time splines
                   self._model_specification_,                                        # outcome_model()
                   ] for i in range(bs_iterations)]                                   # iterations

        # Using pool to multiprocess the bootstrapping procedure
        with Pool(processes=n_cpus) as pool:
            bsd = list(pool.map(_bootstrap_single_iter_pooledgcomputation_,  # Call outside function to run parallel
                                params))                                     # provide packed input list

        # Processing bootstrapped samples
        cols = bsd[0].columns                            # Pull out the column names from the first bootstrap data set
        variance_estimates = pd.DataFrame()              # Create blank data set to store variance results
        for c in cols:                                   # for each column (output parameter to estimate variance)
            bootstraps = pd.concat([b[c] for b in bsd],  # concat the list of results for a specific column
                                   axis=1)               # ...stack as new columns
            bootstraps = bootstraps.ffill(axis=0)        # Forward fill (so everything is on same t axis
            c = "Var_" + c                               # Have column label indicate corresponds to var
            var_estimate = np.var(bootstraps,            # Calculate variance for columns
                                  axis=1,                # ...variance across rows (for each unique t)
                                  ddof=1)                # ...divisor is (n-1)
            variance_estimates[c] = var_estimate         # Store vector of variance estimates under correct label

        return variance_estimates                        # Return the results in a nice data frame

    def _estimate_one_sample_(self, policy, parameter):
        """Background function to conduct estimation under a specific policy (separate call, so can easily evaluate and
        compare two policies).

        Parameters
        ----------
        policy : int, float, array, list, tuple, None
            Fed from `BreslowGComputation.estimate`
        parameter : str, optional
            Fed from `BreslowGComputation.estimate`
        """
        # Estimate conditional survival on extended long data set
        long_data_est = self.long_data.copy()                        # Creating a copy of data to apply the policy to
        if policy is not None:
            long_data_est[self.exposure] = policy                    # Applying policy to data set
        pred = self.fit_pooled_logit.predict(long_data_est)          # Predict from nuisance model under the policy

        # Calculate the parameter function!
        long_data_est['_pred_cond_surv_'] = 1 - pred                                 # Calculate mean by adding pred...
        pred_surv = long_data_est.groupby(self.idvar)['_pred_cond_surv_'].cumprod()  # conditional to marginal...
        if parameter == "hazard":                                                    # If hazard
            long_data_est['_pred_'] = -np.log(pred_surv)                             # ...psi(t) convert to H(t)
        elif parameter == "survival":                                                # If survival
            long_data_est['_pred_'] = pred_surv                                      # ...psi(t) output
        elif parameter == "risk":                                                    # If risk
            long_data_est['_pred_'] = 1 - pred_surv                                  # ...psi(t) covert to F(t)
        else:                                                                        # Else error
            raise ValueError("You entered '" +
                             parameter +
                             "' but only the following options "
                             "are supported: risk, "
                             "survival, and hazard.")
        return long_data_est.groupby(self.time)['_pred_'].mean()   # then take mean by time variable

    def _convert_to_long_data_(self):
        max_t = np.max(self.data[self.time])
        dl = pd.DataFrame(np.repeat(self.data.values, max_t, axis=0), columns=self.data.columns)
        dl['t_in'] = dl.groupby(self.idvar)[self.time].cumcount()
        dl['t_out'] = dl['t_in'] + 1
        dl['_event_'] = np.where(dl['t_out'] == dl[self.time], dl[self.delta], 0)
        dl['_event_'] = np.where(dl['t_out'] > dl[self.time], np.nan, dl['_event_'])
        dl = dl.drop(columns=[self.time, self.delta])
        dl[self.time] = dl['t_out']
        dl[self.delta] = dl['_event_']
        dl = dl.drop(columns=['t_in', 't_out', '_event_'])
        return dl


def _bootstrap_single_iter_pooledgcomputation_(params):
    """Function for BreslowGComputation bootstrapping procedure. This function allows for the bootstrapping iterations
    to be run via Pool, since the function needs to be outside the class for multiprocessing to copy instances
    correctly to each .

    This is only meant as a background function to be called for bootstrapping procedures

    Parameters
    ----------
    params : list
        Packed list of the parameters to be passed to the Pool. The first step expands this out and then passes
        everything to a copied version of the BreslowGComputation call.
    """
    (data, sample_index,                                  # Unpack the list of input params to pass to BreslowGComp
     exposure, time, delta, resolution, verbose,
     t_spline, t_spline_term, t_spline_rstd,
     t_spline_nk, t_spline_k,
     model_spec, ) = params

    # Create a fresh class of the PooledLogitGComputation class with the resample data
    estimator_copy = PooledLogitGComputation(data=data.iloc[sample_index].copy(),   # Select out the resample via index
                                             exposure=exposure,                     # Original exposure value
                                             time=time,                             # Original time value
                                             delta=delta,                           # Original delta value
                                             resolution=resolution,                 # Resolution
                                             verbose=verbose)                       # ALWAYS suppresses verbose here
    # Creating time splines (if done in outer estimator)
    if t_spline:
        estimator_copy.create_time_splines(term=t_spline_term,                      # Create spline with term
                                           restricted=t_spline_rstd,                # ...(un)restricted
                                           knots=t_spline_k)                        # ...knots at x
    # Estimate the same outcome model on resampled data
    estimator_copy.outcome_model(model=model_spec)                                 # Original model specification
    # Estimate the point estimate for the resampled data
    x = estimator_copy.point_estimate()                                            # Original parameter of interest
    # Return the point estimates for the particular resample (gets stacked in a list later)
    return x
