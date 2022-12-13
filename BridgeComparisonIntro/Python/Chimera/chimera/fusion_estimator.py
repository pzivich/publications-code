import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt

from .utilities import BreslowEstimator, twister_plot, bound_probability, area_between_steps


class SurvivalFusionIPW:
    """Bridging inverse probability weighting (IPW) estimator.

    Note
    ----
    This estimator expects survival data in the form of one row for each unique individual.

    The bridging IPW estimator consists of three sets of nuisance functions: the treatment model, the censoring model,
    and the sampling model.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame object containing all variables of interest
    treatment : str
        Column name of the exposure variable. Currently only binary is supported
    outcome : str
        Column name of the outcome variable. Currently only binary is supported
    time : str
        Column name of the time variable
    sample : str
        Column name of the location variable, only binary is currently supported
    censor : str, optional
        Column name of the censoring variable. Leave as None if there is no censoring.
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05, returning the 95% CL
    verbose : bool, optional
        Whether to display all nuisance model results as procedure runs. Default is False, which does not output to
        console

    Examples
    --------

    References
    --------

    """

    def __init__(self, df, treatment, outcome, time, sample, censor=None, alpha=0.05, verbose=False):
        # Setup
        self.treatment = treatment               # action of interest (A_i)
        self.outcome = outcome                   # outcome indicator  (delta_i)
        self.time = time                         # time               (T_i^*)
        self.sample = sample                     # trial indicator    (S_i)
        self.censor = censor                     # censor indicator   (1 - delta_i)

        # Extracting sorted data set
        self.df = df.copy().sort_values(by=[self.sample, self.treatment, self.time]).reset_index(drop=True)

        # Checking provided input data as correctly formatted
        if np.any(~self.df[self.sample].isin([0, 1])):
            raise ValueError("The `sample` column must be all 0 or 1.")
        if np.any(~self.df.loc[self.df[self.sample] == 1, self.treatment].isin([1, 2])):
            raise ValueError("The `treatment` column must be all 1 or 2 when sample=1.")
        if np.any(~self.df.loc[self.df[self.sample] == 0, self.treatment].isin([0, 1])):
            raise ValueError("The `treatment` column must be all 1 or 2 when sample=1.")

        # Number of observations in location of interest
        self.n_local = np.sum(self.df[self.sample])  # n_1
        self.n_distal = self.df.shape[0] - self.n_local  # n_0
        self.n_distal_weighted = None  # hat{n}_0

        # Determining unique event times to generate survival curves
        if np.sum(np.where(self.df[self.time] == 0, 1, 0)) > 0:
            raise ValueError("At least one observation had a time of zero. "
                             "All observation times must be greater than zero")
        self.event_times = [0] + sorted(self.df.loc[self.df[self.outcome] == 1,
                                                    self.time].unique()) + [np.max(self.df[self.time])]

        # Storage for later procedures
        self._alpha_ = alpha
        self._verbose_ = verbose
        self._fit_location_, self._fit_treatment_, self._fit_censor_ = False, False, False
        self._censor_model_custom_, self._censor_model_method_ = False, None
        self._baseline_weight_, self._timed_weight_ = None, None
        self._nuisance_treatment_, self._nuisance_sampling_, self._nuisance_censoring_ = [], [], []

        # Storage for function specifications to feed to the bootstrap
        (self._censor_model_, self._censor_stratify_,
         self._censor_strata_, self._censor_bound_, self._censor_shift) = None, None, None, None, None
        self._treat_model_, self._treat_bound_ = None, None
        self._sample_model_, self._sample_bound_ = None, None

    def sampling_model(self, model, bound=False):
        r"""Sampling model, which predicts the probability of :math:`S=1` given the baseline covariates. The sampling
        model consists of the following

        .. math::

            pi_S(V_i) = \Pr(S=1 | V; \beta)

        Parameters
        ----------
        model : str
            Variables to predict the location differences via the patsy format. For example, 'var1 + var2 + var3'
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            truncating weights leads to additional confounding. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation, where values below
            or above the threshold are set to the threshold value. Alternatively a list of floats can be provided for
            asymmetric trunctation, with the first value being the lower bound and the second being the upper bound
        """
        # Saving specifications (used for the bootstrapped variance)
        self._sample_model_ = model
        self._sample_bound_ = bound

        loc_fm = smf.glm(self.sample + " ~ " + model, self.df, family=sm.families.family.Binomial()).fit()
        self._nuisance_sampling_.append(loc_fm)
        if self._verbose_:
            print('==============================================================================')
            print('Sampling Model')
            print(loc_fm.summary())
            print('==============================================================================')

        # Calculating sampling weight of bridging estimator
        self.df['_pr_local_'] = loc_fm.predict(self.df)

        # Applying bound if requested
        if bound:
            self.df['_pr_local_'] = bound_probability(self.df['_pr_local_'], bounds=bound)

        # Calculating the re-weighted sample size for S=0, hat{n}_0
        self.n_distal_weighted = np.sum((np.asarray(1 - self.df[self.sample]) * self.df['_pr_local_'])
                                        / (1 - self.df['_pr_local_']))
        # Marker to indicate model has been specified and fully fit
        self._fit_location_ = True

    def treatment_model(self, model, bound=False):
        r"""Treatment model, which predicts the probability of treatment within each location (i.e., stratified by
        location). Uses separate logistic regression models for each data source.

        Note
        ----
        For a randomized trial, use a null model. This is implemented via ``model="1"``.

        Parameters
        ----------
        model : str
            Variables to predict censoring via the patsy format. For example, 'var1 + var2 + var3'
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities of remaining uncensored. Helps to avoid near
            positivity violations. Specifying this argument can improve finite sample performance for random
            positivity violations. However, truncating weights leads to additional confounding. Default is False,
            meaning no truncation of predicted probabilities occurs. Providing a single float assumes symmetric
            truncation, where values below or above the threshold are set to the threshold value. Alternatively a list
            of floats can be provided for asymmetric truncation, with the first value being the lower bound and the
            second being the upper bound
        """
        # Saving specifications (used for the bootstrapped variance)
        self._treat_model_ = model
        self._treat_bound_ = bound

        d = self.df.copy()

        # Splitting data set into pieces to fit models
        d0 = d.loc[d[self.sample] == 0].copy()
        d1 = d.loc[d[self.sample] == 1].copy()
        d1[self.treatment] = d1[self.treatment] - 1

        # Fitting treatment model for S=0
        w0_fm = smf.glm(self.treatment + " ~ " + model, d0,              # Fitting nuisance model
                        family=sm.families.family.Binomial()).fit()
        self._nuisance_treatment_.append(w0_fm)                          # Saving nuisance model
        pr_tw0 = w0_fm.predict(d0)                                       # Predicted probabilities
        if self._verbose_:
            print('==============================================================================')
            print('Treatment Model, ' + self.sample + "=0")
            print(w0_fm.summary())
            print('==============================================================================')
        if bound:                                                        # Applying the probability bound
            pr_tw0 = bound_probability(pr_tw0, bounds=bound)

        # Fitting treatment model for S=1
        w1_fm = smf.glm(self.treatment + " ~ " + model, d1,              # Fitting nuisance model
                        family=sm.families.family.Binomial()).fit()
        self._nuisance_treatment_.append(w1_fm)                          # Saving nuisance model
        pr_tw1 = w1_fm.predict(d1)                                       # Predicted probabilities
        if self._verbose_:
            print('==============================================================================')
            print('Treatment Model, ' + self.sample + "=1")
            print(w1_fm.summary())
            print('==============================================================================')
        if bound:                                                        # Applying the probability bound
            pr_tw1 = bound_probability(pr_tw1, bounds=bound)

        # Setting as column in the data set
        self.df['_pr_treat_'] = np.append(np.where(d0[self.treatment] == 1, pr_tw0, 1 - pr_tw0),
                                          np.where(d1[self.treatment] == 1, pr_tw1, 1 - pr_tw1))
        # Marker to indicate model has been specified and fully fit
        self._fit_treatment_ = True

    def censoring_model(self, model, stratify_by_sample=False, strata=None, bound=False, censor_shift=1e-5):
        r"""Censoring model, which predicts the probability of remaining uncensored at time=t as a function of sample,
        covariates, and the action via the Breslow method. This nuisance model corresponds to

        .. math::

            pi_C = \Pr(C>T | W,S; \gamma)

        Note
        ----
        To ensure events happen before censoring, all censoring times are shifted by `censor_shift` (default 1e-5).

        This model is meant to account for informative censoring during follow-up. If there is no loss-to-follow-up
        (unlikely in nearly all contexts), this model is not necessary.

        Parameters
        ----------
        model : str
            Variables to predict censoring via the patsy format. For example, 'var1 + var2 + var3'
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities of remaining uncensored. Helps to avoid near
            positivity violations. Specifying this argument can improve finite sample performance for random
            positivity violations. However, truncating weights leads to additional confounding. Default is False,
            meaning no truncation of predicted probabilities occurs. Providing a single float assumes symmetric
            trunctation, where values below or above the threshold are set to the threshold value. Alternatively a list
            of floats can be provided for asymmetric trunctation, with the first value being the lower bound and the
            second being the upper bound
        stratify_by_sample : bool, optional
            Whether to stratify by study when estimating the Cox Proportional Hazard models for censoring. Default is
            False, which estimates the censoring models using both studies in the same model.
        strata : str, None, optional
            Column to stratify the Cox proportional hazards model. Note that this is different from
            ``stratify_by_sample``, which fits separate models for each study. ``strata`` instead fits a single Cox
            model but the baseline hazard is allowed to vary by ``strata``.
        censor_shift : float, optional
            Float to shift all censoring times by to avoid event/censoring event ties. When tied, events are considered
            to happen right before censoring. Therefore the weights need to respect this. By slightly shifting the
            censoring times by a small positive amount, ties can be prevents. Default is 1e-5.
        """
        # Saving specifications (used for the bootstrapped variance)
        self._censor_model_ = model
        self._censor_stratify_ = stratify_by_sample
        self._censor_strata_ = strata
        self._censor_bound_ = bound
        self._censor_shift = censor_shift

        d = self.df.copy()

        # Applying minor time shift to censor times for calculation of censoring probability (in case of ties)
        d[self.time] = np.where(d[self.censor] == 1,
                                d[self.time] + censor_shift,      # Censored observations are increased a tiny amount
                                d[self.time])                     # All other observations are kept as-is.

        # Stratifying by study for the Cox models (if requested)
        if stratify_by_sample:
            d0 = d.loc[d[self.sample] == 0].copy()
            d1 = d.loc[d[self.sample] == 1].copy()

            if self._verbose_:
                print('==============================================================================')
                print('Censoring Models')
                print('------------------------------------------------------------------------------')
                print(self.sample + "=0")
                print('------------------------------------------------------------------------------')
            cens_fm_s0 = BreslowEstimator(d0, self.time, self.censor, verbose=self._verbose_)
            cens_fm_s0.cox_model(model=model, strata=strata)
            self._nuisance_censoring_.append(cens_fm_s0._cox_model_)
            if self._verbose_:
                print('------------------------------------------------------------------------------')
                print(self.sample + "=1")
                print('------------------------------------------------------------------------------')
            cens_fm_s1 = BreslowEstimator(d1, self.time, self.censor, verbose=self._verbose_)
            cens_fm_s1.cox_model(model=model, strata=strata)
            self._nuisance_censoring_.append(cens_fm_s1._cox_model_)
            if self._verbose_:
                print('==============================================================================')
            # Returning predicted probability of remaining uncensored at T_i^*
            self.df['_pr_uncens_'] = 1 - np.append(cens_fm_s0.predict_risk(),
                                                   cens_fm_s1.predict_risk())

        # Common model for samples / study
        else:
            if self._verbose_:
                print('==============================================================================')
                print('Censoring Models')
            cens_fm = BreslowEstimator(d, self.time, self.censor, verbose=self._verbose_)
            cens_fm.cox_model(model=model, strata=strata)
            if self._verbose_:
                print('==============================================================================')
            # Returning predicted probability of remaining uncensored at T_i^*
            self.df['_pr_uncens_'] = 1 - cens_fm.predict_risk()

        # Applying probability bound if specified
        if bound:
            self.df['_pr_uncens_'] = bound_probability(self.df['_pr_uncens_'], bounds=bound)
        # Marker to indicate model has been specified and fully fit
        self._fit_censor_ = True

    def estimate(self, variance='influence_curve', bs_iterations=200, seed=None, n_cpus=1):
        """Estimate the mean and variance for the fusion parameter of interest. Called after all modeling functions
        have been specified and estimated.

        Parameters
        ----------
        variance : str, None, optional
            Variance estimation method
        bs_iterations : int, optional
            Number of iterations to run the bootstrap for. Default is 200
        seed : int, optional
            Random seed for the bootstrap resample. Default is None
        n_cpus : int, optional
            Number of CPU cores to use for the bootstrapping procedure. Default is 1 (single core)

        Returns
        -------
        pandas.DataFrame
            Data frame containing the estimated risk difference at each unique event time.
        """
        # Error checking
        if not self._fit_location_:
            raise ValueError("`sampling_model()` must be specified before calling `fit()`")
        if not self._fit_treatment_:
            raise ValueError("`treatment_model()` must be specified before calling `fit()`")
        if not self._fit_censor_:
            warnings.warn("The weighted empirical distribution function requires that a censoring model is specified"
                          "when any censoring occurs. It appears that you did not call `censoring_model()` prior to "
                          "calling `fit()`. If any censoring occurs in your data set, you MUST specify that model. "
                          "Otherwise your results are likely to be biased.",
                          UserWarning)
            self.df['_pr_uncens_'] = 1

        # Setup for looping (weights for calculations)
        fuse_weight = np.where(self.df[self.sample] == 1,                              # creating fusion weights
                               1,                                                      # If location, fusion weight=1
                               (1 - self.df['_pr_local_']) / (self.df['_pr_local_']))  # otherwise inverse odds weight
        self._baseline_weight_ = self.df["_pr_treat_"] * fuse_weight                   # creating weights at baseline
        self._timed_weight_ = self.df['_pr_uncens_']                                   # weight that varies with time

        # Estimation
        if variance is None:
            results = self._estimate_risk_function_(return_variance=False)
        elif variance.lower() == 'influence_curve':
            results = self._estimate_risk_function_(return_variance=True)
        elif variance.lower() == 'bootstrap':
            results = self._estimate_risk_function_(return_variance=False)
            results = results[["t", "RD", "RR", "R1D", "R2_S1", "R1_S1", "R1_S0", "R0_S0"]].set_index("t")

            # Getting params setup to pass to the Pool'd function
            rng = np.random.default_rng(seed)                                          # Setting the seed for bootstraps
            ids_s1 = list(self.df.loc[self.df[self.sample] == 1].index)                # Get indices of units with S=1
            ids_s0 = list(self.df.loc[self.df[self.sample] == 0].index)                # Get indices of units with S=0

            params = [[self.df,
                       # Getting random samples (w/ replace) of the S=1 and S=0 indices separately
                       np.concatenate((rng.choice(ids_s1, size=len(ids_s1), replace=True),
                                       rng.choice(ids_s0, size=len(ids_s0), replace=True))),
                       # Extracting the model specifications from above
                       self.treatment, self.outcome, self.time, self.sample, self.censor,
                       self._sample_model_, self._sample_bound_,
                       self._treat_model_, self._treat_bound_,
                       self._censor_model_, self._censor_stratify_, self._censor_strata_, self._censor_bound_,
                       self._censor_shift
                       ] for i in range(bs_iterations)]                                # iterations

            # Using pool to multiprocess the bootstrapping procedure
            with Pool(processes=n_cpus) as pool:
                bsd = list(pool.map(_bootstrap_single_,                         # Call outside function to run parallel
                                    params))                                    # provide packed input list

            # Processing bootstrapped samples into variance estimates
            cols = bsd[0].columns
            variance_estimates = pd.DataFrame()  # Create blank data set to store variance results
            for c in cols:                                     # for each column (output parameter to estimate variance)
                bootstraps = pd.concat([b[c] for b in bsd],    # concat the list of results for a specific column
                                       ignore_index=False,     # ...align on the index (time)
                                       axis=1)                 # ...stack as new columns
                bootstraps = bootstraps.ffill(axis=0)          # Forward fill (so everything is on same t axis)
                var_estimate = np.var(bootstraps,              # Calculate variance for columns
                                      axis=1,                  # ...variance across rows (for each unique t)
                                      ddof=1)                  # ...divisor is (n-1)
                variance_estimates[c + "_SE"] = np.sqrt(var_estimate)

            results = pd.concat((results, variance_estimates),
                                ignore_index=False, axis=1)
            results = results.reset_index(drop=False)
        else:
            raise ValueError("The variance option: '" + str(variance) + "' is not supported. Only "
                             "'bootstrap' and 'influence_curve' are currently supported.")

        # Calculating corresponding confidence intervals
        zalpha = norm.ppf(1 - self._alpha_ / 2, loc=0, scale=1)
        results['RD_LCL'] = results['RD'] - zalpha * results['RD_SE']
        results['RD_UCL'] = results['RD'] + zalpha * results['RD_SE']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            results['RR_LCL'] = np.exp(np.log(results['RR']) - zalpha * results['RR_SE'])
            results['RR_UCL'] = np.exp(np.log(results['RR']) + zalpha * results['RR_SE'])
        results['R1D_LCL'] = results['R1D'] - zalpha * results['R1D_SE']
        results['R1D_UCL'] = results['R1D'] + zalpha * results['R1D_SE']

        # Returning the output results
        return results[["t", "RD", "RD_SE", "RD_LCL", "RD_UCL",
                        "RR", "RR_SE", "RR_LCL", "RR_UCL",
                        "R1D", "R1D_SE", "R1D_LCL", "R1D_UCL",
                        "R2_S1", "R2_S1_SE", "R1_S1", "R1_S1_SE", "R1_S0", "R1_S0_SE", "R0_S0", "R0_S0_SE"]]

    def diagnostic_plot(self, figsize=(5, 7)):
        """Generates diagnostic twister plot for the bridged treatment comparison. The diagnostic compares the risks
        between the shared arms, which should be approximately zero across follow-up.

        Parameters
        ----------
        figsize : set, list, optional
            Adjust the size of the diagnostic twister plot

        Returns
        -------
        matplotlib axes
        """
        estimates = self.estimate()
        ax = twister_plot(data=estimates,
                          xvar='R1D', lcl='R1D_LCL', ucl='R1D_UCL',
                          yvar='t',
                          treat_labs=None, treat_labs_top=True,
                          figsize=figsize)
        m = np.max([np.max(np.abs(estimates['R1D_LCL'])), np.max(np.abs(estimates['R1D_UCL']))])
        ax.set_xlim([-m - 0.05, m + 0.05])
        ax.set_xlabel("Difference in Shared Arms")
        ax.set_ylabel("time")
        return ax

    def permutation_test(self, permutation_n=1000, signed=False, decimal=3, plot_results=True, print_results=True,
                         n_cpus=1, figsize=(7, 5), seed=None):
        """Permutation test to diagnose the validity of the bridged comparison through the shared intermediate shared
        arm between the trials.

        Parameters
        ----------
        permutation_n : int, optional
            Number of permutations to run. Default is to use 2000.
        signed : bool, optional
            Whether to calculate the geometric-area (all non-negative values) or the signed-area (all values).
        decimal: int, optional
            Number of decimal places to display in the printed output.
        plot_results : bool, optional
            Whether to present a histogram of the permutations and the observed area. Default is True
        print_results : bool, optional
            Whether to print the permutation results to the console. Default is True.
        n_cpus : int, optional
            Number of CPUs to pool together for the permutation test. Increasing the number of available CPUs should
            speed up the permutation procedure.
        figsize : set, list, optional
            Adjust the size of the diagnostic twister plot
        seed : None, int, optional
            Random seed.
        """
        # Setting the seed
        rng = np.random.default_rng(seed)

        # Step 1: calculate area between steps
        estimates = self.estimate()
        obs_area = area_between_steps(data=estimates,
                                      time='t', prob1='R1_S1', prob2='R1_S0',
                                      signed=signed)

        # Step 2: Area under permutations
        d = self.df.copy()
        d['_base_weights_'] = 1 / self._baseline_weight_
        d['_full_weights_'] = 1 / (self._baseline_weight_ * self._timed_weight_)

        # Restricting to intermediate (shared) trial arm only
        d = d.loc[d[self.treatment] == 1].copy()

        # Conducting permutations!
        input_params = [[d,                                                       # data set to permute
                         self.event_times, self.sample, self.time, self.outcome,  # dynamic columns to use
                         signed,                                                  # whether to use the signed variation
                         rng.permutation(d[self.sample])                          # shuffle the study indicators
                         ] for i in range(permutation_n)]                         # and permutation number copies
        with Pool(processes=n_cpus) as pool:
            permutation_area = list(pool.map(_permute_,       # Call _permute function from outside to run in parallel
                                             input_params))   # provide packed input list

        # Step 3: calculating P-value
        p_value = np.mean(np.where(permutation_area > obs_area, 1, 0))

        if print_results:
            print('======================================================================')
            print('       Fusion Inverse Probability Weighting Diagnostic Test           ')
            print('======================================================================')
            fmt = 'Observed area:    {:<15} No. Permutations:     {:<20}'
            print(fmt.format(np.round(obs_area, decimal), permutation_n))
            print('----------------------------------------------------------------------')
            print("P-value:", np.round(p_value, 3))
            print('======================================================================')
        if plot_results:
            fig, ax = plt.subplots(figsize=figsize)                          # fig_size is width by height
            ax.hist(permutation_area, bins=50, color="gray", density=True)
            ax.axvline(x=obs_area, color="red")
            return ax
        else:
            return p_value

    def plot(self, figsize=(5, 7)):
        """Generates a twister plot of the risk difference estimates.

        Parameters
        ----------
        figsize : set, list, optional
            Adjust the size of the diagnostic twister plot

        Returns
        -------
        matplotlib axes
        """
        rd_df = self.estimate()
        ax = twister_plot(data=rd_df,
                          xvar='RD', lcl='RD_LCL', ucl='RD_UCL', yvar='t',
                          treat_labs=None, treat_labs_top=True, figsize=figsize)
        m = np.max([np.max(np.abs(rd_df['LCL'])), np.max(np.abs(rd_df['UCL']))])
        ax.set_xlim([-m - 0.05, m + 0.05])
        ax.set_xlabel("Risk Difference")
        ax.set_ylabel("time")
        return ax

    def _estimate_risk_function_(self, return_variance):
        # Setup for looping (storage)
        results = pd.DataFrame(columns=["t",
                                        "RD", "RD_SE", "RR", "RR_SE", "R1D", "R1D_SE",
                                        "R2_S1", "R2_S1_SE", "R1_S1", "R1_S1_SE",
                                        "R1_S0", "R1_S0_SE", "R0_S0", "R0_S0_SE", ])

        # Looping through all events times to generate curves
        for time in self.event_times:
            # Estimating risk functions at a particular time
            psi_t = _estimate_risk_functions_at_t_(current_time=time,
                                                   data=self.df,
                                                   location=self.sample, treatment=self.treatment,
                                                   delta=self.outcome, time_variable=self.time,
                                                   full_weight=self._baseline_weight_ * self._timed_weight_,
                                                   n_local=self.n_local, n_distal=self.n_distal_weighted,
                                                   return_variance=return_variance)
            # Stacking results together
            results = results.append(psi_t, ignore_index=True)

        # Return estimates (possibly with variance estimates
        return results


def _bootstrap_single_(params):
    """

    """
    # Unpacking input parameters
    (data, sample_index,
     treatment, outcome, time, sample, censor,
     sampling_model, sample_pr_bound,
     treatment_model, treat_pr_bound,
     censoring_model, censor_stratify, censor_strata, censor_pr_bound, censor_shift) = params

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # Hides excessive censor=None warns
        # Create a fresh class with the resampled data
        estr = SurvivalFusionIPW(df=data.iloc[sample_index].copy(),
                                 treatment=treatment, outcome=outcome, time=time,
                                 sample=sample, censor=censor, verbose=False)
        # Estimate the same sampling model with the same specifications
        estr.sampling_model(sampling_model, bound=sample_pr_bound)
        # Estimate the same treatment model with the same specifications
        estr.treatment_model(treatment_model, bound=treat_pr_bound)
        # Estimate the same censoring model with the same specifications
        estr.censoring_model(censoring_model, stratify_by_sample=censor_stratify, strata=censor_strata,
                             bound=censor_pr_bound, censor_shift=censor_shift)
        # Estimate the point estimates for the sampled data
        x = estr.estimate(variance=None)
    return x[["t", "RD", "RR", "R1D", "R2_S1", "R1_S1", "R1_S0", "R0_S0"]].set_index("t")


def _permute_(params):
    """self-contained function to permute the data. This function is called by the multiprocessing (so we
    can run in parallel for quicker speed). Speed depends on the number of CPUs given
    """
    # Unpack the parameters
    d, times, local_col, time_col, y_col, signed, shuffled = params

    # Step 2a: permute the study indicator labels
    dc = d.copy()
    dc[local_col] = shuffled

    # Calculating new N's based on the baseline weights (IPTW & IOSW)
    n_local = np.sum(dc[local_col] * dc['_base_weights_'])
    n_distal = np.sum((1 - dc[local_col]) * dc['_base_weights_'])

    # Step 2b: weighted empirical distribution function stratified by study
    distal_r1 = []
    local_r1 = []
    for tau in times:
        r_local, r_distal = _estimate_diagnostic_functions_at_t_(current_time=tau,
                                                                 data=dc, location=local_col, delta=y_col,
                                                                 time_variable=time_col,
                                                                 full_weight=dc['_full_weights_'],
                                                                 n_local=n_local, n_distal=n_distal)
        local_r1.append(r_local)
        distal_r1.append(r_distal)

    edf_curve = pd.DataFrame({"t": times,
                              "pR_p1": distal_r1,
                              "pR_p0": local_r1})

    # Step 2c: calculate area under permuted curves
    area = area_between_steps(data=edf_curve,
                              time=time_col, prob1='pR_p1', prob2='pR_p0',
                              signed=signed)
    return area


def _estimate_risk_functions_at_t_(current_time, data, location, treatment, delta, time_variable,
                                   full_weight, n_local, n_distal, return_variance):
    # Pr_{distal}(Y^{R=1})
    numerator = (np.asarray(1 - data[location]) *  # I(W=d)
                 np.where(data[treatment] == 1, 1, 0) *  # I(R=1)
                 np.where(data[time_variable] <= current_time, 1, 0) *  # I(T<=t)
                 data[delta])  # Y
    pr_fusion_r1_i = numerator / full_weight
    pr_fusion_r1 = np.sum(pr_fusion_r1_i) / n_distal

    # Pr_{distal}(Y^{R=0})
    numerator = (np.asarray(1 - data[location]) *  # I(W=d)
                 np.where(data[treatment] == 0, 1, 0) *  # I(R=0)
                 np.where(data[time_variable] <= current_time, 1, 0) *  # I(T<=t)
                 data[delta])  # Y
    pr_fusion_r0_i = numerator / full_weight
    pr_fusion_r0 = np.sum(pr_fusion_r0_i) / n_distal

    # Pr_{local}(Y^{R=2})
    numerator = (data[location] *  # I(W=l)
                 np.where(data[treatment] == 2, 1, 0) *  # I(R=2)
                 np.where(data[time_variable] <= current_time, 1, 0) *  # I(T<=t)
                 data[delta])  # Y
    pr_local_r2_i = numerator / full_weight
    pr_local_r2 = np.sum(pr_local_r2_i) / n_local

    # Pr_{local}(Y^{R=1})
    numerator = (data[location] *  # I(W=l)
                 np.where(data[treatment] == 1, 1, 0) *  # I(R=1)
                 np.where(data[time_variable] <= current_time, 1, 0) *  # I(T<=t)
                 data[delta])  # Y
    pr_local_r1_i = numerator / full_weight
    pr_local_r1 = np.sum(pr_local_r1_i) / n_local

    # Point estimates
    psi = (pr_local_r2 - pr_local_r1) + (pr_fusion_r1 - pr_fusion_r0)
    psi_diag = pr_local_r1 - pr_fusion_r1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psi_ratio = (pr_local_r2 / pr_fusion_r0) * (pr_fusion_r1 / pr_local_r1)

    # Variance estimates (optional)
    if return_variance:
        var = np.sum(((pr_local_r2_i - pr_local_r1_i) + (pr_fusion_r1_i - pr_fusion_r0_i)
                      - psi) ** 2) / (n_local ** 2)
        diag_var = np.sum((pr_local_r1_i - pr_fusion_r1_i - psi_diag) ** 2) / (n_local ** 2)
        r2s1_var = np.sum((pr_local_r2_i - pr_local_r2) ** 2) / (n_local ** 2)
        r1s1_var = np.sum((pr_local_r1_i - pr_local_r1) ** 2) / (n_local ** 2)
        r1s0_var = np.sum((pr_fusion_r1_i - pr_fusion_r1) ** 2) / (n_local ** 2)
        r0s0_var = np.sum((pr_fusion_r0_i - pr_fusion_r0) ** 2) / (n_local ** 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            log_rr_var = ((1 / pr_local_r2) ** 2 * r2s1_var + (1 / pr_local_r1) ** 2 * r1s1_var
                          + (1 / pr_fusion_r1) ** 2 * r1s0_var + (1 / pr_fusion_r0) ** 2 * r0s0_var)
    else:
        var, log_rr_var, diag_var, r2s1_var, r1s1_var, r1s0_var, r0s0_var = [np.nan, ] * 7

    result_at_t = {"t": current_time,
                   "RD": psi, "RD_SE": np.sqrt(var),
                   "RR": psi_ratio, "RR_SE": np.sqrt(log_rr_var),
                   "R1D": psi_diag, "R1D_SE": np.sqrt(diag_var),
                   "R2_S1": pr_local_r2, "R2_S1_SE": np.sqrt(r2s1_var),
                   "R1_S1": pr_local_r1, "R1_S1_SE": np.sqrt(r1s1_var),
                   "R1_S0": pr_fusion_r1, "R1_S0_SE": np.sqrt(r1s0_var),
                   "R0_S0": pr_fusion_r0, "R0_S0_SE": np.sqrt(r0s0_var),
                   }
    return result_at_t


def _estimate_diagnostic_functions_at_t_(current_time, data, location, delta, time_variable, full_weight,
                                         n_local, n_distal):
    # Pr_{distal}(Y^{R=1})
    numerator = (np.asarray(1 - data[location]) *  # I(W=d)
                 np.where(data[time_variable] <= current_time, 1, 0) *  # I(T<=t)
                 data[delta])  # Y
    pr_fusion_r1_i = numerator * full_weight
    pr_fusion_r1 = np.sum(pr_fusion_r1_i) / n_distal

    # Pr_{local}(Y^{R=1})
    numerator = (data[location] *  # I(W=l)
                 np.where(data[time_variable] <= current_time, 1, 0) *  # I(T<=t)
                 data[delta])  # Y
    pr_local_r1_i = numerator * full_weight
    pr_local_r1 = np.sum(pr_local_r1_i) / n_local

    # Returning results
    return pr_local_r1, pr_fusion_r1
