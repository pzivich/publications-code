import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from amonhen.utils import (probability_to_odds, odds_to_probability, bounding, propensity_score,
                           stochastic_check_conditional,
                           exposure_machine_learner, stochastic_outcome_machine_learner, stochastic_outcome_predict)


class StochasticTMLE:
    r"""Implementation of target maximum likelihood estimator for stochastic treatment plans. This implementation
    calculates TMLE for a time-fixed exposure and a single time-point outcome under a stochastic treatment plan of
    interest. By default, standard parametric regression models are used to calculate the estimate of interest. The
    StochasticTMLE estimator allows users to instead use machine learning algorithms from sklearn and PyGAM.

    Note
    ----
    Valid confidence intervals are only attainable with certain machine learning algorithms. These algorithms must be
    Donsker class for valid confidence intervals. GAM and LASSO are examples of alogorithms that are Donsker class.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    exposure : str
        Column label for the exposure of interest
    outcome : str
        Column label for the outcome of interest
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        Optional argument to control the bounding feature for continuous outcomes. The bounding process may result
        in values of 0,1 which are undefined for logit(x). This parameter adds or substracts from the scenarios of
        0,1 respectively. Default value is 0.0005
    verbose : bool, optional
        Optional argument for verbose estimation. With verbose estimation, the model fits for each result are printed
        to the console. It is highly recommended to turn this parameter to True when conducting model diagnostics

    Following is a general narrative of the estimation procedure for TMLE with stochastic treatments
    1. Initial estimators for exposure and outcome models are fit. By default these estimators are based
    on parametric regression models. Additionally, machine learning algorithms can be used.

    2. The auxiliary covariate is calculated (i.e. IPTW).

    .. math::
        H = \frac{p}{\widehat{\Pr}(A=a)}

    where `p` is the probability of treatment `a` under the stochastic intervention of interest.

    3. Targeting step occurs through estimation of `e` via a logistic regression model. Briefly a weighted logistic
    regression model (weighted by the auxiliary covariates) with the dependent variable as the observed outcome and
    an offset term of the outcome model predictions under the observed treatment (A).

    .. math::
        \text{logit}(Y) = \text{logit}(Q(A, W)) + \epsilon

    4. Stochastic interventions are evaluated through Monte Carlo integration for binary treatments. The different
    treatment plans are randomly applied and evaluated through the outcome model and then the targeting step via

    .. math::
        E[\text{logit}(Q(A=a, W)) + \hat{\epsilon}]

    This process is repeated a large number of times and the point estimate is the average of those individual treatment
    plans.

    Examples
    --------

    Setting up environment

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.doublyrobust import StochasticTMLE
    >>> df = load_sample_data(False).dropna()
    >>> df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)

    Estimating TMLE for 0.2 being treated with ART

    >>> tmle = StochasticTMLE(df, exposure='art', outcome='dead')
    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> tmle.outcome_model('art + male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> tmle.fit(p=0.2)
    >>> tmle.summary()

    Estimating TMLE for conditional plan

    >>> tmle = StochasticTMLE(df, exposure='art', outcome='dead')
    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> tmle.outcome_model('art + male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> tmle.fit(p=[0.6, 0.4], conditional=["df['male']==1", "df['male']==0"])
    >>> tmle.summary()

    Estimating TMLE with machine learning algorithm from sklearn

    >>> from sklearn.linear_model import LogisticRegression
    >>> log1 = LogisticRegression(penalty='l1', random_state=201)
    >>> tmle = StochasticTMLE(df, 'art', 'dead')
    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', custom_model=log1)
    >>> tmle.outcome_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', custom_model=log1)
    >>> tmle.fit(p=0.75)

    References
    ----------

    Muñoz ID, and Van Der Laan MJ. Population intervention causal effects based on stochastic interventions.
    Biometrics 68.2 (2012): 541-549.

    van der Laan MJ, and Sherri R. Targeted learning in data science: causal inference for complex longitudinal
    studies. Springer Science & Business Media, 2011.
    """
    def __init__(self, df, exposure, outcome, alpha=0.05, continuous_bound=0.0005, verbose=False):
        # Dropping ALL missing data (doesn't allow for censored outcomes)
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the data set. StochasticTMLE will drop all missing data. "
                          "StochasticTMLE will fit "
                          + str(df.dropna().shape[0]) +
                          ' of ' + str(df.shape[0]) + ' observations', UserWarning)
            self.df = df.copy().dropna().reset_index()
        else:
            self.df = df.copy().reset_index()

        if not df[exposure].value_counts().index.isin([0, 1]).all():
            raise ValueError("StochasticTMLE only supports binary exposures currently")

        # Manage outcomes
        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
            self._cb = 0.0
        else:
            self._continuous_outcome = True
            self._continuous_min = np.min(df[outcome])
            self._continuous_max = np.max(df[outcome])
            self._cb = continuous_bound
            self.df[outcome] = _tmle_unit_bounds_(y=df[outcome], mini=self._continuous_min,
                                                  maxi=self._continuous_max, bound=self._cb)
            self._q_min_bound = np.min(self.df[outcome])
            self._q_max_bound = np.max(self.df[outcome])

        self.exposure = exposure
        self.outcome = outcome

        # Output attributes
        self.epsilon = None
        self.marginals_vector = None
        self.marginal_outcome = None
        self.alpha = alpha
        self.marginal_se = None
        self.marginal_ci = None
        self.conditional_se = None
        self.conditional_ci = None

        # Storage for items I need later
        self._outcome_model = None
        self._q_model = None
        self._Qinit_ = None
        self._treatment_model = None
        self._g_model = None
        self._resamples_ = None
        self._specified_bound_ = None
        self._denominator_ = None
        self._verbose_ = verbose
        self._out_model_custom = False
        self._exp_model_custom = False
        self._continuous_type = None

        # Custom model / machine learner storage
        self._g_custom_ = None
        self._q_custom_ = None

    def exposure_model(self, model, custom_model=None, bound=None):
        """Estimation of the exposure model, Pr(A=1|W). This value is used as the denominator for the inverse
        probability weights.

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            truncating weights leads to additional confounding. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation, where values below
            or above the threshold are set to the threshold value. Alternatively a list of floats can be provided for
            asymmetric trunctation, with the first value being the lower bound and the second being the upper bound
        """
        self._g_model = self.exposure + ' ~ ' + model

        if custom_model is None:  # Standard parametric regression model
            fitmodel = propensity_score(self.df, self._g_model, print_results=self._verbose_)
            pred = fitmodel.predict(self.df)
        else:  # User-specified prediction model
            self._exp_model_custom = True
            data = patsy.dmatrix(model + ' - 1', self.df)
            pred = exposure_machine_learner(xdata=np.asarray(data), ydata=np.asarray(self.df[self.exposure]),
                                            ml_model=custom_model,
                                            pdata=np.asarray(data))

        if bound is not None:
            pred2 = bounding(ipw=pred, bound=bound)
            self._specified_bound_ = np.sum(np.where(pred2 == pred, 0, 1))
            pred = pred2

        self._denominator_ = np.where(self.df[self.exposure] == 1, pred, 1 - pred)

    def outcome_model(self, model, custom_model=None, continuous_distribution='gaussian'):
        """Estimation of the outcome model, E(Y|A,W).

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
        continuous_distribution : str, optional
            Distribution to use for continuous outcomes. Options are 'gaussian' for normal distributions and 'poisson'
            for Poisson distributions
        """
        self._q_model = self.outcome + ' ~ ' + model

        if custom_model is None:  # Standard parametric regression
            self._out_model_custom = False
            self._continuous_type = continuous_distribution
            if self._continuous_outcome:
                if (continuous_distribution.lower() == 'gaussian') or (continuous_distribution.lower() == 'normal'):
                    f = sm.families.family.Gaussian()
                elif continuous_distribution.lower() == 'poisson':
                    f = sm.families.family.Poisson()
                else:
                    raise ValueError("Only 'gaussian' and 'poisson' distributions are supported for continuous "
                                     "outcomes")
                self._outcome_model = smf.glm(self._q_model, self.df, family=f).fit()
            else:
                f = sm.families.family.Binomial()
                self._outcome_model = smf.glm(self._q_model, self.df, family=f).fit()
            if self._verbose_:
                print('==============================================================================')
                print('Q-model')
                print(self._outcome_model.summary())

            # Step 2) Estimation under the scenarios
            self._Qinit_ = self._outcome_model.predict(self.df)

        else:  # User-specified model
            self._out_model_custom = True
            data = patsy.dmatrix(model + ' - 1', self.df)
            output = stochastic_outcome_machine_learner(xdata=np.asarray(data),
                                                        ydata=np.asarray(self.df[self.outcome]),
                                                        ml_model=custom_model,
                                                        continuous=self._continuous_outcome,
                                                        print_results=self._verbose_)
            self._Qinit_, self._outcome_model = output

        if self._continuous_outcome:  # Ensures all predicted values are bounded
            self._Qinit_ = np.where(self._Qinit_ < self._q_min_bound, self._q_min_bound, self._Qinit_)
            self._Qinit_ = np.where(self._Qinit_ > self._q_max_bound, self._q_max_bound, self._Qinit_)

    def fit(self, p, conditional=None, samples=100, seed=None):
        """Calculate the mean from the predicted exposure probabilities and predicted outcome values using the TMLE
        procedure. Confidence intervals are calculated using influence curves.

        Parameters
        ----------
        p : float, list, tuple
            Proportion that correspond to the number of persons treated (all values must be between 0.0 and 1.0). If
            conditional is specified, p must be a list/tuple of floats of the same length
        conditional : None, list, tuple, optional
            A
        samples : int, optional
            Number of samples to use for the Monte Carlo integration procedure
        seed : None, int, optional
            Seed for the Monte Carlo integration procedure

        Note
        ----
        Exposure and outcome models must be specified prior to `fit()`

        Returns
        -------
        `StochasticTMLE` gains `marginal_vector` and `marginal_outcome` along with `marginal_ci`
        """
        if self._denominator_ is None:
            raise ValueError("The exposure_model() function must be specified before the fit() function")
        if self._Qinit_ is None:
            raise ValueError("The outcome_model() function must be specified before the fit() function")

        if seed is None:
            pass
        else:
            np.random.seed(seed)

        p = np.array(p)
        if np.any(p > 1) or np.any(p < 0):
            raise ValueError("All specified treatment probabilities must be between 0 and 1")
        if conditional is not None:
            if len(p) != len(conditional):
                raise ValueError("'p' and 'conditional' must be the same length")

        # Step 1) Calculating clever covariate (HAW)
        if conditional is None:
            numerator = np.where(self.df[self.exposure] == 1, p, 1 - p)
        else:
            df = self.df.copy()
            stochastic_check_conditional(df=self.df, conditional=conditional)
            numerator = np.array([np.nan] for i in range(self.df.shape[0]))
            for c, prop in zip(conditional, p):
                numerator = np.where(eval(c), np.where(df[self.exposure] == 1, prop, 1 - prop), numerator)

        haw = np.array(numerator / self._denominator_).astype(float)

        # Step 2) Estimate from Q-model
        # process completed in outcome_model() function and stored in self._Qinit_

        # Step 3) Target parameter TMLE
        self.epsilon = self.targeting_step(y=self.df[self.outcome], q_init=self._Qinit_, iptw=haw,
                                           verbose=self._verbose_)

        # Step 4) Monte-Carlo Integration procedure
        q_star_list = []
        q_i_star_list = []
        self._resamples_ = samples
        for i in range(samples):
            # Applying treatment plan
            df = self.df.copy()
            if conditional is None:
                df[self.exposure] = np.random.binomial(n=1, p=p, size=df.shape[0])
            else:
                df[self.exposure] = np.nan
                for c, prop in zip(conditional, p):
                    df[self.exposure] = np.random.binomial(n=1, p=prop, size=df.shape[0])

            # Outcome model under treatment plan
            if self._out_model_custom:
                _, data_star = patsy.dmatrices(self._q_model + ' - 1', self.df)
                y_star = stochastic_outcome_predict(xdata=data_star,
                                                    fit_ml_model=self._outcome_model,
                                                    continuous=self._continuous_outcome)
            else:
                y_star = self._outcome_model.predict(df)

            if self._continuous_outcome:  # Ensures all predicted values are bounded
                y_star = np.where(y_star < self._q_min_bound, self._q_min_bound, y_star)
                y_star = np.where(y_star > self._q_max_bound, self._q_max_bound, y_star)

            # Targeted Estimate
            logit_qstar = np.log(probability_to_odds(y_star)) + self.epsilon  # logit(Y^*) + e
            q_star = odds_to_probability(np.exp(logit_qstar))  # Y^*
            q_i_star_list.append(q_star)  # Saving Y_i^* for marginal variance
            q_star_list.append(np.mean(q_star))  # Saving E[Y^*]

        if self._continuous_outcome:
            self.marginals_vector = _tmle_unit_unbound_(np.array(q_star_list),
                                                        mini=self._continuous_min, maxi=self._continuous_max)
            y_ = np.array(_tmle_unit_unbound_(self.df[self.outcome], mini=self._continuous_min,
                                              maxi=self._continuous_max))
            yq0_ = _tmle_unit_unbound_(self._Qinit_, mini=self._continuous_min, maxi=self._continuous_max)
            yqstar_ = _tmle_unit_unbound_(np.array(q_i_star_list), mini=self._continuous_min, maxi=self._continuous_max)

        else:
            self.marginals_vector = q_star_list
            y_ = np.array(self.df[self.outcome])
            yq0_ = self._Qinit_
            yqstar_ = np.array(q_i_star_list)

        self.marginal_outcome = np.mean(self.marginals_vector)

        # Step 5) Estimating Var(psi)
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        # Marginal variance estimator
        variance_marginal = self.est_marginal_variance(haw=haw, y_obs=y_, y_pred=yq0_,
                                                       y_pred_targeted=np.mean(yqstar_, axis=0),
                                                       psi=self.marginal_outcome)
        self.marginal_se = np.sqrt(variance_marginal) / np.sqrt(self.df.shape[0])
        self.marginal_ci = [self.marginal_outcome - zalpha * self.marginal_se,
                            self.marginal_outcome + zalpha * self.marginal_se]

        # Conditional on W variance estimator (not generally recommended but I need it for other work)
        variance_conditional = self.est_conditional_variance(haw=haw, y_obs=y_, y_pred=yq0_)
        self.conditional_se = np.sqrt(variance_conditional) / np.sqrt(self.df.shape[0])
        self.conditional_ci = [self.marginal_outcome - zalpha * self.conditional_se,
                               self.marginal_outcome + zalpha * self.conditional_se]

    def summary(self, decimal=3):
        """Prints summary of the estimated incidence under the specified treatment plan
        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if self.marginal_outcome is None:
            raise ValueError('The fit() statement must be ran before summary()')

        print('======================================================================')
        print('          Stochastic Targeted Maximum Likelihood Estimator            ')
        print('======================================================================')

        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} No. Truncated:        {:<20}'
        if self._specified_bound_ is None:
            b = 0
        else:
            b = self._specified_bound_
        print(fmt.format(self.outcome, b))
        fmt = 'Q-Model:          {:<15} g-model:              {:<20}'
        print(fmt.format('Logistic', 'Logistic'))
        fmt = 'No. Resamples:    {:<15}'
        print(fmt.format(self._resamples_))

        print('======================================================================')
        print('Overall incidence:      ', np.round(self.marginal_outcome, decimals=decimal))
        print('======================================================================')
        print('Marginal 95% CL:        ', np.round(self.marginal_ci, decimals=decimal))
        print('Conditional 95% CL:     ', np.round(self.conditional_ci, decimals=decimal))
        print('======================================================================')

    @staticmethod
    def targeting_step(y, q_init, iptw, verbose):
        f = sm.families.family.Binomial()
        log = sm.GLM(y,  # Outcome / dependent variable
                     np.repeat(1, y.shape[0]),  # Generating intercept only model
                     offset=np.log(probability_to_odds(q_init)),  # Offset by g-formula predictions
                     freq_weights=iptw,  # Weighted by calculated IPW
                     family=f).fit()

        if verbose:  # Optional argument to print each intermediary result
            print('==============================================================================')
            print('Targeting Model')
            print(log.summary())

        return log.params[0]  # Returns single-step estimated Epsilon term

    @staticmethod
    def est_marginal_variance(haw, y_obs, y_pred, y_pred_targeted, psi):
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4965321/
        doqg_psi_sq = (haw*(y_obs - y_pred) + y_pred_targeted - psi)**2
        var_est = np.mean(doqg_psi_sq)
        return var_est

    @staticmethod
    def est_conditional_variance(haw, y_obs, y_pred):
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4965321/
        doqg_psi_sq = (haw*(y_obs - y_pred))**2
        var_est = np.mean(doqg_psi_sq)
        return var_est


# Functions that all TMLEs can call are below
def _tmle_unit_bounds_(y, mini, maxi, bound):
    # bounding for continuous outcomes
    v = (y - mini) / (maxi - mini)
    v = np.where(np.less(v, bound), bound, v)
    v = np.where(np.greater(v, 1-bound), 1-bound, v)
    return v


def _tmle_unit_unbound_(ystar, mini, maxi):
    # unbounding of bounded continuous outcomes
    return ystar*(maxi - mini) + mini
