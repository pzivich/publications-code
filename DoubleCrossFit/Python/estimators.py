import warnings
import copy
import patsy
import numpy as np
import pandas as pd
from scipy.stats import logistic
import statsmodels.api as sm
import statsmodels.formula.api as smf


class GFormula:
    """Implementation of the g-computation algorithm that uses sklearn-style fitting to estimate the outcome nuisance
    function.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    treatment : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> gform = GFormula(data, treatment='X', outcome='Y')
    >>> gform.outcome_model(covariates="X + Z", estimator=LogisticRegression(penalty='none', solver='lbfgs'))
    >>> gform.fit()
    >>> gform.summary()
    """
    def __init__(self, df, treatment, outcome):
        # Prepping data
        self.df = df.copy().dropna().reset_index()
        self.treatment = treatment
        self.outcome = outcome

        # Results storage
        self.risk_difference = None
        self.risk_all = None
        self.risk_none = None

        # intermediate storage
        self._y_covariates = None
        self._y_estimator = None
        self._fit_outcome_ = False

    def outcome_model(self, covariates, estimator):
        """Sets up the outcome model to be estimated in the .fit() function

        Parameters
        ----------
        covariates : str
            String of covariates to include in the outcome nuisance model, following patsy input
        estimator : sklearn object
            Estimator for the outcome nuisance model. This code supports any functions that follow the sklearn API.
        """
        # Specifying outcome model covariates and estimators
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self):
        """Calculates the average causal effect via the g-computation algorithm. Confidence intervals are not
        calculated. A bootstrapping procedure should be used.

        Returns
        -------
        The class object gains an updated `risk_difference` which corresponds to the average causal effect
        """
        # Extract data in correct format
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', self.df))
        ydata = np.asarray(self.df[self.outcome])

        # Fitting machine learner / super learner to each
        try:
            fm = self._y_estimator.fit(X=xdata, y=ydata)
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Treat-all predictions
        df = self.df.copy()
        df[self.treatment] = 1
        x_all = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', df))
        if hasattr(fm, 'predict_proba'):
            r_all = fm.predict_proba(x_all)[:, 1]
        elif hasattr(fm, 'predict'):
            r_all = fm.predict(x_all)
        else:
            raise ValueError("No predict or predict_proba function")

        # Treat-none predictions
        df[self.treatment] = 0
        x_non = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', df))
        if hasattr(fm, 'predict_proba'):
            r_non = fm.predict_proba(x_non)[:, 1]
        elif hasattr(fm, 'predict'):
            r_non = fm.predict(x_non)
        else:
            raise ValueError("No predict or predict_proba function")

        # storing results
        self.risk_all = np.mean(r_all)
        self.risk_none = np.mean(r_non)
        self.risk_difference = self.risk_all - self.risk_none

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if self._fit_outcome_ is False:
            raise ValueError('The outcome model must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('                            G-formula')
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('======================================================================')


class IPTW:
    """Implementation of inverse probability weighted estimator that uses sklearn-style fitting to estimate the
    treatment nuisance function.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    treatment : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> iptw = IPTW(data, treatment='X', outcome='Y')
    >>> iptw.treatment_model(covariates="Z", estimator=LogisticRegression(penalty='none', solver='lbfgs'))
    >>> iptw.fit()
    >>> iptw.summary()
    """
    def __init__(self, df, treatment, outcome):
        # prepping data
        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
        else:
            self._continuous_outcome = True

        self.df = df.copy().reset_index()
        self.treatment = treatment
        self.outcome = outcome

        # results storage
        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

        # intermediate place-holders
        self._a_covariates = None
        self._a_estimator = None
        self._fit_treatment_ = False
        self._gbounds = None

    def treatment_model(self, covariates, estimator, bound=False):
        """Sets up the treatment model and estimates the weights for use in the .fit() function

        Parameters
        ----------
        covariates : str
            String of covariates to include in the outcome nuisance model, following patsy input
        estimator : sklearn object
            Estimator for the outcome nuisance model. This code supports any functions that follow the sklearn API.
        bound : float, list, set, bool, optional
            Allows for symmetric or asymmetric bounds to be placed on the calculated weights
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True

        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', self.df))
        ydata = np.asarray(self.df[self.treatment])

        # Fitting machine learner / super learner to each
        try:
            fm = self._a_estimator.fit(X=xdata, y=ydata)
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner. If there is a predictive model you would "
                            "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                            "can work on adding support")

        # Treat-all category
        x_all = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', self.df))
        if hasattr(fm, 'predict_proba'):
            pr_a = fm.predict_proba(x_all)[:, 1]
        elif hasattr(fm, 'predict'):
            pr_a = fm.predict(x_all)
        else:
            raise ValueError("No predict or predict_proba function")

        # Calculating weights
        if bound:
            pr_a = _bounding_(pr_a, bounds=bound)

        self.df['_iptw_'] = (self.df[self.treatment] / pr_a) + ((1 - self.df[self.treatment]) / (1 - pr_a))

    def fit(self):
        """Calculates the average causal effect via the inverse probability weighted estimator. Confidence intervals
        are calculated using the robust variance approach, which results in conservative variances

        Returns
        -------
        The class object gains an updated `risk_difference` which corresponds to the average causal effect
        """
        # Estimating Risk Difference via GEE
        ind = sm.cov_struct.Independence()
        if self._continuous_outcome:
            f = sm.families.family.Gaussian(sm.families.links.identity)
        else:
            f = sm.families.family.Binomial(sm.families.links.identity)
        df = self.df.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # ignores the DomainWarning output by statsmodels
            fm = smf.gee('Y ~ statin', df.index, df, cov_struct=ind, family=f, weights=df['_iptw_']).fit()

        self.risk_difference = np.asarray(fm.params)[1]
        self.risk_difference_se = np.asarray(fm.bse)[1]
        self.risk_difference_ci = (self.risk_difference - 1.96 * self.risk_difference_se,
                                   self.risk_difference + 1.96 * self.risk_difference_se)

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if self._fit_treatment_ is False:
            raise ValueError('The outcome model must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('                              IPTW')
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('SE(RD): ', round(float(self.risk_difference_se), decimal))
        print('95% CL: ', np.round(self.risk_difference_ci, decimal))
        print('======================================================================')


class AIPTW:
    """Implementation of augmented inverse probability weighted estimator that uses sklearn-style fitting to estimate
    the treatment and outcome nuisance functions.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    treatment : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> aipw = AIPTW(data, treatment='X', outcome='Y')
    >>> aipw.treatment_model(covariates="Z", estimator=LogisticRegression(penalty='none', solver='lbfgs'))
    >>> aipw.outcome_model(covariates="X + Z", estimator=LogisticRegression(penalty='none', solver='lbfgs'))
    >>> aipw.fit()
    >>> aipw.summary()
    """
    def __init__(self, df, treatment, outcome):
        self.df = df.copy().dropna().reset_index()
        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
        else:
            self._continuous_outcome = True
        self.treatment = treatment
        self.outcome = outcome

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False

        self._ps_ = None
        self._pred_Y_a1 = None
        self._pred_Y_a0 = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

        self.risk_all = None
        self.risk_all_ci = None

        self.risk_none = None
        self.risk_none_ci = None

    def treatment_model(self, covariates, estimator, bound=False):
        """Sets up the treatment model and estimates the weights for use in the .fit() function

        Parameters
        ----------
        covariates : str
            String of covariates to include in the outcome nuisance model, following patsy input
        estimator : sklearn object
            Estimator for the outcome nuisance model. This code supports any functions that follow the sklearn API.
        bound : float, list, set, bool, optional
            Allows for symmetric or asymmetric bounds to be placed on the calculated weights
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True

        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', self.df))
        ydata = np.asarray(self.df[self.treatment])

        # Fitting machine learner / super learner to each
        try:
            fm = self._a_estimator.fit(X=xdata, y=ydata)
            # print(fm.summarize())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        x_all = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', self.df))
        if hasattr(fm, 'predict_proba'):
            pr_a = fm.predict_proba(x_all)[:, 1]
        elif hasattr(fm, 'predict'):
            pr_a = fm.predict(x_all)
        else:
            raise ValueError("No predict or predict_proba function")

        if bound:
            pr_a = _bounding_(pr_a, bounds=bound)

        self._ps_ = pr_a

    def outcome_model(self, covariates, estimator):
        """Sets up the outcome model to be estimated in the .fit() function

        Parameters
        ----------
        covariates : str
            String of covariates to include in the outcome nuisance model, following patsy input
        estimator : sklearn object
            Estimator for the outcome nuisance model. This code supports any functions that follow the sklearn API.
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

        # Fitting machine learner / super learner to each
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', self.df))
        ydata = np.asarray(self.df[self.outcome])
        try:
            fm = self._y_estimator.fit(X=xdata, y=ydata)
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Treat-all & Treat-none combinations
        df = self.df.copy()
        df[self.treatment] = 1
        x_all = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', df))
        df = self.df.copy()
        df[self.treatment] = 0
        x_non = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', df))

        if self._continuous_outcome:
            if hasattr(fm, 'predict'):
                r_all = fm.predict(x_all)
                r_non = fm.predict(x_non)
            else:
                raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

        else:
            if hasattr(fm, 'predict_proba'):
                r_all = fm.predict_proba(x_all)[:, 1]
                r_non = fm.predict_proba(x_non)[:, 1]
            elif hasattr(fm, 'predict'):
                r_all = fm.predict(x_all)
                r_non = fm.predict(x_non)
            else:
                raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

        self._pred_Y_a1 = r_all
        self._pred_Y_a0 = r_non

    def fit(self):
        """Calculates the average causal effect via the augmented inverse probability weighted estimator. Confidence
        intervals are calculated using influence curves

        Returns
        -------
        The class object gains an updated `risk_difference` and `risk_difference_ci` which corresponds to the average
        causal effect and the confidence interval
        """
        y_obs = self.df[self.outcome]
        py_a1 = self._pred_Y_a1
        py_a0 = self._pred_Y_a0
        ps_g1 = self._ps_
        ps_g0 = 1 - self._ps_
        y1 = np.where(self.df[self.treatment] == 1,
                      (y_obs / ps_g1) - ((py_a1 * ps_g0) / ps_g1),
                      py_a1)
        y0 = np.where(self.df[self.treatment] == 1,
                      py_a0,
                      (y_obs / ps_g0 - ((py_a0 * ps_g1) / ps_g0)))

        self.risk_all = np.mean(y1)
        self.risk_none = np.mean(y0)
        self.risk_difference = np.mean(y1 - y0)

        risk_all_var = np.var(y1 - self.risk_all, ddof=1) / self.df.shape[0]
        risk_none_var = np.var(y0 - self.risk_none, ddof=1) / self.df.shape[0]
        self.risk_difference_se = np.sqrt(np.var((y1 - y0) - self.risk_difference, ddof=1) / self.df.shape[0])
        self.risk_all_ci = [self.risk_all - 1.96 * np.sqrt(risk_all_var),
                            self.risk_all + 1.96 * np.sqrt(risk_all_var)]
        self.risk_none_ci = [self.risk_none - 1.96 * np.sqrt(risk_none_var),
                             self.risk_none + 1.96 * np.sqrt(risk_none_var)]
        self.risk_difference_ci = [self.risk_difference - 1.96 * self.risk_difference_se,
                                   self.risk_difference + 1.96 * self.risk_difference_se]

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('The treatment and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('                                AIPTW')
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('SE(RD):          ', round(float(self.risk_difference_se), decimal))
        print('95% CL:          ', round(float(self.risk_difference_ci[0]), decimal),
              round(float(self.risk_difference_ci[1]), decimal))
        print('======================================================================')


class TMLE:
    """Implementation of targeted maximum likelihood estimator that uses sklearn-style fitting to estimate
    the treatment and outcome nuisance functions.

    Note
    ----
    Continuous outcomes haven't been thoroughly tested in this implementation. Use the zEpid library stable
    implementation instead in your actual work

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    treatment : str
        Column label for the exposure of interest
    outcome : str
        Column label for the outcome of interest
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        Optional argument to control the bounding feature for continuous outcomes. The bounding process may result
        in values of 0,1 which are undefined for logit(x). This parameter adds or substracts from the scenarios of
        0,1 respectively. Default value is 0.0005

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> tmle = TMLE(data, treatment='X', outcome='Y')
    >>> tmle.treatment_model(covariates="Z", estimator=LogisticRegression(penalty='none', solver='lbfgs'))
    >>> tmle.outcome_model(covariates="X + Z", estimator=LogisticRegression(penalty='none', solver='lbfgs'))
    >>> tmle.fit()
    >>> tmle.summary()
    """
    def __init__(self, df, treatment, outcome, alpha=0.05, continuous_bound=0.0005):
        self.df = df.copy().dropna().reset_index()
        self.exposure = treatment
        self.outcome = outcome

        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
            self._cb = 0.0
        else:
            self._continuous_outcome = True
            self._continuous_min = np.min(df[outcome])
            self._continuous_max = np.max(df[outcome])
            self._cb = continuous_bound
            self.df[outcome] = self._unit_bounds(y=df[outcome], mini=self._continuous_min,
                                                 maxi=self._continuous_max, bound=self._cb)

        self._out_model = None
        self._exp_model = None
        self._miss_model = None
        self._out_model_custom = False
        self._exp_model_custom = False
        self._fit_exposure_model = False
        self._fit_outcome_model = False
        self.alpha = alpha

        self.QA0W = None
        self.QA1W = None
        self.QAW = None
        self.g1W = None
        self.g0W = None
        self.m1W = None
        self.m0W = None
        self._epsilon = None

        self.risk_difference = None
        self.risk_difference_se = None
        self.risk_difference_ci = None

    def treatment_model(self, covariates, estimator, bound=False):
        """Sets up the treatment model and estimates the weights for use in the .fit() function

        Parameters
        ----------
        covariates : str
            String of covariates to include in the outcome nuisance model, following patsy input
        estimator : sklearn object
            Estimator for the outcome nuisance model. This code supports any functions that follow the sklearn API.
        bound : float, list, set, bool, optional
            Allows for symmetric or asymmetric bounds to be placed on the calculated weights
        """
        self._exp_model_custom = True
        data = patsy.dmatrix(covariates + ' - 1', self.df)
        self.g1W = self.exposure_machine_learner(xdata=np.asarray(data), ydata=np.asarray(self.df[self.exposure]),
                                                 ml_model=estimator)
        self.g0W = 1 - self.g1W
        if bound:  # Bounding predicted probabilities if requested
            self.g1W = _bounding_(self.g1W, bounds=bound)
            self.g0W = _bounding_(self.g0W, bounds=bound)

        self._fit_exposure_model = True

    def outcome_model(self, covariates, estimator, bound=False):
        """Sets up the outcome model to be estimated in the .fit() function

        Parameters
        ----------
        covariates : str
            String of covariates to include in the outcome nuisance model, following patsy input
        estimator : sklearn object
            Estimator for the outcome nuisance model. This code supports any functions that follow the sklearn API.
        """
        cc = self.df.copy()

        # Step 1) Prediction for Q (estimation of Q-model)
        data = patsy.dmatrix(covariates + ' - 1', cc)
        dfx = self.df.copy()
        dfx[self.exposure] = 1
        adata = patsy.dmatrix(covariates + ' - 1', dfx)
        dfx = self.df.copy()
        dfx[self.exposure] = 0
        ndata = patsy.dmatrix(covariates + ' - 1', dfx)

        self.QA1W, self.QA0W = self.outcome_machine_learner(xdata=np.asarray(data),
                                                            ydata=np.asarray(cc[self.outcome]),
                                                            all_a=adata, none_a=ndata,
                                                            ml_model=estimator,
                                                            continuous=self._continuous_outcome)

        if not bound:  # Bounding predicted probabilities if requested
            bound = self._cb

        # This bounding step prevents continuous outcomes from being outside the range
        self.QA1W = _bounding_(self.QA1W, bounds=bound)
        self.QA0W = _bounding_(self.QA0W, bounds=bound)
        self.QAW = self.QA1W * self.df[self.exposure] + self.QA0W * (1 - self.df[self.exposure])
        self._fit_outcome_model = True

    def fit(self):
        """Calculates the average causal effect via the targeted maximum likelihood estimator. Confidence
        intervals are calculated using influence curves

        Returns
        -------
        The class object gains an updated `risk_difference` and `risk_difference_ci` which corresponds to the average
        causal effect and the confidence interval
        """
        if (self._fit_exposure_model is False) or (self._fit_outcome_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')
        # Step 4) Calculating clever covariate (HAW)
        H1W = self.df[self.exposure] / self.g1W
        H0W = -(1 - self.df[self.exposure]) / self.g0W
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        y = self.df[self.outcome]
        log = sm.GLM(y, np.column_stack((H1W, H0W)), offset=np.log(self.probability_to_odds(self.QAW)),
                     family=f, missing='drop').fit()
        self._epsilon = log.params
        Qstar1 = logistic.cdf(np.log(self.probability_to_odds(self.QA1W)) + self._epsilon[0] / self.g1W)
        Qstar0 = logistic.cdf(np.log(self.probability_to_odds(self.QA0W)) - self._epsilon[1] / self.g0W)
        Qstar = log.predict(np.column_stack((H1W, H0W)), offset=np.log(self.probability_to_odds(self.QAW)))

        # Step 6) Calculating Psi
        zalpha = 1.96

        if self._continuous_outcome:
            # Calculating Average Treatment Effect
            Qstar = self._unit_unbound(Qstar, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar1 = self._unit_unbound(Qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar0 = self._unit_unbound(Qstar0, mini=self._continuous_min, maxi=self._continuous_max)

            self.risk_difference = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            y_unbound = self._unit_unbound(self.df[self.outcome], mini=self._continuous_min, maxi=self._continuous_max)
            ic = HAW * (y_unbound - Qstar) + (Qstar1 - Qstar0) - self.risk_difference
            varIC = np.nanvar(ic, ddof=1) / self.df.shape[0]
            self.risk_difference_se = np.sqrt(varIC)
            self.risk_difference_ci = [self.risk_difference - zalpha * np.sqrt(varIC),
                                       self.risk_difference + zalpha * np.sqrt(varIC)]
        else:
            # Calculating Risk Difference
            self.risk_difference = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = HAW * (self.df[self.outcome] - Qstar) + (Qstar1 - Qstar0) - self.risk_difference
            varIC = np.nanvar(ic, ddof=1) / self.df.shape[0]
            self.risk_difference_se = np.sqrt(varIC)
            self.risk_difference_ci = [self.risk_difference - zalpha * np.sqrt(varIC),
                                       self.risk_difference + zalpha * np.sqrt(varIC)]

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_exposure_model is False) or (self._fit_exposure_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')

        print('======================================================================')
        print('                Targeted Maximum Likelihood Estimator                 ')
        print('======================================================================')
        print('Risk Difference:    ', round(float(self.risk_difference), decimal))
        print('SE:                 ', round(float(self.risk_difference_se), decimal))
        print('95% two-sided CI:   (' +
              str(round(self.risk_difference_ci[0], decimal)), ',',
              str(round(self.risk_difference_ci[1], decimal)) + ')')
        print('======================================================================')

    @staticmethod
    def _unit_bounds(y, mini, maxi, bound):
        v = (y - mini) / (maxi - mini)
        v = np.where(np.less(v, bound), bound, v)
        v = np.where(np.greater(v, 1-bound), 1-bound, v)
        return v

    @staticmethod
    def _unit_unbound(ystar, mini, maxi):
        return ystar*(maxi - mini) + mini

    @staticmethod
    def exposure_machine_learner(xdata, ydata, ml_model):
        """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of being
        treated (i.e. Pr(A=1 | L))
        """
        # Trying to fit the Machine Learning model
        try:
            fm = ml_model.fit(X=xdata, y=ydata)
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'")

        # Generating predictions
        if hasattr(fm, 'predict_proba'):
            g = fm.predict_proba(xdata)[:, 1]
            return g
        elif hasattr(fm, 'predict'):
            g = fm.predict(xdata)
            return g
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

    @staticmethod
    def outcome_machine_learner(xdata, ydata, all_a, none_a, ml_model, continuous):
        """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of outcome
        (i.e. Pr(Y=1 | A=1, L) and Pr(Y=1 | A=0, L))
        """
        # Trying to fit Machine Learning model
        try:
            fm = ml_model.fit(X=xdata, y=ydata)
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'")

        # Generating predictions
        if continuous:
            if hasattr(fm, 'predict'):
                qa1 = fm.predict(all_a)
                qa0 = fm.predict(none_a)
                return qa1, qa0
            else:
                raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

        else:
            if hasattr(fm, 'predict_proba'):
                qa1 = fm.predict_proba(all_a)[:, 1]
                qa0 = fm.predict_proba(none_a)[:, 1]
                return qa1, qa0
            elif hasattr(fm, 'predict'):
                qa1 = fm.predict(all_a)
                qa0 = fm.predict(none_a)
                return qa1, qa0
            else:
                raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

    @staticmethod
    def probability_to_odds(prob):
        return prob / (1 - prob)


class DoubleCrossfitAIPTW:
    """Implementation of the augmented inverse probability weighted estimator with a double cross-fit procedure
    happening in the background

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    treatment : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    random_state : int, optional
        Attempt to add a seed for reproducibility of all the algorithms. Not confirmed as stable yet

    Notes
    -----
    To be added to the zEpid library in the near future. TODO's throughout correspond to items needed to be added to
    the zEpid implementation

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> dcaipw = DoubleCrossfitAIPTW(data, 'X', 'Y')
    >>> dcaipw.treatment_model('Z', LogisticRegression(penalty='none', solver='lbfgs'), bound=0.01)
    >>> dcaipw.outcome_model('X + Z', LogisticRegression(penalty='none', solver='lbfgs'))
    >>> dcaipw.fit(resamples=100, method='median')

    References
    ----------
    Newey WK, Robins JR. (2018) "Cross-fitting and fast remainder rates for semiparametric estimation".
    arXiv:1801.09138

    Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, & Robins J. (2018). "Double/debiased machine
    learning for treatment and structural parameters". The Econometrics Journal 21:1; pC1–C6
    """
    def __init__(self, df, treatment, outcome, random_state=None):
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, DCAIPTW will drop all missing data. "
                          "DCAIPTW will fit " + str(df.dropna().shape[0]) + ' of ' + str(df.shape[0]) + ' observations', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self.treatment = treatment
        self.outcome = outcome
        if random_state is None:
            self._seed_ = np.random.randint(1, 10000)
        else:
            self._seed_ = random_state

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

    def treatment_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held until usage
        in the .fit() function. These approaches are for each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held until usage
        in the .fit() function. These approaches are for each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, resamples, method='median'):
        """Runs the double-crossfit estimation procedure with augmented inverse probability weighted estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the average causal effect from each of the different splits. Median is
        used as the default since it is more stable.

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        resamples : int
            Number of times to repeat the sample-splitting estimation process. It is recommended to use at least 100.
            Note that this algorithm can take a long time to run for high values of this parameter. Be sure to test out
            run-times on small numbers first
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        """
        # Creating blank lists
        rd_point = []
        rd_var = []

        # Conducts the re-sampling procedure
        for j in range(resamples):
            split_samples = _sample_split_(self.df, seed=None)

            # Estimating (lots of functions happening in the background)
            result = self._single_crossfit_(sample_split=split_samples)

            # Appending results of this particular split combination
            rd_point.append(result[0])
            rd_var.append(result[1])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        if method == 'median':
            self.risk_difference = np.median(rd_point)
            self.risk_difference_se = np.sqrt(np.median(rd_var + (rd_point - self.risk_difference)**2))
        elif method == 'mean':
            self.risk_difference = np.mean(rd_point)
            self.risk_difference_se = np.sqrt(np.mean(rd_var + (rd_point - self.risk_difference)**2))
        else:
            raise ValueError("Either 'mean' or 'median' must be selected for the pooling of repeated sample splits")

        # TODO add z calculation here for arbitrary alphas
        self.risk_difference_ci = (self.risk_difference - 1.96*self.risk_difference_se,
                                   self.risk_difference + 1.96*self.risk_difference_se)

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('The treatment and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('                    Double-Crossfit AIPTW               ')
        # TODO add overall details regarding run (like number of different splits, etc.)
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('SE(RD):          ', round(float(self.risk_difference_se), decimal))
        print('95% CL:          ', round(float(self.risk_difference_ci[0]), decimal),
              round(float(self.risk_difference_ci[1]), decimal))
        print('======================================================================')

    def _single_crossfit_(self, sample_split):
        """Background function that runs a single crossfit of the split samples
        """

        # Estimating treatment function
        a_models = _treatment_nuisance_(treatment=self.treatment, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)

        # Estimating outcome function
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Calculating predictions for each sample split and each combination
        # TODO generalize this to arbitrary K
        s1_predictions = self._generate_predictions(sample_split[0], a_model_v=a_models[1], y_model_v=y_models[2])
        s2_predictions = self._generate_predictions(sample_split[1], a_model_v=a_models[2], y_model_v=y_models[0])
        s3_predictions = self._generate_predictions(sample_split[2], a_model_v=a_models[0], y_model_v=y_models[1])

        # Observed values of treatment and outcome
        y_obs = np.append(np.asarray(sample_split[0][self.outcome]),
                          np.append(np.asarray(sample_split[1][self.outcome]),
                                    np.asarray(sample_split[2][self.outcome])))
        a_obs = np.append(np.asarray(sample_split[0][self.treatment]),
                          np.append(np.asarray(sample_split[1][self.treatment]),
                                    np.asarray(sample_split[2][self.treatment])))

        # For combination of prediction
        # TODO need to create a generalized algorithm to break into pieces and assign samples for fitting
        split_index = np.asarray([0]*sample_split[0].shape[0] + [1]*sample_split[1].shape[0] +
                                 [2]*sample_split[2].shape[0])

        # Stacking Predicted Pr(A=1)
        pred_a_array = np.append(s1_predictions[0], np.append(s2_predictions[0], s3_predictions[0]))
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = _bounding_(pred_a_array, bounds=self._gbounds)

        # Stacking predicted outcomes under each treatment plan; Y(a=1), Y(a=0)
        pred_treat_array = np.append(s1_predictions[1], np.append(s2_predictions[1], s3_predictions[1]))
        pred_none_array = np.append(s1_predictions[2], np.append(s2_predictions[2], s3_predictions[2]))

        # Calculating point estimates
        riskdifference, var_rd = self._aipw_calculator(y=y_obs, a=a_obs,
                                                       py_a=pred_treat_array, py_n=pred_none_array,
                                                       pa=pred_a_array, splits=split_index)
        return riskdifference, var_rd

    def _generate_predictions(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.treatment] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.treatment] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none

    @staticmethod
    def _aipw_calculator(y, a, py_a, py_n, pa, splits):
        """Background calculator for AIPW and AIPW standard error
        """
        y1 = np.where(a == 1, y/pa - py_a*((1 - pa) / pa), py_a)
        y0 = np.where(a == 0, y/(1 - pa) - py_n*(pa / (1 - pa)), py_n)
        rd = np.mean(y1 - y0)
        # Variance calculations
        var_rd = []
        for i in [0, 1, 2]:  # TODO needs to generalized to s sample splits
            y1s = y1[splits == i]
            y0s = y0[splits == i]

            var_rd.append(np.var((y1s - y0s) - rd, ddof=1))

        return rd, (np.mean(var_rd) / y.shape[0])


class DoubleCrossfitTMLE:
    """Implementation of the targeted maximum likelihood estimator with a double cross-fit procedure happening in the
    background

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    treatment : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    random_state : int, optional
        Attempt to add a seed for reproducibility of all the algorithms. Not confirmed as stable yet

    Notes
    -----
    To be added to the zEpid library in the near future. TODO's throughout correspond to items needed to be added to
    the zEpid implementation

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> dctmle = DoubleCrossfitTMLE(data, 'X', 'Y')
    >>> dctmle.treatment_model('Z', LogisticRegression(penalty='none', solver='lbfgs'), bound=0.01)
    >>> dctmle.outcome_model('X + Z', LogisticRegression(penalty='none', solver='lbfgs'))
    >>> dctmle.fit(resamples=100, method='median')
    """
    def __init__(self, df, treatment, outcome, random_state=None):
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, CrossfitTMLE will drop all missing data. "
                          "Crossfit TMLE will fit " + str(df.dropna().shape[0]) + ' of ' +
                          str(df.shape[0]) + ' observations', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self.treatment = treatment
        self.outcome = outcome
        if random_state is None:
            self._seed_ = np.random.randint(1, 10000)
        else:
            self._seed_ = random_state

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

    def treatment_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held until usage
        in the .fit() function. These approaches are for each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held until usage
        in the .fit() function. These approaches are for each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, resamples, method='median'):
        """Runs the double-crossfit estimation procedure with targeted maximum likelihood estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the average causal effect from each of the different splits. Median is
        used as the default since it is more stable.

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        resamples : int
            Number of times to repeat the sample-splitting estimation process. It is recommended to use at least 100.
            Note that this algorithm can take a long time to run for high values of this parameter. Be sure to test out
            run-times on small numbers first
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        """
        # Creating blank lists
        rd_point = []
        rd_var = []

        # Conducts the re-sampling procedure
        for j in range(resamples):
            split_samples = _sample_split_(self.df, seed=None)

            # Estimating (lots of functions happening in the background
            result = self._single_crossfit_(sample_split=split_samples)

            # Appending this particular split's results
            rd_point.append(result[0])
            rd_var.append(result[1])

        if method == 'median':
            self.risk_difference = np.median(rd_point)
            self.risk_difference_se = np.sqrt(np.median(rd_var + (rd_point - self.risk_difference)**2))
        elif method == 'mean':
            self.risk_difference = np.mean(rd_point)
            self.risk_difference_se = np.sqrt(np.mean(rd_var + (rd_point - self.risk_difference)**2))
        else:
            raise ValueError("Either 'mean' or 'median' must be selected for the pooling of repeated sample splits")

        self.risk_difference_ci = (self.risk_difference - 1.96*self.risk_difference_se,
                                   self.risk_difference + 1.96*self.risk_difference_se)

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('The treatment and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('                       Double-Crossfit TMLE')
        print('======================================================================')
        print('----------------------------------------------------------------------')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('SE(RD):          ', round(float(self.risk_difference_se), decimal))
        print('95% CL:          ', round(float(self.risk_difference_ci[0]), decimal),
              round(float(self.risk_difference_ci[1]), decimal))
        print('======================================================================')

    def _single_crossfit_(self, sample_split):
        """Background function that runs a single crossfit of the split samples
        """

        # Estimating treatment function
        a_models = _treatment_nuisance_(treatment=self.treatment, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)

        # Estimating outcome function
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Calculating predictions for each sample split and each combination
        s1_predictions = self._generate_predictions(sample_split[0], a_model_v=a_models[1], y_model_v=y_models[2])
        s2_predictions = self._generate_predictions(sample_split[1], a_model_v=a_models[2], y_model_v=y_models[0])
        s3_predictions = self._generate_predictions(sample_split[2], a_model_v=a_models[0], y_model_v=y_models[1])

        # Observed values of treatment and outcome
        y_obs = np.append(np.asarray(sample_split[0][self.outcome]),
                          np.append(np.asarray(sample_split[1][self.outcome]),
                                    np.asarray(sample_split[2][self.outcome])))
        a_obs = np.append(np.asarray(sample_split[0][self.treatment]),
                          np.append(np.asarray(sample_split[1][self.treatment]),
                                    np.asarray(sample_split[2][self.treatment])))

        split_index = np.asarray([0]*sample_split[0].shape[0] + [1]*sample_split[1].shape[0] +
                                 [2]*sample_split[2].shape[0])

        # Stacking predicted Pr(A=)
        pred_a_array = np.append(s1_predictions[0], np.append(s2_predictions[0], s3_predictions[0]))
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = _bounding_(pred_a_array, bounds=self._gbounds)

        # Stacking predicted outcomes under each treatment plan
        pred_treat_array = np.append(s1_predictions[1], np.append(s2_predictions[1], s3_predictions[1]))
        pred_none_array = np.append(s1_predictions[2], np.append(s2_predictions[2], s3_predictions[2]))

        ate, var_ate = self._tmle_calculator(y_obs=y_obs, a=a_obs,
                                             qaw=np.where(a_obs == 1, pred_treat_array, pred_none_array),
                                             qa1w=pred_treat_array, qa0w=pred_none_array,
                                             g1w=pred_a_array, g0w=1 - pred_a_array,
                                             splits=split_index, continuous=False)
        return ate, var_ate

    def _generate_predictions(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.treatment] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.treatment] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none

    @staticmethod
    def _tmle_calculator(y_obs, a, qaw, qa1w, qa0w, g1w, g0w, splits, continuous=False):
        """Background targeting step from g-model and Q-model
        """
        h1w = a / g1w
        h0w = -(1 - a) / g0w
        haw = h1w + h0w

        qstar = []
        qstar1 = []
        qstar0 = []

        # Calculating overall estimate
        for i in [0, 1, 2]:
            yb_ = y_obs[splits == i]
            g1s = g1w[splits == i]
            g0s = g0w[splits == i]
            q1s = qa1w[splits == i]
            q0s = qa0w[splits == i]
            qas = qaw[splits == i]
            h1s = h1w[splits == i]
            h0s = h0w[splits == i]

            # Targeting model
            f = sm.families.family.Binomial()
            log = sm.GLM(yb_, np.column_stack((h1s, h0s)), offset=np.log(probability_to_odds(qas)),
                         family=f, missing='drop').fit()
            epsilon = log.params

            qstar1 = np.append(qstar1, logistic.cdf(np.log(probability_to_odds(q1s)) + epsilon[0] / g1s))
            qstar0 = np.append(qstar0, logistic.cdf(np.log(probability_to_odds(q0s)) - epsilon[1] / g0s))
            qstar = np.append(qstar, log.predict(np.column_stack((h1s, h0s)), offset=np.log(probability_to_odds(qas))))

        # TODO bounding bit if continuous
        if continuous:
            raise ValueError("Not completed yet")
            # TODO I do an unbounding step here for the outcomes if necessary
            # y_obs = self._unit_unbound(y_bound, mini=self._continuous_min, maxi=self._continuous_max)
            # qstar = self._unit_unbound(qstar, mini=self._continuous_min, maxi=self._continuous_max)
            # qstar1 = self._unit_unbound(qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            # qstar0 = self._unit_unbound(qstar0, mini=self._continuous_min, maxi=self._continuous_max)

        qstar_est = np.mean(qstar1 - qstar0)

        # Variance estimation
        var_rd = []
        for i in [0, 1, 2]:
            yu_ = y_obs[splits == i]
            qs1s = qstar1[splits == i]
            qs0s = qstar0[splits == i]
            qs = qstar[splits == i]
            has = haw[splits == i]

            ic = has * (yu_ - qs) + (qs1s - qs0s) - qstar_est
            var_rd.append(np.var(ic, ddof=1))

        return qstar_est, (np.mean(var_rd) / splits.shape[0])


###############################################################
# Background utility functions shared by estimators
def probability_to_odds(prob):
    """Converts given probability (proportion) to odds

    Parameters
    ---------------
    prob : float, NumPy array
        Probability or array of probabilities to transform into odds
    """
    return prob / (1 - prob)


def _bounding_(v, bounds):
    """Background function to perform bounding feature for inverse probability weights. Supports both symmetric
    and asymmetric bounding
    """
    if type(bounds) is float:  # Symmetric bounding
        if bounds < 0 or bounds > 1:
            raise ValueError('Bound value must be between (0, 1)')
        v = np.where(v < bounds, bounds, v)
        v = np.where(v > 1 - bounds, 1 - bounds, v)

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
        v = np.where(v < bounds[0], bounds[0], v)
        v = np.where(v > bounds[1], bounds[1], v)
    return v


def _sample_split_(data, seed):
    """Background function to split data into three non-overlapping pieces
    """
    n = int(data.shape[0] / 3)
    s1 = data.sample(n=n, random_state=seed)
    s2 = data.loc[data.index.difference(s1.index)].sample(n=n, random_state=seed)
    s3 = data.loc[data.index.difference(s1.index) & data.index.difference(s2.index)]
    return s1, s2, s3


def _ml_predictor(xdata, fitted_algorithm):
    """Background function to generate predictions of treatments
    """
    if hasattr(fitted_algorithm, 'predict_proba'):
        return fitted_algorithm.predict_proba(xdata)[:, 1]
    elif hasattr(fitted_algorithm, 'predict'):
        return fitted_algorithm.predict(xdata)


def _treatment_nuisance_(treatment, estimator, samples, covariates):
    """Procedure to fit the treatment ML
    """
    treatment_fit_splits = []
    for s in samples:
        # Using patsy to pull out the covariates
        xdata = np.asarray(patsy.dmatrix(covariates + ' - 1', s))
        ydata = np.asarray(s[treatment])

        # Fitting machine learner / super learner to each split
        est = copy.deepcopy(estimator)
        try:
            fm = est.fit(X=xdata, y=ydata)
            # print("Treatment model")
            # print(fm.summarize())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        treatment_fit_splits.append(fm)

    return treatment_fit_splits


def _outcome_nuisance_(outcome, estimator, samples, covariates):
    """Background function to generate predictions of outcomes
    """
    outcome_fit_splits = []
    for s in samples:
        # Using patsy to pull out the covariates
        xdata = np.asarray(patsy.dmatrix(covariates + ' - 1', s))
        ydata = np.asarray(s[outcome])

        # Fitting machine learner / super learner to each
        est = copy.deepcopy(estimator)
        try:
            fm = est.fit(X=xdata, y=ydata)
            # print("Outcome model")
            # print(fm.summarize())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        outcome_fit_splits.append(fm)

    return outcome_fit_splits
