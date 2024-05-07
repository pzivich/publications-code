import numpy as np
import pandas as pd
import statsmodels.api as sm
from delicatessen.utilities import inverse_logit, logit
from formulaic import model_matrix
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from nuisance_estimators import LogitRegression, SuperLearner


class SingleCrossFitTMLE:
    def __init__(self, data, outcome, action, seed=None):
        """A minimal implementation of a single cross-fit targeted maximum likelihood estimator (SCFTMLE). This SCFTMLE
        implementation is intended only for the conduct of the simulation study described in the corresponding
        commentary. This implementation of SCF-TMLE is fairly bare bones, expects treatment to be binary and the outcome
        to be continuous, and uses a particular super-learner setup. The setup is also meant to provide a very
        high-level control over how the random number generator seeds are specified (all seeds are set by ``seed``).
        So, SCFTMLE as implemented here is not meant for use outside of the simulation study reported in the main paper
        (but one could adapt the code provided).

        Parameters
        ----------
        data : pd.DataFrame
            Pandas DataFrame object containing variables of interest
        outcome : str
            Column name for the outcome variable in the DataFrame
        action : str
            Column name for the action (e.g., exposure, treatment, intervention) variable in the DataFrame
        seed : int, None, optional
            RNG seed used to initialize a RNG to create starting values for all cross-fitting procedures and machine
            learning algorithms. This argument essentially sets the seeds of all downstream processes.
        """
        # Storing inputs for later use
        self.data = data.copy()
        self.outcome = outcome
        self.action = action

        # Managing the continuous outcome by setting up bounding procedure
        self._ymin = np.min(self.data[self.outcome])
        self._ymax = np.max(self.data[self.outcome])
        self.data[self.outcome] = self._unit_bound_(self.data[self.outcome],
                                                    mini=self._ymin,
                                                    maxi=self._ymax)

        # Managing random seed for all downstream processes. This RNG is called for the generation of all down-stream
        #   RNG seeds (for sample-splitting and ML algorithms).
        self.rng = np.random.default_rng(seed)   # Instantiate a new RNG
        self.rng_lower = 0                       # Lower value for the possible seed
        self.rng_upper = 2**32 - 1               # Upper value for the possible seed

        # Storage for nuisance models
        self.model_action = None          # Storage for model specification
        self.estimator_action = None      # Storage for SL setup
        self.model_outcome = None         # Storage for model specification
        self.estimator_outcome = None     # Storage for SL setup

    def nuisance_action(self, model):
        """Specify the action nuisance model.

        Parameters
        ----------
        model : str
            Model specification in an R-style formula
        """
        # Store as model that removes the intercept from the formula
        self.model_action = model + " -1"

    def nuisance_outcome(self, model):
        """Specify the outcome nuisance model.

        Parameters
        ----------
        model : str
            Model specification in an R-style formula
        """
        # Store as model that removes the intercept from the formula
        self.model_outcome = model + " -1"

    def estimate(self, n_splits, n_reps):
        """Apply SCFTMLE for the specified number of splits and number of repetitions. The results across repetitions
        are summarized using the median method analog of Rubin's Rules, as described in Zivich & Breskin 2021 and
        Chernozhukov et al. 2018.

        Parameters
        ----------
        n_splits : int
            Number of folds or splits to divide the data into for the cross-fitting procedure
        n_reps : int
            Number of different folds or splits to consider and summarize over.

        Returns
        -------
        list
            Returns an array consisting of the point estimate and variance estimate.
        """
        # Running cross-fitting procedure for the requested number of repetitions
        ace_s, ace_var_s = [], []                          # Storage for point and variance estimates across splits
        for s in range(n_reps):                            # Loop over range of repetitions
            a, a_var = self._single_split_(n_splits)       # ... compute point and var estimate for single split
            ace_s.append(a)                                # ... store point estimate
            ace_var_s.append(a_var)                        # ... store variance estimate

        # Managing the results across the different splits
        if n_reps == 1:
            # If only a single split is requested, return results from that single split
            return ace_s[0], ace_var_s[0]
        else:
            # Otherwise summarize across different splits using the median method
            ace, var = self._summarize_difference_splits_(estimates=ace_s, variances=ace_var_s)
            return ace, var

    def _single_split_(self, splits):
        """Internal function to run SCFTMLE for a single fold. This function is repeatedly called for the repetitions

        Parameters
        ----------
        splits : int
            Number of folds or splits to use.

        Returns
        -------
        list
            Returns an array consisting of the point estimate and variance estimate.
        """
        # Generate division
        data_splits = self._split_data_(splits=splits)   # Split the data into pieces
        y_reordered = []                                 # Storage for how Y is shuffled
        a_reordered = []                                 # Storage for how A is shuffled
        pr_a1s, y1hats, y0hats = [], [], []              # Storage for predictions

        # Run fitting procedure for each of the data splits
        for s in range(len(data_splits)):
            # Separating into fit and predict chunks
            data_predict = data_splits[s]
            data_splits_int = data_splits.copy()
            data_splits_int.pop(s)
            data_fit = pd.concat(data_splits_int, ignore_index=True).reset_index(drop=True)

            # Fitting action nuisance model
            nuisance_estr_action = self._setup_estimators_(continuous=False)
            fnm_action = self._fit_action_nuisance_model_(data=data_fit, estimator=nuisance_estr_action)

            # Fitting outcome nuisance model
            nuisance_estr_outcome = self._setup_estimators_(continuous=True)
            fnm_outcome = self._fit_outcome_nuisance_model_(data=data_fit, estimator=nuisance_estr_outcome)

            # Generating predictions from nuisance models
            pr_a = self._predict_from_action_model_(data=data_predict, estimator=fnm_action)
            y0hat, y1hat = self._predict_from_outcome_model_(data=data_predict, estimator=fnm_outcome)

            # Trimming propensity scores
            pr_a = np.clip(pr_a, 0.01, 0.99)

            # Storing re-ordered variables
            a_reordered.append(np.asarray(data_predict[self.action]))
            y_reordered.append(np.asarray(data_predict[self.outcome]))

            # Storing predictions from fitted nuisance models
            pr_a1s.append(pr_a)
            y1hats.append(y1hat)
            y0hats.append(y0hat)

        # Stacking all n predictions together into a single array according to the shuffled order
        a_obs = np.hstack(a_reordered)                        # A observed
        y_obs = np.hstack(y_reordered)                        # Y observed
        pr_a_full = np.hstack(pr_a1s)                         # Pr(A|W) from model
        y0hat_full = np.hstack(y0hats)                        # Y^0 from model
        y0hat_full = self._trim_predictions_(y0hat_full)      # ... ensuring predicted outcomes are bounded [0,1]
        y1hat_full = np.hstack(y1hats)                        # Y^1 from model
        y1hat_full = self._trim_predictions_(y1hat_full)      # ... ensuring predicted outcomes are bounded [0,1]

        # Combining predictions via recipe for TMLE
        ipw0 = (1 - a_obs) / (1 - pr_a_full)                  # IPW for A=0
        ipw1 = a_obs / pr_a_full                              # IPW for A=1
        y0tilde = self._targeting_step_(y_obs=y_obs, yhat=y0hat_full, ipw=ipw0)
        y0tilde = self._unit_unbound_(y0tilde, mini=self._ymin, maxi=self._ymax)
        y1tilde = self._targeting_step_(y_obs=y_obs, yhat=y1hat_full, ipw=ipw1)
        y1tilde = self._unit_unbound_(y1tilde, mini=self._ymin, maxi=self._ymax)

        # Computing average causal effect
        ace = np.mean(y1tilde - y0tilde)

        # Computing the standard error for the average causal effect
        y_obs_unbound = self._unit_unbound_(y_obs, mini=self._ymin, maxi=self._ymax)
        ace_var = self._infcurve_variance_(y=y_obs_unbound,
                                           a=a_obs,
                                           y1hat=y1tilde,
                                           y0hat=y0tilde,
                                           pr_a1=pr_a_full)

        # Return results from this particular split
        return ace, ace_var

    def _generate_rng_(self):
        # Function to randomly generate a RNG seed according to the master RNG created in __init__
        return self.rng.integers(low=self.rng_lower, high=self.rng_upper, size=1)[0]

    def _setup_estimators_(self, continuous=False):
        # Function to setup the Super-Learner functions
        if continuous:
            # Algorithms for continuous target
            loss_func = 'L2'
            lreg = LinearRegression()
            lasso = LassoCV(random_state=self._generate_rng_())
            tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=self._generate_rng_())
            nnet = MLPRegressor(hidden_layer_sizes=(10, 10, 5),
                                activation='relu',
                                random_state=self._generate_rng_(),
                                max_iter=20000)
        else:
            # Algorithms for binary target
            loss_func = 'nloglik'
            lreg = LogitRegression()
            lasso = LogisticRegressionCV(penalty='l1', solver='saga', random_state=self._generate_rng_())
            tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=self._generate_rng_())
            nnet = MLPClassifier(hidden_layer_sizes=(10, 10, 5),
                                 activation='relu',
                                 random_state=self._generate_rng_(),
                                 max_iter=20000)

        # Stacking estimators together according to what `SuperLearner` expects
        estrs = [lreg, lasso, tree, nnet]
        names = ["LReg", "LASSO", "Tree", "NNET"]
        learner = SuperLearner(library=estrs, libnames=names,
                               K=10,   # Using 10-fold cross-validation
                               loss=loss_func, print_results=False)
        return learner

    def _split_data_(self, splits):
        # Breaking up the rows randomly into approximately equal splits
        row_ids = self.data.index
        shuffled_row_ids = self.rng.permutation(row_ids)
        row_ids_chucks = np.array_split(shuffled_row_ids, indices_or_sections=splits)

        # Stacking sample splits into list
        ssplits = []
        for rids in row_ids_chucks:
            ds = self.data.iloc[rids].copy()
            ssplits.append(ds)

        # Returning how the data is split
        return ssplits

    def _fit_action_nuisance_model_(self, data, estimator):
        # Function to fit the action nuisance model
        a = np.asarray(data[self.action])
        W = model_matrix(self.model_action, data)
        W = np.asarray(W)
        fnm = estimator.fit(X=W, y=a)
        return fnm

    def _predict_from_action_model_(self, data, estimator):
        # Function to generate predictions from the action nuisance model
        W = model_matrix(self.model_action, data)
        W = np.asarray(W)
        pr_a1 = estimator.predict(X=W)
        return pr_a1

    def _fit_outcome_nuisance_model_(self, data, estimator):
        # Function to fit the outcome nuisance model
        y = np.asarray(data[self.outcome])
        X = model_matrix(self.model_outcome, data)
        X = np.asarray(X)
        fnm = estimator.fit(X=X, y=y)
        return fnm

    def _predict_from_outcome_model_(self, data, estimator):
        # Function to generate predictions from the outcome nuisance model
        da = data.copy()
        yhats = []
        for j in [0, 1]:
            da[self.action] = j
            Xa = model_matrix(self.model_outcome, da)
            Xa = np.asarray(Xa)
            yhat = estimator.predict(X=Xa)
            yhats.append(yhat)

        return yhats

    @staticmethod
    def _targeting_step_(y_obs, yhat, ipw):
        # Static function to run the targeting step on all observations at once
        f = sm.families.family.Binomial()

        # Targeting model
        logit_yhat = logit(yhat)
        log = sm.GLM(y_obs,                         # Outcome / dependent variable
                     np.repeat(1, y_obs.shape[0]),  # Generating intercept only model
                     offset=logit_yhat,                   # Offset by outcome predictions
                     freq_weights=ipw,              # Weighted by calculated IPW
                     family=f).fit()
        epsilon = log.params[0]
        ytilde = inverse_logit(logit_yhat + epsilon)

        return ytilde

    @staticmethod
    def _infcurve_variance_(y, a, y1hat, y0hat, pr_a1):
        # Static function to compute the IF variance
        pr_a0 = 1 - pr_a1
        scale = (a - pr_a1) / (pr_a1 * pr_a0)
        ic = ((y*a / pr_a1) - (y*(1-a) / pr_a0)) - scale * (pr_a0*y1hat + pr_a1*y0hat) - np.mean(y1hat - y0hat)
        var = np.var(ic, ddof=1) / y.shape[0]
        return var

    @staticmethod
    def _unit_bound_(y, mini, maxi):
        # Static function to bound outcomes between [0,1]
        v = (y - mini) / (maxi - mini)
        return v

    @staticmethod
    def _unit_unbound_(y, maxi, mini):
        # Static function to unbound outcomes from [0,1]
        return y * (maxi - mini) + mini

    @staticmethod
    def _trim_predictions_(yhat):
        # Static function to bound predicted outcomes between [0,1]
        return np.clip(yhat, a_min=0, a_max=1)

    @staticmethod
    def _summarize_difference_splits_(estimates, variances):
        # Static function to apply the median variation of Rubin's Rules to summarize across different splits
        est = np.median(estimates)
        var = np.median(variances + (np.array(estimates) - est)**2)
        return est, var
