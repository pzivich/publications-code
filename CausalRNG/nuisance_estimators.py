import numpy as np
import statsmodels.api as sm
from sklearn import clone
from sklearn import model_selection as ms
from sklearn.base import BaseEstimator
from scipy.optimize import fmin_l_bfgs_b, nnls, fmin_slsqp


class LogitRegression(BaseEstimator):
    """Implements logistic regression using statsmodels (I find that sci-kit learn's logistic regression does extra
    things, which I don't want, so this is a wrapper class to get statsmodels GLM to work like sci-kit learn with the
    implemented super-learner).
    """
    def __init__(self, family=sm.families.Binomial()):
        self.model = None
        self._family_ = family

    def fit(self, X, y):
        # Error Checking
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows).")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("It looks like there is missing values in X or y.")

        # Final results
        family = sm.families.Binomial()
        self.model = sm.GLM(y,
                            np.hstack([np.zeros([X.shape[0], 1]) + 1, X]),
                            family=family).fit()
        return self

    def predict(self, X):
        Xd = np.hstack([np.zeros([X.shape[0], 1]) + 1, X])
        return self.model.predict(Xd)

    def get_params(self, deep=True):
        """For sklearn.base.clone() compatibility"""
        return {"family": self._family_, }

    def set_params(self, **parameters):
        """For sklearn.base.clone() compatibility"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class SLError(Exception):
    """
    Base class for errors in the SuperLearner package
    """
    pass


class SuperLearner(BaseEstimator):
    """Loss-based super learning. SuperLearner chooses a weighted combination of candidate estimates in a specified
    library using cross-validation.

    Implementation borrowed from: https://github.com/alexpkeil1/SuPyLearner

    Parameters
    ----------
    library : list
        List of scikit-learn style estimators with fit() and predict()
        methods.
    K : Number of folds for cross-validation.
    loss : loss function, 'L2' or 'nloglik'.
    discrete : True to choose the best estimator
               from library ("discrete SuperLearner"), False to choose best
               weighted combination of esitmators in the library.
    coef_method : Method for estimating weights for weighted combination
                  of estimators in the library. 'L_BFGS_B', 'NNLS', or 'SLSQP'.
    """

    def __init__(self, library, libnames=None, K=5, loss='L2', discrete=False, coef_method='SLSQP',
                 save_pred_cv=False, bound=0.00001, print_results=True):
        self.library = library[:]
        self.libnames = libnames
        self.K = K
        self.loss = loss
        self.n_estimators = len(library)
        self.discrete = discrete
        self.coef_method = coef_method
        self.save_pred_cv = save_pred_cv
        self.bound = bound
        self._print = print_results

    def fit(self, X, y):
        """Fit SuperLearner.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            or other object acceptable to the fit() methods
            of all candidates in the library
            Training data
        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        n = len(y)
        folds = ms.KFold(self.K)

        y_pred_cv = np.empty(shape=(n, self.n_estimators))
        for train_index, test_index in folds.split(range(n)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for aa in range(self.n_estimators):
                if self.libnames is not None:
                    if self._print:
                        print('...K-fold fitting ' + self.libnames[aa] + '...')
                est = clone(self.library[aa])
                est.fit(X_train, y_train)

                y_pred_cv[test_index, aa] = self._get_pred(est, X_test)

        self.coef = self._get_coefs(y, y_pred_cv)
        self.fitted_library = clone(self.library)

        for est in self.fitted_library:
            ind = self.fitted_library.index(est)
            if self._print:
                print('...fitting ' + self.libnames[ind] + '...')
            est.fit(X, y)

        self.risk_cv = []
        for aa in range(self.n_estimators):
            self.risk_cv.append(self._get_risk(y, y_pred_cv[:, aa]))
        self.risk_cv.append(self._get_risk(y, self._get_combination(y_pred_cv, self.coef)))

        if self.save_pred_cv:
            self.y_pred_cv = y_pred_cv

        return self

    def predict(self, X):
        """Predict using SuperLearner

        Parameters
        ----------
        X : numpy.array of shape [n_samples, n_features]
           or other object acceptable to the predict() methods
           of all candidates in the library

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """

        n_X = X.shape[0]
        y_pred_all = np.empty((n_X, self.n_estimators))
        for aa in range(self.n_estimators):
            y_pred_all[:, aa] = self._get_pred(self.fitted_library[aa], X)
        y_pred = self._get_combination(y_pred_all, self.coef)
        return y_pred

    def summarize(self):
        """Print CV risk estimates for each candidate estimator in the library,
        coefficients for weighted combination of estimators,
        and estimated risk for the SuperLearner.
        """
        if self.libnames is None:
            libnames = [est.__class__.__name__ for est in self.library]
        else:
            libnames = self.libnames
        print("Cross-validated risk estimates for each estimator in the library:")
        print(np.column_stack((libnames, self.risk_cv[:-1])))
        print("\nCoefficients:")
        print(np.column_stack((libnames, self.coef)))
        print("\n(Not cross-valided) estimated risk for SL:", self.risk_cv[-1])

    def _get_combination(self, y_pred_mat, coef):
        """Calculate weighted combination of predictions
        """
        if self.loss == 'L2':
            comb = np.dot(y_pred_mat, coef)
        elif self.loss == 'nloglik':
            comb = _inv_logit(np.dot(_logit(_trim(y_pred_mat, self.bound)), coef))
        return comb

    def _get_risk(self, y, y_pred):
        """Calculate risk given observed y and predictions
        """
        if self.loss == 'L2':
            risk = np.mean((y - y_pred) ** 2)
        elif self.loss == 'nloglik':
            risk = -np.mean(y * np.log(_trim(y_pred, self.bound)) + \
                            (1 - y) * np.log(1 - (_trim(y_pred, self.bound))))
        return risk

    def _get_coefs(self, y, y_pred_cv):
        """Find coefficients that minimize the estimated risk.
        """
        if self.coef_method == 'L_BFGS_B':
            if self.loss == 'nloglik':
                raise SLError("coef_method 'L_BFGS_B' is only for 'L2' loss")

            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))

            x0 = np.array([1. / self.n_estimators] * self.n_estimators)
            bds = [(0, 1)] * self.n_estimators
            coef_init, b, c = fmin_l_bfgs_b(ff, x0, bounds=bds, approx_grad=True)
            if c['warnflag'] != 0:
                raise SLError("fmin_l_bfgs_b failed when trying to calculate coefficients")

        elif self.coef_method == 'NNLS':
            if self.loss == 'nloglik':
                raise SLError("coef_method 'NNLS' is only for 'L2' loss")
            coef_init, b = nnls(y_pred_cv, y)

        elif self.coef_method == 'SLSQP':
            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))

            def constr(x):
                return np.array([np.sum(x) - 1])

            x0 = np.array([1. / self.n_estimators] * self.n_estimators)
            bds = [(0, 1)] * self.n_estimators
            coef_init, b, c, d, e = fmin_slsqp(ff, x0, f_eqcons=constr, bounds=bds, disp=0, full_output=True)
            if d != 0:
                raise SLError("fmin_slsqp failed when trying to calculate coefficients")

        else:
            raise ValueError("method not recognized")
        coef_init = np.array(coef_init)
        coef_init[coef_init < np.sqrt(np.finfo(np.double).eps)] = 0
        coef = coef_init / np.sum(coef_init)
        return coef

    def _get_pred(self, est, X):
        """
        Get prediction from the estimator.
        Use est.predict if loss is L2.
        If loss is nloglik, use est.predict_proba if possible
        otherwise just est.predict, which hopefully returns something
        like a predicted probability, and not a class prediction.
        """
        if self.loss == 'L2':
            pred = est.predict(X)
        elif self.loss == 'nloglik':
            if hasattr(est, "predict_proba"):
                try:
                    pred = est.predict_proba(X)[:, 1]
                except IndexError:
                    pred = est.predict_proba(X)
            else:
                pred = est.predict(X)
                pred = np.clip(pred, a_min=0, a_max=1)
                # if pred.min() < 0 or pred.max() > 1:
                #     raise SLError("Probability less than zero or greater than one")
        else:
            raise SLError("loss must be 'L2' or 'nloglik'")
        return pred


def _trim(p, bound):
    """Trim a probabilty to be in (bound, 1-bound)
    """
    p[p < bound] = bound
    p[p > 1 - bound] = 1 - bound
    return p


def _logit(p):
    """Calculate the logit of a probability
    """
    return np.log(p / (1 - p))


def _inv_logit(x):
    """Calculate the inverse logit
    """

    return 1 / (1 + np.exp(-x))
