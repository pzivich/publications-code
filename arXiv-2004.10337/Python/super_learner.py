import numpy as np
from scipy.optimize import fmin_l_bfgs_b, nnls, fmin_slsqp
from sklearn import clone
from sklearn.base import BaseEstimator
import sklearn.model_selection as ms

from sklearn.linear_model import LogisticRegression
from pygam import GAM, LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# Super Learner set-up
class SLError(Exception):
    """
    Base class for errors in the SupyLearner package
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
        if self.coef_method is 'L_BFGS_B':
            if self.loss == 'nloglik':
                raise SLError("coef_method 'L_BFGS_B' is only for 'L2' loss")

            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))

            x0 = np.array([1. / self.n_estimators] * self.n_estimators)
            bds = [(0, 1)] * self.n_estimators
            coef_init, b, c = fmin_l_bfgs_b(ff, x0, bounds=bds, approx_grad=True)
            if c['warnflag'] is not 0:
                raise SLError("fmin_l_bfgs_b failed when trying to calculate coefficients")

        elif self.coef_method is 'NNLS':
            if self.loss == 'nloglik':
                raise SLError("coef_method 'NNLS' is only for 'L2' loss")
            coef_init, b = nnls(y_pred_cv, y)

        elif self.coef_method is 'SLSQP':
            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))

            def constr(x):
                return np.array([np.sum(x) - 1])

            x0 = np.array([1. / self.n_estimators] * self.n_estimators)
            bds = [(0, 1)] * self.n_estimators
            coef_init, b, c, d, e = fmin_slsqp(ff, x0, f_eqcons=constr, bounds=bds, disp=0, full_output=True)
            if d is not 0:
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
                if pred.min() < 0 or pred.max() > 1:
                    raise SLError("Probability less than zero or greater than one")
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


class EmpiricalMean(BaseEstimator):
    """Calculates the empirical mean in a format that SuPyLearner is expecting
    """

    def __init__(self):
        self.empirical_mean = np.nan

    def fit(self, X, y):
        self.empirical_mean = np.mean(y)

    def predict(self, X):
        return np.array([self.empirical_mean] * X.shape[0])


def superlearnersetup(var_type, K=5):
    """Super Learner setup for binary and continuous variables"""
    if var_type == 'binary':
        # Binary variable
        log_b = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
        rdf_b = RandomForestClassifier(n_estimators=500, min_samples_leaf=20)  # max features is sqrt(n_features)
        gam1_b = LogisticGAM(n_splines=4, lam=0.6)
        gam2_b = LogisticGAM(n_splines=6, lam=0.6)
        nn1_b = MLPClassifier(hidden_layer_sizes=(4,),
                              activation='relu', solver='lbfgs',
                              max_iter=2000)
        emp_b = EmpiricalMean()

        lib = [log_b, gam1_b, gam2_b, rdf_b, nn1_b, emp_b]
        libnames = ["Logit", "GAM1", "GAM2", "Random Forest", "Neural-Net", "Mean"]
        sl = SuperLearner(lib, libnames, loss="nloglik", K=K, print_results=False)

    elif var_type == 'continuous':
        # Continuous variable
        lin_c = LinearRegression()
        rdf_c = RandomForestRegressor(n_estimators=500, min_samples_leaf=20)
        gam1_c = GAM(link='identity', n_splines=4, lam=0.6)
        gam2_c = GAM(link='identity', n_splines=6, lam=0.6)
        nn1_c = MLPRegressor(hidden_layer_sizes=(4,),
                             activation='relu', solver='lbfgs',
                             max_iter=2000)
        emp_c = EmpiricalMean()

        lib = [lin_c, gam1_c, gam2_c, rdf_c, nn1_c, emp_c]
        libnames = ["Linear", "GAM1", "GAM2", "Random Forest", "Neural-Net", "Mean"]
        sl = SuperLearner(lib, libnames, K=K, print_results=False)

    else:
        raise ValueError("Not Supported")

    return sl
