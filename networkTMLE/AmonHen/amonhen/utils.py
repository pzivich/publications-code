import warnings
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf


def probability_to_odds(prob):
    """Converts given probability (proportion) to odds"""
    return prob / (1 - prob)


def odds_to_probability(odds):
    """Converts given odds to probability"""
    return odds / (1 + odds)


def exp_map(graph, var):
    """Does the exposure mapping functionality
    """
    # get adjacency matrix
    matrix = nx.adjacency_matrix(graph, weight=None)
    # get node attributes
    y_vector = np.array(list(nx.get_node_attributes(graph, name=var).values()))
    # multiply the weight matrix by node attributes
    wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
    return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...


def fast_exp_map(matrix, y_vector, measure):
    """Improved exposure mapping speed (doesn't need to parse adj matrix every time)"""
    if measure.lower() == 'sum':
        # multiply the weight matrix by node attributes
        wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
        return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...
    elif measure.lower() == 'mean':
        rowsum_vector = np.sum(matrix, axis=1)  # calculate row-sum (denominator / degree)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            weight_matrix = matrix / rowsum_vector.reshape((matrix.shape[0]), 1)  # calculate each nodes weight
        wy_matrix = weight_matrix * y_vector.reshape((matrix.shape[0]), 1)  # multiply matrix by node attributes
        return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...
    elif measure.lower() == 'var':
        a = matrix.toarray()  # Convert matrix to array
        a = np.where(a == 0, np.nan, a)  # filling non-edges with NaN's
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(a * y_vector, axis=1)
    elif measure.lower() == 'mean_dist':
        a = matrix.toarray()  # Convert matrix to array
        a = np.where(a == 0, np.nan, a)  # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector  # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(c.transpose(),  # back-transpose
                              axis=1)
    elif measure.lower() == 'var_dist':
        a = matrix.toarray()  # Convert matrix to array
        a = np.where(a == 0, np.nan, a)  # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector  # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():  # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(c.transpose(),  # back-transpose
                             axis=1)
    else:
        raise ValueError("The summary measure mapping" + str(measure) + "is not available")


def exp_map_individual(network, measure, max_degree):
    """Exposure mapping for non-parametric estimation of the gs-model. Generates a dataframe with max_degree columns"""
    attrs = []
    for i in network.nodes:
        j_attrs = []
        for j in network.neighbors(i):
            j_attrs.append(network.nodes[j][measure])
        attrs.append(j_attrs[:max_degree])

    return pd.DataFrame(attrs, columns=[measure+'_map' + str(x+1) for x in range(max_degree)])


def network_to_df(graph):
    """Convert node attributes to pandas dataframe
    """
    return pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')


def bounding(ipw, bound):
    if type(bound) is float or type(bound) is int:  # Symmetric bounding
        if bound > 1:
            ipw = np.where(ipw > bound, bound, ipw)
            ipw = np.where(ipw < 1 / bound, 1 / bound, ipw)
        elif 0 < bound < 1:
            ipw = np.where(ipw < bound, bound, ipw)
            ipw = np.where(ipw > 1 / bound, 1 / bound, ipw)
        else:
            raise ValueError('Bound must be a positive value')
    elif type(bound) is str:  # Catching string inputs
        raise ValueError('Bounds must either be a float or integer, or a collection')
    else:  # Asymmetric bounds
        if bound[0] > bound[1]:
            raise ValueError('Bound thresholds must be listed in ascending order')
        if len(bound) > 2:
            warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                          'specified bounds are used by the bound statement. So only ' +
                          str(bound[0:2]) + ' will be used', UserWarning)
        if type(bound[0]) is str or type(bound[1]) is str:
            raise ValueError('Bounds must be floats or integers')
        if bound[0] < 0 or bound[1] < 0:
            raise ValueError('Both bound values must be positive values')
        ipw = np.where(ipw < bound[0], bound[0], ipw)
        ipw = np.where(ipw > bound[1], bound[1], ipw)
    return ipw


def outcome_learner_fitting(ml_model, xdata, ydata):
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
        print("OUTCOME MODEL")
        fm.summarize()
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers sklearn.")
    return fm


def outcome_learner_predict(ml_model_fit, xdata):
    if hasattr(ml_model_fit, 'predict_proba'):
        g = ml_model_fit.predict_proba(xdata)
        if g.ndim == 1:  # allows support for pygam.LogisticGAM
            return g
        else:
            return g[:, 1]
    elif hasattr(ml_model_fit, 'predict'):
        return ml_model_fit.predict(xdata)
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def exposure_machine_learner(ml_model, xdata, ydata, pdata):
    # Fitting model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers sklearn.")

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        g = fm.predict_proba(pdata)
        if g.ndim == 1:  # allows support for pygam.LogisticGAM
            return g
        else:
            return g[:, 1]
    elif hasattr(fm, 'predict'):
        g = fm.predict(pdata)
        return g
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


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


def tmle_unit_bounds(y, mini, maxi):
    # bounding for continuous outcomes
    return (y - mini) / (maxi - mini)


def tmle_unit_unbound(ystar, mini, maxi):
    # unbounding of bounded continuous outcomes
    return ystar*(maxi - mini) + mini


def propensity_score(df, model, weights=None, print_results=True):
    """Generate propensity scores (probability) based on the model input. Uses logistic regression model
    to calculate

    Parameters
    -----------
    df : DataFrame
        Pandas Dataframe containing the variables of interest
    model : str
        Model to fit the logistic regression to. For example, 'y ~ var1 + var2'
    weights : str, optional
        Whether to estimate the model using weights. Default is None (unweighted)
    print_results : bool, optional
        Whether to print the logistic regression results. Default is True

    Returns
    -------------
    Fitted statsmodels GLM object

    Example
    ------------
    """
    f = sm.families.family.Binomial()
    if weights is None:
        log = smf.glm(model, df, family=f).fit()
    else:
        log = smf.glm(model, df, freq_weights=df[weights], family=f).fit()

    if print_results:
        print('\n----------------------------------------------------------------')
        print('MODEL: ' + model)
        print('-----------------------------------------------------------------')
        print(log.summary())
    return log


def stochastic_outcome_machine_learner(xdata, ydata, ml_model, continuous, print_results=True):
    """Function to fit machine learning predictions. Used by StochasticTMLE to generate predicted probabilities of
    outcome (i.e. Pr(Y=1 | A, L)
    """
    # Trying to fit Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers sklearn.")
    if print_results and hasattr(fm, 'summarize'):  # Nice summarize option from SuPyLearner
        fm.summarize()

    # Generating predictions
    if continuous:
        if hasattr(fm, 'predict'):
            qa = fm.predict(xdata)
            return qa, fm
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

    else:
        if hasattr(fm, 'predict_proba'):
            qa = fm.predict_proba(xdata)
            if qa.ndim == 1:  # Allows for PyGAM
                return qa, fm
            else:
                return qa[:, 1], fm
        elif hasattr(fm, 'predict'):
            qa = fm.predict(xdata)
            return qa, fm
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def stochastic_outcome_predict(xdata, fit_ml_model, continuous):
    """Function to generate predictions machine learning predictions. Used by StochasticTMLE to generate predicted
    probabilities of outcome (i.e. Pr(Y=1 | A=a*, L) in the Monte-Carlo integration procedure
    """
    # Generating predictions
    if continuous:
        if hasattr(fit_ml_model, 'predict'):
            qa_star = fit_ml_model.predict(xdata)
            return qa_star
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")
    else:
        if hasattr(fit_ml_model, 'predict_proba'):
            qa_star = fit_ml_model.predict_proba(xdata)
            if qa_star.ndim == 1:  # Allows for PyGAM
                return qa_star
            else:
                return qa_star[:, 1]
        elif hasattr(fit_ml_model, 'predict'):
            qa_star = fit_ml_model.predict(xdata)
            return qa_star
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def distribution_shift(data, model, shift):
    """Calculate the new probabilities based on a model and returns the shifted values"""
    # Getting predicted probabilities
    f = sm.families.family.Binomial()
    log = smf.glm(model, data, family=f).fit()
    pred_prob = np.array(log.predict(data))

    # Converting to log-odds
    odds = probability_to_odds(pred_prob)
    logodds = np.log(odds)

    # Shifting distribution
    logodds += shift

    # Back transforming and returning
    return odds_to_probability(np.exp(logodds))


def create_threshold(data, variables, thresholds, definitions):
    for v, t, d in zip(variables, thresholds, definitions):
        if type(t) is float:
            label = v + '_tp' + str(int(t * 100))
        else:
            label = v + '_t' + str(t)
        data[label] = np.where(data[v + '_' + d] > t, 1, 0)
