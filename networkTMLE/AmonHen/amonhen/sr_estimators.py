import re
import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import networkx as nx

from scipy.stats import logistic, poisson, binom, norm
from scipy.sparse import lil_matrix

from amonhen.utils import (network_to_df, fast_exp_map, probability_to_odds, odds_to_probability,
                           bounding, exp_map_individual, create_threshold)


class NetworkGFormula:
    def __init__(self, network, exposure, outcome, verbose=False):
        """Implementation of the g-formula estimator described in Sofrygin & van der Laan 2017	
        """
        # Background processing to convert network attribute data to pandas DataFrame
        df = network_to_df(network)

        if not df[exposure].value_counts().index.isin([0, 1]).all():
            raise ValueError("NetworkGFormula only supports binary exposures currently")

        if df[outcome].value_counts().index.isin([0, 1]).all():
            self._continuous_ = False
        else:
            self._continuous_ = True

        network = nx.convert_node_labels_to_integers(network, first_label=0, label_attribute='_original_id_')

        self.network = network
        self.adj_matrix = nx.adjacency_matrix(network, weight=None)
        self.exposure = exposure
        self.outcome = outcome

        # Creating variable mapping for all variables in the network
        for v in [var for var in list(df.columns) if var not in ['_original_id_', outcome]]:
            v_vector = np.asarray(df[v])
            df[v + '_sum'] = fast_exp_map(self.adj_matrix, v_vector, measure='sum')
            df[v + '_mean'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean')
            df[v + '_mean'] = df[v + '_mean'].fillna(0)  # isolates should have mean=0
            df[v + '_var'] = fast_exp_map(self.adj_matrix, v_vector, measure='var')
            df[v + '_var'] = df[v + '_var'].fillna(0)  # isolates should have var=0
            df[v + '_mean_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean_dist')
            df[v + '_mean_dist'] = df[v + '_mean_dist'].fillna(0)  # isolates should have mean_dist=0
            df[v + '_var_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='var_dist')
            df[v + '_var_dist'] = df[v + '_var_dist'].fillna(0)  # isolates should have var_dist=0

        # Calculating Degree
        degree_data = pd.DataFrame.from_dict(dict(network.degree), orient='index').rename(columns={0: 'degree'})
        self.df = pd.merge(df, degree_data, how='left', left_index=True, right_index=True)

        # Output attributes
        self.marginals_vector = None
        self.marginal_outcome = None

        # Storage for items I need later
        self._outcome_model = None
        self._q_model = None
        self._verbose_ = verbose
        self._thresholds_ = []
        self._thresholds_variables_ = []
        self._thresholds_def_ = []
        self._thresholds_any_ = False

    def outcome_model(self, model, distribution='normal'):
        """Outcome model specification	
        Uses special map variables, like `treatment_map`	
        """
        self._q_model = self.outcome + ' ~ ' + model

        if not self._continuous_:
            f = sm.families.family.Binomial()
        elif distribution.lower() == 'normal':
            f = sm.families.family.Gaussian()
        elif distribution.lower() == 'poisson':
            f = sm.families.family.Poisson()
        else:
            raise ValueError("Distribution" + str(distribution) + " is not currently supported")

        self._outcome_model = smf.glm(self._q_model, self.df, family=f).fit()

        if self._verbose_:
            print('==============================================================================')
            print('Outcome model')
            print(self._outcome_model.summary())

    def fit(self, p, samples=100):
        """Estimate the g-formula for network data under weak interference at coverage `p`	
        """
        marginals = []
        for s in range(samples):
            # Selecting and applying binary treatment
            g = self.df.copy()
            g[self.exposure] = np.random.binomial(n=1, p=p, size=g.shape[0])

            # Back-calculate updated exposure mapping
            v_vector = np.asarray(g[self.exposure])
            g[self.exposure + '_sum'] = fast_exp_map(self.adj_matrix, v_vector, measure='sum')
            g[self.exposure + '_mean'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean')
            g[self.exposure + '_mean'] = g[self.exposure + '_mean'].fillna(0)  # isolates should have mean=0
            g[self.exposure + '_var'] = fast_exp_map(self.adj_matrix, v_vector, measure='var')
            g[self.exposure + '_var'] = g[self.exposure + '_var'].fillna(0)  # isolates should have var=0
            g[self.exposure + '_mean_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean_dist')
            g[self.exposure + '_mean_dist'] = g[self.exposure + '_mean_dist'].fillna(0)  # isolates have mean_dist=0
            g[self.exposure + '_var_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='var_dist')
            g[self.exposure + '_var_dist'] = g[self.exposure + '_var_dist'].fillna(0)  # isolates have var_dist=0
            if self._thresholds_any_:
                create_threshold(data=g, variables=self._thresholds_variables_,
                                 thresholds=self._thresholds_, definitions=self._thresholds_def_)

            # Generating predictions for treatment plan
            g[self.outcome] = np.nan
            g[self.outcome] = self._outcome_model.predict(g)
            marginals.append(np.mean(g[self.outcome]))

        self.marginals_vector = marginals
        self.marginal_outcome = np.mean(marginals)

    def define_threshold(self, variable, threshold, definition):
        """Function arbitrarily allows for multiple different defined thresholds
        """
        self._thresholds_any_ = True
        self._thresholds_.append(threshold)
        self._thresholds_variables_.append(variable)
        self._thresholds_def_.append(definition)
        create_threshold(self.df, variables=[variable], thresholds=[threshold], definitions=[definition])

    def summary(self, decimal=3):
        """Prints summary results for the sample average treatment effect under the treatment plan specified in
        the fit procedure

        Parameters
        ----------
        decimal : int
            Number of decimal places to display
        """
        if self.marginal_outcome is None:
            raise ValueError('The fit() statement must be ran before summary()')

        print('======================================================================')
        print('                  Network G-Computation Estimator                     ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15}'
        print(fmt.format(self.outcome))
        print('======================================================================')
        print('Overall incidence:      ', np.round(self.marginal_outcome, decimals=decimal))
        print('======================================================================')


class NetworkIPTW:
    def __init__(self, network, exposure, outcome, verbose=False):
        """Implementation of the IPTW estimator described in Sofrygin & van der Laan 2017
        """
        # Background processing to convert network attribute data to pandas DataFrame
        df = network_to_df(network)

        if not df[exposure].value_counts().index.isin([0, 1]).all():
            raise ValueError("NetworkGFormula only supports binary exposures currently")

        if not df[outcome].value_counts().index.isin([0, 1]).all():
            self._continuous_ = False
        else:
            self._continuous_ = True

        network = nx.convert_node_labels_to_integers(network, first_label=0, label_attribute='_original_id_')
        self.network = network
        self.adj_matrix = nx.adjacency_matrix(network, weight=None)
        self.exposure = exposure
        self.outcome = outcome

        # Creating variable mapping for all variables in the network
        for v in [var for var in list(df.columns) if var not in ['_original_id_', outcome]]:
            v_vector = np.asarray(df[v])
            df[v + '_sum'] = fast_exp_map(self.adj_matrix, v_vector, measure='sum')
            df[v + '_mean'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean')
            df[v + '_mean'] = df[v + '_mean'].fillna(0)  # isolates should have mean=0
            df[v + '_var'] = fast_exp_map(self.adj_matrix, v_vector, measure='var')
            df[v + '_var'] = df[v + '_var'].fillna(0)  # isolates should have var=0
            df[v + '_mean_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean_dist')
            df[v + '_mean_dist'] = df[v + '_mean_dist'].fillna(0)  # isolates should have mean_dist=0
            df[v + '_var_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='var_dist')
            df[v + '_var_dist'] = df[v + '_var_dist'].fillna(0)  # isolates should have var_dist=0

        # Calculating Degree
        degree_data = pd.DataFrame.from_dict(dict(network.degree), orient='index').rename(columns={0: 'degree'})
        self.df = pd.merge(df, degree_data, how='left', left_index=True, right_index=True)

        # Output attributes
        self.marginals_vector = None
        self.marginal_outcome = None
        self.conditional_variance = None
        self.conditional_ci = None
        self.alpha = 0.05

        # Storage for items I need later
        self._denominator_ = None
        self._gi_model = None
        self._gs_model = None
        self._map_dist_ = None
        self._denominator_estimated_ = False
        self._gs_measure_ = None
        self._treatment_models = []
        self._verbose_ = verbose
        self._thresholds_ = []
        self._thresholds_variables_ = []
        self._thresholds_def_ = []
        self._thresholds_any_ = False

        self._max_degree_ = np.max([d for n, d in network.degree])
        exp_map_cols = exp_map_individual(network=network, measure=exposure, max_degree=self._max_degree_)
        self._nonparam_cols_ = list(exp_map_cols.columns)

    def exposure_model(self, model):
        """Outcome model specification	
        Used special map variables, like `treatment_map`	
        """
        self._gi_model = self.exposure + ' ~ ' + model
        self._denominator_estimated_ = False

    def exposure_map_model(self, model, measure=None, distribution=None):
        """Outcome model specification	
        Used special map variables, like `treatment_map`	
        """
        self._check_distribution_measure_(distribution=distribution, measure=measure)

        # Getting distribution for parametric models. Ignored if custom_model is not None
        if distribution is None:
            self._map_dist_ = distribution
        else:
            self._map_dist_ = distribution.lower()

        if measure is not None:
            self._gs_measure_ = self.exposure + '_' + measure
        self._gs_model = model
        self._treatment_models = []
        self._denominator_estimated_ = False

    def fit(self, p, samples=100, bound=None):
        """Estimate IPTW for network data under weak interference at coverage `p`	
        """
        if not self._denominator_estimated_:
            self._denominator_ = self._estimate_g_(data=self.df.copy(), distribution=self._map_dist_)

        # Creating pooled sample to estimate weights
        pooled_df = self._generate_pooled_sample(p=p, samples=samples)

        # Generating numerator weights for treatment plan
        numerator = self._estimate_gstar_(pooled_data=pooled_df.copy(),
                                          data_to_predict=self.df.copy(), distribution=self._map_dist_)

        # Calculating H = g-star(As | Ws) / g(As | Ws)
        iptw = numerator / self._denominator_
        if bound is not None:
            bounding(ipw=iptw, bound=bound)

        # Calculating marginal outcome
        self.marginal_outcome = np.average(self.df[self.outcome], weights=iptw)

        # Estimating Variance
        y_ = np.array(self.df[self.outcome])
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        self.conditional_variance = self._est_variance_conditional_(iptw=iptw, obs_y=y_, psi=self.marginal_outcome)
        self.conditional_ci = [self.marginal_outcome - zalpha*np.sqrt(self.conditional_variance),
                               self.marginal_outcome + zalpha*np.sqrt(self.conditional_variance)]

    def summary(self, decimal=3):
        """Prints summary results for the sample average treatment effect under the treatment plan specified in
        the fit procedure

        Parameters
        ----------
        decimal : int
            Number of decimal places to display
        """
        if self.marginal_outcome is None:
            raise ValueError('The fit() statement must be ran before summary()')

        print('======================================================================')
        print('            Network Inverse Probability Weight Estimator              ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} No. Background Nodes: {:<20}'
        print(fmt.format(self.outcome, 0))
        fmt = 'g-Model:          {:<15} g-Distribution:       {:<20}'
        if self._map_dist_ is None:
            gs = 'Nonparametric'
        else:
            gs = self._map_dist_.capitalize()
        print(fmt.format('Logistic', gs))

        print('======================================================================')
        print('Overall incidence:      ', np.round(self.marginal_outcome, decimals=decimal))
        print('======================================================================')
        print('Conditional')
        print(str(round(100 * (1 - self.alpha), 0)) + '% CL:    ', np.round(self.conditional_ci, decimals=decimal))
        print('======================================================================')

    def _generate_pooled_sample(self, p, samples):
        pooled_sample = []

        for s in range(samples):
            g = self.df.copy()
            g[self.exposure] = np.random.binomial(n=1, p=p, size=g.shape[0])

            g[self.exposure+'_sum'] = fast_exp_map(self.adj_matrix, np.array(g[self.exposure]), measure='sum')
            g[self.exposure + '_mean'] = fast_exp_map(self.adj_matrix, np.array(g[self.exposure]), measure='mean')
            g[self.exposure + '_mean'] = g[self.exposure + '_mean'].fillna(0)  # isolates should have mean=0
            g[self.exposure + '_var'] = fast_exp_map(self.adj_matrix, np.array(g[self.exposure]), measure='var')
            g[self.exposure + '_var'] = g[self.exposure + '_var'].fillna(0)  # isolates should have mean=0
            g[self.exposure + '_mean_dist'] = fast_exp_map(self.adj_matrix,
                                                           np.array(g[self.exposure]), measure='mean_dist')
            g[self.exposure + '_mean_dist'] = g[self.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0
            g[self.exposure + '_var_dist'] = fast_exp_map(self.adj_matrix,
                                                                     np.array(g[self.exposure]), measure='var_dist')
            g[self.exposure + '_mean_dist'] = g[self.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0

            if self._gs_measure_ is None:
                network = self.network.copy()
                a = np.array(g[self.exposure])
                for n in network.nodes():
                    network.node[n][self.exposure] = a[n]
                df = exp_map_individual(network, measure=self.exposure, max_degree=self._max_degree_).fillna(0)
                for c in self._nonparam_cols_:
                    g[c] = df[c]

            if self._thresholds_any_:
                create_threshold(data=g, variables=self._thresholds_variables_,
                                 thresholds=self._thresholds_, definitions=self._thresholds_def_)

            g['_sample_id_'] = s
            pooled_sample.append(g)

        return pd.concat(pooled_sample, axis=0, ignore_index=True)

    def _estimate_g_(self, data, distribution):
        # Treatment of individual
        f = sm.families.family.Binomial()
        treat_i_model = smf.glm(self._gi_model, data, family=f).fit()
        if self._verbose_:
            print('==============================================================================')
            print('g-model: A')
            print(treat_i_model.summary())

        self._treatment_models.append(treat_i_model)
        pred = treat_i_model.predict(data)
        pr_i = np.where(data[self.exposure] == 1, pred, 1 - pred)

        # Summary measure for contacts
        if distribution is None:
            f = sm.families.family.Binomial()
            cond_vars = patsy.dmatrix(self._gs_model, data, return_type='matrix')
            pr_s = np.array([1.] * data.shape[0])

            for c in self._nonparam_cols_:
                # Estimating probability
                treat_s_model = sm.GLM(data[c], cond_vars, family=f).fit()
                if self._verbose_:
                    print('==============================================================================')
                    print('g-model: ' + c)
                    print(treat_s_model.summary())

                self._treatment_models.append(treat_s_model)
                pred = treat_s_model.predict(cond_vars)
                pr_s *= np.where(data[c] == 1, pred, 1 - pred)

                # Stacking vector to the end of the array
                cond_vars = np.c_[cond_vars, np.array(data[c])]

        elif distribution == 'normal':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.ols(gs_model, data).fit()
            self._treatment_models.append(treat_s_model)
            pred = treat_s_model.predict(data)
            pr_s = norm.pdf(data[self._gs_measure_], pred, np.sqrt(treat_s_model.mse_resid))
            if self._verbose_:
                print('==============================================================================')
                print('g-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'poisson':
            f = sm.families.family.Poisson()
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.glm(gs_model, data, family=f).fit()
            self._treatment_models.append(treat_s_model)
            pred = treat_s_model.predict(data)
            pr_s = poisson.pmf(data[self._gs_measure_], pred)
            if self._verbose_:
                print('==============================================================================')
                print('g-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'multinomial':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.mnlogit(gs_model, data).fit(disp=False)
            self._treatment_models.append(treat_s_model)
            pred = treat_s_model.predict(data)
            values = pd.get_dummies(data[self._gs_measure_])
            pr_s = np.array([0] * data.shape[0])
            for i in data[self._gs_measure_].unique():
                pr_s += pred[i] * values[i]

            if self._verbose_:
                print('==============================================================================')
                print('g-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'threshold':
            f = sm.families.family.Binomial()
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.glm(gs_model, data, family=f).fit()
            self._treatment_models.append(treat_s_model)
            pred = treat_s_model.predict(data)
            pr_s = np.where(data[self._gs_measure_] == 1, pred, 1 - pred)
            if self._verbose_:
                print('==============================================================================')
                print('g-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        else:
            raise ValueError("Invalid distribution choice")

        # Creating estimated g(As|Ws)
        return pr_i * pr_s

    def _estimate_gstar_(self, pooled_data, data_to_predict, distribution):
        # Treatment of individual
        f = sm.families.family.Binomial()
        treat_i_model = smf.glm(self._gi_model, pooled_data, family=f).fit()
        pred = treat_i_model.predict(data_to_predict)
        pr_i = np.where(data_to_predict[self.exposure] == 1, pred, 1 - pred)
        if self._verbose_:
            print('==============================================================================')
            print('gstar-model: A')
            print(treat_i_model.summary())

        # Treatment of direct contacts
        if distribution is None:
            f = sm.families.family.Binomial()
            cond_vars = patsy.dmatrix(self._gs_model, pooled_data, return_type='matrix')
            pred_vars = patsy.dmatrix(self._gs_model, data_to_predict, return_type='matrix')
            pr_s = np.array([1.] * data_to_predict.shape[0])

            for c in self._nonparam_cols_:
                # Estimating probability
                treat_s_model = sm.GLM(pooled_data[c], cond_vars, family=f).fit()
                pred = treat_s_model.predict(pred_vars)
                pr_s *= np.where(data_to_predict[c] == 1, pred, 1 - pred)
                if self._verbose_:
                    print('==============================================================================')
                    print('gstar-model: ' + c)
                    print(treat_s_model.summary())

                # Stacking vector to the end of the array
                cond_vars = np.c_[cond_vars, np.array(pooled_data[c])]
                pred_vars = np.c_[pred_vars, np.array(data_to_predict[c])]

        elif distribution == 'normal':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.ols(gs_model, pooled_data).fit()
            pred = treat_s_model.predict(data_to_predict)
            pr_s = norm.pdf(data_to_predict[self._gs_measure_], pred, np.sqrt(treat_s_model.mse_resid))
            if self._verbose_:
                print('==============================================================================')
                print('gstar-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'poisson':
            f = sm.families.family.Poisson()
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.glm(gs_model, pooled_data, family=f).fit()
            pred = treat_s_model.predict(data_to_predict)
            pr_s = poisson.pmf(data_to_predict[self._gs_measure_], pred)
            if self._verbose_:
                print('==============================================================================')
                print('gstar-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'multinomial':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.mnlogit(gs_model, pooled_data).fit(disp=False)
            pred = treat_s_model.predict(data_to_predict)
            values = pd.get_dummies(data_to_predict[self._gs_measure_])
            pr_s = np.array([0] * data_to_predict.shape[0])
            for i in data_to_predict[self._gs_measure_].unique():
                pr_s += pred[i] * values[i]

            if self._verbose_:
                print('==============================================================================')
                print('gstar-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'threshold':
            f = sm.families.family.Binomial()
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            treat_s_model = smf.glm(gs_model, pooled_data, family=f).fit()
            pred = treat_s_model.predict(data_to_predict)
            pr_s = np.where(data_to_predict[self._gs_measure_] == 1, pred, 1 - pred)
            if self._verbose_:
                print('==============================================================================')
                print('gstar-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        else:
            raise ValueError("Invalid distribution choice")

        # Creating estimated probability
        return pr_i * pr_s

    def define_threshold(self, variable, threshold, definition):
        """Function arbitrarily allows for multiple different defined thresholds
        """
        self._thresholds_any_ = True
        self._thresholds_.append(threshold)
        self._thresholds_variables_.append(variable)
        self._thresholds_def_.append(definition)
        create_threshold(self.df, variables=[variable], thresholds=[threshold], definitions=[definition])

    @staticmethod
    def _est_variance_conditional_(iptw, obs_y, psi):
        """Variance estimator from Sofrygin & van der Laan 2017; section 6.3
        Interpretation is conditional on observed W. The advantage is narrower confidence intervals and easier to
        estimate for the cost of reduced interpretation capacity
        """
        return np.mean((iptw*obs_y - psi)**2) / iptw.shape[0]

    @staticmethod
    def _check_distribution_measure_(distribution, measure):
        """Checks whether the distribution and measure specified are compatible"""
        if distribution is None:
            if measure is not None:
                raise ValueError("The distribution `None` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'normal':
            if measure not in ['sum', 'mean']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'poisson':
            if measure not in ['sum', 'mean']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'multinomial':
            if measure not in ['sum']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'threshold':
            # if re.search(r"^t[0-9]+$", measure) is None:
            if re.search(r"^t\d", measure) is None and re.search(r"^tp\d", measure) is None:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        else:
            raise ValueError("The distribution `"+str(distribution)+"` is not currently implemented")
