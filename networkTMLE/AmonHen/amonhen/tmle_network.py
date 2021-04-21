import re
import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import poisson, norm
from scipy.stats.kde import gaussian_kde

from amonhen.utils import (network_to_df, fast_exp_map, exp_map_individual, tmle_unit_bounds, tmle_unit_unbound,
                           probability_to_odds, odds_to_probability, bounding, check_conditional,
                           outcome_learner_fitting, outcome_learner_predict, exposure_machine_learner,
                           targeting_step, create_threshold)


class NetworkTMLE:
    """Implementation of the targeted maximum likelihood estimation for dependent data through summary measures. The
    following procedure estimates the expected incidence under a treatment plan of interest. For stochastic treatment
    plans, the expected incidence is obtained through Monte Carlo integration of a subsample of possible treatment
    allotments that correspond to the plan of interest.

    Note
    ----
    Network TMLE makes the weak dependence assumption, such that only direct contacts' treatment can interfere with
    individual i's outcome.

    Parameters
    ----------
    network : NetworkX Graph
        NetworkX undirected network WITHOUT self-loops. Additionally, all data should be stored for each node. In
        the background, `NetworkTMLE` extracts the node data from the graph and creates a pandas DataFrame object
        from that information. It is important that no nodes have missing data. Currently there is no procedure to
        handle missing data
    exposure : str
        String indicating the exposure variable of interest. Variable should be part of node data in the input
        network
    outcome : str
        String indicating the outcome variable of interest. Variable should be part of node data in the input
        network
    degree_restrict : None, list, tuple, optional
        Restriction on the minimum & maximum degree for the sample average treatment effect. All samples below the first
        value OR above the second level are considered as "background" features. Hence the intervention does not change
        their exposure. Additionally, the SATE no longer includes those individuals. See Ogburn et al. 2017 for
        further details on this.
        Must be a list with a length of two, where the first value corresponds to the lower bound and the second is the
        upper bound for degree. Values are inclusive
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        For continuous outcomes, TMLE needs to bound Y between 0 and 1. However, 0/1 cannot be included in these
        bounded values. This specification sets the bounds for the continuous outcomes. The default is 0.0005.
    verbose : bool, optional
        Whether to print all intermediary model results for the estimation process. When set to True, each of the
        model results are printed to the console. The default is False.

    Notes
    -----
    The NetworkTMLE estimation algorithm uses the following procedure to estimate the marginal incidence under the
    treatment plan of interest: (1) estimate the outcome model, (2) estimate the exposure model,
    (3) perform the targeting step, (4) repeat step 3 for Monte Carlo integration

    Note
    ----
    `NetworkTMLE` calculates exposure mapping variables automatically with the input network. These variables are
    saved as variable-name_map. So for a variable `'A'`, the newly created exposure mapping variable calculated is
    `'A_map'`

    For stochastic treatment plans, there are many alternative treatment distributions. The differences between
    particular allotments are incorporated through Monte Carlo integration. Using the generated data sets in step 2
    the Q* is estimated by calculating the value from the targeted model in step 3 and the treatment plan. This
    process is repeated for all the treatment plans generated as part of `samples`. These different incidences under
    the treatment allotments are summarized through the mean

    Examples
    --------
    Estimation with `NetworkTMLE`

    >>>tmle = NetworkTMLE(network=graph, exposure='A', outcome='Y')
    >>>tmle.exposure_model('W + W_map')
    >>>tmle.exposure_map_model('A + W + W_map', distribution=None)
    >>>tmle.outcome_model('A + W + A_map + W_map', print_results=False)
    >>>tmle.fit(p=0.8, bound=10e5)
    >>>tmle.summary()

    Note
    ----
    For directed networks, the direction of of influence goes from the target node to the source (i.e. opposite of the
    arrow direction). If `A --> B` then B's covariates will be part of the A's summary measures.

    References
    ----------
    Sofrygin O & van der Laan MJ. (2017). Semi-parametric estimation and inference for the mean outcome of
    the single time-point intervention in a causally connected population. Journal of Causal Inference, 5(1).

    Ogburn EL, Sofrygin O, Diaz I, & van der Laan MJ. (2017). Causal inference for social network data.
    arXiv preprint arXiv:1705.08527.

    Sofrygin O, Ogburn EL, & van der Laan MJ. (2018). Single Time Point Interventions in Network-Dependent
    Data. In Targeted Learning in Data Science (pp. 373-396). Springer, Cham.
    """
    def __init__(self, network, exposure, outcome, degree_restrict=None, alpha=0.05,
                 continuous_bound=0.0005, verbose=False):
        if not all([isinstance(x, int) for x in list(network.nodes())]):  # Node IDs *must* be integers for processing
            raise ValueError("NetworkTMLE requires that all node IDs must be integers")  # Probably not needed anymore

        if nx.number_of_selfloops(network) > 0:  # Self-loops don't make sense in the context of interference (to me)
            raise ValueError("NetworkTMLE does not support networks with self-loops")

        if degree_restrict is not None:
            self._check_degree_restrictions_(bounds=degree_restrict)

        # Allowing for non-consecutive IDs
        network = nx.convert_node_labels_to_integers(network, first_label=0, label_attribute='_original_id_')
        self.network = network
        self.exposure = exposure
        self.outcome = outcome

        # Background processing to convert network attribute data to pandas DataFrame
        self.adj_matrix = nx.adjacency_matrix(self.network, weight=None)
        df = network_to_df(self.network)

        if not df[exposure].value_counts().index.isin([0, 1]).all():
            raise ValueError("NetworkTMLE only supports binary exposures currently")

        # Manage outcomes
        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():  # Binary outcomes
            self._continuous_outcome = False
            self._cb = 0.0
        else:  # Continuous outcomes: bounding procedure
            self._continuous_outcome = True
            self._continuous_min = np.min(df[outcome]) - continuous_bound
            self._continuous_max = np.max(df[outcome]) + continuous_bound
            self._cb = continuous_bound
            df[outcome] = tmle_unit_bounds(y=df[self.outcome], mini=self._continuous_min, maxi=self._continuous_max)
            self._q_min_bound = np.min(df[self.outcome])
            self._q_max_bound = np.max(df[self.outcome])

        # Creating variable mapping for all variables in the network
        oid = "_original_id_"
        for v in [var for var in list(df.columns) if var not in [oid, outcome]]:
            v_vector = np.asarray(df[v])
            df[v+'_sum'] = fast_exp_map(self.adj_matrix, v_vector, measure='sum')
            df[v+'_mean'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean')
            df[v + '_mean'] = df[v+'_mean'].fillna(0)  # isolates should have mean=0
            df[v+'_var'] = fast_exp_map(self.adj_matrix, v_vector, measure='var')
            df[v + '_var'] = df[v+'_var'].fillna(0)  # isolates should have var=0
            df[v+'_mean_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='mean_dist')
            df[v + '_mean_dist'] = df[v+'_mean_dist'].fillna(0)  # isolates should have mean_dist=0
            df[v+'_var_dist'] = fast_exp_map(self.adj_matrix, v_vector, measure='var_dist')
            df[v + '_var_dist'] = df[v+'_var_dist'].fillna(0)  # isolates should have var_dist=0

        # Creating individual column mappings (used for non-parametric exposure_map_model()
        if degree_restrict is None:
            if nx.is_directed(network):
                self._max_degree_ = np.max([d for n, d in network.out_degree])
            else:
                self._max_degree_ = np.max([d for n, d in network.degree])
        else:
            self._max_degree_ = degree_restrict[1]
        exp_map_cols = exp_map_individual(network=network, measure=exposure, max_degree=self._max_degree_)
        self._nonparam_cols_ = list(exp_map_cols.columns)  # saving for loop procedure for g_model estimations
        df = pd.merge(df,
                      exp_map_cols.fillna(0),  # fills NaN with 0's to keep same dimension across i
                      how='left', left_index=True, right_index=True)

        # Calculating Degree
        if nx.is_directed(network):  # Out-degree is needed for networks since arrows represent reverse influence direct
            degree_data = pd.DataFrame.from_dict(dict(network.out_degree), orient='index').rename(columns={0: 'degree'})
        else:  # Undirected is just degree
            degree_data = pd.DataFrame.from_dict(dict(network.degree), orient='index').rename(columns={0: 'degree'})
        self.df = pd.merge(df, degree_data, how='left', left_index=True, right_index=True)

        # Subset out nodes by degree that do NOT change treatment status
        if degree_restrict is not None:
            self.df['__degree_flag__'] = self._degree_restrictions_(degree_dist=self.df['degree'],
                                                                    bounds=degree_restrict)
        else:
            self.df['__degree_flag__'] = 0
        self.df_restricted = self.df.loc[self.df['__degree_flag__'] == 0].copy()

        # Output attributes
        self.marginals_vector = None
        self.marginal_outcome = None
        self.conditional_variance = None
        self.conditional_ci = None

        # Storage for items I need later
        self.alpha = alpha
        self._outcome_model = None
        self._q_model = None
        self._Qinit_ = None
        self._treatment_models = []
        self._gi_model = None
        self._gs_model = None
        self._gs_measure_ = None
        self._exposure_measure_ = None
        self._map_dist_ = None
        self._denominator_ = None
        self._denominator_estimated_ = False
        self._verbose_ = verbose
        self._thresholds_ = []
        self._thresholds_variables_ = []
        self._thresholds_def_ = []
        self._thresholds_any_ = False

        # Custom model / machine learner storage
        self._gi_custom_ = None
        self._gi_custom_sim_ = None
        self._gs_custom_ = None
        self._gs_custom_sim_ = None
        self._q_custom_ = None

        # Storage items for summary formatting
        self._specified_p_ = None
        self._specified_bound_ = None
        self._resamples_ = None

    def exposure_model(self, model, custom_model=None, custom_model_sim=None):
        """Exposure model for individual i.  Estimates Pr(A=a|W, W_map) using a logistic regression model.

        Note
        ----
        Only saves the model specifications. IPTW are calculated later during the fit() function

        Parameters
        ----------
        model : str
            Exposure mapping model
        custom_model
            User-specified nuisance model estimator. Format should be like sklearn
        custom_model_sim
            User-specified model. This allows the user to specify a different IPW model to be fit for the numerator.
            That model is fit to the simulated data, so some constraints may be added to speed up the estimation
            procedure. If None and custom_model is not None, copies over the custom_model used.
        """
        self._gi_model = model
        self._treatment_models = []
        self._denominator_estimated_ = False

        # Storing user-specified model
        self._gi_custom_ = custom_model
        if custom_model_sim is None:
            self._gi_custom_sim_ = custom_model
        else:
            self._gi_custom_sim_ = custom_model_sim

    def exposure_map_model(self, model, measure=None, distribution=None, custom_model=None, custom_model_sim=None):
        """Exposure summary measure model for individual i. Estimates Pr(A_map=a|A=a, W, W_map) using a logistic
        regression model.

        Note
        ----
        Only saves the model specifications. IPTW are calculated later during the fit() function

        There are several options for the distributions of the summary measure. One option is a series of logit models
        that estimates the probability for each individual contact (works best for uniform distributions). However, this
        approach may not always be possible to estimate. Instead, parametric distributional assumption can be used
        instead. Currently, implemented are normal and Poisson distributions.

        Parameters
        ----------
        model : str
            Exposure mapping model. Ideally would include treatment for individual i
        measure : None, str, optional
            Exposure mapping to use for the modeling statement. Options include 'mean' and 'sum'. Default is None
            which natively works with the `distribution=None` option
        distribution : None, str, optional
            Distribution to use for exposure mapping model. Options include: series of logit models (None),
            Poisson ('poisson'), Normal ('normal').
        custom_model : None, optional
            User-specified nuisance model estimator. Format should be like sklearn
        custom_model_sim
            User-specified model. This allows the user to specify a different IPW model to be fit for the numerator.
            That model is fit to the simulated data, so some constraints may be added to speed up the estimation
            procedure. If None and custom_model is not None, copies over the custom_model used.
        """
        # Checking that distribution and measure are compatible
        self._check_distribution_measure_(distribution=distribution, measure=measure)

        # Getting distribution for parametric models. Ignored if custom_model is not None
        if distribution is None:
            self._map_dist_ = distribution
        else:
            self._map_dist_ = distribution.lower()

        if measure is not None:
            self._exposure_measure_ = measure
            self._gs_measure_ = self.exposure + '_' + measure
        self._gs_model = model
        self._treatment_models = []
        self._denominator_estimated_ = False

        # Storing user-specified model(s)
        if custom_model is not None:
            if distribution in ["poisson", "threshold"]:
                pass
            elif distribution is None:
                raise ValueError("...generalized conditional distribution super-learner to be added...")
                # TODO for me to later read the implementation in TL for Data Science, Chapter 14.4
            else:
                raise ValueError("Incompatible `distribution` for implemented `custom_model` forms. Select from: "
                                 "None, poisson, binomial, threshold")
        self._gs_custom_ = custom_model
        if custom_model_sim is None:
            self._gs_custom_sim_ = custom_model
        else:
            self._gs_custom_sim_ = custom_model_sim

    def outcome_model(self, model, custom_model=None, distribution='normal'):
        """Estimation of E(Y|A, A_map, W, W_map), which is also written sometimes as Q(A, As, W, Ws).

        Note
        ----
        Estimates the outcome model (g-formula) using the observed data and generates predictions under the observed
        distribution of the exposure.

        Parameters
        ----------
        model : str
            Specified Q-model
        custom_model :
            User-specified nuisance model estimator. Format should be like sklearn
        distribution : optional, str
            For non-binary outcome variables, the distribution of Y must be specified. Default is 'normal'.
        """
        self._q_model = model

        if custom_model is None:
            if not self._continuous_outcome:
                f = sm.families.family.Binomial()
            elif distribution.lower() == 'normal':
                f = sm.families.family.Gaussian()
            elif distribution.lower() == 'poisson':
                f = sm.families.family.Poisson()
            else:
                raise ValueError("Distribution"+str(distribution)+" is not currently supported")

            self._outcome_model = smf.glm(self.outcome + ' ~ ' + self._q_model, self.df_restricted, family=f).fit()
            self._Qinit_ = self._outcome_model.predict(self.df_restricted)
            if self._verbose_:
                print('==============================================================================')
                print('Outcome model')
                print(self._outcome_model.summary())

        else:
            data = patsy.dmatrix(model + ' - 1', self.df_restricted)
            # Estimating model
            self._q_custom_ = outcome_learner_fitting(ml_model=custom_model,
                                                      xdata=np.asarray(data),
                                                      ydata=np.asarray(self.df_restricted[self.outcome]))
            # Generating predictions
            self._Qinit_ = outcome_learner_predict(ml_model_fit=self._q_custom_, xdata=np.asarray(data))

        # Ensures all predicted values are bounded
        if self._continuous_outcome:
            self._Qinit_ = np.where(self._Qinit_ < self._q_min_bound, self._q_min_bound, self._Qinit_)
            self._Qinit_ = np.where(self._Qinit_ > self._q_max_bound, self._q_max_bound, self._Qinit_)

    def fit(self, p, samples=100, bound=None, seed=None):
        """Estimation procedure for the sample mean under the specified treatment plan.

        This function estimates the IPTW for the treatment plan of interest, performs the target steps, and
        performs Monte Carlo integration with the targeted model, and calculates confidence intervals. Confidence
        intervals are obtained from influence curves.

        Parameters
        ----------
        p : float, int, list, set
            Percent of population to treat. For conditional treatment plans, a container object of floats. All values
            must be between 0 and 1
        samples : int
            Number of samples to generate to calculate numerator for weights and for the Monte Carlo integration
            procedure for stochastic treatment plans. For deterministic treatment plans (p==1 or p==0), samples is set
            to 1 to reduce computation burden. Deterministic treatment plan do not require the Monte Carlo integration
            procedure
        bound : None, int, float
            Bounds to truncate calculate weights by. Should be between 0 and 1.
        seed : int, None
            Random seed for the Monte Carlo integration procedure
        """
        if self._gi_model is None:
            raise ValueError("exposure_model() must be specified before fit()")
        if self._gs_model is None:
            raise ValueError("exposure_map_model() must be specified before fit()")
        if self._q_model is None:
            raise ValueError("outcome_model() must be specified before fit()")

        # Checking conditional statement setting up estimation scenario
        if type(p) is int:
            raise ValueError("Input `p` must be float or container of floats")
        if type(p) != float:
            if len(p) != self.df.shape[0]:
                raise ValueError("Vector of `p` must be same length as input data")
            if np.all(np.asarray(p) == 0) or np.all(np.asarray(p) == 1):
                raise ValueError("Deterministic treatment plans not supported")
            if np.any(np.asarray(p) < 0) or np.any(np.asarray(p) > 1):
                raise ValueError("Probabilities for treatment must be between 0 and 1")
        else:
            if p == 0 or p == 1:
                raise ValueError("Deterministic treatment plans not supported")
            if p < 0 or p > 1:
                raise ValueError("Probabilities for treatment must be between 0 and 1")

        # 1) Estimate H -- keeping pooled_data for Monte Carlo integration procedure
        h_iptw, pooled_data = self._estimate_iptw_(p=p, samples=samples, bound=bound, seed=seed)
        pooled_data_restricted = pooled_data.loc[pooled_data['__degree_flag__'] == 0].copy()
        self._resamples_ = samples
        if bound is not None:  # Counting up the number bounded
            self._specified_bound_ = np.sum(np.where(h_iptw == bound, 1, 0)) + np.sum(np.where(h_iptw == 1/bound, 1, 0))
        if self._gs_measure_ is None:
            self._for_diagnostics_ = pooled_data_restricted[[self.exposure, self.exposure+"_sum"]].copy()
        else:
            self._for_diagnostics_ = pooled_data_restricted[[self.exposure, self._gs_measure_]].copy()

        # 2) Estimate from Q-model
        # process completed in outcome_model() function and stored in self._Qinit_

        # 3) target parameter
        epsilon = targeting_step(y=self.df_restricted[self.outcome], q_init=self._Qinit_, iptw=h_iptw,
                                 verbose=self._verbose_)

        # 4) MC integration procedure using generated data
        q_star_list = []
        # TODO may be able to speed up by replacing for-loop here with one-shot predictions
        for s in pooled_data['_sample_id_'].unique():
            # Extracting one exposure allocation plan
            ps = pooled_data_restricted.loc[pooled_data_restricted['_sample_id_'] == s]

            # Estimate Y_star based on un-targeted Q-model
            if self._q_custom_ is None:  # Parametric model
                y_star = self._outcome_model.predict(ps)
            else:  # Custom input model by user
                d = patsy.dmatrix(self._q_model + ' - 1', ps)
                y_star = outcome_learner_predict(ml_model_fit=self._q_custom_, xdata=np.asarray(d))
            if self._continuous_outcome:  # Ensures all predicted values are bounded for continuous
                y_star = np.where(y_star < self._q_min_bound, self._q_min_bound, y_star)
                y_star = np.where(y_star > self._q_max_bound, self._q_max_bound, y_star)

            # Estimate Q_star based on Y_star and target step
            logit_qstar = np.log(probability_to_odds(y_star)) + epsilon  # NOTE: log(Y^*) + e
            q_star = odds_to_probability(np.exp(logit_qstar))
            q_star_list.append(np.mean(q_star))

        if self._continuous_outcome:
            self.marginals_vector = tmle_unit_unbound(np.asarray(q_star_list),
                                                      mini=self._continuous_min, maxi=self._continuous_max)
            y_ = np.array(tmle_unit_unbound(self.df_restricted[self.outcome], mini=self._continuous_min,
                                            maxi=self._continuous_max))
            yq0_ = tmle_unit_unbound(self._Qinit_, mini=self._continuous_min, maxi=self._continuous_max)

        else:
            self.marginals_vector = q_star_list
            y_ = np.array(self.df_restricted[self.outcome])
            yq0_ = self._Qinit_

        self.marginal_outcome = np.mean(self.marginals_vector)
        self._specified_p_ = p

        # 5) Variance estimation
        var_cond = self._est_variance_conditional_(iptw=h_iptw,  # Conditional variance (conditional on W)
                                                   obs_y=y_,
                                                   pred_y=yq0_)
        self.conditional_variance = var_cond
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)
        self.conditional_ci = [self.marginal_outcome - zalpha*np.sqrt(var_cond),
                               self.marginal_outcome + zalpha*np.sqrt(var_cond)]

    def summary(self, decimal=3):
        """Prints summary results for the sample average treatment effect under the treatment plan specified in
        the fit() procedure

        Parameters
        ----------
        decimal : int
            Number of decimal places to display
        """
        if self.marginal_outcome is None:
            raise ValueError('The fit() statement must be ran before summary()')

        print('======================================================================')
        print('            Network Targeted Maximum Likelihood Estimator             ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df_restricted.shape[0]))
        fmt = 'Outcome:          {:<15} No. Background Nodes: {:<20}'
        print(fmt.format(self.outcome, self.df.shape[0] - self.df_restricted.shape[0]))
        fmt = 'Q-Model:          {:<15} No. IPW Truncated:    {:<20}'
        if self._specified_bound_ is None:
            b = 0
        else:
            b = self._specified_bound_
        if self._q_custom_ is None:
            qm = 'Logistic'
        else:
            qm = 'Custom'
        print(fmt.format(qm, b))

        fmt = 'g-Model:          {:<15} No. Resamples:        {:<20}'
        if self._gi_custom_ is None:
            gim = 'Logistic'
        else:
            gim = 'Custom'
        print(fmt.format(gim, self._resamples_))

        fmt = 'gs-Model:         {:<15} g-Distribution:       {:<20}'
        if self._gs_custom_ is None:
            if self._map_dist_ is None:
                gsm = 'Logitistic'
            else:
                gsm = self._map_dist_.capitalize()
        else:
            gsm = 'Custom'
        if self._map_dist_ is None:
            gs = 'Nonparametric'
        else:
            gs = self._map_dist_.capitalize()
        print(fmt.format(gsm, gs))

        print('======================================================================')
        print('Overall incidence:      ', np.round(self.marginal_outcome, decimals=decimal))
        print('======================================================================')
        print('Conditional')
        print(str(round(100 * (1 - self.alpha), 0)) + '% CL:    ', np.round(self.conditional_ci, decimals=decimal))
        print('======================================================================')

    def diagnostics(self):
        """Returns diagnostics for the specified NetworkTMLE. Evaluates common diagnostics for the IPW and g-formula
        models used in the IID setting. Also provides a plot to assess the positivity assumption

        Returns
        -------
        Various diagnostics
        """
        # sum, mean, var, mean_dist, var_dist, threshold
        obs_a1 = self.df_restricted.loc[self.df_restricted[self.exposure] == 1, self._gs_measure_].copy()
        obs_a0 = self.df_restricted.loc[self.df_restricted[self.exposure] == 0, self._gs_measure_].copy()
        sim_a1 = self._for_diagnostics_.loc[self._for_diagnostics_[self.exposure] == 1, self._gs_measure_].copy()
        sim_a0 = self._for_diagnostics_.loc[self._for_diagnostics_[self.exposure] == 0, self._gs_measure_].copy()
        if self._exposure_measure_ in [None, "sum"]:
            min_x = 0
            max_x = int(np.max([np.max(self.df_restricted[self._gs_measure_]),
                                np.max(self._for_diagnostics_[self._gs_measure_])]))

            # Observed values
            plt.subplot(211)
            pa1 = obs_a1.value_counts(normalize=True, dropna=True, ascending=True)
            plt.bar([x-0.2 for x in pa1.index], pa1, width=0.4, color='blue', label=r"$A=1$")
            pa0 = obs_a0.value_counts(normalize=True, dropna=True, ascending=True)
            plt.bar([x+0.2 for x in pa0.index], pa0, width=0.4, color='red', label=r"$A=0$")
            plt.xticks([x for x in range(min_x, max_x+1)], ["" for x in range(min_x, max_x+1)])
            plt.xlim([min_x-1, max_x+1])
            plt.ylim([0, 1])
            plt.ylabel("Proportion")
            plt.title("Observed")
            plt.legend()

            # Values under unobserved intervention
            plt.subplot(212)
            qa1 = sim_a1.value_counts(normalize=True, dropna=True, ascending=True)
            plt.bar([x-0.2 for x in qa1.index], qa1, width=0.4, color='blue')
            qa0 = sim_a0.value_counts(normalize=True, dropna=True, ascending=True)
            plt.bar([x+0.2 for x in qa0.index], qa0, width=0.4, color='red')
            plt.xticks([x for x in range(min_x, max_x+1)])
            plt.xlim([min_x-1, max_x+1])
            plt.xlabel(r"$A^s$")
            plt.ylim([0, 1])
            plt.ylabel("Proportion")
            plt.title(r"Under $\alpha$")

        else:
            if self._exposure_measure_ in ["mean"]:
                min_x = 0
                max_x = 1
            else:
                min_x = np.min([np.min(self.df_restricted[self._gs_measure_]),
                                np.min(self._for_diagnostics_[self._gs_measure_])]) - 0.2
                max_x = np.max([np.max(self.df_restricted[self._gs_measure_]),
                                np.max(self._for_diagnostics_[self._gs_measure_])]) + 0.2
            xticks = np.linspace(min_x, max_x, num=11)
            xvals = np.linspace(min_x, max_x, num=200)

            plt.subplot(211)
            pa1 = gaussian_kde(obs_a1.dropna())
            pa0 = gaussian_kde(obs_a0.dropna())
            plt.fill_between(xvals, pa0(xvals), color='red', alpha=0.2, label=None)
            plt.fill_between(xvals, pa1(xvals), color='blue', alpha=0.2, label=None)
            plt.plot(xvals, pa0(xvals), color='red', alpha=1, label=r'$A=0$')
            plt.plot(xvals, pa1(xvals), color='blue', alpha=1, label=r'$A=1$')
            plt.xticks(xticks, ["" for x in range(len(xticks))])
            plt.xlim([min_x, max_x])
            plt.ylabel("Density")
            plt.yticks([])
            plt.title("Observed")
            plt.legend()

            plt.subplot(212)
            qa1 = gaussian_kde(sim_a1.dropna())
            qa0 = gaussian_kde(sim_a0.dropna())
            plt.fill_between(xvals, qa0(xvals), color='red', alpha=0.2, label=None)
            plt.fill_between(xvals, qa1(xvals), color='blue', alpha=0.2, label=None)
            plt.plot(xvals, qa0(xvals), color='red', alpha=1)
            plt.plot(xvals, qa1(xvals), color='blue', alpha=1)
            plt.xticks(xticks)
            plt.xlim([min_x, max_x])
            plt.xlabel(r"$A^s$")
            plt.ylabel("Density")
            plt.yticks([])
            plt.title(r"Under $\alpha$")

        plt.tight_layout()
        plt.show()

    def define_threshold(self, variable, threshold, definition):
        """Function arbitrarily allows for multiple different defined thresholds
        """
        self._thresholds_any_ = True
        self._thresholds_.append(threshold)
        self._thresholds_variables_.append(variable)
        self._thresholds_def_.append(definition)
        create_threshold(self.df_restricted,
                         variables=[variable], thresholds=[threshold], definitions=[definition])

    def _estimate_iptw_(self, p, samples, bound, seed):
        """Background function to estimate the IPTW based on the algorithm described in Sofrygin & van der Laan (2017)

        IPTW are estimated using the following process.

        For the observed data, models are fit to estimate the Pr(A=a) for individual i (treating as IID data) and then
        the Pr(A=a) for their contacts (treated as IID data). These probabilities are then multiplied together to
        generate the denominator.

        To calculate the numerator, the input data set is replicated `samples` times. To each of the data set copies,
        the treatment plan is repeatedly applied. From this large set of observations under the stochastic treatment
        plan of interest, models are again fit to the data, same as the prior procedure. The corresponding probabilities
        are then multiplied together to generate the numerator.

        Allows different exposure mapping procedures. Mapping procedures can place parametric model requirements on the
        probability of treatment distributions. Options currently include:

            None: iterated logit models for all unique contacts
            Poisson
            Multinomial
            Logistic
            Normal
        """
        if not self._denominator_estimated_:
            self._denominator_ = self._estimate_g_(data=self.df_restricted.copy(), distribution=self._map_dist_)
            self._denominator_estimated_ = True  # Updates flag so no need to be re-calculate denominators for plans

        # Creating pooled sample to estimate weights
        pooled_df = self._generate_pooled_sample(p=p, samples=samples, seed=seed)

        numerator = self._estimate_gstar_(pooled_data=pooled_df.copy(),
                                          data_to_predict=self.df_restricted.copy(),
                                          distribution=self._map_dist_)
        # Calculating H = g-star(As | Ws) / g(As | Ws)
        iptw = numerator / self._denominator_
        if bound is not None:
            iptw = bounding(ipw=iptw, bound=bound)

        return iptw, pooled_df

    def _estimate_g_(self, data, distribution):
        # Restricting to degree subset
        data = data.loc[data['__degree_flag__'] == 0].copy()

        ##################################
        # Treatment of individual
        if self._gi_custom_ is None:
            f = sm.families.family.Binomial()
            treat_i_model = smf.glm(self.exposure + ' ~ ' + self._gi_model, data, family=f).fit()
            if self._verbose_:
                print('==============================================================================')
                print('g-model: A')
                print(treat_i_model.summary())

            self._treatment_models.append(treat_i_model)
            pred = treat_i_model.predict(data)
        else:
            xdata = patsy.dmatrix(self._gi_model + ' - 1', data)
            pred = exposure_machine_learner(ml_model=self._gi_custom_,
                                            xdata=np.asarray(xdata),
                                            ydata=np.asarray(self.df_restricted[self.exposure]),
                                            pdata=np.asarray(xdata))
        pr_i = np.where(data[self.exposure] == 1, pred, 1 - pred)

        ##################################
        # Summary measure for contacts
        if distribution is None:
            if self._gs_custom_ is None:
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
            else:
                raise ValueError("Not available currently...")
                # TODO fill in the super-learner conditional density approach here when possible...

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
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            if self._gs_custom_ is None:
                f = sm.families.family.Poisson()
                treat_s_model = smf.glm(gs_model, data, family=f).fit()
                self._treatment_models.append(treat_s_model)
                pred = treat_s_model.predict(data)
                if self._verbose_:
                    print('==============================================================================')
                    print('g-model: '+self._gs_measure_)
                    print(treat_s_model.summary())
            else:  # Custom model for Poisson
                xdata = patsy.dmatrix(self._gs_model + ' - 1', data)
                pred = exposure_machine_learner(ml_model=self._gs_custom_,
                                                xdata=np.asarray(xdata),
                                                ydata=np.asarray(self.df_restricted[self._gs_measure_]),
                                                pdata=np.asarray(xdata))
            pr_s = poisson.pmf(data[self._gs_measure_], pred)

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

        elif distribution == 'binomial':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            f = sm.families.family.Binomial()
            treat_s_model = smf.glm(gs_model, data, family=f).fit()
            self._treatment_models.append(treat_s_model)
            pr_s = treat_s_model.predict(data)
            if self._verbose_:
                print('==============================================================================')
                print('g-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'threshold':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            if self._gs_custom_ is None:
                f = sm.families.family.Binomial()
                treat_s_model = smf.glm(gs_model, data, family=f).fit()
                self._treatment_models.append(treat_s_model)
                pred = treat_s_model.predict(data)
                if self._verbose_:
                    print('==============================================================================')
                    print('g-model: '+self._gs_measure_)
                    print(treat_s_model.summary())
            else:  # Custom model for Threshold (since it obeys a binary
                xdata = patsy.dmatrix(self._gs_model + ' - 1', data)
                pred = exposure_machine_learner(ml_model=self._gs_custom_,
                                                xdata=np.asarray(xdata),
                                                ydata=np.asarray(self.df_restricted[self._gs_measure_]),
                                                pdata=np.asarray(xdata))
            pr_s = np.where(data[self._gs_measure_] == 1, pred, 1 - pred)

        else:
            raise ValueError("Invalid distribution choice")

        ##################################
        # Creating estimated g(As|Ws)
        return pr_i * pr_s

    def _generate_pooled_sample(self, p, samples, seed):
        pooled_sample = []
        for s in range(samples):
            g = self.df.copy()
            probs = np.random.binomial(n=1, p=p, size=g.shape[0])
            g[self.exposure] = np.where(g['__degree_flag__'] == 1, g[self.exposure], probs)  # Subset out background

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
                    network.nodes[n][self.exposure] = a[n]
                df = exp_map_individual(network, measure=self.exposure, max_degree=self._max_degree_).fillna(0)
                for c in self._nonparam_cols_:
                    g[c] = df[c]

            if self._thresholds_any_:
                create_threshold(data=g, variables=self._thresholds_variables_,
                                 thresholds=self._thresholds_, definitions=self._thresholds_def_)

            g['_sample_id_'] = s
            pooled_sample.append(g)

        return pd.concat(pooled_sample, axis=0, ignore_index=True)

    def _estimate_gstar_(self, pooled_data, data_to_predict, distribution):
        # Restricting to degree subset
        pooled_data = pooled_data.loc[pooled_data['__degree_flag__'] == 0].copy()

        ##################################
        # Treatment of individual
        if self._gi_custom_sim_ is None:
            f = sm.families.family.Binomial()
            treat_i_model = smf.glm(self.exposure + ' ~ ' + self._gi_model, pooled_data, family=f).fit()
            if self._verbose_:
                print('==============================================================================')
                print('gstar-model: A')
                print(treat_i_model.summary())
            pred = treat_i_model.predict(data_to_predict)
        else:
            xdat = patsy.dmatrix(self._gi_model + ' - 1', pooled_data)
            pdat = patsy.dmatrix(self._gi_model + ' - 1', data_to_predict)
            pred = exposure_machine_learner(ml_model=self._gi_custom_sim_,
                                            xdata=np.asarray(xdat),
                                            ydata=np.asarray(pooled_data[self.exposure]),
                                            pdata=np.asarray(pdat))
        pr_i = np.where(data_to_predict[self.exposure] == 1, pred, 1 - pred)

        ##################################
        # Treatment of direct contacts
        if distribution is None:
            if self._gs_custom_sim_ is None:
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
            else:
                raise ValueError("Not available currently...")
                # TODO fill in the super-learner conditional density approach here when possible...

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
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            if self._gs_custom_sim_ is None:
                f = sm.families.family.Poisson()
                treat_s_model = smf.glm(gs_model, pooled_data, family=f).fit()
                pred = treat_s_model.predict(data_to_predict)
                if self._verbose_:
                    print('==============================================================================')
                    print('gstar-model: '+self._gs_measure_)
                    print(treat_s_model.summary())
            else:
                xdat = patsy.dmatrix(self._gi_model + ' - 1', pooled_data)
                pdat = patsy.dmatrix(self._gi_model + ' - 1', data_to_predict)
                pred = exposure_machine_learner(ml_model=self._gs_custom_sim_,
                                                xdata=np.asarray(xdat),
                                                ydata=np.asarray(pooled_data[self._gs_measure_]),
                                                pdata=np.asarray(pdat))
            pr_s = poisson.pmf(data_to_predict[self._gs_measure_], pred)

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

        elif distribution == 'binomial':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            f = sm.families.family.Binomial()
            treat_s_model = smf.glm(gs_model, pooled_data, family=f).fit()
            pr_s = treat_s_model.predict(data_to_predict)
            if self._verbose_:
                print('==============================================================================')
                print('gstar-model: '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'threshold':
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model
            if self._gs_custom_sim_ is None:
                f = sm.families.family.Binomial()
                treat_s_model = smf.glm(gs_model, pooled_data, family=f).fit()
                pred = treat_s_model.predict(data_to_predict)
                if self._verbose_:
                    print('==============================================================================')
                    print('gstar-model: '+self._gs_measure_)
                    print(treat_s_model.summary())
            else:
                xdat = patsy.dmatrix(self._gi_model + ' - 1', pooled_data)
                pdat = patsy.dmatrix(self._gi_model + ' - 1', data_to_predict)
                pred = exposure_machine_learner(ml_model=self._gs_custom_sim_,
                                                xdata=np.asarray(xdat),
                                                ydata=np.asarray(pooled_data[self._gs_measure_]),
                                                pdata=np.asarray(pdat))
            pr_s = np.where(data_to_predict[self._gs_measure_] == 1, pred, 1 - pred)

        else:
            raise ValueError("Invalid distribution choice")

        ##################################
        # Creating estimated probability
        return pr_i * pr_s

    @staticmethod
    def _est_variance_conditional_(iptw, obs_y, pred_y):
        """Variance estimator from Sofrygin & van der Laan 2017; section 6.3
        Interpretation is conditional on observed W. The advantage is narrower confidence intervals and easier to
        estimate for the cost of reduced interpretation capacity
        """
        return np.mean((iptw * (obs_y - pred_y))**2) / iptw.shape[0]

    @staticmethod
    def _check_distribution_measure_(distribution, measure):
        """Checks whether the distribution and measure specified are compatible"""
        if distribution is None:
            if measure is not None:
                raise ValueError("The distribution `None` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'normal':
            if measure not in ['sum', 'mean', 'var', 'mean_dist', 'var_dist']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'poisson':
            if measure not in ['sum', 'mean']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'multinomial':
            if measure not in ['sum']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'binomial':
            if measure not in ['mean']:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        elif distribution.lower() == 'threshold':
            # if re.search(r"^t[0-9]+$", measure) is None:
            if re.search(r"^t\d", measure) is None and re.search(r"^tp\d", measure) is None:
                raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
        else:
            raise ValueError("The distribution `"+str(distribution)+"` is not currently implemented")

    @staticmethod
    def _degree_restrictions_(degree_dist, bounds):
        """Bounds the degree by the specified levels"""
        restrict = np.where(degree_dist < bounds[0], 1, 0)
        restrict = np.where(degree_dist > bounds[1], 1, restrict)
        return restrict

    @staticmethod
    def _check_degree_restrictions_(bounds):
        """Checks degree restrictions"""
        if type(bounds) is not list and type(bounds) is not tuple:
            raise ValueError("`degree_restrict` should be a list/tuple of the upper and lower bounds")
        if len(bounds) != 2:
            raise ValueError("`degree_restrict` should only have two values")
        if bounds[0] > bounds[1]:
            raise ValueError("Degree restrictions must be specified in ascending order")
