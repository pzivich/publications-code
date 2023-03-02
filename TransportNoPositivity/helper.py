#######################################################################################################################
# Transportability without positivity: a synthesis of statistical and simulation modeling
#       Helper functions for applied example and simulations
#
# Paul Zivich
#######################################################################################################################

import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit


def load_data(full=False):
    """Load and format the GetTested trial data from Wilson et al. (2017), then generate the clinic data.

    Parameters
    ----------
    full : bool
        Whether to return the full GetTested trial or the restricted GetTested trial as described in the paper.

    Returns
    -------
    pandas.DataFrame
    """
    # Setting seed for generation of clinic data
    np.random.seed(77777777)

    # Loading trial data
    t = pd.read_csv("wilson.csv")                                     # Reading in file that was pre-processed
    t = t[['group', 'gender', 'age', 'anytest']].dropna().copy()      # Restricting to a subset then drop missing
    t = t.loc[t['gender'] != 2].copy()                                # Dropping transgender (since not clarified)
    t['clinic'] = 0                                                   # Marking as trial data

    # Create clinic data
    p = pd.DataFrame()
    n = 1000
    x = trapezoid(mini=16, mode1=18, mode2=22, maxi=31, size=n)       # Simulating age from trapezoid
    p['age'] = np.floor(x)                                            # Rounding down to integers
    p['gender'] = np.random.binomial(n=1, p=0.6, size=n)              # Simulating gender
    p['clinic'] = 1                                                   # Marking as clinic data

    # Stack trial and clinic data
    d = pd.concat([t, p], ignore_index=True)
    d['age_sq'] = d['age']**2                       # Adding square term for age
    d['intercept'] = 1                              # Adding intercept

    # Logic for full or restricted trial data
    if full:
        return d
    else:
        return d.loc[((d['gender'] == 0) & (d['clinic'] == 0))
                     | d['clinic'] == 1].copy()


def trapezoid(mini, mode1, mode2, maxi, size=None):
    """Function to generate values from a trapezoid distribution.

    Parameters
    ----------
    mini : float, int
        Minimum value of trapezoid
    mode1 : float, int
        Lower value for uniform region
    mode2 : float, int
        Upper value for uniform region
    maxi : float, int
        Maximum value of trapezoid
    size : int, None
        Number of observations to generate

    Returns
    -------
    numpy.array
    """
    u = np.random.uniform(size=size)
    v = (u * (maxi + mode2 - mini - mode1) + (mini + mode1)) / 2
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        v = np.where(v < mode1,
                     mini + np.sqrt((mode1 - mini) * (2 * v - mini - mode1)),
                     v)
        v = np.where(v > mode2, maxi - np.sqrt(2 * (maxi - mode2) * (v - mode2)),
                     v)
    return v


def synthesis_g_computation(data, mc_iters, setup, n_cpus=1, mu=None, cov=None):
    """Function to run the statistical-simulation synthesis g-computation estimator in the simulations.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    mc_iters :
        Number of iterations for the Monte Carlo / semiparametric bootstrap procedure
    setup :
        What simulation model parameters to use (scenario).
    n_cpus :
        Number of CPUs available to use for the Monte Carlo procedure (helps to reduce up run-time of simulations).
    mu :
        Input point parameter for the simulation model (from the secret trial)
    cov :
        Input covariance matrix for parameters (from the secret trial)

    Returns
    -------
    list
    """
    # Subset the data by clinic / trial indicators
    d0 = data.loc[data['S'] == 0].copy()
    d1 = data.loc[data['S'] == 1].copy()

    def psi_outcome_model(theta):
        # Estimating equation for statistical model
        return ee_regression(theta=theta,
                             X=d0[['intercept', 'A', 'V']],
                             y=d0['Y'],
                             model='logistic')

    # Estimating the statistical model parameters
    estr = MEstimator(psi_outcome_model, init=[-5, 1.5, 0])
    estr.estimate(solver='lm')

    # Drawing parameters from a MVN for the statistical model (done the number of Monte Carlo iterations)
    stat_params = np.random.multivariate_normal(estr.theta,
                                                cov=estr.variance,
                                                size=mc_iters)

    # Setup for simulation model parameters
    if setup == 1:
        # Strict null
        mech_params = [[0, 0], ] * mc_iters
    elif setup == 2:
        # Uncertain null
        mech_params = np.asarray([trapezoid(-2, -1, 1, 2, size=mc_iters),
                                  trapezoid(-2, -1, 1, 2, size=mc_iters)]).T
    elif setup in [3, 4]:
        # Accurate & Inaccurate
        mech_params = np.asarray([np.random.normal(mu[0], scale=np.sqrt(cov[0, 0]), size=mc_iters),
                                  np.random.normal(mu[1], scale=np.sqrt(cov[1, 1]), size=mc_iters)]).T
    elif setup == 5:
        # Accurate with covariance
        mech_params = np.random.multivariate_normal(mu, cov=cov, size=mc_iters)
    else:
        # Error if invalid setup option is given
        raise ValueError("Invalid setup option")

    # Packaging data and parameters to give to each process in Pool
    params = [[d1.sample(n=d1.shape[0], replace=True),
               stat_params[j],
               mech_params[j],
               ] for j in range(mc_iters)]

    # Running on multi-CPU with Pool to reduce run-times
    with Pool(processes=n_cpus) as pool:
        estimates = list(pool.map(gcomp_iteration,     # Call outside function (defined below)
                                  params))             # ... and providing prepackaged input

    # Returning point and 95% CI estimates
    return np.median(estimates), np.percentile(estimates, q=[2.5, 97.5])


def synthesis_ipw(data, mc_iters, setup, n_cpus=1, mu=None, cov=None):
    """Function to run the statistical-simulation synthesis IPW estimator in the simulations.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    mc_iters :
        Number of iterations for the Monte Carlo / semiparametric bootstrap procedure
    setup :
        What simulation model parameters to use (scenario).
    n_cpus :
        Number of CPUs available to use for the Monte Carlo procedure (helps to reduce up run-time of simulations).
    mu :
        Input point parameter for the simulation model (from the secret trial)
    cov :
        Input covariance matrix for parameters (from the secret trial)

    Returns
    -------
    list
    """
    # Subset the data
    d1 = data.loc[data['S'] == 1].copy()
    s = np.asarray(data['S'])               # convert to array for estimating equations
    a = np.asarray(data['A'])               # convert to array for estimating equations

    def psi_msm_model(theta):
        # Estimating equations for the statistical model (i.e., marginal structural model)
        alpha, beta, gamma = theta[0], theta[1:4], theta[4:]

        # Estimating the nuisance action model (simple mean among s=0)
        ee_act = np.nan_to_num((1 - s) * (a - alpha), copy=True, nan=0.)
        pi_a = (a == 1) * alpha + (a == 0) * (1 - alpha) + s
        iptw = 1 / pi_a

        # Estimating the nuisance sampling model (using both data sources)
        ee_trp = ee_regression(theta=beta,
                               X=data[['intercept', 'V', 'V_i25']],
                               y=data['S'],
                               model='logistic')
        pi_s = inverse_logit(np.dot(np.asarray(data[['intercept', 'V', 'V_i25']]), beta))
        odds = pi_s / (1 - pi_s)
        iosw = s * 1 + (1 - s) * odds

        # Estimating the marginal structural model among s=0 using the weights
        ee_msm = ee_regression(theta=gamma,
                               X=data[['intercept', 'A']],
                               y=data['Y'],
                               weights=iptw * iosw,
                               model='logistic') * (1 - s)
        ee_msm = np.nan_to_num(ee_msm, copy=True, nan=0.)

        return np.vstack([ee_act, ee_trp, ee_msm])

    # Estimating the statistical model parameters
    estr = MEstimator(psi_msm_model, init=[0.5, -1.2, -0.1, 0.0, -1.3, 1.5])
    estr.estimate(solver='lm', maxiter=2000)

    # Extracting relevant statistical model parameters (only the MSM parameters)
    stat_means = estr.theta[4:]
    stat_covar = estr.variance[4:, 4:]

    # Drawing parameters from a MVN for the statistical model (done the number of Monte Carlo iterations)
    stat_params = np.random.multivariate_normal(stat_means,
                                                cov=stat_covar,
                                                size=mc_iters)

    # Setup for simulation model parameters
    if setup == 1:
        # Strict null
        mech_params = [[0, 0], ] * mc_iters
    elif setup == 2:
        # Uncertain null
        mech_params = np.asarray([trapezoid(-2, -1, 1, 2, size=mc_iters),
                                  trapezoid(-2, -1, 1, 2, size=mc_iters)]).T
    elif setup in [3, 4]:
        # Accurate & Inaccurate
        mech_params = np.asarray([np.random.normal(mu[0], scale=np.sqrt(cov[0, 0]), size=mc_iters),
                                  np.random.normal(mu[1], scale=np.sqrt(cov[1, 1]), size=mc_iters)]).T
    elif setup == 5:
        # Accurate with covariance
        mech_params = np.random.multivariate_normal(mu, cov=cov, size=mc_iters)
    else:
        # Error if invalid setup given
        raise ValueError("Invalid setup option")

    # Packaging data to give to each process in Pool
    params = [[d1.sample(n=d1.shape[0], replace=True),
               stat_params[j],
               mech_params[j],
               ] for j in range(mc_iters)]

    # Running on multi-CPU with Pool to reduce run-times
    with Pool(processes=n_cpus) as pool:
        estimates = list(pool.map(ipw_iteration,      # Call outside function (defined below)
                                  params))            # ... and provide packed input

    # Returning point and 95% CI estimates
    return np.median(estimates), np.percentile(estimates, q=[2.5, 97.5])


def gcomp_iteration(params):
    """Generates the point estimate for the average causal effect for a given set of parameters and resampled clinic
    data

    Parameters
    ----------
    params : list, ndarray
        Packaged inputs as provided in synthesis_gcomputation

    Returns
    -------
    float
    """
    # Unpacking provided inputs
    clinic_data, stat_param, mech_param = params

    # Generate predicted probabilities
    dca = clinic_data.copy()                       # Copy the clinic data input
    dca['A'] = 1                                   # Set A=1
    dca['AW'] = dca['A'] * dca['W']                # Interaction term
    X1 = np.asarray(dca[['intercept', 'A', 'V']])  # Covariates for statistical model
    W1 = np.asarray(dca[['W', 'AW']])              # Covariates for simulation model
    # Generating predicted potential outcomes for a=1
    ya1 = inverse_logit(np.dot(X1, stat_param) + np.dot(W1, mech_param))

    dca['A'] = 0                                   # Set A=0
    dca['AW'] = dca['A'] * dca['W']                # Interaction term update
    X0 = np.asarray(dca[['intercept', 'A', 'V']])  # Covariates for statistical model
    W0 = np.asarray(dca[['W', 'AW']])              # Covariates for simulation model
    # Generating predicted potential outcomes for a=0
    ya0 = inverse_logit(np.dot(X0, stat_param) + np.dot(W0, mech_param))

    # Return the point estimate for the ACE
    return np.mean(ya1) - np.mean(ya0)


def ipw_iteration(params):
    clinic_data, stat_param, mech_param = params

    # Generate predicted probabilities
    dca = clinic_data.copy()                   # Copy the clinic data input
    dca['A'] = 1                               # Set A=1
    dca['AW'] = dca['A'] * dca['W']            # Interaction term
    X1 = np.asarray(dca[['intercept', 'A']])   # Covariates for statistical model
    W1 = np.asarray(dca[['W', 'AW']])          # Covariates for simulation model
    # Generating predicted potential outcomes for a=1
    ya1 = inverse_logit(np.dot(X1, stat_param) + np.dot(W1, mech_param))

    dca['A'] = 0                               # Set A=0
    dca['AW'] = dca['A'] * dca['W']            # Interaction term update
    X0 = np.asarray(dca[['intercept', 'A']])   # Covariates for statistical model
    W0 = np.asarray(dca[['W', 'AW']])          # Covariates for simulation model
    # Generating predicted potential outcomes for a=0
    ya0 = inverse_logit(np.dot(X0, stat_param) + np.dot(W0, mech_param))

    # Return the point estimate for the ACE
    return np.mean(ya1) - np.mean(ya0)
