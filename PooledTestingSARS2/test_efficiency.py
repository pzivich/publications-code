import warnings
import numpy as np


def calculate_dilution(pool_size):
    """Calculates the dilution. Broken out as its own function so its easy to find (in case it needs to be changed).

    Parameters
    ----------
    pool_size : number of specimens contained within a pool

    Returns
    -------
    float
    """
    return np.log10(pool_size) / 7


def calculate_efficiency(pool_size, sensitivity, specificity, q):
    """Efficiency calculator for two-level pooling strategies

    Parameters
    ----------
    pool_size : int
        Number of specimens contained within a pool.
    sensitivity : float
        Input sensitivity of test.
    specificity : float
        Input specificity of test.
    q : float
        1 minus prevalence of the disease.

    Returns
    -------
    float
    """
    # summand is not used here, but becomes necessary for D3-type pooling setups
    summand = 0
    ess = ((np.power(q, pool_size)*np.power((1-specificity), 1)) +
           summand +
           ((1-np.power(q, pool_size)) * sensitivity))
    expect = 1 + ess*pool_size
    return expect / pool_size


def calculate_poolsize_efficiency(pool_size, sensitivity, specificity, prevalence, mad_value):
    """Calculate the efficiency for a given pool size of specimens

    Parameters
    ----------
    pool_size : int
        Number of specimens contained within a pool.
    sensitivity : float
        Input sensitivity for a single test.
    specificity : float
        Input specificity for a single test.
    prevalence : float
        Prevalence of the disease.
    mad_value :
        Maximum allowable dilution for pooling of samples.

    Returns
    -------
    float
    """
    dilute = calculate_dilution(pool_size=pool_size)
    q = 1 - prevalence
    if dilute <= mad_value:
        sensitivity_reduced = sensitivity*(1-dilute)
        return calculate_efficiency(pool_size=pool_size, sensitivity=sensitivity_reduced,
                                    specificity=specificity, q=q)
    else:
        return 999


def optimal_poolsize(max_pool_size, sensitivity, specificity, prevalence, mad_value, ignore_warning=False):
    """Calculate the optimal pool size given a maximum pool size, sensitivity of a test, specificity of a test,
    prevalence of the outcome, and a maximum allowable dilution for the test.

    Parameters
    ----------
    max_pool_size :
        Maximum number of specimens contained within a pool to assess.
    sensitivity :
        Input sensitivity for a single test.
    specificity :
        Input specificity for a single test.
    prevalence :
        Prevalence of the disease.
    mad_value :
        Maximum allowable dilution for pooling of samples.

    Returns
    -------
    ndarray
        Set of optimal pool size, and efficiency for the optimal pool size.

    References
    ----------
    Pilcher CD, Westreich D, & Hudgens MG. "Group testing for severe acute respiratory syndrome-coronavirus 2 to enable
    rapid scale-up of testing and real-time surveillance of incidence". J Infect Dis 2020;222(6):903–909.
    """
    if sensitivity > 1 or sensitivity < 0 or specificity > 1 or specificity < 0:
        raise ValueError("Invalid sensitivity or specificity")
    if prevalence > 1 or prevalence < 0:
        raise ValueError("Invalid prevalence")
    if max_pool_size < 2:
        raise ValueError("Max pool size to explore must be at least 2")
    if mad_value < 0 or mad_value > 1:
        raise ValueError("Invalid MAD value")

    # Efficiency for no pooling (pool_size = 1)
    opt_poolsize = 1              # set original optimal size to 1
    opt_efficiency = 1            # set original efficiency (the one to beat) as 1

    # Check all valid combinations
    for i in range(2, max_pool_size+1):                                      # Consider all pool sizes up to the max
        efficiency = calculate_poolsize_efficiency(pool_size=i,              # Evaluate the pool size at
                                                   sensitivity=sensitivity,  # ... input sensitivity
                                                   specificity=specificity,  # ... input specificity
                                                   prevalence=prevalence,    # ... input prevalence
                                                   mad_value=mad_value)      # ... and input MAD value
        if efficiency < opt_efficiency:                                      # Check new efficiency versus old
            opt_poolsize = i                                                 # ... when better update optim size
            opt_efficiency = efficiency                                      # ... and corresponding efficiency

    # Rough check for prevalence threshold
    #   This rough check is based on a set of experiments I ran. It is an informal check that is probably a bit
    #   conservative. The numbers are based on a simulation I ran that checked against a variety of sensitivity,
    #   specificity, and prevalence values. These were regressed together to get the coefficients in a saturated model
    #   and modified them to be slightly conservative.
    if prevalence > (-0.3550 + 0.3200*sensitivity + 0.4210*specificity - 0.1575*sensitivity*specificity):
        warnings.warn("It looks like the prevalence is higher that a pre-defined threshold. The calculator may "
                      "not function as expected. Therefore, the optimal pool-size and efficiency could be "
                      "incorrect. By default, the optimal pool-size and efficiency are set as 1 when this "
                      "occurs. If you still would like the optimal pool size calculation, set `ignore_warning=True`",
                      UserWarning)
        if not ignore_warning:                                               # When warning is NOT ignored,
            opt_poolsize = 1                                                 # ... set optimal to 1 due to issue
            opt_efficiency = 1                                               # ... and set efficiency to 1

    # Providing optimal pool size, and corresponding efficiency at optimal pool size
    return opt_poolsize, opt_efficiency


def overall_diagnostic_calculator(sensitivity, specificity, pr_s, pr_d_s, pr_d_ns):
    """Calculate the overall (weighted average) of sensitivity and specificity of a test given strata specific values.

    Parameters
    ----------
    sensitivity : ndarray, list
        Set of sensitivity values
    specificity : ndarray, list
        Set of specificity values
    pr_s : float
        Overall probability of symptoms (grouping factor)
    pr_d_s : float
        Conditional probability of disease given symptoms (grouping factor)
    pr_d_ns : float
        Conditional probability of disease given no symptoms (grouping factor)

    Returns
    -------
    ndarray
        Sensitivity and specificity values
    """
    # Setup calculations
    pr_ns = 1 - pr_s                         # Probability of no symptoms
    pr_d = pr_d_s * pr_s + pr_d_ns * pr_ns   # Probability of infection

    # Calculating sensitivity via the following formula
    #   \sum_s P(T|D,s) * P(D|s) * P(s) / P(D)
    sens = sensitivity[0] * pr_d_s * pr_s / pr_d + sensitivity[1] * pr_d_ns * pr_ns / pr_d

    # Calculating sensitivity via the following formula
    #   \sum_s P(not-T|not-D,s) * P(not-D|s) * P(s) / P(not-D)
    spec = specificity[0] * (1 - pr_d_s) * pr_s / (1 - pr_d) + specificity[1] * (1 - pr_d_ns) * pr_ns / (1 - pr_d)
    return sens, spec
