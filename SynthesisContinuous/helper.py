#######################################################################################################################
# Synthesis estimators for positivity violations with a continuous covariate
#   Helper functions used in the simulations
#
# Paul Zivich
#######################################################################################################################

import numpy as np

from dgm import generate_target_trial
from statistical import StatMSM, StatCACE


def trapezoid(mini, mode1, mode2, maxi, size=None):
    """Generate random draws from a trapezoidal distribution.

    Parameters
    ----------
    mini :
        Minimum value
    mode1 :
        Lower mode
    mode2 :
        Upper mode
    maxi :
        Maximum value
    size :
        Number of draws
    """
    if size is None:
        p = np.random.uniform()
        v = (p * (maxi + mode2 - mini - mode1) + (mini + mode1)) / 2
        if v < mode1:
            v = mini + np.sqrt((mode1 - mini) * (2 * v - mini - mode1))
        elif v > mode2:
            v = maxi - np.sqrt(2 * (maxi - mode2) * (v - mode2))
        else:
            pass
        return v
    elif type(size) is int:
        va = []
        for i in range(size):
            va.append(trapezoid(mini=mini, mode1=mode1, mode2=mode2, maxi=maxi, size=None))
        return np.array(va)
    else:
        raise ValueError('"size" must be an integer')


def math_parameters_msm(n, scenario):
    """Compute the mathematical model parameters for the marginal structural model.

    Parameters
    ----------
    n : int
        Number of observations to generate
    scenario : int
        Scenario to generate observations for
    """
    # Generating a target population trial
    d = generate_target_trial(n=n, scenario=scenario)

    # Estimating the full marginal structural model
    msm = StatMSM(data=d, outcome='Y')
    msm.marginal_structural_model("A + V + A:V + A:V_s300 + A:V_s800")
    msm.estimate()
    params = msm.params
    std_dev = np.sqrt(msm.params_var)

    # Packaging up parameters for outputting
    return (params[4], std_dev[4]), (params[5], std_dev[5])


def math_parameters_cace(n, scenario):
    """Compute the mathematical model parameters for the conditional average causal effect model.

    Parameters
    ----------
    n : int
        Number of observations to generate
    scenario : int
        Scenario to generate observations for
    """
    # Generating a target population trial
    d = generate_target_trial(n=n, scenario=scenario)

    # Estimating the full marginal structural model
    cace = StatCACE(data=d, outcome='Y', action='A')
    cace.outcome_model("A + V + A:V + A:V_s300 + A:V_s800 + W + A:W")
    cace.cace_model("V + V_s300 + V_s800")
    cace.estimate()
    params = cace.params
    std_dev = np.sqrt(cace.params_var)

    # Packaging up parameters for outputting
    return (params[2], std_dev[2]), (params[3], std_dev[3])
