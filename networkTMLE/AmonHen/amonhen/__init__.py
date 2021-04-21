"""AmonHen is a personal library containing targeted maximum likelihood estimators for my dissertation. AmonHen comes
from Lord of the Rings's Amon Hen, which translate to the 'hill of sight' or 'hill of the eye'. Essentially the TMLE
implementations in this library are a hill with which to expand our (in)sight into causal inference (particularly for
contexts of interference).
"""

from .version import __version__

from .tmle_network import NetworkTMLE
from .tmle_stochastic import StochasticTMLE
from .sr_estimators import NetworkGFormula, NetworkIPTW

from .utils import (network_to_df, fast_exp_map, tmle_unit_bounds, tmle_unit_unbound,
                    probability_to_odds, odds_to_probability, bounding,
                    outcome_learner_fitting, outcome_learner_predict, exposure_machine_learner,
                    targeting_step, distribution_shift)
