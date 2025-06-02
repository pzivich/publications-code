#####################################################################################################################
# Data generating mechanisms for Case Studies
#####################################################################################################################

import numpy as np
import pandas as pd
from delicatessen.utilities import inverse_logit


def dgm_example1(n, truth=False):
    d = pd.DataFrame()
    d['A'] = np.random.binomial(n=1, p=0.5, size=n)
    d['U'] = np.random.normal(size=n)
    d['W'] = 2*d['A'] + 1*d['U'] + np.random.normal(-1, size=n)
    d['S'] = np.random.binomial(n=1, p=inverse_logit(2 - 1*d['W']), size=n)
    d['Y'] = np.random.binomial(n=1, p=inverse_logit(0.5 + 0.75*d['U'] - 1*d['A']), size=n)
    d['I'] = 1
    if not truth:
        d['Y'] = np.where(d['S'] == 1, d['Y'], -9999)
    return d


def dgm_example2(n, truth=False):
    d = pd.DataFrame()
    d['U1'] = np.random.binomial(n=1, p=0.5, size=n)
    d['U2'] = np.random.binomial(n=1, p=0.5, size=n)
    d['Z'] = np.random.binomial(n=1, p=0.5, size=n)
    # d['Z'] = np.random.normal(size=n)

    if truth:
        d['A'] = np.random.binomial(n=1, p=0.5, size=n)
    else:
        d['A'] = np.random.binomial(n=1, p=inverse_logit(-2.3 + np.log(2)*d['Z'] + np.log(4)*d['U1']),
                                    size=n)

    d['X'] = 4*d['U1'] - 4*d['U2'] + np.random.normal(size=n)
    d['S'] = np.random.binomial(n=1, p=inverse_logit(0. + .25*d['X']), size=n)
    d['Y'] = np.random.binomial(n=1, p=inverse_logit(-2.0 - 2*d['A'] + np.log(2)*d['Z']
                                                     + np.log(2)*d['A']*d['Z'] + np.log(4)*d['U2']),
                                size=n)
    d['I'] = 1
    if not truth:
        d['Y'] = np.where(d['S'] == 1, d['Y'], -9999)
    return d
