import numpy as np
import pandas as pd


def weierstrass(x, a, b, n_approx, cos=True):
    """Approximation of the Weierstrass function.

    Parameters
    ----------
    x :
        Input continuous variables
    a :
        First hyperparameter for the Weierstrass function
    b :
        Second hyperparameter for the Weierstrass function
    n_approx :
        Finite number for the summation (hence why this function is an approximation)
    cos :
        Whether to use the cosine or sine variations.

    Returns
    -------
    array
    """
    y = np.zeros(x.shape[0])
    if cos:
        x = 2 * x
        for i in range(n_approx):
            y += (a**i) * np.cos(b**i * np.pi * x)
        return y
    else:
        for i in range(1, n_approx+1):
            y += np.sin(np.pi * i**a * x) / (np.pi * i**a)
        return (1.2*y) + 0.20


def data_generation(n, rng, truth=False):
    """Generate n observations given a NumPy RNG object.

    Parameters
    ----------
    n :
        Number of observations
    rng :
        NumPy random number generator object
    truth :
        Whether to compute the ACE using the potential outcomes (True) or return the generated data (False)
    """
    d = pd.DataFrame()
    d['W'] = rng.uniform(0, 1, size=n)

    pr_a = 0.95 - weierstrass(np.asarray(d['W']) - 0.1, a=3., b=1., n_approx=100, cos=False)
    d['A'] = rng.binomial(n=1, p=pr_a, size=n)
    y1 = weierstrass(np.asarray(d['W']), a=0.5, b=5, n_approx=100, cos=True)*5
    y0 = weierstrass(np.asarray(d['W'])*0.5, a=0.5, b=5, n_approx=100, cos=True)*5
    if truth:
        return np.mean(y1 - y0)

    d['Y'] = np.where(d['A'] == 1, y1, y0) + rng.normal(size=n)
    return d
