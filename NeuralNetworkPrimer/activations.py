import numpy as np


def identity(x):
    return x


def logistic(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (2 / (1 + np.exp(-2*x))) - 1


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, scale=0.01):
    return np.where(x < 0, scale*x, x)


def elu(x, scale=0.01):
    return np.where(x < 0, scale*(np.exp(x) - 1), x)


def softplus(x):
    return np.log(1 + np.exp(x))



