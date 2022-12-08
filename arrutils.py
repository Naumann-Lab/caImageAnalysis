import numpy as np


def pretty(x, n=3):
    return np.convolve(x, np.ones(n) / n, mode="same")
