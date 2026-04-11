"""
Membership Functions for Fuzzy Logic
Reference: https://github.com/twmeggs/anfis (uses scikit-fuzzy style naming)

These functions map crisp input values to fuzzy membership degrees [0, 1].
"""
import numpy as np


def gaussian_mf(x, center, sigma):
    """
    Gaussian membership function.
    Parameters:
        x      : input value (scalar or numpy array)
        center : center of the Gaussian curve
        sigma  : width (standard deviation) of the curve
    Returns:
        Membership degree in range [0, 1]
    """
    return np.exp(-0.5 * ((x - center) / (sigma + 1e-8)) ** 2)


def bell_mf(x, a, b, c):
    """
    Generalized bell-shaped membership function.
    Parameters:
        x : input value
        a : width of the curve
        b : slope of the curve
        c : center of the curve
    Returns:
        Membership degree in range [0, 1]
    """
    return 1.0 / (1.0 + np.abs((x - c) / (a + 1e-8)) ** (2 * b))


def sigmoid_mf(x, a, c):
    """
    Sigmoid membership function.
    Parameters:
        x : input value
        a : slope (controls steepness)
        c : crossover point (where output = 0.5)
    Returns:
        Membership degree in range [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-a * (x - c)))
