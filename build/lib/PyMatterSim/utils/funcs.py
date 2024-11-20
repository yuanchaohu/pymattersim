# coding = utf-8

"""mathmatical functions for feasible computation"""

import numpy as np
import numpy.typing as npt
from sympy.physics.wigner import wigner_3j

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name


def kronecker(i: int, j: int) -> int:
    """Kronecker function"""
    return int(i == j)


def nidealfac(ndim: int = 3) -> float:
    """
    Choose factor of Nideal in g(r) calculation

    Inputs:
        ndim (int): system dimensionality, default 3

    Return:
        (float): Nideal
    """
    if ndim == 3:
        return 4.0 / 3
    elif ndim == 2:
        return 1.0
    else:
        raise ValueError("Wrong input dimensionality")


def areafac(ndim: int = 3) -> float:
    """
    Choose factor of area in S2 calculation

    Inputs:
        ndim (int): system dimensionality, default 3

    Return:
        (float): Areafac
    """
    if ndim == 3:
        return 4.0
    elif ndim == 2:
        return 2.0
    else:
        raise ValueError("Wrong input dimensionality")


def alpha2factor(ndim: int = 3) -> float:
    """
    Choose factor in alpha2 calculation

    Inputs:
        ndim (int): system dimensionality, default 3

    Return:
        (float): alpha2factor
    """
    if ndim == 3:
        return 3.0 / 5.0
    elif ndim == 2:
        return 1.0 / 2.0
    else:
        raise ValueError("Wrong input dimensionality")


def moment_of_inertia(positions: npt.NDArray, m: int = 1, matrix: bool = False) -> npt.NDArray:
    """
    moment of inertia for a rigid body made of n points / particles

    Inputs:
        1. positions (npt.NDArray): positions of the point particles as [numofatoms, 3]
        2. m (int): assuming each point mass is 1.0/numofatoms
        3. matrix (bool): return the results as a matrix of [ixx iyy izz ixy ixz iyz]

    Return:
        moment of inertia (npt.NDArray)
    """

    Iij = np.zeros((3, 3))
    distance2 = np.square(positions).sum(axis=1)
    for i in range(3):
        for j in range(3):
            Iij[i, j] = m * (distance2 * kronecker(i, j) - positions[:, i] * positions[:, j]).sum()

    Iij /= positions.shape[0]
    if matrix:
        return Iij
    else:
        return np.array([Iij[0, 0], Iij[1, 1], Iij[2, 2], Iij[0, 1], Iij[0, 2], Iij[1, 2]])


def Wignerindex(l: int) -> npt.NDArray:
    """
    Define Wigner 3-j symbol

    Inputs:
        l (int): degree of harmonics

    Return:
        Wigner 3-j symbol (npt.NDArray)
    """
    selected = []
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            for m3 in range(-l, l + 1):
                if m1 + m2 + m3 == 0:
                    windex = wigner_3j(l, l, l, m1, m2, m3).evalf()
                    selected.append(np.array([m1, m2, m3, windex]))

    return np.ravel(np.array(selected)).reshape(-1, 4)


def grid_gaussian(distances: npt.NDArray, sigma: float = 1) -> npt.NDArray:
    """
    Calculate the gaussian distribution from the zero center,
    give the gaussian probability based on distance and sigma

    Inputs:
        1. distances (npt.NDArray): grid distances to the zero center
        2. sigma (float): standard deviation in the standard gaussian function

    Return:
        gaussian probability at various distances in numpy array
    """
    sigma2 = 2 * sigma**2
    return np.exp(-np.square(distances) / sigma2) / np.sqrt(sigma2 * np.pi)


def Legendre_polynomials(x, ndim):
    return (ndim * x**2 - 1) / 2
