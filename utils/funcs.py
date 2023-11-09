# coding = utf-8

"""mathmatical functions for feasible computation"""

import numpy as np
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name

def kronecker(i: int, j: int) -> int:
    """Kronecker function"""
    return int(i==j)

def nidealfac(ndim: int=3) -> float:
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

def moment_of_inertia(
    positions: np.ndarray,
    m: int=1,
    matrix: bool=False
) -> np.ndarray:
    """
    moment of inertia for a rigid body made of n points / particles

    Inputs:
        1. positions (np.ndarray): positions of the point particles as [numofatoms, 3]
        2. m (int): assuming each point mass is 1.0/numofatoms
        3. matrix (bool): return the results as a matrix of [ixx iyy izz ixy ixz iyz]
    
    Return:
        moment of inertia (np.ndarray)
    """

    Iij = np.zeros((3, 3))
    distance2 = np.square(positions).sum(axis=1)
    for i in range(3):
        for j in range(3):
            Iij[i, j] = m*(distance2*kronecker(i, j)-positions[:, i]*positions[:, j]).sum()
    
    Iij /= positions.shape[0]
    if matrix:
        return Iij
    else:
        return np.array([Iij[0,0], Iij[1,1], Iij[2,2], Iij[0,1], Iij[0,2], Iij[1,2]])
