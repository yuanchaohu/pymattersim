# coding = utf-8

"""see documentation @ ../docs/utils.md"""

import numpy as np


def remove_pbc(RIJ: np.array, hmatrix: np.array, ppp: np.ndarray=np.array([1,1,1])) -> np.ndarray:
    """
    remove periodic boundary conditions.
    This module is usually embedded in other analysis modules.

    Inputs:
        1. RIJ (np.array):position difference between particle i (center) and j (neighbors) with PBC
        2. hmatrix (np.array): h-matrix of the box
        3. ppp (np.ndarray): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default np.array([1, 1, 1]), that is, PBC is applied in all three dimensions for 3D box
 
    Return:
        (np.array): position difference between particle i (center) and j (neighbors)
                    after removing PBC
    """

    # pylint: disable=invalid-name
    ppp = np.array(ppp)[np.newaxis, :]
    hmatrixinv = np.linalg.inv(hmatrix)
    matrixij = np.dot(RIJ, hmatrixinv)
    return np.dot(matrixij - np.rint(matrixij)*ppp, hmatrix)
    