 #coding = utf-8

import numpy as np 

def remove_pbc(RJI: array, hmatrix: array, ppp: list) -> array:
    """
    remove periodic boundary conditions.
    This module is usually embedded in other analysis modules.

    Inputs:
        1. RJI (array): position difference between particle i (center) and j (neighbors) with PBC
        2. hmatrix (array): h-matrix of the box
        3. ppp (list): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box
 
    Return:
        RJI (array): position difference between particle i (center) and j (neighbors) without PBC
    """

    ppp = np.array(ppp)[np.newaxis, :]
    hmatrixinv = np.linalg.inv(hmatrix)
    matrixij = np.dot(RJI, hmatrixinv)
    RJI = np.dot(matrixij - np.rint(matrixij)*ppp, hmatrix)

    return RJI
    