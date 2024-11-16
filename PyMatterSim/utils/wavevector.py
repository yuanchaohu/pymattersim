# coding = utf-8
"""
This module is used to generate wave-vector for calculations
like static/dynamic structure factor
"""

from math import modf, sqrt

import numpy as np
import numpy.typing as npt

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=too-many-branches


def wavevector3d(numofq: int = 500) -> npt.NDArray:
    """
    Define wave vector for three dimensional systems

    Inputs:
        numofq (int): number of q

    Return:
        wavevector (npt.NDArray)
    """

    wavenumber = np.square(np.arange(numofq))
    wavevector = []
    for a in range(numofq):
        for b in range(numofq):
            for c in range(numofq):
                d = a**2 + b**2 + c**2
                if d in wavenumber:
                    wavevector.append(np.array([d, a, b, c]))
    wavevector = np.ravel(np.array(wavevector))[4:].reshape((-1, 4))
    wavevector = wavevector[wavevector[:, 0].argsort()]
    return np.array(wavevector)


def wavevector2d(numofq: int = 500) -> npt.NDArray:
    """
    Define Wave Vector for two dimensional system

    Inputs:
        numofq (int): number of q

    Return:
        wavevector (npt.NDArray)
    """

    wavenumber = np.square(np.arange(numofq))
    wavevector = []
    for a in range(numofq):
        for b in range(numofq):
            d = a**2 + b**2
            if d in wavenumber:
                wavevector.append(np.array([d, a, b]))
    wavevector = np.ravel(np.array(wavevector))[3:].reshape((-1, 3))
    wavevector = wavevector[wavevector[:, 0].argsort()]
    return np.array(wavevector)


def choosewavevector(ndim: int, numofq: int, onlypositive: bool = False) -> npt.NDArray:
    """
    define wave vector for [nx, ny, nz] as long as they are integers
    considering qvector values from [-N/2, N/2] or from [0, N/2] (onlypositive=True)
    only get the sqrt-able wave vector

    Inputs:
        1. ndim (int): dimensionality
        2. numofq (int): number of q
        3. onlypositive (bool): whether only consider positive wave vectors
                                default False

    Return:
        qvectors (npt.NDArray)
    """

    qvectors = np.zeros((numofq**ndim, ndim), dtype=np.int32)
    nhalf = int(numofq / 2)

    if ndim == 2:
        index = 0
        for i in range(-nhalf, nhalf):
            for j in range(-nhalf, nhalf):
                if modf(sqrt(i**2 + j**2))[0] == 0:
                    qvectors[index] = [i, j]
                    index += 1
        # choose wavevector along a specific dimension 'x', 'y', or 'z'
        if onlypositive == "x":
            # [x, 0]
            condition = (qvectors[:, 0] > 0) * (qvectors[:, 1] == 0)
            qvectors = qvectors[condition]
        elif onlypositive == "y":
            # [0, y]
            condition = (qvectors[:, 0] == 0) * (qvectors[:, 1] > 0)
            qvectors = qvectors[condition]

    if ndim == 3:
        index = 0
        for i in range(-nhalf, nhalf):
            for j in range(-nhalf, nhalf):
                for k in range(-nhalf, nhalf):
                    if modf(sqrt(i**2 + j**2 + k**2))[0] == 0:
                        qvectors[index] = [i, j, k]
                        index += 1
        # choose wavevector along a specific dimension 'x', 'y', or 'z'
        if onlypositive == "x":
            # [x, 0, 0]
            condition = (qvectors[:, 0] > 0) * (qvectors[:, 1] == 0) * (qvectors[:, 2] == 0)
            qvectors = qvectors[condition]
        elif onlypositive == "y":
            # [0, y, 0]
            condition = (qvectors[:, 0] == 0) * (qvectors[:, 1] > 0) * (qvectors[:, 2] == 0)
            qvectors = qvectors[condition]
        elif onlypositive == "z":
            # [0, 0, z]
            condition = (qvectors[:, 0] == 0) * (qvectors[:, 1] == 0) * (qvectors[:, 2] > 0)
            qvectors = qvectors[condition]

    condition = (qvectors == 0).all(axis=1)
    qvectors = qvectors[~condition]

    # choose only postive integers as the wavevector
    if isinstance(onlypositive, bool) and onlypositive:
        condition = (qvectors >= 0).all(axis=1)
        qvectors = qvectors[condition]

    return qvectors


def continuousvector(ndim: int, numofq: int = 100, onlypositive: bool = False) -> npt.NDArray:
    """
    define wave vector for [nx, ny, nz] as long as they are integers
    considering qvector values from [-N/2, N/2]

    Inputs:
        1. ndim (int): dimensionality
        2. numofq (int): number of q, default 100
        3. onlypositive (bool): whether only consider positive wave vectors,
                                default False

    Return:
        qvectors (npt.NDArray)
    """

    qvectors = np.zeros((numofq**ndim, ndim), dtype=np.int32)
    nhalf = int(numofq / 2)

    if ndim == 2:
        index = 0
        for i in range(-nhalf, nhalf):
            for j in range(-nhalf, nhalf):
                qvectors[index] = [i, j]
                index += 1

    if ndim == 3:
        index = 0
        for i in range(-nhalf, nhalf):
            for j in range(-nhalf, nhalf):
                for k in range(-nhalf, nhalf):
                    qvectors[index] = [i, j, k]
                    index += 1

    condition = (qvectors == 0).all(axis=1)
    qvectors = qvectors[~condition]
    if onlypositive:
        condition = (qvectors >= 0).all(axis=1)
        qvectors = qvectors[condition]

    return qvectors
