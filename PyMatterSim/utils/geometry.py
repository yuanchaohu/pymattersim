# coding = utf-8

"""
math geometrical functions to assist other analysis
see documentation @ ../../docs/utils.md
"""

import numpy as np
import numpy.typing as npt

from ..utils.logging import get_logger_handle
from ..utils.pbc import remove_pbc

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value


def triangle_area(positions: npt.NDArray, hmatrix: npt.NDArray, ppp: npt.NDArray = np.array([1, 1])) -> float:
    """
    claculate the area of a triangle using Heron's equation

    Inputs:
        1. positions (npt.NDArray): numpy array of particle positions,
                                 shape=(3, 2) for 2D, shape=(3, 3) for 3D
        2. hmatrix (npt.NDArray): h-matrix of the box
        3. ppp (npt.NDArray): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default np.array([1,1]), that is, PBC is applied in all two dimensions

    Return:
        area of triangle (float)
    """
    logger.info(f"Calculate triangle area in a {len(ppp)}-dimensional system")

    point1 = positions[0]
    point2 = positions[1]
    point3 = positions[2]

    R12 = remove_pbc(point1 - point2, hmatrix, ppp)
    R12 = np.linalg.norm(R12)
    R13 = remove_pbc(point1 - point3, hmatrix, ppp)
    R13 = np.linalg.norm(R13)
    R23 = remove_pbc(point2 - point3, hmatrix, ppp)
    R23 = np.linalg.norm(R23)

    p = (R12 + R13 + R23) / 2
    S = np.sqrt(p * (p - R12) * (p - R13) * (p - R23))
    return S


def triangle_angle(a: float, b: float, c: float) -> float:
    """
    calculate the angle of a triangle based on side lengths

    Inputs:
        1. a, b, c (float): side length,

    Return:
        corresponding angles: C (float)
    """
    logger.info("Calculate triangle angle")

    cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)
    return np.arccos(cos_theta)


def lines_intersection(P1: npt.NDArray, P2: npt.NDArray, P3: npt.NDArray, P4: npt.NDArray) -> npt.NDArray:
    """
    extract the line-line intersection for two lines [P1, P2] and [P3, P4]
    in two dimensions

    Inputs:
        1. P1 (npt.NDArray): one point on line 1
        2. P2 (npt.NDArray): another point on line 1
        3. P3 (npt.NDArray): one point on line 2
        4. P4 (npt.NDArray): another point on line 2

    Return:
        intersection point (npt.NDArray)
    """
    logger.info("Extract the line-line intersection for two lines")

    x1, y1 = P1  # one point on line 1
    x2, y2 = P2  # another point on line 1
    x3, y3 = P3  # one point on line 2
    x4, y4 = P4  # another point on line 2

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    Px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    Px /= D
    Py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    Py /= D
    return np.array([Px, Py])


def LineWithinSquare(
    P1: npt.NDArray,
    P2: npt.NDArray,
    P3: npt.NDArray,
    P4: npt.NDArray,
    R0: npt.NDArray,
    vector: npt.NDArray,
) -> npt.NDArray:
    """
    calculate the line segment within a square defined by [P1, P2, P3, P4]
    in anti-clockwise manner

    Inputs:
        1. P1 (npt.NDArray): first point within a square
        2. P2 (npt.NDArray): second point within a square
        3. P3 (npt.NDArray): third point within a square
        4. P4 (npt.NDArray): fourth point within a square
        5. R0 (npt.NDArray): point within the sqaure
        6. vector (npt.NDArray): pointing to R0 from R1 outside the square

    Return:
        line segment (npt.NDArray)
    """
    logger.info("Calculate the line segment within a square")

    R1 = R0 - vector
    theta = np.arctan2(-vector[1], -vector[0])

    # threshold angles from R0 to P's
    points = np.zeros((4, 2))
    points[0, :] = P1
    points[1, :] = P2
    points[2, :] = P3
    points[3, :] = P4
    RtoP = points - R0
    angles = np.arctan2(RtoP[:, 1], RtoP[:, 0])

    if (theta > angles[0]) and (theta <= angles[1]):
        return lines_intersection(P1, P2, R0, R1)
    elif (theta > angles[1]) and (theta <= angles[2]):
        return lines_intersection(P2, P3, R0, R1)
    elif (theta > angles[2]) and (theta <= angles[3]):
        return lines_intersection(P3, P4, R0, R1)
    else:
        return lines_intersection(P4, P1, R0, R1)
