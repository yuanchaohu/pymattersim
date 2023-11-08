#coding = utf-8

"""
develop tiny (math) functions to assist other analysis
see documentation @ ../docs/utils.md
"""

import numpy as np
from utils.pbc import remove_pbc

# pylint: disable=invalid-name

def triangle_area(positions: np.array, hmatrix: np.array, ppp: list=[1,1]) -> np.array:
    """
    claculate the area of a triangle by using Heron's equation

    Inputs:
        1. positions (np.array): numpy array of particle positions,
                                 shape=(3, 2) for 2D, shape=(3, 3) for 3D
        2. hmatrix (np.array): h-matrix of the box
        3. ppp (list): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default [1, 1], that is, PBC is applied in all three dimensions for 3D box

    Return:
        area of triangle (np.array)             
    """
    point1 = positions[0]
    point2 = positions[1]
    point3 = positions[2]

    R12 = remove_pbc(point1-point2, hmatrix, ppp)
    R12 = np.linalg.norm(R12)
    R13 = remove_pbc(point1-point3, hmatrix, ppp)
    R13 = np.linalg.norm(R13)
    R23 = remove_pbc(point2-point3, hmatrix, ppp)
    R23 = np.linalg.norm(R23)

    p = (R12+R13+R23)/2
    S = np.sqrt(p*(p-R12)*(p-R13)*(p-R23))
    return S

def triangle_angle(a: float, b: float, c: float) -> np.array:
    """
    calculate the angle of a triangle based on side lengths

    Inputs:
        1. a, b, c (float): side length, 
    
    Return:
        corresponding angles: A, B, C (np.array)
    """

    cos_theta = (a**2 + b**2 - c**2) / (2*a*b)
    return np.arccos(cos_theta)

def lines_intersection(P1: np.array, P2: np.array, P3: np.array, P4: np.array) -> np.array:
    """
    extract the line-line intersection for two lines [P1, P2] and [P3, P4]

    Inputs:
        1. P1 (np.array): one point on line 1
        2. P2 (np.array): another point on line 1
        3. P3 (np.array): one point on line 2
        4. P4 (np.array): another point on line 2

    Return:
        intersection point (np.array)
    """
    x1, y1 = P1  #
    x2, y2 = P2  #another point on line 1
    x3, y3 = P3  #one point on line 2
    x4, y4 = P4  #another point on line 2

    D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    Px = (x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)
    Px /= D 
    Py = (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)
    Py /= D 
    return np.array([Px, Py])

def LineWithinSquare(P1: np.array, P2: np.array, P3: np.array, P4: np.array,
                     R0: np.array, vector: np.array) -> np.array:
    """
    calculate the line segment within a square defined by [P1, P2, P3, P4] in anti-clockwise manner

    Inputs:
        1. P1 (np.array): first point within a square
        2. P2 (np.array): second point within a square
        3. P3 (np.array): third point within a square
        4. P4 (np.array): fourth point within a square
        5. R0 (np.array): point within the sqaure
        6. vector (np.array): pointing to R0 from R1 outside the square
    
    Return:
        line segment (np.array)
    """

    R1 = R0 - vector
    theta = np.arctan2(-vector[1], -vector[0])

    #threshold angles from R0 to P's 
    points = np.zeros((4, 2))
    points[0, :] = P1 
    points[1, :] = P2 
    points[2, :] = P3 
    points[3, :] = P4
    RtoP = points - R0
    angles = np.arctan2(RtoP[:, 1], RtoP[:, 0])
    
    if (theta>angles[0]) and (theta<=angles[1]):
        return lines_intersection(P1, P2, R0, R1)
    elif (theta>angles[1]) and (theta<=angles[2]):
        return lines_intersection(P2, P3, R0, R1)
    elif (theta>angles[2]) and (theta<=angles[3]):
        return lines_intersection(P3, P4, R0, R1)
    else:
        return lines_intersection(P4, P1, R0, R1)

def Kronecker(i, j):
    if i==j:
        return 1 
    else:
        return 0

def MomentOfInertia(positions: np.array, m: int=1, matrix: bool=False) -> np.array:
    """
    moment of inertia for a rigid body made of n points

    Inputs:
        1. positions (np.array): positions of the point particles as [numofatoms, 3]
        2. m (int): assuming each point mass is 1.0/numofatoms
        3. matrix (bool): return the results as a matrix of just [ixx iyy izz ixy ixz iyz]
    
    Return:
        moment of inertia (np.array)
    """

    Iij = np.zeros((3, 3))
    distance2 = np.square(positions).sum(axis=1)
    for i in range(3):
        for j in range(3):
            Iij[i, j] = m*(distance2*Kronecker(i, j)-positions[:, i]*positions[:, j]).sum()
    
    Iij /= positions.shape[0]
    if matrix:
        return Iij
    else:
        return np.array([Iij[0,0], Iij[1,1], Iij[2,2], Iij[0,1], Iij[0,2], Iij[1,2]])
