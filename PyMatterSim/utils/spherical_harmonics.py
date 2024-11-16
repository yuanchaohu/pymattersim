# coding = utf-8

"""
see documentation @../../docs/utils.md

Calculate spherical harmonics of given (theta,phi) from (l = 0) to (l = 10)
From SphHarm0() to SphHarm10() a list of [-l, l] values will be returned
if l>10 use scipy.special.sph_harm (this may be slower)
"""

import cmath

import numpy as np
import numpy.typing as npt
from scipy.special import sph_harm

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=no-name-in-module


def SphHarm0() -> float:
    """
    Spherical Harmonics l = 0

    Inputs:
        None

    Return:
        spherical harmonics for l=0 (npt.NDArray)
    """
    return (1 / 2) * np.sqrt(1 / np.pi)


def SphHarm1(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 1

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=1 (npt.NDArray)
    """
    results = []

    mN1 = (1 / 2) * np.sqrt(3 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta)
    results.append(mN1)
    m0 = (1 / 2) * np.sqrt(3 / np.pi) * np.cos(theta)
    results.append(m0)
    m1 = -(1 / 2) * np.sqrt(3 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta)
    results.append(m1)
    return np.array(results)


def SphHarm2(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 2

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=2 (npt.NDArray)
    """
    results = []

    mN2 = (1 / 4) * np.sqrt(15 / 2 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2
    results.append(mN2)
    mN1 = (1 / 2) * np.sqrt(15 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * np.cos(theta)
    results.append(mN1)
    m0 = (1 / 4) * np.sqrt(5 / np.pi) * (3 * (np.cos(theta)) ** 2 - 1)
    results.append(m0)
    m1 = -(1 / 2) * np.sqrt(15 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * np.cos(theta)
    results.append(m1)
    m2 = (1 / 4) * np.sqrt(15 / 2 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2
    results.append(m2)
    return np.array(results)


def SphHarm3(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 3

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=3 (npt.NDArray)
    """
    results = []

    mN3 = (1 / 8) * np.sqrt(35 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3
    results.append(mN3)
    mN2 = (1 / 4) * np.sqrt(105 / 2 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * np.cos(theta)
    results.append(mN2)
    mN1 = (1 / 8) * np.sqrt(21 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (5 * (np.cos(theta)) ** 2 - 1)
    results.append(mN1)
    m0 = (1 / 4) * np.sqrt(7 / np.pi) * (5 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(m0)
    m1 = -(1 / 8) * np.sqrt(21 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (5 * (np.cos(theta)) ** 2 - 1)
    results.append(m1)
    m2 = (1 / 4) * np.sqrt(105 / 2 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * np.cos(theta)
    results.append(m2)
    m3 = -(1 / 8) * np.sqrt(35 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3
    results.append(m3)
    return np.array(results)


def SphHarm4(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 4

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=4 (npt.NDArray)
    """
    results = []

    mN4 = (3 / 16) * np.sqrt(35 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4
    results.append(mN4)
    mN3 = (3 / 8) * np.sqrt(35 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * np.cos(theta)
    results.append(mN3)
    mN2 = (3 / 8) * np.sqrt(5 / 2 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (7 * (np.cos(theta)) ** 2 - 1)
    results.append(mN2)
    mN1 = (3 / 8) * np.sqrt(5 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (7 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(mN1)
    m0 = (3 / 16) * np.sqrt(1 / np.pi) * (35 * (np.cos(theta)) ** 4 - 30 * (np.cos(theta)) ** 2 + 3)
    results.append(m0)
    m1 = -(3 / 8) * np.sqrt(5 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (7 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(m1)
    m2 = (3 / 8) * np.sqrt(5 / 2 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (7 * (np.cos(theta)) ** 2 - 1)
    results.append(m2)
    m3 = -(3 / 8) * np.sqrt(35 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * np.cos(theta)
    results.append(m3)
    m4 = (3 / 16) * np.sqrt(35 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4
    results.append(m4)
    return np.array(results)


def SphHarm5(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 5

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=5 (npt.NDArray)
    """
    results = []

    mN5 = (3 / 32) * np.sqrt(77 / np.pi) * cmath.exp(-5j * phi) * (np.sin(theta)) ** 5
    results.append(mN5)
    mN4 = (3 / 16) * np.sqrt(385 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4 * np.cos(theta)
    results.append(mN4)
    mN3 = (1 / 32) * np.sqrt(385 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * (9 * (np.cos(theta)) ** 2 - 1)
    results.append(mN3)
    mN2 = (1 / 8) * np.sqrt(1155 / 2 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (3 * (np.cos(theta)) ** 3 - np.cos(theta))
    results.append(mN2)
    mN1 = (1 / 16) * np.sqrt(165 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (21 * (np.cos(theta)) ** 4 - 14 * (np.cos(theta)) ** 2 + 1)
    results.append(mN1)
    m0 = (1 / 16) * np.sqrt(11 / np.pi) * (63 * (np.cos(theta)) ** 5 - 70 * (np.cos(theta)) ** 3 + 15 * np.cos(theta))
    results.append(m0)
    m1 = -(1 / 16) * np.sqrt(165 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (21 * (np.cos(theta)) ** 4 - 14 * (np.cos(theta)) ** 2 + 1)
    results.append(m1)
    m2 = (1 / 8) * np.sqrt(1155 / 2 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (3 * (np.cos(theta)) ** 3 - np.cos(theta))
    results.append(m2)
    m3 = -(1 / 32) * np.sqrt(385 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * (9 * (np.cos(theta)) ** 2 - 1)
    results.append(m3)
    m4 = (3 / 16) * np.sqrt(385 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4 * np.cos(theta)
    results.append(m4)
    m5 = -(3 / 32) * np.sqrt(77 / np.pi) * cmath.exp(5j * phi) * (np.sin(theta)) ** 5
    results.append(m5)
    return np.array(results)


def SphHarm6(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 6

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=6 (npt.NDArray)
    """
    results = []

    mN6 = (1 / 64) * np.sqrt(3003 / np.pi) * cmath.exp(-6j * phi) * (np.sin(theta)) ** 6
    results.append(mN6)
    mN5 = (3 / 32) * np.sqrt(1001 / np.pi) * cmath.exp(-5j * phi) * (np.sin(theta)) ** 5 * np.cos(theta)
    results.append(mN5)
    mN4 = (3 / 32) * np.sqrt(91 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4 * (11 * (np.cos(theta)) ** 2 - 1)
    results.append(mN4)
    mN3 = (1 / 32) * np.sqrt(1365 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * (11 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(mN3)
    mN2 = (1 / 64) * np.sqrt(1365 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (33 * (np.cos(theta)) ** 4 - 18 * (np.cos(theta)) ** 2 + 1)
    results.append(mN2)
    mN1 = (1 / 16) * np.sqrt(273 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (33 * (np.cos(theta)) ** 5 - 30 * (np.cos(theta)) ** 3 + 5 * np.cos(theta))
    results.append(mN1)
    m0 = (1 / 32) * np.sqrt(13 / np.pi) * (231 * (np.cos(theta)) ** 6 - 315 * (np.cos(theta)) ** 4 + 105 * (np.cos(theta)) ** 2 - 5)
    results.append(m0)
    m1 = -(1 / 16) * np.sqrt(273 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (33 * (np.cos(theta)) ** 5 - 30 * (np.cos(theta)) ** 3 + 5 * np.cos(theta))
    results.append(m1)
    m2 = (1 / 64) * np.sqrt(1365 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (33 * (np.cos(theta)) ** 4 - 18 * (np.cos(theta)) ** 2 + 1)
    results.append(m2)
    m3 = -(1 / 32) * np.sqrt(1365 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * (11 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(m3)
    m4 = (3 / 32) * np.sqrt(91 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4 * (11 * (np.cos(theta)) ** 2 - 1)
    results.append(m4)
    m5 = -(3 / 32) * np.sqrt(1001 / np.pi) * cmath.exp(5j * phi) * (np.sin(theta)) ** 5 * np.cos(theta)
    results.append(m5)
    m6 = (1 / 64) * np.sqrt(3003 / np.pi) * cmath.exp(6j * phi) * (np.sin(theta)) ** 6
    results.append(m6)
    return np.array(results)


def SphHarm7(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 7

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=7 (npt.NDArray)
    """
    results = []

    mN7 = (3 / 64) * np.sqrt(715 / 2 / np.pi) * cmath.exp(-7j * phi) * (np.sin(theta)) ** 7
    results.append(mN7)
    mN6 = (3 / 64) * np.sqrt(5005 / np.pi) * cmath.exp(-6j * phi) * (np.sin(theta)) ** 6 * np.cos(theta)
    results.append(mN6)
    mN5 = (3 / 64) * np.sqrt(385 / 2 / np.pi) * cmath.exp(-5j * phi) * (np.sin(theta)) ** 5 * (13 * (np.cos(theta)) ** 2 - 1)
    results.append(mN5)
    mN4 = (3 / 32) * np.sqrt(385 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4 * (13 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(mN4)
    mN3 = (3 / 64) * np.sqrt(35 / 2 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * (143 * (np.cos(theta)) ** 4 - 66 * (np.cos(theta)) ** 2 + 3)
    results.append(mN3)
    mN2 = (3 / 64) * np.sqrt(35 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (143 * (np.cos(theta)) ** 5 - 110 * (np.cos(theta)) ** 3 + 15 * np.cos(theta))
    results.append(mN2)
    mN1 = (1 / 64) * np.sqrt(105 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (429 * (np.cos(theta)) ** 6 - 495 * (np.cos(theta)) ** 4 + 135 * (np.cos(theta)) ** 2 - 5)
    results.append(mN1)
    m0 = (1 / 32) * np.sqrt(15 / np.pi) * (429 * (np.cos(theta)) ** 7 - 693 * (np.cos(theta)) ** 5 + 315 * (np.cos(theta)) ** 3 - 35 * np.cos(theta))
    results.append(m0)
    m1 = -(1 / 64) * np.sqrt(105 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (429 * (np.cos(theta)) ** 6 - 495 * (np.cos(theta)) ** 4 + 135 * (np.cos(theta)) ** 2 - 5)
    results.append(m1)
    m2 = (3 / 64) * np.sqrt(35 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (143 * (np.cos(theta)) ** 5 - 110 * (np.cos(theta)) ** 3 + 15 * np.cos(theta))
    results.append(m2)
    m3 = -(3 / 64) * np.sqrt(35 / 2 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * (143 * (np.cos(theta)) ** 4 - 66 * (np.cos(theta)) ** 2 + 3)
    results.append(m3)
    m4 = (3 / 32) * np.sqrt(385 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4 * (13 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(m4)
    m5 = -(3 / 64) * np.sqrt(385 / 2 / np.pi) * cmath.exp(5j * phi) * (np.sin(theta)) ** 5 * (13 * (np.cos(theta)) ** 2 - 1)
    results.append(m5)
    m6 = (3 / 64) * np.sqrt(5005 / np.pi) * cmath.exp(6j * phi) * (np.sin(theta)) ** 6 * np.cos(theta)
    results.append(m6)
    m7 = -(3 / 64) * np.sqrt(715 / 2 / np.pi) * cmath.exp(7j * phi) * (np.sin(theta)) ** 7
    results.append(m7)
    return np.array(results)


def SphHarm8(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 8

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=8 (npt.NDArray)
    """
    results = []

    mN8 = (3 / 256) * np.sqrt(12155 / 2 / np.pi) * cmath.exp(-8j * phi) * (np.sin(theta)) ** 8
    results.append(mN8)
    mN7 = (3 / 64) * np.sqrt(12155 / 2 / np.pi) * cmath.exp(-7j * phi) * (np.sin(theta)) ** 7 * np.cos(theta)
    results.append(mN7)
    mN6 = (1 / 128) * np.sqrt(7293 / np.pi) * cmath.exp(-6j * phi) * (np.sin(theta)) ** 6 * (15 * (np.cos(theta)) ** 2 - 1)
    results.append(mN6)
    mN5 = (3 / 64) * np.sqrt(17017 / 2 / np.pi) * cmath.exp(-5j * phi) * (np.sin(theta)) ** 5 * (5 * (np.cos(theta)) ** 3 - np.cos(theta))
    results.append(mN5)
    mN4 = (3 / 128) * np.sqrt(1309 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4 * (65 * (np.cos(theta)) ** 4 - 26 * (np.cos(theta)) ** 2 + 1)
    results.append(mN4)
    mN3 = (1 / 64) * np.sqrt(19635 / 2 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * (39 * (np.cos(theta)) ** 5 - 26 * (np.cos(theta)) ** 3 + 3 * np.cos(theta))
    results.append(mN3)
    mN2 = (3 / 128) * np.sqrt(595 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (143 * (np.cos(theta)) ** 6 - 143 * (np.cos(theta)) ** 4 + 33 * (np.cos(theta)) ** 2 - 1)
    results.append(mN2)
    mN1 = (3 / 64) * np.sqrt(17 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (715 * (np.cos(theta)) ** 7 - 1001 * (np.cos(theta)) ** 5 + 385 * (np.cos(theta)) ** 3 - 35 * np.cos(theta))
    results.append(mN1)
    m0 = (1 / 256) * np.sqrt(17 / np.pi) * (6435 * (np.cos(theta)) ** 8 - 12012 * (np.cos(theta)) ** 6 + 6930 * (np.cos(theta)) ** 4 - 1260 * (np.cos(theta)) ** 2 + 35)
    results.append(m0)
    m1 = -(3 / 64) * np.sqrt(17 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (715 * (np.cos(theta)) ** 7 - 1001 * (np.cos(theta)) ** 5 + 385 * (np.cos(theta)) ** 3 - 35 * np.cos(theta))
    results.append(m1)
    m2 = (3 / 128) * np.sqrt(595 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (143 * (np.cos(theta)) ** 6 - 143 * (np.cos(theta)) ** 4 + 33 * (np.cos(theta)) ** 2 - 1)
    results.append(m2)
    m3 = -(1 / 64) * np.sqrt(19635 / 2 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * (39 * (np.cos(theta)) ** 5 - 26 * (np.cos(theta)) ** 3 + 3 * np.cos(theta))
    results.append(m3)
    m4 = (3 / 128) * np.sqrt(1309 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4 * (65 * (np.cos(theta)) ** 4 - 26 * (np.cos(theta)) ** 2 + 1)
    results.append(m4)
    m5 = -(3 / 64) * np.sqrt(17017 / 2 / np.pi) * cmath.exp(5j * phi) * (np.sin(theta)) ** 5 * (5 * (np.cos(theta)) ** 3 - np.cos(theta))
    results.append(m5)
    m6 = (1 / 128) * np.sqrt(7293 / np.pi) * cmath.exp(6j * phi) * (np.sin(theta)) ** 6 * (15 * (np.cos(theta)) ** 2 - 1)
    results.append(m6)
    m7 = -(3 / 64) * np.sqrt(12155 / 2 / np.pi) * cmath.exp(7j * phi) * (np.sin(theta)) ** 7 * np.cos(theta)
    results.append(m7)
    m8 = (3 / 256) * np.sqrt(12155 / 2 / np.pi) * cmath.exp(8j * phi) * (np.sin(theta)) ** 8
    results.append(m8)
    return np.array(results)


def SphHarm9(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 9

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=9 (npt.NDArray)
    """
    results = []

    mN9 = (1 / 512) * np.sqrt(230945 / np.pi) * cmath.exp(-9j * phi) * (np.sin(theta)) ** 9
    results.append(mN9)
    mN8 = (3 / 256) * np.sqrt(230945 / 2 / np.pi) * cmath.exp(-8j * phi) * (np.sin(theta)) ** 8 * np.cos(theta)
    results.append(mN8)
    mN7 = (3 / 512) * np.sqrt(13585 / np.pi) * cmath.exp(-7j * phi) * (np.sin(theta)) ** 7 * (17 * (np.cos(theta)) ** 2 - 1)
    results.append(mN7)
    mN6 = (1 / 128) * np.sqrt(40755 / np.pi) * cmath.exp(-6j * phi) * (np.sin(theta)) ** 6 * (17 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(mN6)
    mN5 = (3 / 256) * np.sqrt(2717 / np.pi) * cmath.exp(-5j * phi) * (np.sin(theta)) ** 5 * (85 * (np.cos(theta)) ** 4 - 30 * (np.cos(theta)) ** 2 + 1)
    results.append(mN5)
    mN4 = (3 / 128) * np.sqrt(95095 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4 * (17 * (np.cos(theta)) ** 5 - 10 * (np.cos(theta)) ** 3 + np.cos(theta))
    results.append(mN4)
    mN3 = (1 / 256) * np.sqrt(21945 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * (221 * (np.cos(theta)) ** 6 - 195 * (np.cos(theta)) ** 4 + 39 * (np.cos(theta)) ** 2 - 1)
    results.append(mN3)
    mN2 = (3 / 128) * np.sqrt(1045 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (221 * (np.cos(theta)) ** 7 - 273 * (np.cos(theta)) ** 5 + 91 * (np.cos(theta)) ** 3 - 7 * np.cos(theta))
    results.append(mN2)
    mN1 = (3 / 256) * np.sqrt(95 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (2431 * (np.cos(theta)) ** 8 - 4004 * (np.cos(theta)) ** 6 + 2002 * (np.cos(theta)) ** 4 - 308 * (np.cos(theta)) ** 2 + 7)
    results.append(mN1)
    m0 = (1 / 256) * np.sqrt(19 / np.pi) * (12155 * (np.cos(theta)) ** 9 - 25740 * (np.cos(theta)) ** 7 + 18018 * (np.cos(theta)) ** 5 - 4620 * (np.cos(theta)) ** 3 + 315 * np.cos(theta))
    results.append(m0)
    m1 = -(3 / 256) * np.sqrt(95 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (2431 * (np.cos(theta)) ** 8 - 4004 * (np.cos(theta)) ** 6 + 2002 * (np.cos(theta)) ** 4 - 308 * (np.cos(theta)) ** 2 + 7)
    results.append(m1)
    m2 = (3 / 128) * np.sqrt(1045 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (221 * (np.cos(theta)) ** 7 - 273 * (np.cos(theta)) ** 5 + 91 * (np.cos(theta)) ** 3 - 7 * np.cos(theta))
    results.append(m2)
    m3 = -(1 / 256) * np.sqrt(21945 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * (221 * (np.cos(theta)) ** 6 - 195 * (np.cos(theta)) ** 4 + 39 * (np.cos(theta)) ** 2 - 1)
    results.append(m3)
    m4 = (3 / 128) * np.sqrt(95095 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4 * (17 * (np.cos(theta)) ** 5 - 10 * (np.cos(theta)) ** 3 + np.cos(theta))
    results.append(m4)
    m5 = -(3 / 256) * np.sqrt(2717 / np.pi) * cmath.exp(5j * phi) * (np.sin(theta)) ** 5 * (85 * (np.cos(theta)) ** 4 - 30 * (np.cos(theta)) ** 2 + 1)
    results.append(m5)
    m6 = (1 / 128) * np.sqrt(40755 / np.pi) * cmath.exp(6j * phi) * (np.sin(theta)) ** 6 * (17 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(m6)
    m7 = -(3 / 512) * np.sqrt(13585 / np.pi) * cmath.exp(7j * phi) * (np.sin(theta)) ** 7 * (17 * (np.cos(theta)) ** 2 - 1)
    results.append(m7)
    m8 = (3 / 256) * np.sqrt(230945 / 2 / np.pi) * cmath.exp(8j * phi) * (np.sin(theta)) ** 8 * np.cos(theta)
    results.append(m8)
    m9 = -(1 / 512) * np.sqrt(230945 / np.pi) * cmath.exp(9j * phi) * (np.sin(theta)) ** 9
    results.append(m9)
    return np.array(results)


def SphHarm10(theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonics l = 10

    Inputs:
        1. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        2. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l=10 (npt.NDArray)
    """
    results = []

    mN10 = (1 / 1024) * np.sqrt(969969 / np.pi) * cmath.exp(-10j * phi) * (np.sin(theta)) ** 10
    results.append(mN10)
    mN9 = (1 / 512) * np.sqrt(4849845 / np.pi) * cmath.exp(-9j * phi) * (np.sin(theta)) ** 9 * np.cos(theta)
    results.append(mN9)
    mN8 = (1 / 512) * np.sqrt(255255 / 2 / np.pi) * cmath.exp(-8j * phi) * (np.sin(theta)) ** 8 * (19 * (np.cos(theta)) ** 2 - 1)
    results.append(mN8)
    mN7 = (3 / 512) * np.sqrt(85085 / np.pi) * cmath.exp(-7j * phi) * (np.sin(theta)) ** 7 * (19 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(mN7)
    mN6 = (3 / 1024) * np.sqrt(5005 / np.pi) * cmath.exp(-6j * phi) * (np.sin(theta)) ** 6 * (323 * (np.cos(theta)) ** 4 - 102 * (np.cos(theta)) ** 2 + 3)
    results.append(mN6)
    mN5 = (3 / 256) * np.sqrt(1001 / np.pi) * cmath.exp(-5j * phi) * (np.sin(theta)) ** 5 * (323 * (np.cos(theta)) ** 5 - 170 * (np.cos(theta)) ** 3 + 15 * np.cos(theta))
    results.append(mN5)
    mN4 = (3 / 256) * np.sqrt(5005 / 2 / np.pi) * cmath.exp(-4j * phi) * (np.sin(theta)) ** 4 * (323 * (np.cos(theta)) ** 6 - 255 * (np.cos(theta)) ** 4 + 45 * (np.cos(theta)) ** 2 - 1)
    results.append(mN4)
    mN3 = (3 / 256) * np.sqrt(5005 / np.pi) * cmath.exp(-3j * phi) * (np.sin(theta)) ** 3 * (323 * (np.cos(theta)) ** 7 - 357 * (np.cos(theta)) ** 5 + 105 * (np.cos(theta)) ** 3 - 7 * np.cos(theta))
    results.append(mN3)
    mN2 = (3 / 512) * np.sqrt(385 / 2 / np.pi) * cmath.exp(-2j * phi) * (np.sin(theta)) ** 2 * (4199 * (np.cos(theta)) ** 8 - 6188 * (np.cos(theta)) ** 6 + 2730 * (np.cos(theta)) ** 4 - 364 * (np.cos(theta)) ** 2 + 7)
    results.append(mN2)
    mN1 = (1 / 256) * np.sqrt(1155 / 2 / np.pi) * cmath.exp(-1j * phi) * np.sin(theta) * (4199 * (np.cos(theta)) ** 9 - 7956 * (np.cos(theta)) ** 7 + 4914 * (np.cos(theta)) ** 5 - 1092 * (np.cos(theta)) ** 3 + 63 * np.cos(theta))
    results.append(mN1)
    m0 = (1 / 512) * np.sqrt(21 / np.pi) * (46189 * (np.cos(theta)) ** 10 - 109395 * (np.cos(theta)) ** 8 + 90090 * (np.cos(theta)) ** 6 - 30030 * (np.cos(theta)) ** 4 + 3465 * (np.cos(theta)) ** 2 - 63)
    results.append(m0)
    m1 = -(1 / 256) * np.sqrt(1155 / 2 / np.pi) * cmath.exp(1j * phi) * np.sin(theta) * (4199 * (np.cos(theta)) ** 9 - 7956 * (np.cos(theta)) ** 7 + 4914 * (np.cos(theta)) ** 5 - 1092 * (np.cos(theta)) ** 3 + 63 * np.cos(theta))
    results.append(m1)
    m2 = (3 / 512) * np.sqrt(385 / 2 / np.pi) * cmath.exp(2j * phi) * (np.sin(theta)) ** 2 * (4199 * (np.cos(theta)) ** 8 - 6188 * (np.cos(theta)) ** 6 + 2730 * (np.cos(theta)) ** 4 - 364 * (np.cos(theta)) ** 2 + 7)
    results.append(m2)
    m3 = -(3 / 256) * np.sqrt(5005 / np.pi) * cmath.exp(3j * phi) * (np.sin(theta)) ** 3 * (323 * (np.cos(theta)) ** 7 - 357 * (np.cos(theta)) ** 5 + 105 * (np.cos(theta)) ** 3 - 7 * np.cos(theta))
    results.append(m3)
    m4 = (3 / 256) * np.sqrt(5005 / 2 / np.pi) * cmath.exp(4j * phi) * (np.sin(theta)) ** 4 * (323 * (np.cos(theta)) ** 6 - 255 * (np.cos(theta)) ** 4 + 45 * (np.cos(theta)) ** 2 - 1)
    results.append(m4)
    m5 = -(3 / 256) * np.sqrt(1001 / np.pi) * cmath.exp(5j * phi) * (np.sin(theta)) ** 5 * (323 * (np.cos(theta)) ** 5 - 170 * (np.cos(theta)) ** 3 + 15 * np.cos(theta))
    results.append(m5)
    m6 = (3 / 1024) * np.sqrt(5005 / np.pi) * cmath.exp(6j * phi) * (np.sin(theta)) ** 6 * (323 * (np.cos(theta)) ** 4 - 102 * (np.cos(theta)) ** 2 + 3)
    results.append(m6)
    m7 = -(3 / 512) * np.sqrt(85085 / np.pi) * cmath.exp(7j * phi) * (np.sin(theta)) ** 7 * (19 * (np.cos(theta)) ** 3 - 3 * np.cos(theta))
    results.append(m7)
    m8 = (1 / 512) * np.sqrt(255255 / 2 / np.pi) * cmath.exp(8j * phi) * (np.sin(theta)) ** 8 * (19 * (np.cos(theta)) ** 2 - 1)
    results.append(m8)
    m9 = -(1 / 512) * np.sqrt(4849845 / np.pi) * cmath.exp(9j * phi) * (np.sin(theta)) ** 9 * np.cos(theta)
    results.append(m9)
    m10 = (1 / 1024) * np.sqrt(969969 / np.pi) * cmath.exp(10j * phi) * (np.sin(theta)) ** 10
    results.append(m10)
    return np.array(results)


def SphHarm_above(l: int, theta: float, phi: float) -> npt.NDArray:
    """
    Spherical Harmonicsl > 10
    be aware of theta and phi used for scipy is inverse to the above equations
    change phi from [-np.pi, np.pi] to [0, 2PI]

    Inputs:
    1. l (int): degree of harmonics, > 10
    2. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    3. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics for l>10 (npt.NDArray)
    """

    if phi < 0:
        phi += 2 * np.pi

    results = []
    for m in range(-l, l + 1):
        results.append(sph_harm(m, l, phi, theta))
    return np.array(results)


def sph_harm_l(l: int, theta: float, phi: float) -> npt.NDArray:
    """
    Choose Spherical Harmonics for order l

    Inputs:
        1. l (int): degree of spherical harmonics
        2. theta (float): Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
        3. phi (float): Polar (colatitudinal) coordinate; must be in [0, pi]

    Return:
        spherical harmonics (npt.NDArray)
    """
    if l == 2:
        return SphHarm2(theta, phi)
    if l == 3:
        return SphHarm3(theta, phi)
    if l == 4:
        return SphHarm4(theta, phi)
    if l == 5:
        return SphHarm5(theta, phi)
    if l == 6:
        return SphHarm6(theta, phi)
    if l == 7:
        return SphHarm7(theta, phi)
    if l == 8:
        return SphHarm8(theta, phi)
    if l == 9:
        return SphHarm9(theta, phi)
    if l == 10:
        return SphHarm10(theta, phi)
    if l > 10:
        return SphHarm_above(l, theta, phi)
