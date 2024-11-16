# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.utils.funcs import (Legendre_polynomials, Wignerindex,
                                     alpha2factor, areafac, grid_gaussian,
                                     kronecker, nidealfac)
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace

READ_TEST_FILE_PATH = "tests/sample_test_data"


class Test_funcs(unittest.TestCase):
    """
    Test functions
    """

    def setUp(self) -> None:
        super().setUp()

    def test_kronecker(self) -> None:
        """
        Test kronecker function
        """
        k1 = kronecker(2, 1)
        k2 = kronecker(3, 3)
        self.assertEqual(k1, 0)
        self.assertEqual(k2, 1)

    def test_nidealfac(self) -> None:
        """
        Test nidealfac function
        """
        n1 = nidealfac(2)
        n2 = nidealfac(3)
        self.assertEqual(n1, 1.0)
        self.assertEqual(n2, 4.0 / 3)

    def test_areafac(self) -> None:
        """
        Test areafac function
        """
        a1 = areafac(2)
        a2 = areafac(3)
        self.assertEqual(a1, 2.0)
        self.assertEqual(a2, 4.0)

    def test_alpha2factor(self) -> None:
        """
        Test alpha2factor function
        """
        a1 = alpha2factor(2)
        a2 = alpha2factor(3)
        self.assertEqual(a1, 1 / 2.0)
        self.assertEqual(a2, 3 / 5.0)

    def test_Wignerindex(self) -> None:
        """
        Test Wignerindex function
        """
        w = Wignerindex(6)
        np.testing.assert_almost_equal(
            w[:, 3][:10],
            np.array(
                [
                    0.0511827251162099,
                    -0.0957541107531435,
                    0.129114816600119,
                    -0.141438195120055,
                    0.129114816600119,
                    -0.0957541107531435,
                    0.0511827251162099,
                    -0.0957541107531435,
                    0.127956812790525,
                    -0.106078646340041,
                ]
            ),
        )

    def test_grid_gaussian(self) -> None:
        """
        Test grid_gaussian function
        """
        sigma = 2.0
        bins = np.arange(5)
        expected = grid_gaussian(bins, sigma)
        output = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins**2) / (2 * sigma**2))
        np.testing.assert_almost_equal(expected, output)

    def test_Legendre_polynomials(self) -> None:
        """
        Test Legendre_polynomials function
        """
        Lp = Legendre_polynomials(6, 3)
        self.assertEqual(Lp, 53.5)
