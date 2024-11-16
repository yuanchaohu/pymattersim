# coding = utf-8

import os
import unittest

import numpy as np
import pandas as pd

from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.static.sq import conditional_sq, sq
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestSq(unittest.TestCase):
    """
    Test class for Structure factor
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_unary = f"{READ_TEST_FILE_PATH}/unary.dump"
        self.test_file_binary = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_ternary = f"{READ_TEST_FILE_PATH}/ternary.dump"

    def test_Sq_unary(self) -> None:
        """
        Test Sq works properly for unary system
        """
        logger.info(f"Starting test Sq using {self.test_file_unary}...")
        readdump = DumpReader(self.test_file_unary, ndim=3)
        readdump.read_onefile()
        sq(readdump.snapshots, qrange=10, outputfile="sq_unary.csv").getresults()

        result = pd.read_csv("sq_unary.csv")
        np.testing.assert_almost_equal(
            [
                0.036721,
                0.098449,
                0.0966,
                0.087589,
                0.102995,
                0.167968,
                0.32392,
                0.858877,
                2.936611,
                1.497239,
                1.106665,
                0.797491,
                0.413192,
                0.609405,
                1.123366,
                2.695403,
                1.442278,
                0.899162,
                1.271647,
                0.856863,
                0.784563,
                0.876665,
                0.836927,
                1.062343,
                0.956568,
                0.869305,
                1.051758,
                1.04188,
                0.848116,
                1.233345,
                0.745138,
                0.564783,
                0.903626,
                1.217604,
                0.964599,
                0.982553,
                0.960175,
                0.923612,
                0.993903,
                1.321256,
                0.981675,
                1.113917,
                1.082202,
                0.990689,
                0.960879,
                1.252601,
                1.195605,
                1.155925,
                1.142982,
            ],
            result["Sq"].values,
        )
        os.remove("sq_unary.csv")
        logger.info(f"Finishing test Sq using {self.test_file_unary}...")

    def test_Sq_binary(self) -> None:
        """
        Test Sq works properly for binary system
        """
        logger.info(f"Starting test Sq using {self.test_file_binary}...")
        readdump = DumpReader(self.test_file_binary, ndim=2)
        readdump.read_onefile()
        sq(readdump.snapshots, qrange=10, outputfile="sq_binary.csv").getresults()

        result = pd.read_csv("sq_binary.csv")
        np.testing.assert_almost_equal(
            [
                0.038646,
                0.008722,
                0.01594,
                0.032078,
                0.02863,
                0.024956,
                0.031567,
                0.057656,
                0.074003,
                0.148906,
                0.151015,
                0.14022,
                0.214574,
                0.279497,
                0.235961,
                0.120868,
                0.214343,
                1.214033,
                2.545452,
                1.647541,
                0.652246,
                1.258145,
                0.981317,
                0.933233,
                1.353214,
                2.464631,
                2.410273,
                1.156073,
                0.845396,
                0.745672,
                0.486289,
                1.023237,
                1.093542,
                0.755509,
                0.711677,
            ],
            result["Sq"][::5].values,
        )
        np.testing.assert_almost_equal(
            [
                0.010016,
                0.006038,
                0.007666,
                0.004854,
                0.009,
                0.009333,
                0.00685,
                0.014441,
                0.023205,
                0.03844,
                0.044314,
                0.091537,
                0.135216,
                0.320094,
                0.198287,
                0.408314,
                1.119073,
                3.479528,
                4.118514,
                1.510478,
                0.503482,
                0.700712,
                0.417218,
                0.406507,
                0.385748,
                0.813065,
                0.771933,
                0.458545,
                0.421176,
                0.446941,
                0.416637,
                0.800454,
                1.666437,
                1.33979,
                0.625133,
            ],
            result["Sq11"][::5].values,
        )
        np.testing.assert_almost_equal(
            [
                0.180699,
                0.021479,
                0.038509,
                0.112131,
                0.095764,
                0.079835,
                0.110043,
                0.251578,
                0.361017,
                0.663501,
                0.796434,
                1.013789,
                1.478495,
                2.565442,
                1.193986,
                0.925219,
                1.078914,
                1.294598,
                0.240288,
                0.621367,
                0.288448,
                0.793002,
                0.841751,
                0.822287,
                1.417872,
                2.38481,
                2.387517,
                1.079952,
                0.616024,
                0.792896,
                0.439233,
                0.896328,
                1.218659,
                1.200145,
                0.942791,
            ],
            result["Sq22"][::5].values,
        )
        os.remove("sq_binary.csv")
        logger.info(f"Finishing test Sq using {self.test_file_binary}...")

    def test_Sq_ternary(self) -> None:
        """
        Test Sq works properly for ternary system
        """
        logger.info(f"Starting test Sq using {self.test_file_ternary}...")
        readdump = DumpReader(self.test_file_ternary, ndim=3)
        readdump.read_onefile()
        sq(readdump.snapshots, qrange=10, outputfile="sq_ternary.csv").getresults()

        result = pd.read_csv("sq_ternary.csv")
        np.testing.assert_almost_equal(
            [
                0.093943,
                0.043634,
                0.06753,
                0.131979,
                1.701307,
                1.393587,
                0.685079,
                1.03455,
                1.20976,
                1.10204,
                0.806972,
                0.713105,
                1.082076,
                1.033043,
                1.069499,
                0.99042,
                1.018081,
                1.000083,
                1.023933,
                0.728486,
                1.055094,
                0.838312,
                0.923572,
                0.978838,
                0.890376,
            ],
            result["Sq"][::5].values,
        )
        np.testing.assert_almost_equal(
            [
                0.076259,
                0.076837,
                0.106201,
                0.13778,
                1.911361,
                0.91595,
                0.604181,
                0.987004,
                1.272153,
                1.099729,
                0.823135,
                0.740196,
                1.144762,
                1.025841,
                1.05647,
                1.021978,
                1.046636,
                0.93741,
                1.036012,
                0.952977,
                1.022905,
                0.872951,
                0.954655,
                1.038649,
                0.982382,
            ],
            result["Sq11"][::5].values,
        )
        np.testing.assert_almost_equal(
            [
                1.045742,
                0.72311,
                1.017411,
                0.54045,
                0.919682,
                1.360917,
                1.005263,
                1.083289,
                0.945532,
                1.016932,
                1.000361,
                1.039523,
                1.071631,
                1.148394,
                0.994904,
                1.079142,
                0.98099,
                1.277323,
                0.905934,
                1.588855,
                0.836732,
                1.016146,
                1.112143,
                1.07673,
                0.842176,
            ],
            result["Sq22"][::5].values,
        )
        np.testing.assert_almost_equal(
            [
                0.316158,
                0.506681,
                0.715554,
                0.842354,
                0.933485,
                0.95514,
                0.860464,
                0.820994,
                1.019856,
                1.018377,
                0.974524,
                0.904845,
                0.900496,
                0.930537,
                1.171108,
                0.895679,
                1.051473,
                0.963053,
                0.926793,
                1.014876,
                0.983708,
                0.934353,
                0.797349,
                0.864113,
                0.897462,
            ],
            result["Sq33"][::5].values,
        )
        os.remove("sq_ternary.csv")
        logger.info(f"Finishing test Sq using {self.test_file_ternary}...")

    def test_sq_condition(self) -> None:
        """
        Test sq condition works properly for ternary system
        """
        logger.info(f"Starting test conditional_sq using {self.test_file_ternary}...")
        readdump = DumpReader(self.test_file_ternary, ndim=3)
        readdump.read_onefile()
        snapshot = readdump.snapshots.snapshots[0]

        sq22_selected = conditional_sq(
            snapshot,
            qvector=np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
            condition=snapshot.particle_type == 2,
        )[
            1
        ]["Sq"].values
        sq22_results = sq(readdump.snapshots, qvector=np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).getresults()["Sq22"].values

        np.testing.assert_almost_equal(sq22_selected.round(4), sq22_results.round(4))
        logger.info(f"Finishing test conditional_sq using {self.test_file_ternary}...")
