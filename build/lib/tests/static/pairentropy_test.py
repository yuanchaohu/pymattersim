# coding = utf-8

import os
import sys
import unittest

import numpy as np
import pandas as pd
from tables import atom

from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.static.pairentropy import S2
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestS2(unittest.TestCase):
    """
    Test case for pair entropy S2
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_unary = f"{READ_TEST_FILE_PATH}/unary.dump"
        self.test_file_binary_2d = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.atom"
        self.test_file_binary_3d = f"{READ_TEST_FILE_PATH}/3d/3ddump.s.v.atom"

    def test_s2_unary(self) -> None:
        """
        Test S2 for single-component system
        """
        logger.info(f"Starting test S2 using {self.test_file_unary}...")
        readdump = DumpReader(self.test_file_unary, ndim=3)
        readdump.read_onefile()

        s2 = S2(
            snapshots=readdump.snapshots,
            sigmas=np.array([[1.5]]),
            ppp=np.array([1, 1, 1]),
            rdelta=0.02,
            ndelta=500,
        )
        atomic_s2 = s2.particle_s2()
        np.testing.assert_almost_equal(
            [
                -5.69614799,
                -6.07658979,
                -7.13261826,
                -5.30170028,
                -6.14242868,
                -6.07373129,
                -5.28509506,
                -6.02746132,
                -5.44502006,
            ],
            atomic_s2[0, 10:100:10],
        )

    def test_s2_binary_2D(self) -> None:
        """
        Test S2 for binary systems in two dimensions
        """
        logger.info(f"Starting test S2 using {self.test_file_binary_2d}...")
        readdump = DumpReader(self.test_file_binary_2d, ndim=2)
        readdump.read_onefile()

        s2 = S2(
            snapshots=readdump.snapshots,
            sigmas=np.array(
                [
                    [1.5, 2.0],
                    [2.0, 1.5],
                ]
            ),
            ppp=np.array([1, 1]),
            rdelta=0.02,
            ndelta=500,
        )
        atomic_s2 = s2.particle_s2()
        np.testing.assert_almost_equal(
            [
                -6.94583538,
                -7.30615146,
                -6.20884165,
                -6.42701815,
                -6.82957428,
                -6.45916439,
                -6.86828755,
                -7.83382262,
                -6.84823652,
            ],
            atomic_s2[2, 10:100:10],
        )

        spatial_corr_s2 = s2.spatial_corr()
        np.testing.assert_almost_equal(
            [
                0.0,
                74.94370051,
                99.1316091,
                25.63123923,
                22.8294834,
                48.70640496,
                74.07406626,
            ],
            spatial_corr_s2["gA"][30:100:10].values,
        )

        time_corr_s2 = s2.time_corr()
        np.testing.assert_almost_equal(
            [
                1.0,
                0.9991545,
                0.9990423,
                0.99934783,
                0.99878096,
                0.9987917,
                0.99885383,
                0.99761731,
                0.99820612,
                0.99729555,
            ],
            time_corr_s2["time_corr"].values,
        )

    def test_s2_binary_3D(self) -> None:
        """
        Test S2 for binary systems in three dimensions
        """
        logger.info(f"Starting test S2 using {self.test_file_binary_3d}...")
        readdump = DumpReader(self.test_file_binary_3d, ndim=3)
        readdump.read_onefile()

        s2 = S2(
            snapshots=readdump.snapshots,
            sigmas=np.array(
                [
                    [0.2, 0.3],
                    [0.3, 0.4],
                ]
            ),
            ppp=np.array([1, 1, 1]),
            rdelta=0.02,
            ndelta=500,
        )
        atomic_s2 = s2.particle_s2()
        np.testing.assert_almost_equal(
            [
                -1807.45099173,
                -1803.4923892,
                -1808.32461111,
                -1811.64374028,
                -1811.51611006,
                -1809.28859867,
                -1804.86773536,
                -1801.86772559,
                -1811.03800581,
            ],
            atomic_s2[2, 10:100:10],
        )

        spatial_corr_s2 = s2.spatial_corr()
        np.testing.assert_almost_equal(
            [
                0.0,
                767137.14797639,
                5847563.60523199,
                2864236.20132432,
                1568460.02304579,
                3010162.95115112,
                3937367.17411343,
            ],
            spatial_corr_s2["gA"][30:100:10].values,
        )

        time_corr_s2 = s2.time_corr()
        np.testing.assert_almost_equal(
            [
                1.0,
                0.99999649,
                0.99998859,
                0.99998887,
                0.99999361,
                0.99999948,
                1.00000824,
                1.00002434,
                1.00004311,
                1.0000273,
            ],
            time_corr_s2["time_corr"].values,
        )
