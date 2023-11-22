# coding = utf-8

import os
import unittest
import numpy as np
from reader.dump_reader import DumpReader
from static.packing_capability import theta_2D
from neighbors.freud_neighbors import cal_neighbors
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestTheta2D(unittest.TestCase):
    """
    Test class for theta2d
    """
    def setUp(self) -> None:
        super().setUp()
        self.test_file_2D = f"{READ_TEST_FILE_PATH}/dump_2D.atom"

    def test_theta2d(self) -> None:
        """
        Test theta2d works properly for 2D system
        """
        logger.info(f"Starting test theta2d using {self.test_file_2D}...")
        readdump = DumpReader(self.test_file_2D, 2)
        readdump.read_onefile()

        cal_neighbors(snapshots=readdump.snapshots, outputfile='test')

        theta = theta_2D(readdump.snapshots, sigmas=np.array([[1.5,1.5],[1.5,1.5]]), neighborfile='test.neighbor.dat')
        np.testing.assert_almost_equal(
            np.array([[0.52359878, 0.52359878, 0.3876635 , 0.35924647, 0.52359878,
            0.52359878, 0.21647106, 0.26179939, 0.19641053, 0.16182063,
            0.14709638, 0.25303882, 0.18446597, 0.26036996, 0.27088255,
            0.15627046, 0.2129428 , 0.08295416, 0.20201806, 0.17225084],
            [0.52359878, 0.52359878, 0.52359878, 0.37062511, 0.52359878,
            0.52359878, 0.52359878, 0.18944879, 0.18344326, 0.19435052,
            0.16774308, 0.26393261, 0.15548638, 0.26179939, 0.17209369,
            0.18634357, 0.19271306, 0.07699365, 0.26179939, 0.21607533],
            [0.20943951, 0.52359878, 0.52359878, 0.4500801 , 0.23241144,
            0.24664117, 0.30358029, 0.26179939, 0.25870591, 0.15297146,
            0.20439641, 0.18068971, 0.14959965, 0.22721239, 0.14959965,
            0.12862106, 0.2082548 , 0.15399902, 0.18074812, 0.2032766 ],
            [0.52359878, 0.52359878, 0.52359878, 0.52359878, 0.52359878,
            0.37799655, 0.52359878, 0.1637549 , 0.16468338, 0.16176945,
            0.16858183, 0.14959965, 0.15239127, 0.18689122, 0.21434623,
            0.18582464, 0.26179939, 0.19124819, 0.15759018, 0.15043563],
            [0.52359878, 0.52359878, 0.52359878, 0.22193366, 0.52359878,
            0.52359878, 0.52359878, 0.26179939, 0.21235748, 0.20362447,
            0.20033214, 0.14959965, 0.1752682 , 0.26828088, 0.15050744,
            0.20309237, 0.1613257 , 0.15427717, 0.20216858, 0.1819369 ]]),
            theta[:, ::500])
        os.remove('test.edgelength.dat')
        os.remove('test.neighbor.dat')
        os.remove('test.overall.dat')
        logger.info(f"Finishing test theta2d using {self.test_file_2D}...")
