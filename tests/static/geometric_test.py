# coding = utf-8

import os
import unittest

import numpy as np

from PyMatterSim.neighbors.freud_neighbors import cal_neighbors
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.static.geometric import packing_capability_2d, q8_tetrahedral
from PyMatterSim.utils.logging import get_logger_handle

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

        theta = packing_capability_2d(readdump.snapshots, sigmas=np.array(
            [[1.5, 1.5], [1.5, 1.5]]), neighborfile='test.neighbor.dat')
        np.testing.assert_almost_equal(
            np.array([[0.52359878, 0.52359878, 0.3876635, 0.35924647, 0.52359878,
                       0.52359878, 0.21647106, 0.26179939, 0.19641053, 0.16182063,
                       0.14709638, 0.25303882, 0.18446597, 0.26036996, 0.27088255,
                       0.15627046, 0.2129428, 0.08295416, 0.20201806, 0.17225084],
                      [0.52359878, 0.52359878, 0.52359878, 0.37062511, 0.52359878,
                       0.52359878, 0.52359878, 0.18944879, 0.18344326, 0.19435052,
                       0.16774308, 0.26393261, 0.15548638, 0.26179939, 0.17209369,
                       0.18634357, 0.19271306, 0.07699365, 0.26179939, 0.21607533],
                      [0.20943951, 0.52359878, 0.52359878, 0.4500801, 0.23241144,
                       0.24664117, 0.30358029, 0.26179939, 0.25870591, 0.15297146,
                       0.20439641, 0.18068971, 0.14959965, 0.22721239, 0.14959965,
                       0.12862106, 0.2082548, 0.15399902, 0.18074812, 0.2032766],
                      [0.52359878, 0.52359878, 0.52359878, 0.52359878, 0.52359878,
                       0.37799655, 0.52359878, 0.1637549, 0.16468338, 0.16176945,
                       0.16858183, 0.14959965, 0.15239127, 0.18689122, 0.21434623,
                       0.18582464, 0.26179939, 0.19124819, 0.15759018, 0.15043563],
                      [0.52359878, 0.52359878, 0.52359878, 0.22193366, 0.52359878,
                       0.52359878, 0.52359878, 0.26179939, 0.21235748, 0.20362447,
                       0.20033214, 0.14959965, 0.1752682, 0.26828088, 0.15050744,
                       0.20309237, 0.1613257, 0.15427717, 0.20216858, 0.1819369]]),
            theta[:, ::500])
        os.remove('test.edgelength.dat')
        os.remove('test.neighbor.dat')
        os.remove('test.overall.dat')
        logger.info(f"Finishing test theta2d using {self.test_file_2D}...")


class TestTetrahedral(unittest.TestCase):
    """
    Test class for tetrahedral
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_3D = f"{READ_TEST_FILE_PATH}/test_additional_columns.dump"

    def test_tetrahedral(self) -> None:
        """
        Test tetrahedral works properly for 3D system
        """
        logger.info(f"Starting test tetrahedral using {self.test_file_3D}...")
        readdump = DumpReader(self.test_file_3D, 3)
        readdump.read_onefile()
        tetrahedral = q8_tetrahedral(readdump.snapshots)
        np.testing.assert_almost_equal(
            np.array([[0.79553053, 0.87693344, 0.80694122, 0.8021498, 0.78445042,
                       0.82128644, 0.8267357, 0.82825242, 0.85446152, 0.78723782,
                       0.91187236, 0.95982658, 0.82207, 0.89368492, 0.92176236,
                       0.81613609, 0.88312225, 0.78026667, 0.78700256, 0.9084372,
                       0.76013874, 0.84776699, 0.92610592, 0.86596065, 0.83644922,
                       0.93345373, 0.7424407, 0.85787562, 0.84222525, 0.90899426,
                       0.81283177, 0.88652565, 0.89460427, 0.8145703, 0.83746283,
                       0.81115578, 0.81614046, 0.90185306, 0.87648113, 0.91973314,
                       0.81851143],
                      [0.78023514, 0.78866256, 0.85469208, 0.83678135, 0.78683257,
                       0.80899596, 0.85186268, 0.78381519, 0.86064959, 0.90620284,
                       0.92179804, 0.80363381, 0.80844456, 0.83152024, 0.8870659,
                       0.82157773, 0.92530334, 0.86496757, 0.82228795, 0.86166149,
                       0.74768952, 0.89838833, 0.77258799, 0.94369645, 0.8946883,
                       0.81438987, 0.82619657, 0.85071086, 0.8121586, 0.86868307,
                       0.86678466, 0.84724192, 0.80091934, 0.86722429, 0.85641036,
                       0.89014217, 0.82300112, 0.84661378, 0.85579741, 0.85526542,
                       0.83750587]]),
            tetrahedral[:, ::200])
        logger.info(f"Finishing test tetrahedral using {self.test_file_3D}...")
