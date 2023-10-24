# coding = utf-8

import unittest
import numpy as np
from utils.pbc import remove_pbc
from reader.dump_reader import DumpReader

from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"

class TestPBC(unittest.TestCase):
    """
    Test class for PBC
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_pbc_2d(self) -> None:
        """
        Test pbc works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        positions = readdump.snapshots.snapshots[0].positions
        hmatrix = readdump.snapshots.snapshots[0].hmatrix
        boxlength = readdump.snapshots.snapshots[0].boxlength
        RJI = positions[1:] - positions[0]
        RJI_removed = remove_pbc(RJI = RJI, hmatrix = hmatrix, ppp = [1, 1])

        np.testing.assert_almost_equal(np.array([-41.7933, -14.9538]), RJI_removed[10])
        np.testing.assert_almost_equal(np.array([10.3199, 21.3197]), RJI_removed[100])


    def test_pbc_3d(self) -> None:
        """
        Test pbc works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        positions = readdump.snapshots.snapshots[0].positions
        hmatrix = readdump.snapshots.snapshots[0].hmatrix
        boxlength = readdump.snapshots.snapshots[0].boxlength
        RJI = positions[1:] - positions[0]
        RJI_removed = remove_pbc(RJI = RJI, hmatrix = hmatrix, ppp = [1, 1, 1])

        np.testing.assert_almost_equal(np.array([-4.2156,  4.564109, -0.03429]), RJI_removed[10])
        np.testing.assert_almost_equal(np.array([-10.7619,  -5.11166,  -1.70274]), RJI_removed[100])
