# coding = utf-8

import os
import unittest

import numpy as np

from PyMatterSim.neighbors.calculate_neighbors import Nnearests
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.reader.reader_utils import DumpFileType
from PyMatterSim.static.nematic import NematicOrder
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestNematic(unittest.TestCase):
    """
    Test class for Structure factor
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/2d/dump.nematic.atom"

    def test_NematicOrder_no_nei(self) -> None:
        """
        Test tensor on Nematic
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        input_x = DumpReader(self.test_file_2d, ndim=2)
        input_x.read_onefile()
        input_or = DumpReader(
            self.test_file_2d,
            ndim=2,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[5, 6],
        )
        input_or.read_onefile()

        Nematic = NematicOrder(input_or.snapshots, input_x.snapshots)

        t = Nematic.tensor(outputfile="test")
        np.testing.assert_almost_equal(
            t[0, :10],
            np.array(
                [
                    0.99999981,
                    0.99999927,
                    0.99999963,
                    1.00000003,
                    1.00000075,
                    0.99999974,
                    1.00000073,
                    1.00000019,
                    1.00000002,
                    0.99999957,
                ]
            ),
        )

        sc = Nematic.spatial_corr()
        np.testing.assert_almost_equal(
            (sc["gA"] / sc["gr"])[-10:].values,
            np.array(
                [
                    0.0040589,
                    0.0002056,
                    0.0059644,
                    0.0016238,
                    0.0026199,
                    0.0066682,
                    -0.0003822,
                    -0.0021303,
                    0.0052707,
                    -0.002656,
                ]
            ),
        )

        tc = Nematic.time_corr()
        np.testing.assert_almost_equal(
            tc.values[:, 1],
            np.array([1.0, 0.50650211, 0.40303197, 0.3336441, 0.28467284]),
        )

        os.remove("test.QIJ_raw.npy")
        os.remove("test.Qtrace.npy")
        logger.info(f"Finishing test Nematic using {self.test_file_2d}...")

    def test_NematicOrder_w_nei(self) -> None:
        """
        Test tensor on Nematic
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        input_x = DumpReader(self.test_file_2d, ndim=2)
        input_x.read_onefile()
        input_or = DumpReader(
            self.test_file_2d,
            ndim=2,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[5, 6],
        )
        input_or.read_onefile()
        Nnearests(input_x.snapshots, 6, np.array([1, 1]))
        Nematic = NematicOrder(input_or.snapshots, input_x.snapshots)

        t = Nematic.tensor(neighborfile="neighborlist.dat", outputfile="test")
        np.testing.assert_almost_equal(
            t[0, :10],
            np.array(
                [
                    0.57335225,
                    0.3403988,
                    0.63966623,
                    0.43469572,
                    0.28913239,
                    0.11301948,
                    0.53520132,
                    0.40710082,
                    0.25573131,
                    0.29231007,
                ]
            ),
        )

        sc = Nematic.spatial_corr()
        np.testing.assert_almost_equal(
            (sc["gA"] / sc["gr"])[-10:].values,
            np.array(
                [
                    0.00045802,
                    -0.00010462,
                    0.00106089,
                    -0.00066537,
                    0.00048919,
                    0.00088129,
                    0.00048185,
                    0.00164163,
                    0.00181375,
                    -0.00106318,
                ]
            ),
        )

        tc = Nematic.time_corr()
        np.testing.assert_almost_equal(
            tc.values[:, 1],
            np.array([1.0, 0.7083274, 0.61410368, 0.54925373, 0.46704894]),
        )

        os.remove("test.QIJ_cg.npy")
        os.remove("test.Qtrace.npy")
        os.remove("neighborlist.dat")
        logger.info(f"Finishing test Nematic using {self.test_file_2d}...")
