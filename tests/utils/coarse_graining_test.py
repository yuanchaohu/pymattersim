# coding = utf-8

import os
import unittest

import numpy as np

from PyMatterSim.neighbors.calculate_neighbors import Nnearests
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.reader.reader_utils import DumpFileType
from PyMatterSim.utils.coarse_graining import gaussian_blurring, spatial_average
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class Test_Coarse_Graining(unittest.TestCase):
    """
    Test Coarse_Graining file
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/3d/3dkaljdump.s.atom"
        self.test_file_2d_v = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.v.atom"

        self.condition = f"{READ_TEST_FILE_PATH}/condition/condition.npy"
        self.condition = np.load(self.condition)

    def test_spatial_average(self) -> None:
        """
        Test spatial_average
        """
        logger.info(f"Starting test spatial_average using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()

        Nnearests(readdump.snapshots, N=6, ppp=np.array([1, 1]), fnfile="neighborlist.dat")
        neighborfile = "neighborlist.dat"
        sa = spatial_average(self.condition, neighborfile)
        os.remove(neighborfile)

        np.testing.assert_almost_equal(
            np.array(
                [
                    0.57277224,
                    0.53034253,
                    0.63984439,
                    0.49477918,
                    0.75474255,
                    0.7225598,
                    0.59054183,
                    0.54235565,
                    0.59980761,
                    0.5684533,
                ]
            ),
            sa[:, 0],
        )

    def test_gaussian_blurring_2d(self) -> None:
        """
        Test gaussian_blurring works properly for 2D lammps
        """
        logger.info(f"Starting test gaussian_blurring using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        ppp = ppp = np.array([1, 1])
        ngrids = [20, 20]
        gb_2d = gaussian_blurring(readdump.snapshots, self.condition, ngrids, 2.0, ppp)[1]

        np.testing.assert_almost_equal(
            np.array(
                [
                    0.58704447,
                    0.56470494,
                    0.56153822,
                    0.57766662,
                    0.60697736,
                    0.61962383,
                    0.59228598,
                    0.54755265,
                    0.51933923,
                    0.52482187,
                ]
            ),
            gb_2d[0][:10] / np.sqrt(8 * np.pi),
        )

    def test_gaussian_blurring_3d(self) -> None:
        """
        Test gaussian_blurring works properly for 3D lammps
        """
        logger.info(f"Starting test gaussian_blurring using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        ppp = np.array([1, 1, 1])
        ngrids = [20, 20, 20]
        gb_3d = gaussian_blurring(readdump.snapshots, self.condition, ngrids, 2.0, ppp)[1]

        np.testing.assert_almost_equal(
            np.array(
                (
                    [
                        2.89425578,
                        2.85272144,
                        2.81284045,
                        2.77540638,
                        2.7436408,
                        2.72687641,
                        2.71807427,
                        2.71660162,
                        2.72853308,
                        2.7535948,
                    ]
                )
            ),
            gb_3d[0][:10] / np.sqrt(8 * np.pi),
        )

    def test_gaussian_blurring_2d_vector(self) -> None:
        """
        Test gaussian_blurring works properly for 2D vector
        """
        logger.info(f"Starting test gaussian_blurring using {self.test_file_2d_v}...")
        input_x = DumpReader(self.test_file_2d_v, ndim=2)
        input_x.read_onefile()
        input_v = DumpReader(
            self.test_file_2d_v,
            ndim=2,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[5, 6],
        )
        input_v.read_onefile()
        ppp = ppp = np.array([1, 1])
        ngrids = np.array([20, 20])
        condition = []
        for snapshot in input_v.snapshots.snapshots:
            condition.append(snapshot.positions)
        condition = np.array(condition)
        gb_2d_v = gaussian_blurring(input_x.snapshots, condition, ngrids, 2.0, ppp)[1]

        np.testing.assert_almost_equal(
            np.array(
                [
                    [0.0770776, 0.07731522],
                    [0.0207033, 0.08065932],
                    [0.01881688, 0.04155422],
                    [0.07705881, 0.00741947],
                    [0.10508042, -0.01351502],
                    [0.06591994, -0.03737106],
                    [0.00814159, -0.04600918],
                    [-0.0398673, -0.0244735],
                    [-0.07096676, 0.00590275],
                    [-0.07054937, 0.0355927],
                ]
            ),
            gb_2d_v[0, :10] / np.sqrt(8 * np.pi),
        )
