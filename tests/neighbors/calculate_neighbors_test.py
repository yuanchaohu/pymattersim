# coding = utf-8

import os
import unittest
import numpy as np
from reader.dump_reader import DumpReader
from neighbors.calculate_neighbors import Nnearests
from neighbors.calculate_neighbors import cutoffneighbors
from neighbors.calculate_neighbors import cutoffneighbors_particletype

from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestNnearests(unittest.TestCase):
    """
    Test class for Nnearests
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_Nnearests_2d(self) -> None:
        """
        Test Nnearests works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        Nnearests(snapshots = readdump.snapshots, N=12, ppp=[1, 1], fnfile='neighborlist.dat')

        with open(r'neighborlist.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[789].split()]
        self.assertEqual(789, item[0])
        self.assertEqual(12, item[1])
        # benchmarking with old code and ovito
        self.assertEqual([4039, 5385, 8949, 3946, 2059, 2819, 2384, 9131, 399, 2629, 691, 3416],
                         item[2:])
        os.remove("neighborlist.dat")

    def test_Nnearests_3d(self) -> None:
        """
        Test Nnearests works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        Nnearests(readdump.snapshots, N=12, ppp=[1, 1, 1], fnfile='neighborlist.dat')

        with open(r'neighborlist.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[456].split()]
        self.assertEqual(456, item[0])
        self.assertEqual(12, item[1])
        # benchmarking with old code and ovito
        self.assertEqual([3801, 1238, 5135, 1973, 1535, 6238, 435, 281, 3327, 2350, 1056, 5263],
                         item[2:])
        os.remove("neighborlist.dat")


class TestCutoffNeighbors(unittest.TestCase):
    """
    Test class for cutoffneighbors
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_cutoffneighbors_2d(self) -> None:
        """
        Test cutoffneighbors works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        cutoffneighbors(readdump.snapshots, r_cut=1.9, ppp=[1, 1], fnfile='neighborlist.dat')

        with open(r'neighborlist.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[789].split()]
        self.assertEqual(789, item[0])
        self.assertEqual(11, item[1])
        # benchmarking with old code and ovito
        self.assertEqual([4039, 5385, 8949, 3946, 2059, 2819, 2384, 9131, 399, 2629, 691],
                         item[2:])
        os.remove("neighborlist.dat")

    def test_cutoffneighbors_3d(self) -> None:
        """
        Test cutoffneighbors works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        cutoffneighbors(readdump.snapshots, r_cut=1.5, ppp=[1, 1, 1], fnfile='neighborlist.dat')

        with open(r'neighborlist.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[456].split()]
        self.assertEqual(456, item[0])
        self.assertEqual(11, item[1])
        # benchmarking with old code and ovito
        self.assertEqual([3801, 1238, 5135, 1973, 1535, 6238, 435, 281, 3327, 2350, 1056],
                         item[2:])
        os.remove("neighborlist.dat")


class TestCutoffNeighbors_particletype(unittest.TestCase):
    """
    Test class for cutoffneighbors_particletype
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_cutoffneighbors_particletype_2d(self) -> None:
        """
        Test cutoffneighbors_particletype works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        cutoffneighbors_particletype(readdump.snapshots,
                                     r_cut=np.array([[1.8, 1.6], [1.6, 1.9]]),
                                     ppp=[1, 1], fnfile='neighborlist.dat')

        with open(r'neighborlist.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[789].split()]
        self.assertEqual(789, item[0])
        self.assertEqual(11, item[1])
        # benchmaking with old code
        self.assertEqual([4039, 5385, 8949, 3946, 2059, 2819, 2384, 9131, 399, 2629, 691],
                         item[2:])
        os.remove("neighborlist.dat")

    def test_cutoffneighbors_particletype_3d(self) -> None:
        """
        Test cutoffneighbors_particletype works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        cutoffneighbors_particletype(readdump.snapshots,
                                     r_cut=np.array([[1.8, 1.6], [1.6, 1.9]]),
                                     ppp=[1, 1, 1], fnfile='neighborlist.dat')

        with open(r'neighborlist.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[456].split()]
        self.assertEqual(456, item[0])
        self.assertEqual(15, item[1])
        # benchmaking with old code
        self.assertEqual([3801, 1238, 5135, 1973, 1535, 6238, 435, 281, 3327, 2350,
                          1056, 5263, 6135, 3126, 2388],
                         item[2:])
        os.remove("neighborlist.dat")
