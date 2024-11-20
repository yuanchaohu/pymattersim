# coding = utf-8

import os
import unittest

from PyMatterSim.neighbors.freud_neighbors import cal_neighbors
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestFreudNeighbors(unittest.TestCase):
    """
    Test class for freud_neighbors
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_FreudNeighbors_2d(self) -> None:
        """
        Test FreudNeighbors works properly for 2D lammps
        """
        logger.info(f"Starting freud_neighbors test using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        cal_neighbors(readdump.snapshots, outputfile="dump")

        with open(r"dump.neighbor.dat") as f:
            content = f.readlines()
        item = [int(i) for i in content[789].split()]
        self.assertEqual(789, item[0])
        self.assertEqual(5, item[1])
        # benchmaking with old code
        self.assertEqual([3946, 4039, 5385, 8949, 9131], item[2:])
        os.remove("dump.neighbor.dat")

        with open(r"dump.edgelength.dat") as f:
            content = f.readlines()
        item = [float(i) for i in content[789].split()]
        self.assertEqual(789.0, item[0])
        self.assertEqual(5.0, item[1])
        # benchmaking with old code
        self.assertEqual([0.967596, 0.849745, 1.078595, 0.82854, 0.077368], item[2:])
        os.remove("dump.edgelength.dat")

        with open(r"dump.overall.dat") as f:
            content = f.readlines()
        item = [float(i) for i in content[789].split()]
        self.assertEqual(789.0, item[0])
        self.assertEqual(5.0, item[1])
        # benchmaking with old code
        self.assertEqual(0.90653, item[2])
        os.remove("dump.overall.dat")

        logger.info(f"Finishing freud_neighbors test using {self.test_file_2d}")

    def test_FreudNeighbors_3d(self) -> None:
        """
        Test FreudNeighbors works properly for 3D lammps
        """
        logger.info(f"Starting freud_neighbors test using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        cal_neighbors(readdump.snapshots, outputfile="dump")

        with open(r"dump.neighbor.dat") as f:
            content = f.readlines()
        item = [int(i) for i in content[456].split()]
        self.assertEqual(456, item[0])
        self.assertEqual(14, item[1])
        # benchmaking with old code and ovito
        self.assertEqual(
            [
                281,
                435,
                1056,
                1238,
                1535,
                1973,
                2350,
                3126,
                3327,
                3801,
                5135,
                5263,
                6135,
                6238,
            ],
            item[2:],
        )
        os.remove("dump.neighbor.dat")

        with open(r"dump.facearea.dat") as f:
            content = f.readlines()
        item = [float(i) for i in content[456].split()]
        self.assertEqual(456.0, item[0])
        self.assertEqual(14.0, item[1])
        # benchmaking with old code
        self.assertEqual(
            [
                0.43709,
                0.496726,
                0.239007,
                0.560916,
                0.816323,
                0.673743,
                0.323476,
                0.085478,
                0.401778,
                0.757736,
                0.567622,
                0.014058,
                0.096788,
                0.600128,
            ],
            item[2:],
        )
        os.remove("dump.facearea.dat")

        with open(r"dump.overall.dat") as f:
            content = f.readlines()
        item = [float(i) for i in content[456].split()]
        self.assertEqual(456.0, item[0])
        self.assertEqual(14.0, item[1])
        # benchmaking with old code and ovito
        self.assertEqual(1.178814, item[2])
        os.remove("dump.overall.dat")

        logger.info(f"Finishing freud_neighbors test using {self.test_file_3d}")
