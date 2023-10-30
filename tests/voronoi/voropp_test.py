# coding = utf-8

import os
import unittest
from reader.dump_reader import DumpReader

from voronoi.voropp import cal_voro, indicehis

from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestVoropp(unittest.TestCase):
    """
    Test class for voropp
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_Voropp_3d(self) -> None:
        """
        Test voropp works properly for 3D lammps
        """
        logger.info(f"Starting voropp test using {self.test_file_3d}...")
        readdump = DumpReader(self.test_file_3d, ndim=3)
        readdump.read_onefile()
        cal_voro(readdump.snapshots, outputfile='dump')

        with open(r'dump.neighbor.dat') as f:
            content = f.readlines()
        item = [int(i) for i in content[1].split()]
        self.assertEqual(766, item[0])
        self.assertEqual(17, item[1])
        # benchmaking with old code
        self.assertEqual([2718, 6125, 2647, 660, 3420, 3510, 5258, 1381, 3421,
                          5145, 4257, 5442, 6005, 5169, 1348, 359, 4092],
                         item[2:])
        os.remove('dump.neighbor.dat')

        with open(r'dump.facearea.dat') as f:
            content = f.readlines()
        item = [float(i) for i in content[1].split()]
        self.assertEqual(766, item[0])
        self.assertEqual(17, item[1])
        # benchmaking with old code
        self.assertEqual([0.43136, 0.478498, 0.410967, 0.489784, 0.290117, 0.455774,
                          0.606689, 0.285958, 0.648885, 0.152756, 0.590822, 0.431302,
                          0.0129808, 0.356881, 0.449561, 0.469773, 0.0171054],
                         item[2:])
        os.remove('dump.facearea.dat')

        with open(r'dump.voroindex.dat') as f:
            content = f.readlines()
        item = [float(i) for i in content[1].split()]
        self.assertEqual(766, item[0])
        # benchmaking with old code and ovito
        self.assertEqual([0, 0, 0, 0, 1, 10, 6],
                         item[1:])

        with open(r'dump.overall.dat') as f:
            content = f.readlines()
        item = [float(i) for i in content[1].split()]
        self.assertEqual(766, item[0])
        self.assertEqual(17, item[1])
        # benchmaking with old code and ovito
        self.assertEqual(1.39299, item[2])
        self.assertEqual(6.57921, item[3])
        os.remove('dump.overall.dat')

        indicehis('dump.voroindex.dat', outputfile='voroindex_frction.dat')
        with open(r'voroindex_frction.dat') as f:
            content = f.readlines()
        self.assertEqual([0, 2, 8, 2],
                         [int(i) for i in content[1].split()[0:-1]])
        self.assertEqual(0.059531, float(content[1].split()[-1]))
        os.remove('dump.voroindex.dat')
        os.remove('voroindex_frction.dat')

        logger.info(f"Finishing voropp test using {self.test_file_3d}")
