# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.utils import geometry
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestGeometry(unittest.TestCase):
    """
    Test class for geometry
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"

    def test_triangle_area(self) -> None:
        """
        Test triangle_area works properly for 2D lammps
        """
        logger.info(f"Starting test triangle_area using {self.test_file_2d}...")
        readdump = DumpReader(self.test_file_2d, ndim=2)
        readdump.read_onefile()
        result = geometry.triangle_area(
            positions=readdump.snapshots.snapshots[0].positions[:3],
            hmatrix=readdump.snapshots.snapshots[0].hmatrix,
            ppp=[1, 1],
        )
        np.testing.assert_almost_equal(591.3143460374572, result)

    def test_triangle_angle(self) -> None:
        """
        Test triangle_angle works properly for 2D lammps
        """
        logger.info("Starting test triangle_angle...")
        result = geometry.triangle_angle(a=3, b=4, c=5)
        np.testing.assert_almost_equal(1.5707963267948966, result)

    def test_lines_intersection(self) -> None:
        """
        Test lines_intersection works properly for 2D lammps
        """
        logger.info("Starting test lines_intersection...")
        result = geometry.lines_intersection(
            P1=np.array([0, 0]),
            P2=np.array([1, 1]),
            P3=np.array([1, 0]),
            P4=np.array([0, 1]),
        )
        np.testing.assert_almost_equal(np.array([0.5, 0.5]), result)

    def test_LineWithinSquare(self) -> None:
        """
        Test LineWithinSquare works properly for 2D lammps
        """
        logger.info("Starting test LineWithinSquare...")
        result = geometry.LineWithinSquare(
            P1=np.array([0, 0]),
            P2=np.array([1, 0]),
            P3=np.array([1, 1]),
            P4=np.array([0, 1]),
            R0=np.array([0.5, 0.5]),
            vector=np.array([-1.0, 0.0]),
        )
        np.testing.assert_almost_equal(np.array([1.0, 0.5]), result)
