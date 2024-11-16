# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.neighbors.read_neighbors import read_neighbors
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestReadNeighbors(unittest.TestCase):
    """
    Test class for read_neighbors
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_neighborlist = f"{READ_TEST_FILE_PATH}/neighborlist.dat"

    def test_read_neighborlist(self) -> None:
        """
        Test read_neighprop works properly
        """
        logger.info(f"Starting test using {self.test_file_neighborlist}...")
        f = open(self.test_file_neighborlist, "r")
        neighborprop = read_neighbors(f, nparticle=8100, Nmax=200)
        f.close()

        self.assertEqual(10, neighborprop[1789][0])
        np.testing.assert_almost_equal(
            [6816, 7845, 769, 2934, 6428, 3829, 2071, 3712, 1433, 4440],
            neighborprop[1789][1:11],
        )
