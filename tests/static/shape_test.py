# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.static.shape import gyration_tensor
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestShape(unittest.TestCase):
    """
    Test class for Structure factor
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/3d/3dkaljdump.s.atom"

    def test_gyration_tensor(self) -> None:
        """
        Test gyration_tensor
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        input_3d = DumpReader(self.test_file_3d, ndim=3)
        input_3d.read_onefile()

        gt1 = gyration_tensor(input_3d.snapshots.snapshots[0].positions)
        gt2 = gyration_tensor(input_3d.snapshots.snapshots[9].positions)

        np.testing.assert_almost_equal(
            gt1,
            [
                4.706419206451776,
                0.28228365518691767,
                0.14422817056004078,
                0.00019420654687231966,
                4.459702522482764,
            ],
        )

        np.testing.assert_almost_equal(
            gt2,
            [
                4.710464986859923,
                0.18363043137025592,
                0.15827590448870943,
                0.00010665351384344504,
                4.457229892782171,
            ],
        )

        logger.info(f"Finishing test gyration_tensor using {self.test_file_3d}...")
