import unittest

import numpy as np

from PyMatterSim.reader.gsd_reader_helper import read_gsd_wrapper
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestGsdReaderHelper(unittest.TestCase):
    """
    Test class for gsd reader helper
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_gsd = f"{READ_TEST_FILE_PATH}/example.gsd"

    def test_read_gsd_wrapper(self) -> None:
        """
        Test read gsd wrapper gives expecte results for gsd data
        """
        logger.info(f"Starting test using {self.test_file_gsd}...")
        snapshots = read_gsd_wrapper(self.test_file_gsd, ndim=3)
        self.assertEqual(1, snapshots.nsnapshots)

        snapshot = snapshots.snapshots[0]

        np.testing.assert_almost_equal(snapshot.positions[2], np.array([0.313574, 0.1959437, 0.5766102]))
