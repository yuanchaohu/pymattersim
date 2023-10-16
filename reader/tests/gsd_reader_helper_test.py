import unittest
import numpy as np
from reader.gsd_reader_helper import read_gsd_wrapper
from reader.tests.old_dump import readdump

from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "reader/tests/sample_test_data"


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
        self.assertEqual(1, snapshots.snapshots_number)

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_gsd, ndim=3, filetype='gsd')
        old_d.read_onefile()

        self.assertEqual(snapshots.snapshots_number, old_d.SnapshotNumber)
        for i, snapshot in enumerate(snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.particle_number, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])
