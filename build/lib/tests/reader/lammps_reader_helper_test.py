import unittest

import numpy as np

from PyMatterSim.reader.lammps_reader_helper import (read_additions,
                                                     read_lammps_wrapper)
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestLammpsReaderHelper(unittest.TestCase):
    """
    Test class for lammps reader helper
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"
        self.test_file_triclinic = f"{READ_TEST_FILE_PATH}/2d_triclinic.atom"
        self.test_file_xu = f"{READ_TEST_FILE_PATH}/test_xu.dump"
        self.test_file_additions = f"{READ_TEST_FILE_PATH}/test_additional_columns.dump"

    def test_read_lammps_wrapper_2d(self) -> None:
        """
        Test read lammps wrapper gives expected results for 2D data
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        snapshots = read_lammps_wrapper(self.test_file_2d, ndim=2)
        self.assertEqual(5, snapshots.nsnapshots)

        for n in range(snapshots.nsnapshots):
            snapshot = snapshots.snapshots[n]
            self.assertEqual(2, snapshot.particle_type[14])
            self.assertEqual(1, snapshot.particle_type[9999])
            if n == 0:
                np.testing.assert_almost_equal(snapshot.positions[14], np.array([18.9739, 24.6161]))
            if n == 4:
                np.testing.assert_almost_equal(snapshot.positions[14], np.array([18.2545, 24.9591]))
            # TODO check others manually

    def test_read_lammps_wrapper_3d(self) -> None:
        """
        Test read lammps wrapper gives expected results for 3D data
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        snapshots = read_lammps_wrapper(self.test_file_3d, ndim=3)
        self.assertEqual(1, snapshots.nsnapshots)

        snapshot = snapshots.snapshots[0]

        self.assertEqual(2, snapshot.particle_type[6389])
        np.testing.assert_almost_equal(snapshot.positions[6389], np.array([13.6638, 5.51246, 0.161101]))

        self.assertEqual(1, snapshot.particle_type[2554])
        np.testing.assert_almost_equal(snapshot.positions[2554], np.array([26.0894, 5.22851, 5.16113]))

    def test_read_lammps_wrapper_triclinic(self) -> None:
        """
        Test read lammps wrapper gives expected results for triclinic data
        """
        logger.info(f"Starting test using {self.test_file_triclinic}...")
        snapshots = read_lammps_wrapper(self.test_file_triclinic, ndim=2)
        self.assertEqual(1, snapshots.nsnapshots)

        snapshot = snapshots.snapshots[0]

        self.assertEqual(1, snapshot.particle_type[46])
        np.testing.assert_almost_equal(snapshot.positions[46], np.array([9.76023747277, 56.7858319844]))

        self.assertEqual(2, snapshot.particle_type[115])
        np.testing.assert_almost_equal(snapshot.positions[115], np.array([88.4406157094, 86.49953195029]))

    def test_read_lammps_wrapper_xu(self) -> None:
        """
        Test read lammps wrapper gives expected results for 3D-xu data
        """
        logger.info(f"Starting test using {self.test_file_xu}...")
        snapshots = read_lammps_wrapper(self.test_file_xu, ndim=3)
        self.assertEqual(1, snapshots.nsnapshots)

        snapshot = snapshots.snapshots[0]

        self.assertEqual(1, snapshot.particle_type[6])
        np.testing.assert_almost_equal(snapshot.positions[6], np.array([-130.269, 26.7809, -42.8578]))

        self.assertEqual(3, snapshot.particle_type[6570])
        np.testing.assert_almost_equal(snapshot.positions[6570], np.array([-30.0126, -135.635, -55.8192]))

    def test_read_lammps_wrapper_additions(self) -> None:
        """
        Test read lammps wrapper gives expected results for dump file
        with additional columns
        """
        logger.info(f"Starting test using {self.test_file_additions}...")
        results = read_additions(self.test_file_additions, ncol=5)
        self.assertEqual(2, results.shape[0])
        self.assertEqual(8100, results.shape[1])

        np.testing.assert_almost_equal(results[0, 5241], 3.7185)

        np.testing.assert_almost_equal(results[1, 3586], 8.88975)
