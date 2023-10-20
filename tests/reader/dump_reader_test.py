# coding = utf-8

import unittest
import numpy as np
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType

from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/reader/sample_test_data"


class TestDumpReader(unittest.TestCase):
    """
    Test class for dump reader
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"
        self.test_file_triclinic = f"{READ_TEST_FILE_PATH}/2d_triclinic.atom"
        self.test_file_xu = f"{READ_TEST_FILE_PATH}/test_xu.dump"
        self.test_file_gsd_3d = f"{READ_TEST_FILE_PATH}/example.gsd"

    def test_dump_reader_lammps_2d(self) -> None:
        """
        Test dump reader works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        read_dump = DumpReader(self.test_file_2d, ndim=2)
        read_dump.read_onefile()
        self.assertEqual(5, read_dump.snapshots.nsnapshots)
        
        for n in range(read_dump.snapshots.nsnapshots):
            snapshot = read_dump.snapshots.snapshots[n]
            self.assertEqual(2, snapshot.particle_type[14])
            self.assertEqual(1, snapshot.particle_type[9999])
            if n==0:
                np.testing.assert_almost_equal(
                    snapshot.positions[14],
                    np.array([18.9739, 24.6161])
                )
            if n==4:
                np.testing.assert_almost_equal(
                    snapshot.positions[14],
                    np.array([18.2545, 24.9591])
                )
            # TODO check others manually

    def test_dump_reader_lammps_3d(self) -> None:
        """
        Test dump reader works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        read_dump = DumpReader(self.test_file_3d, ndim=3)
        read_dump.read_onefile()
        self.assertEqual(1, read_dump.snapshots.nsnapshots)

        snapshot = read_dump.snapshots.snapshots[0]

        self.assertEqual(2, snapshot.particle_type[6389])
        np.testing.assert_almost_equal(
            snapshot.positions[6389],
            np.array([13.6638, 5.51246,0.161101])
        )

        self.assertEqual(1, snapshot.particle_type[2554])
        np.testing.assert_almost_equal(
            snapshot.positions[2554],
            np.array([26.0894, 5.22851, 5.16113])
        )

    def test_dump_reader_lammps_triclinic(self) -> None:
        """
        Test dump reader works properly for triclinic lammps
        """
        logger.info(f"Starting test using {self.test_file_triclinic}...")
        read_dump = DumpReader(self.test_file_triclinic, ndim=2)
        read_dump.read_onefile()
        self.assertEqual(1, read_dump.snapshots.nsnapshots)

        snapshot = read_dump.snapshots.snapshots[0]

        self.assertEqual(1, snapshot.particle_type[46])
        np.testing.assert_almost_equal(
            snapshot.positions[46],
            np.array([9.76023747277, 56.7858319844])
        )

        self.assertEqual(2, snapshot.particle_type[115])
        np.testing.assert_almost_equal(
            snapshot.positions[115],
            np.array([88.4406157094, 86.49953195029])
        )

    def test_dump_reader_lammps_xu(self) -> None:
        """
        Test dump reader works properly for xu lammps
        """
        logger.info(f"Starting test using {self.test_file_xu}...")
        read_dump = DumpReader(self.test_file_xu, ndim=3)
        read_dump.read_onefile()
        self.assertEqual(1, read_dump.snapshots.nsnapshots)

        snapshot = read_dump.snapshots.snapshots[0]

        self.assertEqual(1, snapshot.particle_type[6])
        np.testing.assert_almost_equal(
            snapshot.positions[6],
            np.array([-130.269, 26.7809, -42.8578])
        )

        self.assertEqual(3, snapshot.particle_type[6570])
        np.testing.assert_almost_equal(
            snapshot.positions[6570],
            np.array([-30.0126, -135.635, -55.8192])
        )

    def test_dump_reader_gsd_3d(self) -> None:
        """
        Test dump reader works properly for 3D gsd
        """
        logger.info(f"Starting test using {self.test_file_gsd_3d}...")
        read_gsd = DumpReader(
            self.test_file_gsd_3d,
            ndim=3,
            filetype=DumpFileType.GSD)
        read_gsd.read_onefile()
        self.assertEqual(1, read_gsd.snapshots.nsnapshots)

        snapshot = read_gsd.snapshots.snapshots[0]

        np.testing.assert_almost_equal(
            snapshot.positions[2],
            np.array([0.313574, 0.1959437, 0.5766102])
        )