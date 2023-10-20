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
        self.test_file_lammps_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_lammps_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"
        self.test_file_gsd_3d = f"{READ_TEST_FILE_PATH}/example.gsd"

    def test_dump_reader_lammps_2d(self) -> None:
        """
        Test dump reader works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_lammps_2d}...")
        read_dump = DumpReader(self.test_file_lammps_2d, ndim=2)
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
            #TODO check others manually

    def test_dump_reader_lammps_3d(self) -> None:
        """
        Test dump reader works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_lammps_3d}...")
        read_dump = DumpReader(self.test_file_lammps_3d, ndim=3)
        read_dump.read_onefile()
        self.assertEqual(1, read_dump.snapshots.nsnapshots)

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