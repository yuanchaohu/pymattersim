# coding = utf-8

import unittest
import numpy as np
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from tests.reader.old_dump import readdump

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

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_lammps_2d, ndim=2)
        old_d.read_onefile()

        self.assertEqual(
            read_dump.snapshots.nsnapshots,
            old_d.SnapshotNumber)

        for i, snapshot in enumerate(read_dump.snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxbounds, old_d.Boxbounds[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])

    def test_dump_reader_lammps_3d(self) -> None:
        """
        Test dump reader works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_lammps_3d}...")
        read_dump = DumpReader(self.test_file_lammps_3d, ndim=3)
        read_dump.read_onefile()
        self.assertEqual(1, read_dump.snapshots.nsnapshots)

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_lammps_3d, ndim=3)
        old_d.read_onefile()

        self.assertEqual(
            read_dump.snapshots.nsnapshots,
            old_d.SnapshotNumber)

        for i, snapshot in enumerate(read_dump.snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxbounds, old_d.Boxbounds[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])

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

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_gsd_3d, ndim=3, filetype="gsd")
        old_d.read_onefile()

        self.assertEqual(
            read_gsd.snapshots.nsnapshots,
            old_d.SnapshotNumber)

        for i, snapshot in enumerate(read_gsd.snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxbounds, old_d.Boxbounds[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])
