import unittest
import numpy as np
from reader.lammps_reader_helper import read_lammps_wrapper
from tests.reader.old_dump import readdump

from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/reader/sample_test_data"


class TestLammpsReaderHelper(unittest.TestCase):
    """
    Test class for lammps reader helper
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D_test.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/glass.IS.n22.atom"
        self.test_file_triclinic = f"{READ_TEST_FILE_PATH}/2d_triclinic.atom"
        self.test_file_xu = f"{READ_TEST_FILE_PATH}/test_xu.dump"

    def test_read_lammps_wrapper_2d(self) -> None:
        """
        Test read lammps wrapper gives expecte results for 2D data
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        snapshots = read_lammps_wrapper(self.test_file_2d, ndim=2)
        self.assertEqual(5, snapshots.nsnapshots)

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_2d, ndim=2)
        old_d.read_onefile()

        self.assertEqual(snapshots.nsnapshots, old_d.SnapshotNumber)
        for i, snapshot in enumerate(snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])

    def test_read_lammps_wrapper_3d(self) -> None:
        """
        Test read lammps wrapper gives expecte results for 3D data
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        snapshots = read_lammps_wrapper(self.test_file_3d, ndim=3)
        self.assertEqual(1, snapshots.nsnapshots)

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_3d, ndim=3)
        old_d.read_onefile()

        self.assertEqual(snapshots.nsnapshots, old_d.SnapshotNumber)
        for i, snapshot in enumerate(snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])

    def test_read_lammps_wrapper_triclinic(self) -> None:
        """
        Test read lammps wrapper gives expecte results for triclinic data
        """
        logger.info(f"Starting test using {self.test_file_triclinic}...")
        snapshots = read_lammps_wrapper(self.test_file_triclinic, ndim=2)
        self.assertEqual(1, snapshots.nsnapshots)

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_triclinic, ndim=2)
        old_d.read_onefile()

        self.assertEqual(snapshots.nsnapshots, old_d.SnapshotNumber)
        for i, snapshot in enumerate(snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])

    def test_read_lammps_wrapper_xu(self) -> None:
        """
        Test read lammps wrapper gives expecte results for 3D-xu data
        """
        logger.info(f"Starting test using {self.test_file_xu}...")
        snapshots = read_lammps_wrapper(self.test_file_xu, ndim=3)
        self.assertEqual(1, snapshots.nsnapshots)

        # Comparison with old dump, will delete once fully tested
        old_d = readdump(self.test_file_xu, ndim=3)
        old_d.read_onefile()

        self.assertEqual(snapshots.nsnapshots, old_d.SnapshotNumber)
        for i, snapshot in enumerate(snapshots.snapshots):
            self.assertEqual(snapshot.timestep, old_d.TimeStep[i])
            self.assertEqual(snapshot.nparticle, old_d.ParticleNumber[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.particle_type, old_d.ParticleType[i])
            np.testing.assert_almost_equal(
                snapshot.boxlength, old_d.Boxlength[i])
            np.testing.assert_almost_equal(snapshot.hmatrix, old_d.hmatrix[i])
            np.testing.assert_almost_equal(
                snapshot.positions, old_d.Positions[i])
