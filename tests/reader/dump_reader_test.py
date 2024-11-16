# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.reader.reader_utils import DumpFileType
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


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
        self.test_file_v = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.v.atom"

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
            if n == 0:
                np.testing.assert_almost_equal(snapshot.positions[14], np.array([18.9739, 24.6161]))
            if n == 4:
                np.testing.assert_almost_equal(snapshot.positions[14], np.array([18.2545, 24.9591]))
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
        np.testing.assert_almost_equal(snapshot.positions[6389], np.array([13.6638, 5.51246, 0.161101]))

        self.assertEqual(1, snapshot.particle_type[2554])
        np.testing.assert_almost_equal(snapshot.positions[2554], np.array([26.0894, 5.22851, 5.16113]))

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
        np.testing.assert_almost_equal(snapshot.positions[46], np.array([9.76023747277, 56.7858319844]))

        self.assertEqual(2, snapshot.particle_type[115])
        np.testing.assert_almost_equal(snapshot.positions[115], np.array([88.4406157094, 86.49953195029]))

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
        np.testing.assert_almost_equal(snapshot.positions[6], np.array([-130.269, 26.7809, -42.8578]))

        self.assertEqual(3, snapshot.particle_type[6570])
        np.testing.assert_almost_equal(snapshot.positions[6570], np.array([-30.0126, -135.635, -55.8192]))

    def test_dump_reader_gsd_3d(self) -> None:
        """
        Test dump reader works properly for 3D gsd
        """
        logger.info(f"Starting test using {self.test_file_gsd_3d}...")
        read_gsd = DumpReader(self.test_file_gsd_3d, ndim=3, filetype=DumpFileType.GSD)
        read_gsd.read_onefile()
        self.assertEqual(1, read_gsd.snapshots.nsnapshots)

        snapshot = read_gsd.snapshots.snapshots[0]

        np.testing.assert_almost_equal(snapshot.positions[2], np.array([0.313574, 0.1959437, 0.5766102]))

    def test_dump_reader_lammps_2d(self) -> None:
        """
        Test dump reader works properly for 2D v
        """
        logger.info(f"Starting test using {self.test_file_v}...")
        read_columns = DumpReader(self.test_file_v, ndim=2, filetype=DumpFileType.LAMMPSVECTOR, columnsids=[5])
        read_columns.read_onefile()

        snapshot = read_columns.snapshots.snapshots[0]
        self.assertEqual(10, read_columns.snapshots.nsnapshots)
        np.testing.assert_almost_equal(
            snapshot.positions[20:25],
            np.array(
                [[0.573728], [-0.669961], [0.422935], [-0.465058], [0.857502]],
            ),
        )

        snapshot = read_columns.snapshots.snapshots[5]
        np.testing.assert_almost_equal(
            snapshot.positions[20:25],
            np.array([[0.552083], [-0.69841], [0.696141], [-0.325622], [-0.159647]]),
        )

        read_columns = DumpReader(
            self.test_file_v,
            ndim=2,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[5, 6],
        )
        read_columns.read_onefile()

        snapshot = read_columns.snapshots.snapshots[0]
        self.assertEqual(10, read_columns.snapshots.nsnapshots)
        np.testing.assert_almost_equal(
            snapshot.positions[68:75],
            np.array(
                [
                    [0.990524, 0.770018],
                    [0.1157, 0.049943],
                    [-0.572635, 0.756168],
                    [-0.438618, -0.658475],
                    [-1.63457, -0.659318],
                    [0.495837, 0.429668],
                    [0.493329, 0.111817],
                ]
            ),
        )

        snapshot = read_columns.snapshots.snapshots[9]
        self.assertEqual(10, read_columns.snapshots.nsnapshots)
        np.testing.assert_almost_equal(
            snapshot.positions[70:75],
            np.array(
                [
                    [0.45497, 1.01626],
                    [0.30524, -0.558264],
                    [1.01384, 0.355063],
                    [0.429141, -0.276826],
                    [0.23194, 1.63782],
                ]
            ),
        )
