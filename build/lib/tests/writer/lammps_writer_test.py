# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.utils.logging import get_logger_handle
from PyMatterSim.writer.lammps_writer import (write_data_header,
                                              write_dump_header)

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestDumpWriter(unittest.TestCase):
    """
    Test class for lammps dump writer
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/dump_3D.atom"

    def test_lammps_dump_writer_2d(self) -> None:
        """
        Test dump writer works properly for 2D lammps
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        header_writer = write_dump_header(
            timestep=26000,
            nparticle=10000,
            boxbounds=np.array([[0.0, 100.0], [0.0, 100.0]]),
            addson="",
        )
        with open(self.test_file_2d, "r") as file:
            header_dump = file.readlines()[:9]
        header_dump = "".join(header_dump)

        self.assertEqual(header_dump, header_writer)

    def test_lammps_dump_writer_3d(self) -> None:
        """
        Test dump writer works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        header_writer = write_dump_header(
            timestep=5451182,
            nparticle=6400,
            boxbounds=np.array([[0.000000, 67.304182], [0.000000, 10.768669], [0.000000, 10.768669]]),
            addson="",
        )
        with open(self.test_file_3d, "r") as file:
            header_dump = file.readlines()[:9]
        header_dump = "".join(header_dump)

        self.assertEqual(header_dump, header_writer)


class TestDataWriter(unittest.TestCase):
    """
    Test class for lammps data writer
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_data = f"{READ_TEST_FILE_PATH}/lammps.data"

    def test_lammps_data_writer_3d(self) -> None:
        """
        Test data writer works properly for 3D lammps
        """
        logger.info(f"Starting test using {self.test_file_data}...")
        header_writer = write_data_header(
            nparticle=8100,
            nparticle_type=2,
            boxbounds=[
                [-17.598971, 35.584472],
                [-17.598971, 35.584472],
                [-17.598971, 35.584472],
            ],
        )
        with open(self.test_file_data, "r") as file:
            header_dump = file.readlines()[:11]
        header_dump = "".join(header_dump)

        self.assertEqual(header_dump, header_writer)
