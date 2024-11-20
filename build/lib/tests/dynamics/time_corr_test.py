# coding = utf-8

import os
import unittest

import numpy as np
import pandas as pd

from PyMatterSim.dynamic.time_corr import time_correlation
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.reader.reader_utils import DumpFileType
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestTime_Corr(unittest.TestCase):
    """
    Test class for Time_Corr
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2D = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.atom"
        self.test_file_log_2D = f"{READ_TEST_FILE_PATH}/2d/2ddump.log.s.atom"
        self.test_file_velocity_2D = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.v.atom"

    def test_time_correlation_linear(self) -> None:
        """
        Test time_correlation with 2d linear
        """
        readdump = DumpReader(self.test_file_2D, ndim=2)
        readdump.read_onefile()

        condition = f"{READ_TEST_FILE_PATH}/condition/condition.npy"
        condition = np.load(condition)

        tc = time_correlation(readdump.snapshots, condition)

        np.testing.assert_almost_equal(
            np.array(
                [
                    1.0,
                    0.75234395,
                    0.75612582,
                    0.75330628,
                    0.75052712,
                    0.75438126,
                    0.74786922,
                    0.75729044,
                    0.74032374,
                    0.77309031,
                ]
            ),
            tc["time_corr"].values[:10],
        )

    def test_time_correlation_log(self) -> None:
        """
        Test time_correlation with 2d log
        """
        readdump = DumpReader(self.test_file_log_2D, ndim=2)
        readdump.read_onefile()

        log_condition = f"{READ_TEST_FILE_PATH}/condition/log_condition.npy"
        log_condition = np.load(log_condition)

        tc = time_correlation(readdump.snapshots, log_condition)

        np.testing.assert_almost_equal(
            np.array(
                [
                    1.0,
                    0.75515639,
                    0.77256705,
                    0.74662384,
                    0.76126827,
                    0.74987555,
                    0.76818978,
                    0.74839397,
                    0.76873851,
                    0.76019467,
                ]
            ),
            tc["time_corr"].values[:10],
        )

    def test_time_correlation_velocity(self) -> None:
        """
        Test time_correlation with 2d velocity
        """
        readdump = DumpReader(self.test_file_velocity_2D, ndim=2)
        readdump.read_onefile()

        velocity_file = DumpReader(
            self.test_file_velocity_2D,
            ndim=2,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[5, 6],
        )
        velocity_file.read_onefile()

        velocity = []
        for snapshot in velocity_file.snapshots.snapshots:
            v = snapshot.positions
            velocity.append(v)
        velocity = np.array(velocity)

        tc = time_correlation(readdump.snapshots, velocity)

        np.testing.assert_almost_equal(
            np.array(
                [
                    1.00000000e00,
                    -2.27614949e-03,
                    4.80819141e-03,
                    -6.89284254e-03,
                    6.39502967e-03,
                    7.75986796e-04,
                    8.35486352e-03,
                    2.03459422e-03,
                    6.53853687e-03,
                    1.01471001e-02,
                ]
            ),
            tc["time_corr"].values[:10],
        )
