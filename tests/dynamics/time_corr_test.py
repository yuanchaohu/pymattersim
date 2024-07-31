# coding = utf-8

import os
import unittest
import numpy as np
import pandas as pd
from reader.dump_reader import DumpReader
from dynamic.time_corr import time_correlation
from reader.reader_utils import DumpFileType


from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestTime_Corr(unittest.TestCase):
    """
    Test class for BOO
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

        condition = f"{READ_TEST_FILE_PATH}/condition/condition.csv"
        condition = pd.read_csv(condition).values
        
        tc = time_correlation(readdump.snapshots,condition)
        
        np.testing.assert_almost_equal(np.array([1.        , 0.74966714, 0.74953381, 0.74911929, 0.75031043,
                                                 0.74931458, 0.75004692, 0.75014047, 0.75091767, 0.75020255]),
                                      tc["time_corr"].values[:10])
        
    def test_time_correlation_log(self) -> None:
        """
        Test time_correlation with 2d log
        """
        readdump = DumpReader(self.test_file_log_2D, ndim=2)
        readdump.read_onefile()

        log_condition = f"{READ_TEST_FILE_PATH}/condition/log_condition.csv"
        log_condition = pd.read_csv(log_condition).values
        
        tc = time_correlation(readdump.snapshots,log_condition)
        
        np.testing.assert_almost_equal(np.array([1.        , 0.78268714, 0.74989003, 0.75928282, 0.75614993,
                                                 0.74647565, 0.74691127, 0.78370166, 0.75222268, 0.75946388]),
                                       tc["time_corr"].values[:10])

    def test_time_correlation_velocity(self) -> None:
        """
        Test time_correlation with 2d velocity
        """
        readdump = DumpReader(self.test_file_velocity_2D, ndim=2)
        readdump.read_onefile()
        
        velocity_file = DumpReader(self.test_file_velocity_2D, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
        velocity_file.read_onefile()
        
        velocity =[]
        for snapshot in velocity_file.snapshots.snapshots:
            v = snapshot.positions
            velocity.append(v)
        velocity = np.array(velocity)
        
        tc = time_correlation(readdump.snapshots,velocity)

        np.testing.assert_almost_equal(np.array([ 1.00000000e+00,  7.50523802e-04, -5.47238324e-04, -4.17835029e-03,-1.56200091e-03, 
                                                 -2.93248605e-03,  1.98983850e-03, -2.56205398e-03,-5.84346852e-06, -2.95951304e-03]),
                                       tc["time_corr"].values[:10])