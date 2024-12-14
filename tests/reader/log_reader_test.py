# coding = utf-8

import unittest

import pandas as pd
from PyMatterSim.reader.simulation_log import read_lammpslog
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"

class TestLogReader(unittest.TestCase):
    """
    Test class for log reader
    """ 

    def setUp(self) -> None:
        super().setUp()
        self.test_lammps_log = f"{READ_TEST_FILE_PATH}/lammps_log.dat"
    
    def test_lammps_log_reader(self) -> None:
        """
        Test lammps log reader to get thermodynamic properties
        """ 
        logger.info(f"Start reading the lammps log output {self.test_lammps_log}")
        test_data = read_lammpslog(self.test_lammps_log)
        self.assertEqual(len(test_data), 4)
        self.assertIs(type(test_data[0]), pd.DataFrame)
