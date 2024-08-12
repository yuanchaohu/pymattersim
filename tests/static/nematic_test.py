# coding = utf-8

import os
import unittest
import numpy as np
import pandas as pd
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from static.nematic import NematicOrder
from utils.logging import get_logger_handle
from neighbors.calculate_neighbors import Nnearests

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestNematic(unittest.TestCase):
    """
    Test class for Structure factor
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/2d/dump.f.n1.atom"

    def test_NematicOrder_no_nei(self) -> None:
        """
        Test tensor on Nematic
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        input_x = DumpReader(self.test_file_2d, ndim=2)
        input_x.read_onefile()
        input_or = DumpReader(self.test_file_2d, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
        input_or.read_onefile()

        Nematic = NematicOrder(input_or.snapshots,input_x.snapshots)
        t = Nematic.tensor(outputfile='test')
        
        sc = Nematic.spatial_corr()
        
        tc = Nematic.time_corr()
        
        np.testing.assert_almost_equal(t[0,:10],np.array([0.99999981, 0.99999927, 0.99999963, 1.00000003, 1.00000075,
                                                          0.99999974, 1.00000073, 1.00000019, 1.00000002, 0.99999957])
                                       )
        np.testing.assert_almost_equal((sc["gA"] / sc["gr"])[-10:].values,
                                       np.array([-0.0022156 , -0.00265594, -0.00152945, -0.00094876, -0.0002543 ,
                                                 -0.00058289, -0.00043138, -0.00198928, -0.00216833, -0.00235915])
                                       )
        np.testing.assert_almost_equal(tc[:10],np.array([[0.00000000e+00, 1.00000000e+00],
                                                        [1.00000000e+02, 5.33281516e-01],
                                                        [2.00000000e+02, 4.33875953e-01],
                                                        [3.00000000e+02, 3.79339545e-01],
                                                        [4.00000000e+02, 3.44007112e-01],
                                                        [5.00000000e+02, 3.15871540e-01],
                                                        [6.00000000e+02, 2.98257213e-01],
                                                        [7.00000000e+02, 2.81872719e-01],
                                                        [8.00000000e+02, 2.67315027e-01],
                                                        [9.00000000e+02, 2.56310122e-01]]))
        
        os.remove("test.QIJ_raw.npy")
        os.remove("test.Qtrace.npy")
        logger.info(f"Finishing test Nematic using {self.test_file_2d}...")


    def test_NematicOrder_w_nei(self) -> None:
        """
        Test tensor on Nematic
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        input_x = DumpReader(self.test_file_2d, ndim=2)
        input_x.read_onefile()
        input_or = DumpReader(self.test_file_2d, ndim=2, filetype=DumpFileType.LAMMPSVECTOR, columnsids=[5,6])
        input_or.read_onefile()
        Nnearests(input_x.snapshots,6,np.array([1,1]))
        Nematic = NematicOrder(input_or.snapshots,input_x.snapshots)
        t = Nematic.tensor(neighborfile="neighborlist.dat",outputfile='test')
        
        sc = Nematic.spatial_corr()
        
        tc = Nematic.time_corr()
        
        np.testing.assert_almost_equal(t[0,:10],np.array([0.57335225, 0.3403988 , 0.63966623, 0.43469572, 0.28913239,
                                                          0.11301948, 0.53520132, 0.40710082, 0.25573131, 0.29231007])

                                       )
        np.testing.assert_almost_equal((sc["gA"] / sc["gr"])[-10:].values,
                                       np.array([-0.0018792 , -0.00264097, -0.00235424, -0.0017951 , -0.00232703,
                                                 -0.00259306, -0.00286805, -0.00261975, -0.00224098, -0.00233309])
                                       )
        np.testing.assert_almost_equal(tc[:10],np.array([[0.00000000e+00, 1.00000000e+00],
                                                        [1.00000000e+02, 7.29692860e-01],
                                                        [2.00000000e+02, 6.53624809e-01],
                                                        [3.00000000e+02, 6.03957602e-01],
                                                        [4.00000000e+02, 5.68729538e-01],
                                                        [5.00000000e+02, 5.41106416e-01],
                                                        [6.00000000e+02, 5.20367864e-01],
                                                        [7.00000000e+02, 5.03631321e-01],
                                                        [8.00000000e+02, 4.87210040e-01],
                                                        [9.00000000e+02, 4.74962864e-01]]))
        
        os.remove("test.QIJ_cg.npy")
        os.remove("test.Qtrace.npy")
        os.remove("neighborlist.dat")
        
        logger.info(f"Finishing test Nematic using {self.test_file_2d}...")
        
        