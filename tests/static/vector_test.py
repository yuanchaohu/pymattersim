# coding = utf-8

import os
import unittest
import numpy as np
import pandas as pd
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from neighbors.calculate_neighbors import Nnearests
from static.vector import *
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestVector(unittest.TestCase):
    """
    Test class for Structure factor
    """
    #,,phase_quotient,divergence_curl,vibrability,vector_fft_corr
    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.v.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/3d/3ddump.s.v.atom"

        self.input_vp = DumpReader(self.test_file_2d, ndim=2)
        self.input_vp.read_onefile()
        self.input_v = DumpReader(self.test_file_2d, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
        self.input_v.read_onefile()
        
        
    def test_participation_ratio(self) -> None:
        """
        Test participation_ratio
        """
        logger.info(f"Starting test using {self.input_v}...")

        logger.info(f"Starting test using {self.input_v}...")
        v1 = self.input_v.snapshots.snapshots[0].positions
        v2 = self.input_v.snapshots.snapshots[25].positions
        pr1 = participation_ratio(v1)
        pr2 = participation_ratio(v2)
        np.testing.assert_almost_equal(pr1,0.521491080361457)
        np.testing.assert_almost_equal(pr2,0.45703413095905115)
        logger.info(f"Finishing test gyration_tensor using {self.input_v}...")
        
    def test_local_vector_alignment(self) -> None:
        """
        Test local_vector_alignment
        """
        logger.info(f"Starting test using {self.input_v}...")

        Nnearests(self.input_vp.snapshots, N = 6, ppp = np.array([1,1]),  fnfile = 'neighborlist.dat')
        neighborfile='neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions
        v2 = self.input_v.snapshots.snapshots[25].positions
        lva1 = local_vector_alignment(v1,neighborfile)
        lva2 = local_vector_alignment(v2,neighborfile)
        
        np.testing.assert_almost_equal(lva1[:10],np.array([-0.11154581, -0.01004038, -0.01522551,  0.09333871, -0.10149098,
                                                           0.25865722, -0.05835916, -0.28434863, -0.01074069, -0.79158841]))
        np.testing.assert_almost_equal(lva2[:10],np.array([ 0.12815051,  0.0367848 , -0.00344609,  0.08879375, -0.08366008,
                                                           0.17879077, -0.11552321, -0.31021834,  0.09241588, -0.16292794]))
        
        os.remove(neighborfile)
        logger.info(f"Finishing test gyration_tensor using {self.input_v}...")

    def test_phase_quotient(self) -> None:
        """
        Test phase_quotient
        """
        logger.info(f"Starting test using {self.input_v}...")

        Nnearests(self.input_vp.snapshots, N = 6, ppp = np.array([1,1]),  fnfile = 'neighborlist.dat')
        neighborfile='neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions
        v2 = self.input_v.snapshots.snapshots[25].positions
        pq1 = phase_quotient(v1,neighborfile)
        pq2 = phase_quotient(v2,neighborfile)
        
        np.testing.assert_almost_equal(pq1,-0.05594896698423584)
        np.testing.assert_almost_equal(pq2,-0.020400239619083677)
        
        os.remove(neighborfile)
        logger.info(f"Finishing test gyration_tensor using {self.input_v}...")
    
    def test_divergence_curl_2d(self) -> None:
        """
        Test divergence_curl
        """
        logger.info(f"Starting test using {self.input_v}...")
        ppp = np.array([1,1])
        Nnearests(self.input_vp.snapshots, N=6, ppp=ppp, fnfile='neighborlist.dat')
        neighborfile='neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions
        #v2 = self.input_v.snapshots.snapshots[25].positions
        
        divergence1 = divergence_curl(self.input_vp.snapshots,v1,ppp,neighborfile)[:10]
        #divergence2 = divergence_curl(self.input_vp.snapshots,v2,ppp,neighborfile)[:10]
        
        np.testing.assert_almost_equal(divergence1[:10],np.array([-0.29712397, -0.16234507, -0.0749001 ,  0.35241748, -0.082197  ,
                                                                -0.01734587,  0.34023927, -0.22716885,  0.28200287, -0.14452179]))
        #np.testing.assert_almost_equal(dc1[:10],np.array([ 0.12815051,  0.0367848 , -0.00344609,  0.08879375, -0.08366008,
           
        
        os.remove(neighborfile)
        logger.info(f"Finishing test gyration_tensor using {self.input_v}...")
        
        
    def test_divergence_curl_3d(self) -> None:
        """
        Test divergence_curl
        """
        logger.info(f"Starting test using {self.input_v}...")
        self.input_vp = DumpReader(self.test_file_3d, ndim=3)
        self.input_vp.read_onefile()
        self.input_v = DumpReader(self.test_file_3d, ndim=3,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[6,7,8])
        self.input_v.read_onefile()
        ppp = np.array([1,1,1])
        Nnearests(self.input_vp.snapshots, N=6, ppp=ppp, fnfile='neighborlist.dat')
        neighborfile='neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions
        #v2 = self.input_v.snapshots.snapshots[25].positions
        
        divergence,curl = divergence_curl(self.input_vp.snapshots,v1,ppp,neighborfile)[:10]
        #dc2 = divergence_curl(self.input_vp.snapshots,v2,ppp,neighborfile)[:10]
        
        np.testing.assert_almost_equal(divergence[:10],np.array([ 0.34231648,  0.4718925 ,  0.13566191, -0.05720669, -0.2043613 ,
                                                                 -0.44275235,  0.69731253, -0.32565264, -0.35823199, -0.79280378]))
        np.testing.assert_almost_equal(curl[:10], np.array([[ 0.15055975, -0.01814468, -0.25594547],
                                                            [ 0.10550859, -0.27002945, -0.02233084],
                                                            [ 0.11285955,  0.05347471, -0.26716981],
                                                            [-0.32636462,  0.16987166, -0.22669574],
                                                            [-0.00609662, -0.03148098, -0.27336277],
                                                            [ 0.03586442, -0.47330915, -0.02624738],
                                                            [-0.16691416,  0.06553677,  0.06989233],
                                                            [-0.23148536, -0.0965238 ,  0.25047707],
                                                            [ 0.21048035,  0.28944822,  0.43194454],
                                                            [-0.47023573, -0.08404571, -0.37585792]]))
        #np.testing.assert_almost_equal(dc1[:10],np.array([ 0.12815051,  0.0367848 , -0.00344609,  0.08879375, -0.08366008,
           
        
        os.remove(neighborfile)
        logger.info(f"Finishing test gyration_tensor using {self.input_v}...")
        
        