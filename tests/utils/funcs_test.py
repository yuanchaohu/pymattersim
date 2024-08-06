# coding = utf-8

import unittest
import numpy as np
import pandas as pd
from utils.funcs import kronecker,nidealfac,areafac,alpha2factor,moment_of_inertia,Wignerindex,grid_gaussian,Legendre_polynomials
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class Test_funcs(unittest.TestCase):
    """
    Test functions
    """

    def setUp(self) -> None:
        super().setUp()

    def test_kronecker(self) -> None:
        """
        Test kronecker function
        """
        k1 = kronecker(2,1)
        k2 = kronecker(3,3)
        self.assertEqual(k1,0)
        self.assertEqual(k2,1)
        
    def test_nidealfac(self) -> None:
        """
        Test nidealfac function
        """
        n1 = nidealfac(2)
        n2 = nidealfac(3)
        self.assertEqual(n1,1.0)
        self.assertEqual(n2,4.0 / 3)

    def test_areafac(self) -> None:
        """
        Test areafac function
        """
        a1 = areafac(2)
        a2 = areafac(3)
        self.assertEqual(a1,2.0)
        self.assertEqual(a2,4.0)


    def test_alpha2factor(self) -> None:
        """
        Test alpha2factor function
        """
        a1 = alpha2factor(2)
        a2 = alpha2factor(3)
        self.assertEqual(a1,1/2.0)
        self.assertEqual(a2,3/5.0)
        
    def test_moment_of_inertia(self) -> None:
        """
        Test moment_of_inertia function
        """
        position = f"{READ_TEST_FILE_PATH}/condition/condition.csv"
        position = pd.read_csv(position).values[:,:3]
        moi = moment_of_inertia(position)
        np.testing.assert_almost_equal(moi,np.array([ 0.59060436,  0.63124302,  0.58061571, -0.20567031, -0.2288654 ,-0.1977408 ]))

    def test_Wignerindex(self) -> None:
        """
        Test Wignerindex function
        """
        w = Wignerindex(6)
        np.testing.assert_almost_equal(w[:,3][:10], np.array([0.0511827251162099, -0.0957541107531435, 0.129114816600119,
                                                             -0.141438195120055, 0.129114816600119, -0.0957541107531435,
                                                             0.0511827251162099, -0.0957541107531435, 0.127956812790525,
                                                             -0.106078646340041]))
    
    def test_grid_gaussian(self) -> None:
        """
        Test grid_gaussian function
        """
        
        a = np.arange(10)
        a = grid_gaussian(a,2.0)
        np.testing.assert_almost_equal(a,np.array([3.97887358e-02, 3.51134361e-02, 2.41330882e-02, 1.29175112e-02, 5.38481983e-03,
                                                   1.74819504e-03, 4.42012928e-04, 8.70375061e-05, 1.33476339e-05, 1.59414753e-06]))
        
    def test_Legendre_polynomials(self) -> None:
        """
        Test Legendre_polynomials function
        """
        Lp = Legendre_polynomials(6,3)
        self.assertEqual(Lp,53.5)
