# coding = utf-8

import os
import unittest
import numpy as np
from reader.dump_reader import DumpReader
from static.sf import sq
from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestSF(unittest.TestCase):
    """
    Test class for Structure factor
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_unary = f"{READ_TEST_FILE_PATH}/unary.dump"
        self.test_file_binary = f"{READ_TEST_FILE_PATH}/dump_2D.atom"
        self.test_file_ternary = f"{READ_TEST_FILE_PATH}/ternary.dump"
        self.test_file_quarternary = f"{READ_TEST_FILE_PATH}/quarternary.dump"

    def test_SF_unary(self) -> None:
        """
        Test SF works properly for unary system
        """
        logger.info(f"Starting test using {self.test_file_unary}...")
        readdump = DumpReader(self.test_file_unary, ndim=3)
        readdump.read_onefile()
        sq(readdump.snapshots).getresults(outputfile='sq_unary.dat')

        result = np.loadtxt('sq_unary.dat', skiprows=1)
        np.testing.assert_almost_equal([0.036721, 0.098449, 0.0966  , 0.087589, 0.102995, 0.167968,
                                        0.32392 , 0.858877, 2.936611, 1.497239, 1.106665, 0.797491,
                                        0.413192, 0.609405, 1.123366, 2.695403, 1.442278, 0.899162,
                                        1.271647, 0.856863, 0.784563, 0.876665, 0.836927, 1.062343,
                                        0.956568, 0.869305, 1.051758, 1.04188 , 0.848116, 1.233345,
                                        0.745138, 0.564783, 0.903626, 1.2132  , 0.963814, 0.978026,
                                        1.009958, 0.847154, 0.981675, 1.092816, 1.052644, 1.0337  ,
                                        0.999711, 0.960958, 1.066947, 0.947499, 1.126202, 1.231742,
                                        1.017693, 1.088642, 0.908585, 1.078991, 1.01862 , 0.942123,
                                        1.001273, 0.902436, 0.917058, 1.166255, 0.92512 , 0.821388,
                                        1.069375, 1.027532, 0.978479, 0.573045, 0.956662, 1.057167,
                                        0.998963, 0.882252, 0.94872 , 0.959364, 0.969324, 1.179223,
                                        1.150363, 1.047789, 0.964387, 0.922825, 0.986515, 1.065991,
                                        1.125633, 1.045019, 1.065602, 1.053564, 1.095384, 0.852839,
                                        0.9024  , 0.835995, 1.01313 , 0.837291, 1.060634, 1.022636,
                                        0.944078, 0.970188, 1.129809, 1.083366, 1.096643, 1.010755,
                                        0.841045, 0.92336 , 0.99297 , 0.996054, 0.756993],
                                        result[:, 1])
        os.remove("sq_unary.dat")

    def test_SF_binary(self) -> None:
        """
        Test SF works properly for binary system
        """
        logger.info(f"Starting test using {self.test_file_binary}...")
        readdump = DumpReader(self.test_file_binary, ndim=2)
        readdump.read_onefile()
        sq(readdump.snapshots).getresults(outputfile='sq_binary.dat')

        result = np.loadtxt('sq_binary.dat', skiprows=1)
        np.testing.assert_almost_equal([0.038646, 0.008722, 0.01594 , 0.032078, 0.02863 , 0.024956,
                                        0.031567, 0.057656, 0.074003, 0.148906, 0.151015, 0.14022 ,
                                        0.214574, 0.279497, 0.235961, 0.120868, 0.214343, 1.214033,
                                        2.545452, 1.647541, 0.652246, 1.258145, 0.981317, 0.933233,
                                        1.353214, 2.464631, 2.410273, 1.156073, 0.845396, 0.745672,
                                        0.486289, 1.023237, 1.173747, 1.540049, 2.468668, 1.250946,
                                        1.103847, 0.57272 , 0.842023, 0.471667, 0.555763, 0.47876 ,
                                        0.678351, 0.847546, 0.789655, 0.699198, 0.820703, 3.023497,
                                        1.540094, 1.34863 , 0.654181, 0.980287, 1.208656, 1.026452,
                                        1.214759, 0.765653, 0.640338, 0.704302, 0.643513, 0.846699,
                                        0.799153, 0.878344, 0.463479, 1.252944, 0.889977, 0.629526,
                                        1.46365 , 0.947766, 0.811045, 0.959797, 1.135036],
                                        result[:, 1][::5])
        np.testing.assert_almost_equal([0.010016, 0.006038, 0.007666, 0.004854, 0.009   , 0.009333,
                                        0.00685 , 0.014441, 0.023205, 0.03844 , 0.044314, 0.091537,
                                        0.135216, 0.320094, 0.198287, 0.408314, 1.119073, 3.479528,
                                        4.118514, 1.510478, 0.503482, 0.700712, 0.417218, 0.406507,
                                        0.385748, 0.813065, 0.771933, 0.458545, 0.421176, 0.446941,
                                        0.416637, 0.800454, 1.201763, 2.6262  , 4.434829, 2.004551,
                                        1.503136, 0.674482, 0.857498, 0.775989, 0.502021, 0.445643,
                                        0.520797, 0.680262, 0.716059, 0.627144, 0.522774, 1.630218,
                                        1.232619, 1.002104, 0.723407, 1.045293, 1.190059, 0.671507,
                                        1.661634, 0.970724, 0.572636, 0.705335, 0.736121, 0.584916,
                                        0.917069, 0.917691, 0.669146, 0.987906, 1.372867, 1.068688,
                                        1.281827, 0.791265, 0.68322 , 1.056873, 1.155707],
                                        result[:, 2][::5])
        os.remove("sq_binary.dat")

    def test_SF_ternary(self) -> None:
        """
        Test SF works properly for ternary system
        """
        logger.info(f"Starting test using {self.test_file_ternary}...")
        readdump = DumpReader(self.test_file_ternary, ndim=3)
        readdump.read_onefile()
        sq(readdump.snapshots).getresults(outputfile='sq_ternary.dat')

        result = np.loadtxt('sq_ternary.dat', skiprows=1)
        np.testing.assert_almost_equal([0.093943, 0.043634, 0.06753 , 0.131979, 1.701307, 1.393587,
                                        0.685079, 1.03455 , 1.20976 , 1.10204 , 0.806972, 0.713105,
                                        1.082076, 1.033043, 1.069499, 0.99042 , 1.018081, 0.990646,
                                        1.034189, 0.640437, 1.038184, 0.908587, 0.969191, 0.952278,
                                        1.055503, 0.919458, 0.943673, 0.891127, 0.949031, 1.03929 ,
                                        0.946677, 1.053279, 0.950944, 0.928278, 0.995442, 0.94138 ,
                                        0.996732, 0.988594, 1.011948, 1.00888 , 1.022701, 1.029733,
                                        0.957761, 1.049779, 1.144282, 1.014977, 0.985122, 1.070951,
                                        0.97958 , 0.989718, 0.906144, 1.010518],
                                        result[:, 1][::5])
        np.testing.assert_almost_equal([0.076259, 0.076837, 0.106201, 0.13778 , 1.911361, 0.91595 ,
                                        0.604181, 0.987004, 1.272153, 1.099729, 0.823135, 0.740196,
                                        1.144762, 1.025841, 1.05647 , 1.021978, 1.046636, 0.962825,
                                        1.002986, 0.829825, 1.004992, 0.876763, 0.988487, 0.913787,
                                        1.027218, 0.876975, 0.978522, 0.787353, 0.953519, 1.029986,
                                        0.932427, 1.003169, 0.953318, 0.979533, 1.012987, 1.01091 ,
                                        1.009593, 1.00006 , 0.999433, 1.054794, 0.997656, 1.037928,
                                        0.88346 , 1.10382 , 1.129535, 1.102673, 0.955759, 1.087523,
                                        0.95976 , 1.028373, 0.91371 , 0.982419],
                                        result[:, 2][::5])
        np.testing.assert_almost_equal([1.045742, 0.72311 , 1.017411, 0.54045 , 0.919682, 1.360917,
                                        1.005263, 1.083289, 0.945532, 1.016932, 1.000361, 1.039523,
                                        1.071631, 1.148394, 0.994904, 1.079142, 0.98099 , 1.182375,
                                        0.9681  , 1.366004, 0.89032 , 1.177933, 1.018109, 1.014412,
                                        0.999663, 0.980937, 0.996195, 1.119598, 0.983539, 0.975951,
                                        0.99951 , 0.941948, 1.005045, 0.98015 , 1.015898, 0.720947,
                                        0.988979, 0.851227, 0.911214, 0.896719, 0.948397, 0.906889,
                                        0.824817, 0.968911, 1.09985 , 1.149507, 1.082683, 0.830487,
                                        1.096303, 0.980659, 1.227392, 0.913208],
                                        result[:, 3][::5])
        os.remove("sq_ternary.dat")

    def test_SF_quarternary(self) -> None:
        """
        Test SF works properly for quarternary system
        """
        logger.info(f"Starting test using {self.test_file_quarternary}...")
        readdump = DumpReader(self.test_file_quarternary, ndim=3)
        readdump.read_onefile()
        sq(readdump.snapshots).getresults(outputfile='sq_quarternary.dat')

        result = np.loadtxt('sq_quarternary.dat', skiprows=1)
        np.testing.assert_almost_equal([0.066408, 0.203192, 1.116248, 0.708799, 0.93104 , 1.011818,
                                        0.94916 , 1.040496, 1.007997, 0.933482, 1.00845 , 0.958165,
                                        1.029006, 0.996047, 0.98938 , 1.048381, 0.991096, 0.921751,
                                        0.989588, 1.021397, 1.104659],
                                        result[:, 1][::5])
        np.testing.assert_almost_equal([0.082545, 0.238347, 1.086682, 0.704452, 0.918969, 0.995946,
                                        0.954125, 1.049149, 1.021458, 0.941336, 1.009117, 0.957264,
                                        1.034583, 0.98732 , 0.995162, 1.021548, 1.003039, 0.918468,
                                        0.985576, 1.021792, 1.094048],
                                        result[:, 2][::5])
        os.remove("sq_quarternary.dat")
