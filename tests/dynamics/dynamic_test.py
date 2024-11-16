# coding = utf-8
import os
import unittest

import numpy as np
import pandas as pd

from PyMatterSim.dynamic.dynamics import Dynamics, LogDynamics
from PyMatterSim.neighbors.calculate_neighbors import Nnearests
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)
READ_TEST_FILE_PATH = "tests/sample_test_data"


'''
file name:
2d:
/2d/2ddump.log.s.atom
/2d/2ddump.log.u.atom
/2d/2ddump.s.atom
/2d/2ddump.u.atom

3d:
/3d/3dkaljdump.log.s.atom
/3d/3dkaljdump.log.u.atom
/3d/3dkaljdump.s.atom
/3d/3dkaljdump.u.atom

tests:
2d: w/ ot w/o condition
xu, x, slow, no neighbor
xu, x, slow, neighbor
xu, x, fast, no neighbor
xu, x, slow, neighbor
x, slow, no neighbor
x, slow, neighbor
x, fast, no neighbor
x, slow, neighbor
xu, slow, no neighbor
xu, slow, neighbor
xu, fast, no neighbor
xu, slow, neighbor

3d:
xu, x, slow, no neighbor
xu, x, fast, no neighbor
x, slow, no neighbor
x, fast, no neighbor
xu, slow, no neighbor
xu, fast, no neighbor

log 2d: same kind of tests of 2d but w/o condition
log 3d:same kind of tests of 3d but w/o condition

'''


class TestDynamics(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # 2d files
        self.test_file_2d_x = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.atom"
        self.dump_2d_x = DumpReader(self.test_file_2d_x, ndim=2)
        self.dump_2d_x.read_onefile()

        self.test_file_2d_xu = f"{READ_TEST_FILE_PATH}/2d/2ddump.u.atom"
        self.dump_2d_xu = DumpReader(self.test_file_2d_xu, ndim=2)
        self.dump_2d_xu.read_onefile()

        self.test_file_2d_log_x = f"{READ_TEST_FILE_PATH}/2d/2ddump.log.s.atom"
        self.dump_2d_log_x = DumpReader(self.test_file_2d_log_x, ndim=2)
        self.dump_2d_log_x.read_onefile()

        self.test_file_2d_log_xu = f"{READ_TEST_FILE_PATH}/2d/2ddump.log.u.atom"
        self.dump_2d_log_xu = DumpReader(self.test_file_2d_log_xu, ndim=2)
        self.dump_2d_log_xu.read_onefile()

        # 3d files
        self.test_file_3d_x = f"{READ_TEST_FILE_PATH}/3d/3dkaljdump.s.atom"
        self.dump_3d_x = DumpReader(self.test_file_3d_x, ndim=3)
        self.dump_3d_x.read_onefile()

        self.test_file_3d_xu = f"{READ_TEST_FILE_PATH}/3d/3dkaljdump.u.atom"
        self.dump_3d_xu = DumpReader(self.test_file_3d_xu, ndim=3)
        self.dump_3d_xu.read_onefile()

        self.test_file_3d_log_x = f"{READ_TEST_FILE_PATH}/3d/3dkaljdump.log.s.atom"
        self.dump_3d_log_x = DumpReader(self.test_file_3d_log_x, ndim=3)
        self.dump_3d_log_x.read_onefile()

        self.test_file_3d_log_xu = f"{READ_TEST_FILE_PATH}/3d/3dkaljdump.log.u.atom"
        self.dump_3d_log_xu = DumpReader(self.test_file_3d_log_xu, ndim=3)
        self.dump_3d_log_xu.read_onefile()

    def test_Dynamics_2d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_x, self.test_file_2d_xu}...")
        xu_snapshots = self.dump_2d_xu.snapshots
        x_snapshots = self.dump_2d_x.snapshots
        ppp = np.array([0, 0])
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=ppp)
        neighborfile = 'neighborlist.dat'
        t = 10
        qrange = 10.0

        condition = []
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type == 1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)

        # xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=None,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.36702354e-01,
                                                  7.57777778e-01,
                                                  4.26750617e+00,
                                                  7.21997640e-02,
                                                  4.90984204e-01],
                                                 [2.00000000e+01,
                                                  4.36370666e-01,
                                                  6.51000000e-01,
                                                  3.62550000e+00,
                                                  1.06063946e-01,
                                                  6.70055277e-01],
                                                 [3.00000000e+01,
                                                  3.57860739e-01,
                                                  5.68285714e-01,
                                                  3.70963265e+00,
                                                  1.43668541e-01,
                                                  6.61990611e-01],
                                                 [4.00000000e+01,
                                                  2.62785498e-01,
                                                  4.73666667e-01,
                                                  4.28988889e+00,
                                                  1.79493505e-01,
                                                  5.96811308e-01],
                                                 [5.00000000e+01,
                                                  1.99583267e-01,
                                                  4.12000000e-01,
                                                  1.69720000e+00,
                                                  2.11959966e-01,
                                                  5.45471374e-01],
                                                 [6.00000000e+01,
                                                  1.71309759e-01,
                                                  3.67000000e-01,
                                                  6.96500000e-01,
                                                  2.47070228e-01,
                                                  4.93235540e-01],
                                                 [7.00000000e+01,
                                                  1.13552234e-01,
                                                  3.22000000e-01,
                                                  3.23266667e+00,
                                                  2.87827169e-01,
                                                  4.06790205e-01],
                                                 [8.00000000e+01,
                                                  9.46026864e-02,
                                                  2.97500000e-01,
                                                  6.50250000e-01,
                                                  3.11575071e-01,
                                                  3.38406147e-01],
                                                 [9.00000000e+01,
                                                  5.68581742e-02,
                                                  2.50000000e-01,
                                                  0.00000000e+00,
                                                  3.77274037e-01,
                                                  1.97250757e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 3.519954],
                                                [0.43531205, 2.86888997],
                                                [0.65296808, 0.95457744],
                                                [0.8706241, 0.7165144],
                                                [1.08828013, 0.56184561],
                                                [1.30593615, 0.36182742],
                                                [1.52359218, 0.26146693],
                                                [1.7412482, 0.22490118],
                                                [1.95890423, 0.23431208],
                                                [2.17656025, 0.16052936]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.39399520e-01,
                                                  7.61025641e-01,
                                                  2.53504274e+00,
                                                  7.00196227e-02,
                                                  4.05403038e-01],
                                                 [2.00000000e+01,
                                                  4.37400251e-01,
                                                  6.55576923e-01,
                                                  2.08862981e+00,
                                                  1.01804767e-01,
                                                  5.71507971e-01],
                                                 [3.00000000e+01,
                                                  3.53978066e-01,
                                                  5.69890110e-01,
                                                  2.70367347e+00,
                                                  1.37686462e-01,
                                                  5.43445426e-01],
                                                 [4.00000000e+01,
                                                  2.62928647e-01,
                                                  4.82051282e-01,
                                                  3.13316239e+00,
                                                  1.70835400e-01,
                                                  4.60609495e-01],
                                                 [5.00000000e+01,
                                                  2.00631936e-01,
                                                  4.13846154e-01,
                                                  1.41600000e+00,
                                                  2.02754631e-01,
                                                  4.20218708e-01],
                                                 [6.00000000e+01,
                                                  1.74335145e-01,
                                                  3.71538462e-01,
                                                  7.28076923e-01,
                                                  2.34997824e-01,
                                                  3.83567328e-01],
                                                 [7.00000000e+01,
                                                  1.20837336e-01,
                                                  3.30769231e-01,
                                                  1.90871795e+00,
                                                  2.73899580e-01,
                                                  3.04333652e-01],
                                                 [8.00000000e+01,
                                                  9.99311311e-02,
                                                  3.11538462e-01,
                                                  4.71153846e-01,
                                                  2.95532107e-01,
                                                  2.40889558e-01],
                                                 [9.00000000e+01,
                                                  5.44840218e-02,
                                                  2.56923077e-01,
                                                  0.00000000e+00,
                                                  3.65678095e-01,
                                                  1.76536414e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 3.519954],
                                                [0.43531205, 2.86888997],
                                                [0.65296808, 0.95457744],
                                                [0.8706241, 0.7165144],
                                                [1.08828013, 0.56184561],
                                                [1.30593615, 0.36182742],
                                                [1.52359218, 0.26146693],
                                                [1.7412482, 0.22490118],
                                                [1.95890423, 0.23431208],
                                                [2.17656025, 0.16052936]]),
                                       result_4[:10].values)

        # xu, x, slow, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=neighborfile,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.65204663e-01,
                                                  9.21666667e-01,
                                                  7.19555556e-01,
                                                  3.23645292e-02,
                                                  1.61972149e+00],
                                                 [2.00000000e+01,
                                                  6.93944564e-01,
                                                  8.64625000e-01,
                                                  1.05673438e+00,
                                                  4.98962402e-02,
                                                  1.89198119e+00],
                                                 [3.00000000e+01,
                                                  6.40073583e-01,
                                                  8.14000000e-01,
                                                  1.61771429e+00,
                                                  6.63103261e-02,
                                                  1.97428727e+00],
                                                 [4.00000000e+01,
                                                  5.89936558e-01,
                                                  7.70666667e-01,
                                                  2.29288889e+00,
                                                  8.34851999e-02,
                                                  1.92218066e+00],
                                                 [5.00000000e+01,
                                                  5.47050929e-01,
                                                  7.28000000e-01,
                                                  1.73680000e+00,
                                                  9.89184345e-02,
                                                  1.75993710e+00],
                                                 [6.00000000e+01,
                                                  5.05727954e-01,
                                                  6.95250000e-01,
                                                  1.12218750e+00,
                                                  1.14735074e-01,
                                                  1.46947779e+00],
                                                 [7.00000000e+01,
                                                  4.71357566e-01,
                                                  6.63333333e-01,
                                                  1.02822222e+00,
                                                  1.28015348e-01,
                                                  1.26171669e+00],
                                                 [8.00000000e+01,
                                                  4.40575184e-01,
                                                  6.33500000e-01,
                                                  4.62250000e-01,
                                                  1.40207960e-01,
                                                  1.06769727e+00],
                                                 [9.00000000e+01,
                                                  3.79127338e-01,
                                                  5.72000000e-01,
                                                  0.00000000e+00,
                                                  1.57737474e-01,
                                                  7.29730815e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 0.72949926],
                                                [0.43531205, 0.7781586],
                                                [0.65296808, 0.4070524],
                                                [0.8706241, 0.37793433],
                                                [1.08828013, 0.24201151],
                                                [1.30593615, 0.157564],
                                                [1.52359218, 0.07824194],
                                                [1.7412482, 0.11350748],
                                                [1.95890423, 0.11724181],
                                                [2.17656025, 0.09197248]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.77933876e-01,
                                                  9.31965812e-01,
                                                  4.05735992e-01,
                                                  2.92658067e-02,
                                                  1.28133005e+00],
                                                 [2.00000000e+01,
                                                  7.05587035e-01,
                                                  8.75192308e-01,
                                                  7.25168269e-01,
                                                  4.48186089e-02,
                                                  1.74683556e+00],
                                                 [3.00000000e+01,
                                                  6.53915175e-01,
                                                  8.28571429e-01,
                                                  8.96640502e-01,
                                                  5.92745865e-02,
                                                  1.95655648e+00],
                                                 [4.00000000e+01,
                                                  5.99156844e-01,
                                                  7.79743590e-01,
                                                  1.30688034e+00,
                                                  7.50221343e-02,
                                                  1.86465605e+00],
                                                 [5.00000000e+01,
                                                  5.56515326e-01,
                                                  7.43076923e-01,
                                                  9.11384615e-01,
                                                  8.82504861e-02,
                                                  1.67712662e+00],
                                                 [6.00000000e+01,
                                                  5.12722456e-01,
                                                  7.06153846e-01,
                                                  6.56153846e-01,
                                                  1.02764507e-01,
                                                  1.37912296e+00],
                                                 [7.00000000e+01,
                                                  4.83333013e-01,
                                                  6.77948718e-01,
                                                  5.09059829e-01,
                                                  1.14628538e-01,
                                                  1.20792745e+00],
                                                 [8.00000000e+01,
                                                  4.49161228e-01,
                                                  6.44615385e-01,
                                                  9.84615385e-02,
                                                  1.27062238e-01,
                                                  1.13241204e+00],
                                                 [9.00000000e+01,
                                                  3.97271621e-01,
                                                  5.86153846e-01,
                                                  0.00000000e+00,
                                                  1.44813413e-01,
                                                  8.14938463e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 0.72949926],
                                                [0.43531205, 0.7781586],
                                                [0.65296808, 0.4070524],
                                                [0.8706241, 0.37793433],
                                                [1.08828013, 0.24201151],
                                                [1.30593615, 0.157564],
                                                [1.52359218, 0.07824194],
                                                [1.7412482, 0.11350748],
                                                [1.95890423, 0.11724181],
                                                [2.17656025, 0.09197248]]),
                                       result_4[:10].values,
                                       decimal=5)

        # xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=None,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.36702354e-01,
                                                  2.42222222e-01,
                                                  4.26750617e+00,
                                                  7.21997640e-02,
                                                  4.90984204e-01],
                                                 [2.00000000e+01,
                                                  4.36370666e-01,
                                                  3.49000000e-01,
                                                  3.62550000e+00,
                                                  1.06063946e-01,
                                                  6.70055277e-01],
                                                 [3.00000000e+01,
                                                  3.57860739e-01,
                                                  4.31714286e-01,
                                                  3.70963265e+00,
                                                  1.43668541e-01,
                                                  6.61990611e-01],
                                                 [4.00000000e+01,
                                                  2.62785498e-01,
                                                  5.26333333e-01,
                                                  4.28988889e+00,
                                                  1.79493505e-01,
                                                  5.96811308e-01],
                                                 [5.00000000e+01,
                                                  1.99583267e-01,
                                                  5.88000000e-01,
                                                  1.69720000e+00,
                                                  2.11959966e-01,
                                                  5.45471374e-01],
                                                 [6.00000000e+01,
                                                  1.71309759e-01,
                                                  6.33000000e-01,
                                                  6.96500000e-01,
                                                  2.47070228e-01,
                                                  4.93235540e-01],
                                                 [7.00000000e+01,
                                                  1.13552234e-01,
                                                  6.78000000e-01,
                                                  3.23266667e+00,
                                                  2.87827169e-01,
                                                  4.06790205e-01],
                                                 [8.00000000e+01,
                                                  9.46026864e-02,
                                                  7.02500000e-01,
                                                  6.50250000e-01,
                                                  3.11575071e-01,
                                                  3.38406147e-01],
                                                 [9.00000000e+01,
                                                  5.68581742e-02,
                                                  7.50000000e-01,
                                                  0.00000000e+00,
                                                  3.77274037e-01,
                                                  1.97250757e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 10.41603694],
                                                [0.43531205, 9.07891855],
                                                [0.65296808, 3.03401767],
                                                [0.8706241, 1.55584839],
                                                [1.08828013, 1.59992708],
                                                [1.30593615, 0.87060532],
                                                [1.52359218, 0.87338807],
                                                [1.7412482, 0.67466083],
                                                [1.95890423, 0.64249829],
                                                [2.17656025, 0.47550529]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.39399520e-01,
                                                  2.38974359e-01,
                                                  2.53504274e+00,
                                                  7.00196227e-02,
                                                  4.05403038e-01],
                                                 [2.00000000e+01,
                                                  4.37400251e-01,
                                                  3.44423077e-01,
                                                  2.08862981e+00,
                                                  1.01804767e-01,
                                                  5.71507971e-01],
                                                 [3.00000000e+01,
                                                  3.53978066e-01,
                                                  4.30109890e-01,
                                                  2.70367347e+00,
                                                  1.37686462e-01,
                                                  5.43445426e-01],
                                                 [4.00000000e+01,
                                                  2.62928647e-01,
                                                  5.17948718e-01,
                                                  3.13316239e+00,
                                                  1.70835400e-01,
                                                  4.60609495e-01],
                                                 [5.00000000e+01,
                                                  2.00631936e-01,
                                                  5.86153846e-01,
                                                  1.41600000e+00,
                                                  2.02754631e-01,
                                                  4.20218708e-01],
                                                 [6.00000000e+01,
                                                  1.74335145e-01,
                                                  6.28461538e-01,
                                                  7.28076923e-01,
                                                  2.34997824e-01,
                                                  3.83567328e-01],
                                                 [7.00000000e+01,
                                                  1.20837336e-01,
                                                  6.69230769e-01,
                                                  1.90871795e+00,
                                                  2.73899580e-01,
                                                  3.04333652e-01],
                                                 [8.00000000e+01,
                                                  9.99311311e-02,
                                                  6.88461538e-01,
                                                  4.71153846e-01,
                                                  2.95532107e-01,
                                                  2.40889558e-01],
                                                 [9.00000000e+01,
                                                  5.44840218e-02,
                                                  7.43076923e-01,
                                                  0.00000000e+00,
                                                  3.65678095e-01,
                                                  1.76536414e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 10.41603694],
                                                [0.43531205, 9.07891855],
                                                [0.65296808, 3.03401767],
                                                [0.8706241, 1.55584839],
                                                [1.08828013, 1.59992708],
                                                [1.30593615, 0.87060532],
                                                [1.52359218, 0.87338807],
                                                [1.7412482, 0.67466083],
                                                [1.95890423, 0.64249829],
                                                [2.17656025, 0.47550529]]),
                                       result_4[:10].values)

        # xu, x, fast, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=neighborfile,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.65204663e-01,
                                                  7.83333333e-02,
                                                  7.19555556e-01,
                                                  3.23645292e-02,
                                                  1.61972149e+00],
                                                 [2.00000000e+01,
                                                  6.93944564e-01,
                                                  1.35375000e-01,
                                                  1.05673438e+00,
                                                  4.98962402e-02,
                                                  1.89198119e+00],
                                                 [3.00000000e+01,
                                                  6.40073583e-01,
                                                  1.86000000e-01,
                                                  1.61771429e+00,
                                                  6.63103261e-02,
                                                  1.97428727e+00],
                                                 [4.00000000e+01,
                                                  5.89936558e-01,
                                                  2.29333333e-01,
                                                  2.29288889e+00,
                                                  8.34851999e-02,
                                                  1.92218066e+00],
                                                 [5.00000000e+01,
                                                  5.47050929e-01,
                                                  2.72000000e-01,
                                                  1.73680000e+00,
                                                  9.89184345e-02,
                                                  1.75993710e+00],
                                                 [6.00000000e+01,
                                                  5.05727954e-01,
                                                  3.04750000e-01,
                                                  1.12218750e+00,
                                                  1.14735074e-01,
                                                  1.46947779e+00],
                                                 [7.00000000e+01,
                                                  4.71357566e-01,
                                                  3.36666667e-01,
                                                  1.02822222e+00,
                                                  1.28015348e-01,
                                                  1.26171669e+00],
                                                 [8.00000000e+01,
                                                  4.40575184e-01,
                                                  3.66500000e-01,
                                                  4.62250000e-01,
                                                  1.40207960e-01,
                                                  1.06769727e+00],
                                                 [9.00000000e+01,
                                                  3.79127338e-01,
                                                  4.28000000e-01,
                                                  0.00000000e+00,
                                                  1.57737474e-01,
                                                  7.29730815e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 7.75447626],
                                                [0.43531205, 8.11407149],
                                                [0.65296808, 4.0554892],
                                                [0.8706241, 2.69828166],
                                                [1.08828013, 1.91444841],
                                                [1.30593615, 1.58646509],
                                                [1.52359218, 1.08255088],
                                                [1.7412482, 1.18694595],
                                                [1.95890423, 1.10744962],
                                                [2.17656025, 0.91132367]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.77933876e-01,
                                                  6.80341880e-02,
                                                  4.05735992e-01,
                                                  2.92658067e-02,
                                                  1.28133005e+00],
                                                 [2.00000000e+01,
                                                  7.05587035e-01,
                                                  1.24807692e-01,
                                                  7.25168269e-01,
                                                  4.48186089e-02,
                                                  1.74683556e+00],
                                                 [3.00000000e+01,
                                                  6.53915175e-01,
                                                  1.71428571e-01,
                                                  8.96640502e-01,
                                                  5.92745865e-02,
                                                  1.95655648e+00],
                                                 [4.00000000e+01,
                                                  5.99156844e-01,
                                                  2.20256410e-01,
                                                  1.30688034e+00,
                                                  7.50221343e-02,
                                                  1.86465605e+00],
                                                 [5.00000000e+01,
                                                  5.56515326e-01,
                                                  2.56923077e-01,
                                                  9.11384615e-01,
                                                  8.82504861e-02,
                                                  1.67712662e+00],
                                                 [6.00000000e+01,
                                                  5.12722456e-01,
                                                  2.93846154e-01,
                                                  6.56153846e-01,
                                                  1.02764507e-01,
                                                  1.37912296e+00],
                                                 [7.00000000e+01,
                                                  4.83333013e-01,
                                                  3.22051282e-01,
                                                  5.09059829e-01,
                                                  1.14628538e-01,
                                                  1.20792745e+00],
                                                 [8.00000000e+01,
                                                  4.49161228e-01,
                                                  3.55384615e-01,
                                                  9.84615385e-02,
                                                  1.27062238e-01,
                                                  1.13241204e+00],
                                                 [9.00000000e+01,
                                                  3.97271621e-01,
                                                  4.13846154e-01,
                                                  0.00000000e+00,
                                                  1.44813413e-01,
                                                  8.14938463e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 7.75447626],
                                                [0.43531205, 8.11407149],
                                                [0.65296808, 4.0554892],
                                                [0.8706241, 2.69828166],
                                                [1.08828013, 1.91444841],
                                                [1.30593615, 1.58646509],
                                                [1.52359218, 1.08255088],
                                                [1.7412482, 1.18694595],
                                                [1.95890423, 1.10744962],
                                                [2.17656025, 0.91132367]]),
                                       result_4[:10].values,
                                       decimal=5)

        os.remove("neighborlist.dat")
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_xu, self.test_file_2d_x}...")

    def test_Dynamics_2d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_x, }...")
        xu_snapshots = None
        x_snapshots = self.dump_2d_x.snapshots
        ppp = np.array([1, 0])
        if xu_snapshots:
            Nnearests(
                snapshots=xu_snapshots,
                N=6,
                ppp=ppp,
                fnfile="test_dynamics_2d_x")
        else:
            Nnearests(
                snapshots=x_snapshots,
                N=6,
                ppp=ppp,
                fnfile="test_dynamics_2d_x")
        neighborfile = "test_dynamics_2d_x"
        t = 10
        qrange = 10.0

        condition = []
        for snapshot in x_snapshots.snapshots:
            condition.append(snapshot.particle_type == 1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)

        # xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=None,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.35301319e-01,
                                                  7.54666667e-01,
                                                  4.30800000e+00,
                                                  3.72711777e+00,
                                                  1.07219268e+02],
                                                 [2.00000000e+01,
                                                  4.34722653e-01,
                                                  6.48500000e-01,
                                                  3.62125000e+00,
                                                  4.50564915e+00,
                                                  8.77591688e+01],
                                                 [3.00000000e+01,
                                                  3.56100939e-01,
                                                  5.65714286e-01,
                                                  3.75877551e+00,
                                                  6.07642669e+00,
                                                  6.44930076e+01],
                                                 [4.00000000e+01,
                                                  2.61576026e-01,
                                                  4.71500000e-01,
                                                  4.31091667e+00,
                                                  6.50781949e+00,
                                                  5.94693289e+01],
                                                 [5.00000000e+01,
                                                  1.98332436e-01,
                                                  4.09600000e-01,
                                                  1.69464000e+00,
                                                  7.43853560e+00,
                                                  5.15791876e+01],
                                                 [6.00000000e+01,
                                                  1.69989406e-01,
                                                  3.64750000e-01,
                                                  7.10687500e-01,
                                                  7.61598493e+00,
                                                  4.98002572e+01],
                                                 [7.00000000e+01,
                                                  1.12189767e-01,
                                                  3.20333333e-01,
                                                  3.09355556e+00,
                                                  7.40071213e+00,
                                                  5.05475373e+01],
                                                 [8.00000000e+01,
                                                  9.23705719e-02,
                                                  2.96500000e-01,
                                                  6.50250000e-01,
                                                  7.82315877e+00,
                                                  4.77312287e+01],
                                                 [9.00000000e+01,
                                                  5.49308340e-02,
                                                  2.48000000e-01,
                                                  0.00000000e+00,
                                                  9.93036691e+00,
                                                  3.76993216e+01]]),
                                       result_1[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal(np.array([[0.21765603, 3.54148318],
                                                [0.43531205, 2.94233424],
                                                [0.65296808, 0.96605736],
                                                [0.8706241, 0.72040618],
                                                [1.08828013, 0.55886752],
                                                [1.30593615, 0.35050565],
                                                [1.52359218, 0.26601472],
                                                [1.7412482, 0.23604591],
                                                [1.95890423, 0.2609116],
                                                [2.17656025, 0.15945146]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.37718181e-01,
                                                  7.57264957e-01,
                                                  2.50795821e+00,
                                                  4.56730285e+00,
                                                  8.76538136e+01],
                                                 [2.00000000e+01,
                                                  4.35524810e-01,
                                                  6.52884615e-01,
                                                  2.06459135e+00,
                                                  4.97586653e+00,
                                                  7.95385746e+01],
                                                 [3.00000000e+01,
                                                  3.52030653e-01,
                                                  5.66813187e-01,
                                                  2.74323391e+00,
                                                  6.76323381e+00,
                                                  5.80641213e+01],
                                                 [4.00000000e+01,
                                                  2.61424993e-01,
                                                  4.79487179e-01,
                                                  3.16239316e+00,
                                                  7.43670447e+00,
                                                  5.22605440e+01],
                                                 [5.00000000e+01,
                                                  1.99271120e-01,
                                                  4.11076923e-01,
                                                  1.43040000e+00,
                                                  8.63751286e+00,
                                                  4.46626228e+01],
                                                 [6.00000000e+01,
                                                  1.72684776e-01,
                                                  3.68461538e-01,
                                                  7.73461538e-01,
                                                  8.56234210e+00,
                                                  4.46584549e+01],
                                                 [7.00000000e+01,
                                                  1.18935634e-01,
                                                  3.28205128e-01,
                                                  1.77675214e+00,
                                                  8.86239337e+00,
                                                  4.27322890e+01],
                                                 [8.00000000e+01,
                                                  9.68963011e-02,
                                                  3.10000000e-01,
                                                  4.71153846e-01,
                                                  9.50719077e+00,
                                                  3.97888760e+01],
                                                 [9.00000000e+01,
                                                  5.13807657e-02,
                                                  2.53846154e-01,
                                                  0.00000000e+00,
                                                  1.27473051e+01,
                                                  2.97301842e+01]]),
                                       result_3[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal(np.array([[0.21765603, 3.54148318],
                                                [0.43531205, 2.94233424],
                                                [0.65296808, 0.96605736],
                                                [0.8706241, 0.72040618],
                                                [1.08828013, 0.55886752],
                                                [1.30593615, 0.35050565],
                                                [1.52359218, 0.26601472],
                                                [1.7412482, 0.23604591],
                                                [1.95890423, 0.2609116],
                                                [2.17656025, 0.15945146]]),
                                       result_4[:10].values)

        # xu, x, slow, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=neighborfile,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[10.,
                                                  0.76747571,
                                                  0.90844444,
                                                  0.63180247,
                                                  3.97317699,
                                                  90.03992762],
                                                 [20.,
                                                  0.69480596,
                                                  0.847625,
                                                  1.00298438,
                                                  4.71000485,
                                                  73.24554407],
                                                 [30.,
                                                  0.64231138,
                                                  0.796,
                                                  1.84142857,
                                                  6.31873346,
                                                  54.09691829],
                                                 [40.,
                                                  0.59214862,
                                                  0.75516667,
                                                  1.97080556,
                                                  6.39755071,
                                                  50.34846194],
                                                 [50.,
                                                  0.55074371,
                                                  0.7134,
                                                  1.79784,
                                                  7.30815457,
                                                  43.02003622],
                                                 [60.,
                                                  0.50868798,
                                                  0.68125,
                                                  1.5796875,
                                                  7.33487211,
                                                  41.93783697],
                                                 [70.,
                                                  0.47493082,
                                                  0.65,
                                                  0.92466667,
                                                  6.90817355,
                                                  42.48350004],
                                                 [80.,
                                                  0.44217942,
                                                  0.618,
                                                  0.576,
                                                  7.15186229,
                                                  39.8051956],
                                                 [90.,
                                                  0.37817124,
                                                  0.556,
                                                  0.,
                                                  8.30325423,
                                                  31.23269749]]),
                                       result_1[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal(np.array([[0.21765603, 1.05436341],
                                                [0.43531205, 0.95455698],
                                                [0.65296808, 0.68321072],
                                                [0.8706241, 0.3576023],
                                                [1.08828013, 0.36165315],
                                                [1.30593615, 0.2147224],
                                                [1.52359218, 0.21481405],
                                                [1.7412482, 0.16966207],
                                                [1.95890423, 0.13946985],
                                                [2.17656025, 0.09861592]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[10.,
                                                  0.78094286,
                                                  0.91846154,
                                                  0.34051282,
                                                  4.77164817,
                                                  75.33496176],
                                                 [20.,
                                                  0.70753545,
                                                  0.85788462,
                                                  0.74651442,
                                                  5.05653776,
                                                  67.0628784],
                                                 [30.,
                                                  0.65553671,
                                                  0.80813187,
                                                  1.1277865,
                                                  6.85050461,
                                                  49.56433845],
                                                 [40.,
                                                  0.60173689,
                                                  0.76410256,
                                                  1.13418803,
                                                  7.02193565,
                                                  45.22208469],
                                                 [50.,
                                                  0.55851768,
                                                  0.72584615,
                                                  1.07101538,
                                                  8.18937205,
                                                  38.41286419],
                                                 [60.,
                                                  0.51358819,
                                                  0.69076923,
                                                  0.95153846,
                                                  8.06740011,
                                                  38.65671041],
                                                 [70.,
                                                  0.48254452,
                                                  0.66358974,
                                                  0.39008547,
                                                  8.2021927,
                                                  37.40401588],
                                                 [80.,
                                                  0.44559039,
                                                  0.62307692,
                                                  0.15384615,
                                                  8.71924724,
                                                  34.1717133],
                                                 [90.,
                                                  0.38703538,
                                                  0.56307692,
                                                  0.,
                                                  10.73245162,
                                                  25.77077298]]),
                                       result_3[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal(np.array([[0.21765603, 1.05436341],
                                                [0.43531205, 0.95455698],
                                                [0.65296808, 0.68321072],
                                                [0.8706241, 0.3576023],
                                                [1.08828013, 0.36165315],
                                                [1.30593615, 0.2147224],
                                                [1.52359218, 0.21481405],
                                                [1.7412482, 0.16966207],
                                                [1.95890423, 0.13946985],
                                                [2.17656025, 0.09861592]]),
                                       result_4[:10].values)

        # xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=None,
                           max_neighbors=100
                           )

        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.35301319e-01,
                                                  2.45333333e-01,
                                                  4.30800000e+00,
                                                  3.72711777e+00,
                                                  1.07219268e+02],
                                                 [2.00000000e+01,
                                                  4.34722653e-01,
                                                  3.51500000e-01,
                                                  3.62125000e+00,
                                                  4.50564915e+00,
                                                  8.77591688e+01],
                                                 [3.00000000e+01,
                                                  3.56100939e-01,
                                                  4.34285714e-01,
                                                  3.75877551e+00,
                                                  6.07642669e+00,
                                                  6.44930076e+01],
                                                 [4.00000000e+01,
                                                  2.61576026e-01,
                                                  5.28500000e-01,
                                                  4.31091667e+00,
                                                  6.50781949e+00,
                                                  5.94693289e+01],
                                                 [5.00000000e+01,
                                                  1.98332436e-01,
                                                  5.90400000e-01,
                                                  1.69464000e+00,
                                                  7.43853560e+00,
                                                  5.15791876e+01],
                                                 [6.00000000e+01,
                                                  1.69989406e-01,
                                                  6.35250000e-01,
                                                  7.10687500e-01,
                                                  7.61598493e+00,
                                                  4.98002572e+01],
                                                 [7.00000000e+01,
                                                  1.12189767e-01,
                                                  6.79666667e-01,
                                                  3.09355556e+00,
                                                  7.40071213e+00,
                                                  5.05475373e+01],
                                                 [8.00000000e+01,
                                                  9.23705719e-02,
                                                  7.03500000e-01,
                                                  6.50250000e-01,
                                                  7.82315877e+00,
                                                  4.77312287e+01],
                                                 [9.00000000e+01,
                                                  5.49308340e-02,
                                                  7.52000000e-01,
                                                  0.00000000e+00,
                                                  9.93036691e+00,
                                                  3.76993216e+01]]),
                                       result_1[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal(np.array([[0.21765603, 10.23940479],
                                                [0.43531205, 9.21250751],
                                                [0.65296808, 3.04372682],
                                                [0.8706241, 1.5453276],
                                                [1.08828013, 1.5549888],
                                                [1.30593615, 0.92162831],
                                                [1.52359218, 0.88553164],
                                                [1.7412482, 0.71619628],
                                                [1.95890423, 0.72553231],
                                                [2.17656025, 0.47386498]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.37718181e-01,
                                                  2.42735043e-01,
                                                  2.50795821e+00,
                                                  4.56730285e+00,
                                                  8.76538136e+01],
                                                 [2.00000000e+01,
                                                  4.35524810e-01,
                                                  3.47115385e-01,
                                                  2.06459135e+00,
                                                  4.97586653e+00,
                                                  7.95385746e+01],
                                                 [3.00000000e+01,
                                                  3.52030653e-01,
                                                  4.33186813e-01,
                                                  2.74323391e+00,
                                                  6.76323381e+00,
                                                  5.80641213e+01],
                                                 [4.00000000e+01,
                                                  2.61424993e-01,
                                                  5.20512821e-01,
                                                  3.16239316e+00,
                                                  7.43670447e+00,
                                                  5.22605440e+01],
                                                 [5.00000000e+01,
                                                  1.99271120e-01,
                                                  5.88923077e-01,
                                                  1.43040000e+00,
                                                  8.63751286e+00,
                                                  4.46626228e+01],
                                                 [6.00000000e+01,
                                                  1.72684776e-01,
                                                  6.31538462e-01,
                                                  7.73461538e-01,
                                                  8.56234210e+00,
                                                  4.46584549e+01],
                                                 [7.00000000e+01,
                                                  1.18935634e-01,
                                                  6.71794872e-01,
                                                  1.77675214e+00,
                                                  8.86239337e+00,
                                                  4.27322890e+01],
                                                 [8.00000000e+01,
                                                  9.68963011e-02,
                                                  6.90000000e-01,
                                                  4.71153846e-01,
                                                  9.50719077e+00,
                                                  3.97888760e+01],
                                                 [9.00000000e+01,
                                                  5.13807657e-02,
                                                  7.46153846e-01,
                                                  0.00000000e+00,
                                                  1.27473051e+01,
                                                  2.97301842e+01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 10.23940479],
                                                [0.43531205, 9.21250751],
                                                [0.65296808, 3.04372682],
                                                [0.8706241, 1.5453276],
                                                [1.08828013, 1.5549888],
                                                [1.30593615, 0.92162831],
                                                [1.52359218, 0.88553164],
                                                [1.7412482, 0.71619628],
                                                [1.95890423, 0.72553231],
                                                [2.17656025, 0.47386498]]),
                                       result_4[:10].values)

        # xu, x, fast, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=neighborfile,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[10.,
                                                  0.76747571,
                                                  0.09155556,
                                                  0.63180247,
                                                  3.97317699,
                                                  90.03992762],
                                                 [20.,
                                                  0.69480596,
                                                  0.152375,
                                                  1.00298438,
                                                  4.71000485,
                                                  73.24554407],
                                                 [30.,
                                                  0.64231138,
                                                  0.204,
                                                  1.84142857,
                                                  6.31873346,
                                                  54.09691829],
                                                 [40.,
                                                  0.59214862,
                                                  0.24483333,
                                                  1.97080556,
                                                  6.39755071,
                                                  50.34846194],
                                                 [50.,
                                                  0.55074371,
                                                  0.2866,
                                                  1.79784,
                                                  7.30815457,
                                                  43.02003622],
                                                 [60.,
                                                  0.50868798,
                                                  0.31875,
                                                  1.5796875,
                                                  7.33487211,
                                                  41.93783697],
                                                 [70.,
                                                  0.47493082,
                                                  0.35,
                                                  0.92466667,
                                                  6.90817355,
                                                  42.48350004],
                                                 [80.,
                                                  0.44217942,
                                                  0.382,
                                                  0.576,
                                                  7.15186229,
                                                  39.8051956],
                                                 [90.,
                                                  0.37817124,
                                                  0.444,
                                                  0.,
                                                  8.30325423,
                                                  31.23269749]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 8.5826075],
                                                [0.43531205, 8.55958774],
                                                [0.65296808, 5.69371142],
                                                [0.8706241, 2.78104677],
                                                [1.08828013, 2.50197449],
                                                [1.30593615, 2.67568978],
                                                [1.52359218, 2.26862401],
                                                [1.7412482, 1.75039804],
                                                [1.95890423, 1.59986474],
                                                [2.17656025, 0.93941497]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.80942858e-01,
                                                  8.15384615e-02,
                                                  3.40512821e-01,
                                                  4.77164817e+00,
                                                  7.53349618e+01],
                                                 [2.00000000e+01,
                                                  7.07535452e-01,
                                                  1.42115385e-01,
                                                  7.46514423e-01,
                                                  5.05653776e+00,
                                                  6.70628784e+01],
                                                 [3.00000000e+01,
                                                  6.55536709e-01,
                                                  1.91868132e-01,
                                                  1.12778650e+00,
                                                  6.85050461e+00,
                                                  4.95643384e+01],
                                                 [4.00000000e+01,
                                                  6.01736888e-01,
                                                  2.35897436e-01,
                                                  1.13418803e+00,
                                                  7.02193565e+00,
                                                  4.52220847e+01],
                                                 [5.00000000e+01,
                                                  5.58517683e-01,
                                                  2.74153846e-01,
                                                  1.07101538e+00,
                                                  8.18937205e+00,
                                                  3.84128642e+01],
                                                 [6.00000000e+01,
                                                  5.13588192e-01,
                                                  3.09230769e-01,
                                                  9.51538462e-01,
                                                  8.06740011e+00,
                                                  3.86567104e+01],
                                                 [7.00000000e+01,
                                                  4.82544519e-01,
                                                  3.36410256e-01,
                                                  3.90085470e-01,
                                                  8.20219270e+00,
                                                  3.74040159e+01],
                                                 [8.00000000e+01,
                                                  4.45590391e-01,
                                                  3.76923077e-01,
                                                  1.53846154e-01,
                                                  8.71924724e+00,
                                                  3.41717133e+01],
                                                 [9.00000000e+01,
                                                  3.87035376e-01,
                                                  4.36923077e-01,
                                                  0.00000000e+00,
                                                  1.07324516e+01,
                                                  2.57707730e+01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 8.5826075],
                                                [0.43531205, 8.55958774],
                                                [0.65296808, 5.69371142],
                                                [0.8706241, 2.78104677],
                                                [1.08828013, 2.50197449],
                                                [1.30593615, 2.67568978],
                                                [1.52359218, 2.26862401],
                                                [1.7412482, 1.75039804],
                                                [1.95890423, 1.59986474],
                                                [2.17656025, 0.93941497]]),
                                       result_4[:10].values)

        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_x}...")

    def test_Dynamics_2d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_xu}...")
        xu_snapshots = self.dump_2d_xu.snapshots
        x_snapshots = None
        ppp = np.array([0, 0])
        if xu_snapshots:
            Nnearests(
                snapshots=xu_snapshots,
                N=6,
                ppp=ppp,
                fnfile="test_Dynamics_2d_xu")
        else:
            Nnearests(
                snapshots=x_snapshots,
                N=6,
                ppp=ppp,
                fnfile="test_Dynamics_2d_xu")
        neighborfile = 'test_Dynamics_2d_xu'
        t = 10
        qrange = 10.0

        condition = []
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type == 1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)

        # xu, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=None,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.36702354e-01,
                                                  7.57777778e-01,
                                                  4.26750617e+00,
                                                  7.21997640e-02,
                                                  4.90984204e-01],
                                                 [2.00000000e+01,
                                                  4.36370666e-01,
                                                  6.51000000e-01,
                                                  3.62550000e+00,
                                                  1.06063946e-01,
                                                  6.70055277e-01],
                                                 [3.00000000e+01,
                                                  3.57860739e-01,
                                                  5.68285714e-01,
                                                  3.70963265e+00,
                                                  1.43668541e-01,
                                                  6.61990611e-01],
                                                 [4.00000000e+01,
                                                  2.62785498e-01,
                                                  4.73666667e-01,
                                                  4.28988889e+00,
                                                  1.79493505e-01,
                                                  5.96811308e-01],
                                                 [5.00000000e+01,
                                                  1.99583267e-01,
                                                  4.12000000e-01,
                                                  1.69720000e+00,
                                                  2.11959966e-01,
                                                  5.45471374e-01],
                                                 [6.00000000e+01,
                                                  1.71309759e-01,
                                                  3.67000000e-01,
                                                  6.96500000e-01,
                                                  2.47070228e-01,
                                                  4.93235540e-01],
                                                 [7.00000000e+01,
                                                  1.13552234e-01,
                                                  3.22000000e-01,
                                                  3.23266667e+00,
                                                  2.87827169e-01,
                                                  4.06790205e-01],
                                                 [8.00000000e+01,
                                                  9.46026864e-02,
                                                  2.97500000e-01,
                                                  6.50250000e-01,
                                                  3.11575071e-01,
                                                  3.38406147e-01],
                                                 [9.00000000e+01,
                                                  5.68581742e-02,
                                                  2.50000000e-01,
                                                  0.00000000e+00,
                                                  3.77274037e-01,
                                                  1.97250757e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 3.51995254],
                                                [0.43531205, 2.86888985],
                                                [0.65296808, 0.95457642],
                                                [0.8706241, 0.71651359],
                                                [1.08828013, 0.56184595],
                                                [1.30593615, 0.3618279],
                                                [1.52359218, 0.26146727],
                                                [1.7412482, 0.22490306],
                                                [1.95890423, 0.23431135],
                                                [2.17656025, 0.16052942]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.39399520e-01,
                                                  7.61025641e-01,
                                                  2.53504274e+00,
                                                  7.00196227e-02,
                                                  4.05403038e-01],
                                                 [2.00000000e+01,
                                                  4.37400251e-01,
                                                  6.55576923e-01,
                                                  2.08862981e+00,
                                                  1.01804767e-01,
                                                  5.71507971e-01],
                                                 [3.00000000e+01,
                                                  3.53978066e-01,
                                                  5.69890110e-01,
                                                  2.70367347e+00,
                                                  1.37686462e-01,
                                                  5.43445426e-01],
                                                 [4.00000000e+01,
                                                  2.62928647e-01,
                                                  4.82051282e-01,
                                                  3.13316239e+00,
                                                  1.70835400e-01,
                                                  4.60609495e-01],
                                                 [5.00000000e+01,
                                                  2.00631936e-01,
                                                  4.13846154e-01,
                                                  1.41600000e+00,
                                                  2.02754631e-01,
                                                  4.20218708e-01],
                                                 [6.00000000e+01,
                                                  1.74335145e-01,
                                                  3.71538462e-01,
                                                  7.28076923e-01,
                                                  2.34997824e-01,
                                                  3.83567328e-01],
                                                 [7.00000000e+01,
                                                  1.20837336e-01,
                                                  3.30769231e-01,
                                                  1.90871795e+00,
                                                  2.73899580e-01,
                                                  3.04333652e-01],
                                                 [8.00000000e+01,
                                                  9.99311311e-02,
                                                  3.11538462e-01,
                                                  4.71153846e-01,
                                                  2.95532107e-01,
                                                  2.40889558e-01],
                                                 [9.00000000e+01,
                                                  5.44840218e-02,
                                                  2.56923077e-01,
                                                  0.00000000e+00,
                                                  3.65678095e-01,
                                                  1.76536414e-01]]),
                                       result_3[:10].values,
                                       )
        np.testing.assert_almost_equal(np.array([[0.21765603, 3.51995254],
                                                [0.43531205, 2.86888985],
                                                [0.65296808, 0.95457642],
                                                [0.8706241, 0.71651359],
                                                [1.08828013, 0.56184595],
                                                [1.30593615, 0.3618279],
                                                [1.52359218, 0.26146727],
                                                [1.7412482, 0.22490306],
                                                [1.95890423, 0.23431135],
                                                [2.17656025, 0.16052942]]),
                                       result_4[:10].values,
                                       decimal=5)

        # xu, slow, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=neighborfile,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.65204663e-01,
                                                  9.21666667e-01,
                                                  7.19555556e-01,
                                                  3.23645292e-02,
                                                  1.61972149e+00],
                                                 [2.00000000e+01,
                                                  6.93944564e-01,
                                                  8.64625000e-01,
                                                  1.05673438e+00,
                                                  4.98962402e-02,
                                                  1.89198119e+00],
                                                 [3.00000000e+01,
                                                  6.40073583e-01,
                                                  8.14000000e-01,
                                                  1.61771429e+00,
                                                  6.63103261e-02,
                                                  1.97428727e+00],
                                                 [4.00000000e+01,
                                                  5.89936558e-01,
                                                  7.70666667e-01,
                                                  2.29288889e+00,
                                                  8.34851999e-02,
                                                  1.92218066e+00],
                                                 [5.00000000e+01,
                                                  5.47050929e-01,
                                                  7.28000000e-01,
                                                  1.73680000e+00,
                                                  9.89184345e-02,
                                                  1.75993710e+00],
                                                 [6.00000000e+01,
                                                  5.05727954e-01,
                                                  6.95250000e-01,
                                                  1.12218750e+00,
                                                  1.14735074e-01,
                                                  1.46947779e+00],
                                                 [7.00000000e+01,
                                                  4.71357566e-01,
                                                  6.63333333e-01,
                                                  1.02822222e+00,
                                                  1.28015348e-01,
                                                  1.26171669e+00],
                                                 [8.00000000e+01,
                                                  4.40575184e-01,
                                                  6.33500000e-01,
                                                  4.62250000e-01,
                                                  1.40207960e-01,
                                                  1.06769727e+00],
                                                 [9.00000000e+01,
                                                  3.79127338e-01,
                                                  5.72000000e-01,
                                                  0.00000000e+00,
                                                  1.57737474e-01,
                                                  7.29730815e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 0.72949926],
                                                 [0.43531205, 0.7781586],
                                                 [0.65296808, 0.4070524],
                                                 [0.8706241, 0.37793433],
                                                 [1.08828013, 0.24201151],
                                                 [1.30593615, 0.157564],
                                                 [1.52359218, 0.07824194],
                                                 [1.7412482, 0.11350748],
                                                 [1.95890423, 0.11724181],
                                                 [2.17656025, 0.09197248]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.77933876e-01,
                                                  9.31965812e-01,
                                                  4.05735992e-01,
                                                  2.92658067e-02,
                                                  1.28133005e+00],
                                                 [2.00000000e+01,
                                                  7.05587035e-01,
                                                  8.75192308e-01,
                                                  7.25168269e-01,
                                                  4.48186089e-02,
                                                  1.74683556e+00],
                                                 [3.00000000e+01,
                                                  6.53915175e-01,
                                                  8.28571429e-01,
                                                  8.96640502e-01,
                                                  5.92745865e-02,
                                                  1.95655648e+00],
                                                 [4.00000000e+01,
                                                  5.99156844e-01,
                                                  7.79743590e-01,
                                                  1.30688034e+00,
                                                  7.50221343e-02,
                                                  1.86465605e+00],
                                                 [5.00000000e+01,
                                                  5.56515326e-01,
                                                  7.43076923e-01,
                                                  9.11384615e-01,
                                                  8.82504861e-02,
                                                  1.67712662e+00],
                                                 [6.00000000e+01,
                                                  5.12722456e-01,
                                                  7.06153846e-01,
                                                  6.56153846e-01,
                                                  1.02764507e-01,
                                                  1.37912296e+00],
                                                 [7.00000000e+01,
                                                  4.83333013e-01,
                                                  6.77948718e-01,
                                                  5.09059829e-01,
                                                  1.14628538e-01,
                                                  1.20792745e+00],
                                                 [8.00000000e+01,
                                                  4.49161228e-01,
                                                  6.44615385e-01,
                                                  9.84615385e-02,
                                                  1.27062238e-01,
                                                  1.13241204e+00],
                                                 [9.00000000e+01,
                                                  3.97271621e-01,
                                                  5.86153846e-01,
                                                  0.00000000e+00,
                                                  1.44813413e-01,
                                                  8.14938463e-01]]),
                                       result_3[:10].values,
                                       )
        np.testing.assert_almost_equal(np.array([[0.21765603, 0.72949926],
                                                 [0.43531205, 0.7781586],
                                                 [0.65296808, 0.4070524],
                                                 [0.8706241, 0.37793433],
                                                 [1.08828013, 0.24201151],
                                                 [1.30593615, 0.157564],
                                                 [1.52359218, 0.07824194],
                                                 [1.7412482, 0.11350748],
                                                 [1.95890423, 0.11724181],
                                                 [2.17656025, 0.09197248]]),
                                       result_4[:10].values,
                                       decimal=5)

        # xu, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=None,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.36702354e-01,
                                                  2.42222222e-01,
                                                  4.26750617e+00,
                                                  7.21997640e-02,
                                                  4.90984204e-01],
                                                 [2.00000000e+01,
                                                  4.36370666e-01,
                                                  3.49000000e-01,
                                                  3.62550000e+00,
                                                  1.06063946e-01,
                                                  6.70055277e-01],
                                                 [3.00000000e+01,
                                                  3.57860739e-01,
                                                  4.31714286e-01,
                                                  3.70963265e+00,
                                                  1.43668541e-01,
                                                  6.61990611e-01],
                                                 [4.00000000e+01,
                                                  2.62785498e-01,
                                                  5.26333333e-01,
                                                  4.28988889e+00,
                                                  1.79493505e-01,
                                                  5.96811308e-01],
                                                 [5.00000000e+01,
                                                  1.99583267e-01,
                                                  5.88000000e-01,
                                                  1.69720000e+00,
                                                  2.11959966e-01,
                                                  5.45471374e-01],
                                                 [6.00000000e+01,
                                                  1.71309759e-01,
                                                  6.33000000e-01,
                                                  6.96500000e-01,
                                                  2.47070228e-01,
                                                  4.93235540e-01],
                                                 [7.00000000e+01,
                                                  1.13552234e-01,
                                                  6.78000000e-01,
                                                  3.23266667e+00,
                                                  2.87827169e-01,
                                                  4.06790205e-01],
                                                 [8.00000000e+01,
                                                  9.46026864e-02,
                                                  7.02500000e-01,
                                                  6.50250000e-01,
                                                  3.11575071e-01,
                                                  3.38406147e-01],
                                                 [9.00000000e+01,
                                                  5.68581742e-02,
                                                  7.50000000e-01,
                                                  0.00000000e+00,
                                                  3.77274037e-01,
                                                  1.97250757e-01]]),
                                       result_1[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal(np.array([[0.21765603, 10.41603589],
                                                 [0.43531205, 9.07891827],
                                                 [0.65296808, 3.03401944],
                                                 [0.8706241, 1.55584835],
                                                 [1.08828013, 1.59992851],
                                                 [1.30593615, 0.87060821],
                                                 [1.52359218, 0.87338879],
                                                 [1.7412482, 0.6746603],
                                                 [1.95890423, 0.64249216],
                                                 [2.17656025, 0.47550557]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.39399520e-01,
                                                  2.38974359e-01,
                                                  2.53504274e+00,
                                                  7.00196227e-02,
                                                  4.05403038e-01],
                                                 [2.00000000e+01,
                                                  4.37400251e-01,
                                                  3.44423077e-01,
                                                  2.08862981e+00,
                                                  1.01804767e-01,
                                                  5.71507971e-01],
                                                 [3.00000000e+01,
                                                  3.53978066e-01,
                                                  4.30109890e-01,
                                                  2.70367347e+00,
                                                  1.37686462e-01,
                                                  5.43445426e-01],
                                                 [4.00000000e+01,
                                                  2.62928647e-01,
                                                  5.17948718e-01,
                                                  3.13316239e+00,
                                                  1.70835400e-01,
                                                  4.60609495e-01],
                                                 [5.00000000e+01,
                                                  2.00631936e-01,
                                                  5.86153846e-01,
                                                  1.41600000e+00,
                                                  2.02754631e-01,
                                                  4.20218708e-01],
                                                 [6.00000000e+01,
                                                  1.74335145e-01,
                                                  6.28461538e-01,
                                                  7.28076923e-01,
                                                  2.34997824e-01,
                                                  3.83567328e-01],
                                                 [7.00000000e+01,
                                                  1.20837336e-01,
                                                  6.69230769e-01,
                                                  1.90871795e+00,
                                                  2.73899580e-01,
                                                  3.04333652e-01],
                                                 [8.00000000e+01,
                                                  9.99311311e-02,
                                                  6.88461538e-01,
                                                  4.71153846e-01,
                                                  2.95532107e-01,
                                                  2.40889558e-01],
                                                 [9.00000000e+01,
                                                  5.44840218e-02,
                                                  7.43076923e-01,
                                                  0.00000000e+00,
                                                  3.65678095e-01,
                                                  1.76536414e-01]]),
                                       result_3[:10].values,
                                       )
        np.testing.assert_almost_equal(np.array([[0.21765603, 10.41603589],
                                                 [0.43531205, 9.07891827],
                                                 [0.65296808, 3.03401944],
                                                 [0.8706241, 1.55584835],
                                                 [1.08828013, 1.59992851],
                                                 [1.30593615, 0.87060821],
                                                 [1.52359218, 0.87338879],
                                                 [1.7412482, 0.6746603],
                                                 [1.95890423, 0.64249216],
                                                 [2.17656025, 0.47550557]]),
                                       result_4[:10].values,
                                       decimal=5)

        # xu, fast, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=neighborfile,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.65204663e-01,
                                                  7.83333333e-02,
                                                  7.19555556e-01,
                                                  3.23645292e-02,
                                                  1.61972149e+00],
                                                 [2.00000000e+01,
                                                  6.93944564e-01,
                                                  1.35375000e-01,
                                                  1.05673438e+00,
                                                  4.98962402e-02,
                                                  1.89198119e+00],
                                                 [3.00000000e+01,
                                                  6.40073583e-01,
                                                  1.86000000e-01,
                                                  1.61771429e+00,
                                                  6.63103261e-02,
                                                  1.97428727e+00],
                                                 [4.00000000e+01,
                                                  5.89936558e-01,
                                                  2.29333333e-01,
                                                  2.29288889e+00,
                                                  8.34851999e-02,
                                                  1.92218066e+00],
                                                 [5.00000000e+01,
                                                  5.47050929e-01,
                                                  2.72000000e-01,
                                                  1.73680000e+00,
                                                  9.89184345e-02,
                                                  1.75993710e+00],
                                                 [6.00000000e+01,
                                                  5.05727954e-01,
                                                  3.04750000e-01,
                                                  1.12218750e+00,
                                                  1.14735074e-01,
                                                  1.46947779e+00],
                                                 [7.00000000e+01,
                                                  4.71357566e-01,
                                                  3.36666667e-01,
                                                  1.02822222e+00,
                                                  1.28015348e-01,
                                                  1.26171669e+00],
                                                 [8.00000000e+01,
                                                  4.40575184e-01,
                                                  3.66500000e-01,
                                                  4.62250000e-01,
                                                  1.40207960e-01,
                                                  1.06769727e+00],
                                                 [9.00000000e+01,
                                                  3.79127338e-01,
                                                  4.28000000e-01,
                                                  0.00000000e+00,
                                                  1.57737474e-01,
                                                  7.29730815e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.21765603, 7.75447626],
                                                 [0.43531205, 8.11407149],
                                                 [0.65296808, 4.0554892],
                                                 [0.8706241, 2.69828166],
                                                 [1.08828013, 1.91444841],
                                                 [1.30593615, 1.58646509],
                                                 [1.52359218, 1.08255088],
                                                 [1.7412482, 1.18694595],
                                                 [1.95890423, 1.10744962],
                                                 [2.17656025, 0.91132367]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  7.77933876e-01,
                                                  6.80341880e-02,
                                                  4.05735992e-01,
                                                  2.92658067e-02,
                                                  1.28133005e+00],
                                                 [2.00000000e+01,
                                                  7.05587035e-01,
                                                  1.24807692e-01,
                                                  7.25168269e-01,
                                                  4.48186089e-02,
                                                  1.74683556e+00],
                                                 [3.00000000e+01,
                                                  6.53915175e-01,
                                                  1.71428571e-01,
                                                  8.96640502e-01,
                                                  5.92745865e-02,
                                                  1.95655648e+00],
                                                 [4.00000000e+01,
                                                  5.99156844e-01,
                                                  2.20256410e-01,
                                                  1.30688034e+00,
                                                  7.50221343e-02,
                                                  1.86465605e+00],
                                                 [5.00000000e+01,
                                                  5.56515326e-01,
                                                  2.56923077e-01,
                                                  9.11384615e-01,
                                                  8.82504861e-02,
                                                  1.67712662e+00],
                                                 [6.00000000e+01,
                                                  5.12722456e-01,
                                                  2.93846154e-01,
                                                  6.56153846e-01,
                                                  1.02764507e-01,
                                                  1.37912296e+00],
                                                 [7.00000000e+01,
                                                  4.83333013e-01,
                                                  3.22051282e-01,
                                                  5.09059829e-01,
                                                  1.14628538e-01,
                                                  1.20792745e+00],
                                                 [8.00000000e+01,
                                                  4.49161228e-01,
                                                  3.55384615e-01,
                                                  9.84615385e-02,
                                                  1.27062238e-01,
                                                  1.13241204e+00],
                                                 [9.00000000e+01,
                                                  3.97271621e-01,
                                                  4.13846154e-01,
                                                  0.00000000e+00,
                                                  1.44813413e-01,
                                                  8.14938463e-01]]),
                                       result_3[:10].values,
                                       )
        np.testing.assert_almost_equal(np.array([[0.21765603, 7.75447626],
                                                 [0.43531205, 8.11407149],
                                                 [0.65296808, 4.0554892],
                                                 [0.8706241, 2.69828166],
                                                 [1.08828013, 1.91444841],
                                                 [1.30593615, 1.58646509],
                                                 [1.52359218, 1.08255088],
                                                 [1.7412482, 1.18694595],
                                                 [1.95890423, 1.10744962],
                                                 [2.17656025, 0.91132367]]),
                                       result_4[:10].values,
                                       decimal=5)

        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_xu}...")

    def test_Dynamics_3d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_x, self.test_file_3d_xu}...")
        xu_snapshots = self.dump_3d_xu.snapshots
        x_snapshots = self.dump_3d_x.snapshots
        ppp = np.array([0, 0, 0])
        t = 10
        qrange = 10.0
        condition = []
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type == 1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)

        # xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=None,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.41337495e-01,
                                                  6.16444444e-01,
                                                  5.59224691e+00,
                                                  1.11494660e-01,
                                                  6.42042349e-01],
                                                 [2.00000000e+01,
                                                  4.42470289e-01,
                                                  5.01750000e-01,
                                                  7.78893750e+00,
                                                  1.63896915e-01,
                                                  7.29737591e-01],
                                                 [3.00000000e+01,
                                                  3.61157742e-01,
                                                  4.01571429e-01,
                                                  4.66910204e+00,
                                                  2.13100071e-01,
                                                  6.84680761e-01],
                                                 [4.00000000e+01,
                                                  3.02034667e-01,
                                                  3.43833333e-01,
                                                  5.28080556e+00,
                                                  2.63609639e-01,
                                                  6.42069210e-01],
                                                 [5.00000000e+01,
                                                  2.42622250e-01,
                                                  2.83400000e-01,
                                                  3.26584000e+00,
                                                  3.12858357e-01,
                                                  5.69164565e-01],
                                                 [6.00000000e+01,
                                                  1.93722985e-01,
                                                  2.32250000e-01,
                                                  1.13418750e+00,
                                                  3.61507843e-01,
                                                  5.29289260e-01],
                                                 [7.00000000e+01,
                                                  1.56989020e-01,
                                                  1.97000000e-01,
                                                  3.12000000e-01,
                                                  4.16759105e-01,
                                                  4.77159918e-01],
                                                 [8.00000000e+01,
                                                  1.18113321e-01,
                                                  1.61000000e-01,
                                                  1.00000000e-03,
                                                  4.73546209e-01,
                                                  4.42058084e-01],
                                                 [9.00000000e+01,
                                                  9.47865013e-02,
                                                  1.32000000e-01,
                                                  0.00000000e+00,
                                                  5.45079210e-01,
                                                  3.78672766e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 3.47872162],
                                                [1.33537618, 1.28511381],
                                                [2.00306427, 0.46925638],
                                                [2.67075237, 0.34470482],
                                                [3.33844046, 0.36478221],
                                                [4.00612855, 0.33486395],
                                                [4.67381664, 0.35057394],
                                                [5.34150473, 0.22428304],
                                                [6.00919282, 0.43405835],
                                                [6.67688091, 1.0276538]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.66752497e-01,
                                                  6.45972222e-01,
                                                  5.24438272e+00,
                                                  9.66083489e-02,
                                                  4.21591290e-01],
                                                 [2.00000000e+01,
                                                  4.65372892e-01,
                                                  5.26250000e-01,
                                                  6.55375000e+00,
                                                  1.40979734e-01,
                                                  4.84004151e-01],
                                                 [3.00000000e+01,
                                                  3.83810580e-01,
                                                  4.23571429e-01,
                                                  4.31301020e+00,
                                                  1.82537107e-01,
                                                  4.45724641e-01],
                                                 [4.00000000e+01,
                                                  3.21135461e-01,
                                                  3.65416667e-01,
                                                  4.34777778e+00,
                                                  2.25065174e-01,
                                                  4.08617620e-01],
                                                 [5.00000000e+01,
                                                  2.60769450e-01,
                                                  3.03250000e-01,
                                                  2.90980000e+00,
                                                  2.64845124e-01,
                                                  3.62285330e-01],
                                                 [6.00000000e+01,
                                                  2.08824946e-01,
                                                  2.50625000e-01,
                                                  1.13281250e+00,
                                                  3.05573799e-01,
                                                  3.40291418e-01],
                                                 [7.00000000e+01,
                                                  1.71330036e-01,
                                                  2.14166667e-01,
                                                  2.54444444e-01,
                                                  3.52397096e-01,
                                                  3.14565380e-01],
                                                 [8.00000000e+01,
                                                  1.33292499e-01,
                                                  1.75000000e-01,
                                                  1.12500000e-02,
                                                  3.97987653e-01,
                                                  2.82139510e-01],
                                                 [9.00000000e+01,
                                                  1.06593521e-01,
                                                  1.45000000e-01,
                                                  0.00000000e+00,
                                                  4.60740318e-01,
                                                  2.21040917e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 3.47872162],
                                                [1.33537618, 1.28511381],
                                                [2.00306427, 0.46925638],
                                                [2.67075237, 0.34470482],
                                                [3.33844046, 0.36478221],
                                                [4.00612855, 0.33486395],
                                                [4.67381664, 0.35057394],
                                                [5.34150473, 0.22428304],
                                                [6.00919282, 0.43405835],
                                                [6.67688091, 1.0276538]]),
                                       result_4[:10].values,
                                       decimal=5)

        # xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=None,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.41337495e-01,
                                                  3.83555556e-01,
                                                  5.59224691e+00,
                                                  1.11494660e-01,
                                                  6.42042349e-01],
                                                 [2.00000000e+01,
                                                  4.42470289e-01,
                                                  4.98250000e-01,
                                                  7.78893750e+00,
                                                  1.63896915e-01,
                                                  7.29737591e-01],
                                                 [3.00000000e+01,
                                                  3.61157742e-01,
                                                  5.98428571e-01,
                                                  4.66910204e+00,
                                                  2.13100071e-01,
                                                  6.84680761e-01],
                                                 [4.00000000e+01,
                                                  3.02034667e-01,
                                                  6.56166667e-01,
                                                  5.28080556e+00,
                                                  2.63609639e-01,
                                                  6.42069210e-01],
                                                 [5.00000000e+01,
                                                  2.42622250e-01,
                                                  7.16600000e-01,
                                                  3.26584000e+00,
                                                  3.12858357e-01,
                                                  5.69164565e-01],
                                                 [6.00000000e+01,
                                                  1.93722985e-01,
                                                  7.67750000e-01,
                                                  1.13418750e+00,
                                                  3.61507843e-01,
                                                  5.29289260e-01],
                                                 [7.00000000e+01,
                                                  1.56989020e-01,
                                                  8.03000000e-01,
                                                  3.12000000e-01,
                                                  4.16759105e-01,
                                                  4.77159918e-01],
                                                 [8.00000000e+01,
                                                  1.18113321e-01,
                                                  8.39000000e-01,
                                                  1.00000000e-03,
                                                  4.73546209e-01,
                                                  4.42058084e-01],
                                                 [9.00000000e+01,
                                                  9.47865013e-02,
                                                  8.68000000e-01,
                                                  0.00000000e+00,
                                                  5.45079210e-01,
                                                  3.78672766e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 4.78626138],
                                                [1.33537618, 1.64329299],
                                                [2.00306427, 0.7321021],
                                                [2.67075237, 0.58253948],
                                                [3.33844046, 0.56209812],
                                                [4.00612855, 0.59535834],
                                                [4.67381664, 0.54222052],
                                                [5.34150473, 0.40989901],
                                                [6.00919282, 0.63606593],
                                                [6.67688091, 1.11655258]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.66752497e-01,
                                                  3.54027778e-01,
                                                  5.24438272e+00,
                                                  9.66083489e-02,
                                                  4.21591290e-01],
                                                 [2.00000000e+01,
                                                  4.65372892e-01,
                                                  4.73750000e-01,
                                                  6.55375000e+00,
                                                  1.40979734e-01,
                                                  4.84004151e-01],
                                                 [3.00000000e+01,
                                                  3.83810580e-01,
                                                  5.76428571e-01,
                                                  4.31301020e+00,
                                                  1.82537107e-01,
                                                  4.45724641e-01],
                                                 [4.00000000e+01,
                                                  3.21135461e-01,
                                                  6.34583333e-01,
                                                  4.34777778e+00,
                                                  2.25065174e-01,
                                                  4.08617620e-01],
                                                 [5.00000000e+01,
                                                  2.60769450e-01,
                                                  6.96750000e-01,
                                                  2.90980000e+00,
                                                  2.64845124e-01,
                                                  3.62285330e-01],
                                                 [6.00000000e+01,
                                                  2.08824946e-01,
                                                  7.49375000e-01,
                                                  1.13281250e+00,
                                                  3.05573799e-01,
                                                  3.40291418e-01],
                                                 [7.00000000e+01,
                                                  1.71330036e-01,
                                                  7.85833333e-01,
                                                  2.54444444e-01,
                                                  3.52397096e-01,
                                                  3.14565380e-01],
                                                 [8.00000000e+01,
                                                  1.33292499e-01,
                                                  8.25000000e-01,
                                                  1.12500000e-02,
                                                  3.97987653e-01,
                                                  2.82139510e-01],
                                                 [9.00000000e+01,
                                                  1.06593521e-01,
                                                  8.55000000e-01,
                                                  0.00000000e+00,
                                                  4.60740318e-01,
                                                  2.21040917e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 4.78626138],
                                                [1.33537618, 1.64329299],
                                                [2.00306427, 0.7321021],
                                                [2.67075237, 0.58253948],
                                                [3.33844046, 0.56209812],
                                                [4.00612855, 0.59535834],
                                                [4.67381664, 0.54222052],
                                                [5.34150473, 0.40989901],
                                                [6.00919282, 0.63606593],
                                                [6.67688091, 1.11655258]]),
                                       result_4[:10].values,
                                       decimal=5)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_x,self.test_file_3d_xu}...")

    def test_Dynamics_3d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_x}...")
        xu_snapshots = None
        x_snapshots = self.dump_3d_x.snapshots
        ppp = np.array([1, 0, 1])
        t = 10
        qrange = 10.0
        condition = []
        for snapshot in x_snapshots.snapshots:
            condition.append(snapshot.particle_type == 1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)

        # xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=None,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.42433291e-01,
                                                  6.11111111e-01,
                                                  5.49787654e+00,
                                                  1.32230493e+00,
                                                  3.40259239e+01],
                                                 [2.00000000e+01,
                                                  4.44323658e-01,
                                                  4.96125000e-01,
                                                  7.68035938e+00,
                                                  1.73202292e+00,
                                                  2.52658999e+01],
                                                 [3.00000000e+01,
                                                  3.64348746e-01,
                                                  3.97000000e-01,
                                                  4.43171429e+00,
                                                  1.93230391e+00,
                                                  2.18684722e+01],
                                                 [4.00000000e+01,
                                                  3.05491364e-01,
                                                  3.41166667e-01,
                                                  5.18313889e+00,
                                                  2.07811323e+00,
                                                  1.97853386e+01],
                                                 [5.00000000e+01,
                                                  2.46859275e-01,
                                                  2.80200000e-01,
                                                  3.09176000e+00,
                                                  2.28117722e+00,
                                                  1.76317550e+01],
                                                 [6.00000000e+01,
                                                  1.97709558e-01,
                                                  2.30750000e-01,
                                                  1.18118750e+00,
                                                  2.40285919e+00,
                                                  1.63298665e+01],
                                                 [7.00000000e+01,
                                                  1.60984684e-01,
                                                  1.94666667e-01,
                                                  3.00222222e-01,
                                                  2.49979772e+00,
                                                  1.52289773e+01],
                                                 [8.00000000e+01,
                                                  1.22202663e-01,
                                                  1.58500000e-01,
                                                  2.50000000e-04,
                                                  2.94961497e+00,
                                                  1.28042417e+01],
                                                 [9.00000000e+01,
                                                  1.00984724e-01,
                                                  1.31000000e-01,
                                                  0.00000000e+00,
                                                  3.01090360e+00,
                                                  1.20330600e+01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 3.45901494],
                                                [1.33537618, 1.29148094],
                                                [2.00306427, 0.46599746],
                                                [2.67075237, 0.38623844],
                                                [3.33844046, 0.3725604],
                                                [4.00612855, 0.33526793],
                                                [4.67381664, 0.35638944],
                                                [5.34150473, 0.25154654],
                                                [6.00919282, 0.434603],
                                                [6.67688091, 1.00770254]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.67502531e-01,
                                                  6.40833333e-01,
                                                  5.21638889e+00,
                                                  1.25260376e+00,
                                                  3.63493513e+01],
                                                 [2.00000000e+01,
                                                  4.67030805e-01,
                                                  5.20468750e-01,
                                                  6.51998047e+00,
                                                  1.70305517e+00,
                                                  2.61105135e+01],
                                                 [3.00000000e+01,
                                                  3.86949004e-01,
                                                  4.18928571e-01,
                                                  4.10479592e+00,
                                                  1.90026319e+00,
                                                  2.26977113e+01],
                                                 [4.00000000e+01,
                                                  3.24396578e-01,
                                                  3.63125000e-01,
                                                  4.31947917e+00,
                                                  1.97470020e+00,
                                                  2.12493073e+01],
                                                 [5.00000000e+01,
                                                  2.64798811e-01,
                                                  3.00000000e-01,
                                                  2.78350000e+00,
                                                  2.20533333e+00,
                                                  1.86739331e+01],
                                                 [6.00000000e+01,
                                                  2.12784943e-01,
                                                  2.48750000e-01,
                                                  1.18562500e+00,
                                                  2.33876380e+00,
                                                  1.72494183e+01],
                                                 [7.00000000e+01,
                                                  1.74870932e-01,
                                                  2.11666667e-01,
                                                  2.54444444e-01,
                                                  2.37125689e+00,
                                                  1.65229646e+01],
                                                 [8.00000000e+01,
                                                  1.36569876e-01,
                                                  1.72500000e-01,
                                                  1.12500000e-02,
                                                  2.77169618e+00,
                                                  1.40515444e+01],
                                                 [9.00000000e+01,
                                                  1.11330445e-01,
                                                  1.43750000e-01,
                                                  0.00000000e+00,
                                                  2.49453964e+00,
                                                  1.47121982e+01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 3.45901494],
                                                [1.33537618, 1.29148094],
                                                [2.00306427, 0.46599746],
                                                [2.67075237, 0.38623844],
                                                [3.33844046, 0.3725604],
                                                [4.00612855, 0.33526793],
                                                [4.67381664, 0.35638944],
                                                [5.34150473, 0.25154654],
                                                [6.00919282, 0.434603],
                                                [6.67688091, 1.00770254]]),
                                       result_4[:10].values)

        # xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=None,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.42433291e-01,
                                                  3.88888889e-01,
                                                  5.49787654e+00,
                                                  1.32230493e+00,
                                                  3.40259239e+01],
                                                 [2.00000000e+01,
                                                  4.44323658e-01,
                                                  5.03875000e-01,
                                                  7.68035938e+00,
                                                  1.73202292e+00,
                                                  2.52658999e+01],
                                                 [3.00000000e+01,
                                                  3.64348746e-01,
                                                  6.03000000e-01,
                                                  4.43171429e+00,
                                                  1.93230391e+00,
                                                  2.18684722e+01],
                                                 [4.00000000e+01,
                                                  3.05491364e-01,
                                                  6.58833333e-01,
                                                  5.18313889e+00,
                                                  2.07811323e+00,
                                                  1.97853386e+01],
                                                 [5.00000000e+01,
                                                  2.46859275e-01,
                                                  7.19800000e-01,
                                                  3.09176000e+00,
                                                  2.28117722e+00,
                                                  1.76317550e+01],
                                                 [6.00000000e+01,
                                                  1.97709558e-01,
                                                  7.69250000e-01,
                                                  1.18118750e+00,
                                                  2.40285919e+00,
                                                  1.63298665e+01],
                                                 [7.00000000e+01,
                                                  1.60984684e-01,
                                                  8.05333333e-01,
                                                  3.00222222e-01,
                                                  2.49979772e+00,
                                                  1.52289773e+01],
                                                 [8.00000000e+01,
                                                  1.22202663e-01,
                                                  8.41500000e-01,
                                                  2.50000000e-04,
                                                  2.94961497e+00,
                                                  1.28042417e+01],
                                                 [9.00000000e+01,
                                                  1.00984724e-01,
                                                  8.69000000e-01,
                                                  0.00000000e+00,
                                                  3.01090360e+00,
                                                  1.20330600e+01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 4.68162491],
                                                [1.33537618, 1.66576039],
                                                [2.00306427, 0.71463603],
                                                [2.67075237, 0.60058827],
                                                [3.33844046, 0.57077318],
                                                [4.00612855, 0.5987266],
                                                [4.67381664, 0.53774771],
                                                [5.34150473, 0.44092069],
                                                [6.00919282, 0.63161618],
                                                [6.67688091, 1.12292559]]),
                                       result_2[:10].values)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.67502531e-01,
                                                  3.59166667e-01,
                                                  5.21638889e+00,
                                                  1.25260376e+00,
                                                  3.63493513e+01],
                                                 [2.00000000e+01,
                                                  4.67030805e-01,
                                                  4.79531250e-01,
                                                  6.51998047e+00,
                                                  1.70305517e+00,
                                                  2.61105135e+01],
                                                 [3.00000000e+01,
                                                  3.86949004e-01,
                                                  5.81071429e-01,
                                                  4.10479592e+00,
                                                  1.90026319e+00,
                                                  2.26977113e+01],
                                                 [4.00000000e+01,
                                                  3.24396578e-01,
                                                  6.36875000e-01,
                                                  4.31947917e+00,
                                                  1.97470020e+00,
                                                  2.12493073e+01],
                                                 [5.00000000e+01,
                                                  2.64798811e-01,
                                                  7.00000000e-01,
                                                  2.78350000e+00,
                                                  2.20533333e+00,
                                                  1.86739331e+01],
                                                 [6.00000000e+01,
                                                  2.12784943e-01,
                                                  7.51250000e-01,
                                                  1.18562500e+00,
                                                  2.33876380e+00,
                                                  1.72494183e+01],
                                                 [7.00000000e+01,
                                                  1.74870932e-01,
                                                  7.88333333e-01,
                                                  2.54444444e-01,
                                                  2.37125689e+00,
                                                  1.65229646e+01],
                                                 [8.00000000e+01,
                                                  1.36569876e-01,
                                                  8.27500000e-01,
                                                  1.12500000e-02,
                                                  2.77169618e+00,
                                                  1.40515444e+01],
                                                 [9.00000000e+01,
                                                  1.11330445e-01,
                                                  8.56250000e-01,
                                                  0.00000000e+00,
                                                  2.49453964e+00,
                                                  1.47121982e+01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 4.68162491],
                                                [1.33537618, 1.66576039],
                                                [2.00306427, 0.71463603],
                                                [2.67075237, 0.60058827],
                                                [3.33844046, 0.57077318],
                                                [4.00612855, 0.5987266],
                                                [4.67381664, 0.53774771],
                                                [5.34150473, 0.44092069],
                                                [6.00919282, 0.63161618],
                                                [6.67688091, 1.12292559]]),
                                       result_4[:10].values)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_x}...")

    def test_Dynamics_3d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_xu}...")
        xu_snapshots = self.dump_3d_xu.snapshots
        x_snapshots = None
        ppp = np.array([0, 0, 0])
        t = 10
        qrange = 10.0
        condition = []
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type == 1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)

        # xu, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="slow",
                           neighborfile=None,
                           max_neighbors=30
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.41337495e-01,
                                                  6.16444444e-01,
                                                  5.59224691e+00,
                                                  1.11494660e-01,
                                                  6.42042349e-01],
                                                 [2.00000000e+01,
                                                  4.42470289e-01,
                                                  5.01750000e-01,
                                                  7.78893750e+00,
                                                  1.63896915e-01,
                                                  7.29737591e-01],
                                                 [3.00000000e+01,
                                                  3.61157742e-01,
                                                  4.01571429e-01,
                                                  4.66910204e+00,
                                                  2.13100071e-01,
                                                  6.84680761e-01],
                                                 [4.00000000e+01,
                                                  3.02034667e-01,
                                                  3.43833333e-01,
                                                  5.28080556e+00,
                                                  2.63609639e-01,
                                                  6.42069210e-01],
                                                 [5.00000000e+01,
                                                  2.42622250e-01,
                                                  2.83400000e-01,
                                                  3.26584000e+00,
                                                  3.12858357e-01,
                                                  5.69164565e-01],
                                                 [6.00000000e+01,
                                                  1.93722985e-01,
                                                  2.32250000e-01,
                                                  1.13418750e+00,
                                                  3.61507843e-01,
                                                  5.29289260e-01],
                                                 [7.00000000e+01,
                                                  1.56989020e-01,
                                                  1.97000000e-01,
                                                  3.12000000e-01,
                                                  4.16759105e-01,
                                                  4.77159918e-01],
                                                 [8.00000000e+01,
                                                  1.18113321e-01,
                                                  1.61000000e-01,
                                                  1.00000000e-03,
                                                  4.73546209e-01,
                                                  4.42058084e-01],
                                                 [9.00000000e+01,
                                                  9.47865013e-02,
                                                  1.32000000e-01,
                                                  0.00000000e+00,
                                                  5.45079210e-01,
                                                  3.78672766e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 3.47872129],
                                                [1.33537618, 1.28511335],
                                                [2.00306427, 0.46925527],
                                                [2.67075237, 0.34470296],
                                                [3.33844046, 0.36478205],
                                                [4.00612855, 0.33486496],
                                                [4.67381664, 0.35057252],
                                                [5.34150473, 0.22428512],
                                                [6.00919282, 0.4340591],
                                                [6.67688091, 1.02765053]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.66752497e-01,
                                                  6.45972222e-01,
                                                  5.24438272e+00,
                                                  9.66083489e-02,
                                                  4.21591290e-01],
                                                 [2.00000000e+01,
                                                  4.65372892e-01,
                                                  5.26250000e-01,
                                                  6.55375000e+00,
                                                  1.40979734e-01,
                                                  4.84004151e-01],
                                                 [3.00000000e+01,
                                                  3.83810580e-01,
                                                  4.23571429e-01,
                                                  4.31301020e+00,
                                                  1.82537107e-01,
                                                  4.45724641e-01],
                                                 [4.00000000e+01,
                                                  3.21135461e-01,
                                                  3.65416667e-01,
                                                  4.34777778e+00,
                                                  2.25065174e-01,
                                                  4.08617620e-01],
                                                 [5.00000000e+01,
                                                  2.60769450e-01,
                                                  3.03250000e-01,
                                                  2.90980000e+00,
                                                  2.64845124e-01,
                                                  3.62285330e-01],
                                                 [6.00000000e+01,
                                                  2.08824946e-01,
                                                  2.50625000e-01,
                                                  1.13281250e+00,
                                                  3.05573799e-01,
                                                  3.40291418e-01],
                                                 [7.00000000e+01,
                                                  1.71330036e-01,
                                                  2.14166667e-01,
                                                  2.54444444e-01,
                                                  3.52397096e-01,
                                                  3.14565380e-01],
                                                 [8.00000000e+01,
                                                  1.33292499e-01,
                                                  1.75000000e-01,
                                                  1.12500000e-02,
                                                  3.97987653e-01,
                                                  2.82139510e-01],
                                                 [9.00000000e+01,
                                                  1.06593521e-01,
                                                  1.45000000e-01,
                                                  0.00000000e+00,
                                                  4.60740318e-01,
                                                  2.21040917e-01]]),
                                       result_3[:10].values,
                                       )
        np.testing.assert_almost_equal(np.array([[0.66768809, 3.47872129],
                                                [1.33537618, 1.28511335],
                                                [2.00306427, 0.46925527],
                                                [2.67075237, 0.34470296],
                                                [3.33844046, 0.36478205],
                                                [4.00612855, 0.33486496],
                                                [4.67381664, 0.35057252],
                                                [5.34150473, 0.22428512],
                                                [6.00919282, 0.4340591],
                                                [6.67688091, 1.02765053]]),
                                       result_4[:10].values,
                                       decimal=5)

        # xu, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1: 1.0, 2: 1.0},
                           a=0.3,
                           cal_type="fast",
                           neighborfile=None,
                           max_neighbors=100
                           )
        # no condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None,
                    outputfile="test_no_condition_sq4.csv")
        # condition
        dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4,
                    outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.41337495e-01,
                                                  3.83555556e-01,
                                                  5.59224691e+00,
                                                  1.11494660e-01,
                                                  6.42042349e-01],
                                                 [2.00000000e+01,
                                                  4.42470289e-01,
                                                  4.98250000e-01,
                                                  7.78893750e+00,
                                                  1.63896915e-01,
                                                  7.29737591e-01],
                                                 [3.00000000e+01,
                                                  3.61157742e-01,
                                                  5.98428571e-01,
                                                  4.66910204e+00,
                                                  2.13100071e-01,
                                                  6.84680761e-01],
                                                 [4.00000000e+01,
                                                  3.02034667e-01,
                                                  6.56166667e-01,
                                                  5.28080556e+00,
                                                  2.63609639e-01,
                                                  6.42069210e-01],
                                                 [5.00000000e+01,
                                                  2.42622250e-01,
                                                  7.16600000e-01,
                                                  3.26584000e+00,
                                                  3.12858357e-01,
                                                  5.69164565e-01],
                                                 [6.00000000e+01,
                                                  1.93722985e-01,
                                                  7.67750000e-01,
                                                  1.13418750e+00,
                                                  3.61507843e-01,
                                                  5.29289260e-01],
                                                 [7.00000000e+01,
                                                  1.56989020e-01,
                                                  8.03000000e-01,
                                                  3.12000000e-01,
                                                  4.16759105e-01,
                                                  4.77159918e-01],
                                                 [8.00000000e+01,
                                                  1.18113321e-01,
                                                  8.39000000e-01,
                                                  1.00000000e-03,
                                                  4.73546209e-01,
                                                  4.42058084e-01],
                                                 [9.00000000e+01,
                                                  9.47865013e-02,
                                                  8.68000000e-01,
                                                  0.00000000e+00,
                                                  5.45079210e-01,
                                                  3.78672766e-01]]),
                                       result_1[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 4.78626074],
                                                [1.33537618, 1.6432925],
                                                [2.00306427, 0.73210104],
                                                [2.67075237, 0.58253944],
                                                [3.33844046, 0.56209741],
                                                [4.00612855, 0.59535815],
                                                [4.67381664, 0.54222002],
                                                [5.34150473, 0.40989809],
                                                [6.00919282, 0.63606613],
                                                [6.67688091, 1.11655405]]),
                                       result_2[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal(np.array([[1.00000000e+01,
                                                  5.66752497e-01,
                                                  3.54027778e-01,
                                                  5.24438272e+00,
                                                  9.66083489e-02,
                                                  4.21591290e-01],
                                                 [2.00000000e+01,
                                                  4.65372892e-01,
                                                  4.73750000e-01,
                                                  6.55375000e+00,
                                                  1.40979734e-01,
                                                  4.84004151e-01],
                                                 [3.00000000e+01,
                                                  3.83810580e-01,
                                                  5.76428571e-01,
                                                  4.31301020e+00,
                                                  1.82537107e-01,
                                                  4.45724641e-01],
                                                 [4.00000000e+01,
                                                  3.21135461e-01,
                                                  6.34583333e-01,
                                                  4.34777778e+00,
                                                  2.25065174e-01,
                                                  4.08617620e-01],
                                                 [5.00000000e+01,
                                                  2.60769450e-01,
                                                  6.96750000e-01,
                                                  2.90980000e+00,
                                                  2.64845124e-01,
                                                  3.62285330e-01],
                                                 [6.00000000e+01,
                                                  2.08824946e-01,
                                                  7.49375000e-01,
                                                  1.13281250e+00,
                                                  3.05573799e-01,
                                                  3.40291418e-01],
                                                 [7.00000000e+01,
                                                  1.71330036e-01,
                                                  7.85833333e-01,
                                                  2.54444444e-01,
                                                  3.52397096e-01,
                                                  3.14565380e-01],
                                                 [8.00000000e+01,
                                                  1.33292499e-01,
                                                  8.25000000e-01,
                                                  1.12500000e-02,
                                                  3.97987653e-01,
                                                  2.82139510e-01],
                                                 [9.00000000e+01,
                                                  1.06593521e-01,
                                                  8.55000000e-01,
                                                  0.00000000e+00,
                                                  4.60740318e-01,
                                                  2.21040917e-01]]),
                                       result_3[:10].values)
        np.testing.assert_almost_equal(np.array([[0.66768809, 4.78626074],
                                                [1.33537618, 1.6432925],
                                                [2.00306427, 0.73210104],
                                                [2.67075237, 0.58253944],
                                                [3.33844046, 0.56209741],
                                                [4.00612855, 0.59535815],
                                                [4.67381664, 0.54222002],
                                                [5.34150473, 0.40989809],
                                                [6.00919282, 0.63606613],
                                                [6.67688091, 1.11655405]]),
                                       result_4[:10].values,
                                       decimal=5)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_xu}...")

    def test_LogDynamics_2d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_log_x, self.test_file_2d_log_xu}...")
        xu_snapshots = self.dump_2d_log_xu.snapshots
        x_snapshots = self.dump_2d_log_x.snapshots
        ppp = np.array([0, 0])
        condition = (xu_snapshots.snapshots[0].particle_type == 1)

        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=ppp)
        neighborfile = 'neighborlist.dat'

        # xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )

        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.99004127e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.00951932e-04,
                                         -1.65671614e-02],
                                        [1.10000000e-01,
                                         9.73961692e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.67278761e-03,
                                         2.47922998e-03],
                                        [2.00000000e-01,
                                         9.36689188e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         6.62964957e-03,
                                         2.08616903e-02],
                                        [1.10000000e+00,
                                         7.91826907e-01,
                                         9.73000000e-01,
                                         0.00000000e+00,
                                         2.38147105e-02,
                                         7.30150947e-02],
                                        [2.00000000e+00,
                                         6.90313836e-01,
                                         8.99000000e-01,
                                         0.00000000e+00,
                                         3.87689761e-02,
                                         1.79901429e-01],
                                        [1.10000000e+01,
                                         4.56127411e-01,
                                         6.74000000e-01,
                                         0.00000000e+00,
                                         8.87644276e-02,
                                         3.47244235e-01],
                                        [2.00000000e+01,
                                         3.31481504e-01,
                                         5.56000000e-01,
                                         0.00000000e+00,
                                         1.30796577e-01,
                                         2.59409224e-01],
                                        [1.10000000e+02,
                                         2.73535257e-02,
                                         2.34000000e-01,
                                         0.00000000e+00,
                                         3.78405353e-01,
                                         2.02337029e-01],
                                        [2.00000000e+02,
                                         5.50599396e-02,
                                         2.33000000e-01,
                                         0.00000000e+00,
                                         3.81775819e-01,
                                         2.68392690e-01],
                                        [1.10000000e+03,
                                         4.32414155e-03,
                                         1.12000000e-01,
                                         0.00000000e+00,
                                         9.71862967e-01,
                                         1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.99033164e-01, 1.00000000e+00, 0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [1.10000000e-01, 9.74236370e-01, 1.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [2.00000000e-01, 9.35823089e-01, 1.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [1.10000000e+00, 7.93078510e-01, 9.75384615e-01, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [2.00000000e+00, 7.04602830e-01, 9.07692308e-01, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [1.10000000e+01, 4.76322541e-01, 6.95384615e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [2.00000000e+01, 3.30483359e-01, 5.63076923e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [1.10000000e+02, 4.01591949e-02, 2.41538462e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [2.00000000e+02, 7.67369390e-02, 2.36923077e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [1.10000000e+03, -2.04444645e-03, 1.06153846e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)

        # xu, x, slow, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        # log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        # result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        # os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02, 9.98862702e-01, 1.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [1.10000000e-01, 9.70926555e-01, 1.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [2.00000000e-01, 9.33593192e-01, 1.00000000e+00, 0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [1.10000000e+00, 8.61662385e-01, 9.91000000e-01, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [2.00000000e+00, 8.27684416e-01, 9.63000000e-01, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [1.10000000e+01, 7.17897230e-01, 8.92000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [2.00000000e+01, 6.38977561e-01, 8.19000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [1.10000000e+02, 3.57359912e-01, 5.69000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [2.00000000e+02, 3.64432615e-01, 5.73000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [1.10000000e+03, 1.65484175e-01, 3.14000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        # np.testing.assert_almost_equal([],result_2[:10].values)

        # xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.99004127e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.00951932e-04,
                                         -1.65671614e-02],
                                        [1.10000000e-01,
                                         9.73961692e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.67278761e-03,
                                         2.47922998e-03],
                                        [2.00000000e-01,
                                         9.36689188e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         6.62964957e-03,
                                         2.08616903e-02],
                                        [1.10000000e+00,
                                         7.91826907e-01,
                                         2.70000000e-02,
                                         0.00000000e+00,
                                         2.38147105e-02,
                                         7.30150947e-02],
                                        [2.00000000e+00,
                                         6.90313836e-01,
                                         1.01000000e-01,
                                         0.00000000e+00,
                                         3.87689761e-02,
                                         1.79901429e-01],
                                        [1.10000000e+01,
                                         4.56127411e-01,
                                         3.26000000e-01,
                                         0.00000000e+00,
                                         8.87644276e-02,
                                         3.47244235e-01],
                                        [2.00000000e+01,
                                         3.31481504e-01,
                                         4.44000000e-01,
                                         0.00000000e+00,
                                         1.30796577e-01,
                                         2.59409224e-01],
                                        [1.10000000e+02,
                                         2.73535257e-02,
                                         7.66000000e-01,
                                         0.00000000e+00,
                                         3.78405353e-01,
                                         2.02337029e-01],
                                        [2.00000000e+02,
                                         5.50599396e-02,
                                         7.67000000e-01,
                                         0.00000000e+00,
                                         3.81775819e-01,
                                         2.68392690e-01],
                                        [1.10000000e+03,
                                         4.32414155e-03,
                                         8.88000000e-01,
                                         0.00000000e+00,
                                         9.71862967e-01,
                                         1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.99033164e-01, 0.00000000e+00, 0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [1.10000000e-01, 9.74236370e-01, 0.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [2.00000000e-01, 9.35823089e-01, 0.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [1.10000000e+00, 7.93078510e-01, 2.46153846e-02, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [2.00000000e+00, 7.04602830e-01, 9.23076923e-02, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [1.10000000e+01, 4.76322541e-01, 3.04615385e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [2.00000000e+01, 3.30483359e-01, 4.36923077e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [1.10000000e+02, 4.01591949e-02, 7.58461538e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [2.00000000e+02, 7.67369390e-02, 7.63076923e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [1.10000000e+03, -2.04444645e-03, 8.93846154e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)

        # xu, x, fast, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )

        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        # log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        # result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        # os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02, 9.98862702e-01, 0.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [1.10000000e-01, 9.70926555e-01, 0.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [2.00000000e-01, 9.33593192e-01, 0.00000000e+00, 0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [1.10000000e+00, 8.61662385e-01, 9.00000000e-03, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [2.00000000e+00, 8.27684416e-01, 3.70000000e-02, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [1.10000000e+01, 7.17897230e-01, 1.08000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [2.00000000e+01, 6.38977561e-01, 1.81000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [1.10000000e+02, 3.57359912e-01, 4.31000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [2.00000000e+02, 3.64432615e-01, 4.27000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [1.10000000e+03, 1.65484175e-01, 6.86000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        # np.testing.assert_almost_equal([],result_2[:10].values)

        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_log_x,self.test_file_2d_log_xu}...")

    def test_LogDynamics_2d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_log_x}...")
        xu_snapshots = None
        x_snapshots = self.dump_2d_log_x.snapshots
        condition = (x_snapshots.snapshots[0].particle_type == 1)
        ppp = np.array([1, 0])
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=ppp)
        neighborfile = 'neighborlist.dat'

        # x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )

        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.99004127e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.00951937e-04,
                                         -1.66244715e-02],
                                        [1.10000000e-01,
                                         9.73407171e-01,
                                         9.98000000e-01,
                                         0.00000000e+00,
                                         1.66307070e+00,
                                         2.48199661e+02],
                                        [2.00000000e-01,
                                         9.35825056e-01,
                                         9.97000000e-01,
                                         0.00000000e+00,
                                         2.49616085e+00,
                                         1.64784970e+02],
                                        [1.10000000e+00,
                                         7.90936761e-01,
                                         9.70000000e-01,
                                         0.00000000e+00,
                                         2.51239315e+00,
                                         1.62529909e+02],
                                        [2.00000000e+00,
                                         6.88625594e-01,
                                         8.95000000e-01,
                                         0.00000000e+00,
                                         4.15913517e+00,
                                         9.71594000e+01],
                                        [1.10000000e+01,
                                         4.54025701e-01,
                                         6.71000000e-01,
                                         0.00000000e+00,
                                         5.04226199e+00,
                                         7.94421760e+01],
                                        [2.00000000e+01,
                                         3.30174788e-01,
                                         5.53000000e-01,
                                         0.00000000e+00,
                                         3.43019255e+00,
                                         1.14669853e+02],
                                        [1.10000000e+02,
                                         2.58498614e-02,
                                         2.33000000e-01,
                                         0.00000000e+00,
                                         1.14907876e+01,
                                         3.25131387e+01],
                                        [2.00000000e+02,
                                         5.54434609e-02,
                                         2.33000000e-01,
                                         0.00000000e+00,
                                         1.07529011e+01,
                                         3.48872588e+01],
                                        [1.10000000e+03,
                                         3.91603498e-03,
                                         1.12000000e-01,
                                         0.00000000e+00,
                                         2.15698479e+01,
                                         1.60366508e+01]],
                                       result_1[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.99033173e-01, 1.00000000e+00, 0.00000000e+00, 9.80036933e-05, -7.26690771e-02],
                                        [1.10000000e-01, 9.73383227e-01, 9.96923077e-01, 0.00000000e+00, 2.55710163e+00, 1.61165899e+02],
                                        [2.00000000e-01, 9.34493858e-01, 9.95384615e-01, 0.00000000e+00, 3.83677352e+00, 1.06955457e+02],
                                        [1.10000000e+00, 7.91708997e-01, 9.70769231e-01, 0.00000000e+00, 3.85211907e+00, 1.06018666e+02],
                                        [2.00000000e+00, 7.02398006e-01, 9.01538462e-01, 0.00000000e+00, 5.12252716e+00, 7.90971317e+01],
                                        [1.10000000e+01, 4.73089678e-01, 6.90769231e-01, 0.00000000e+00, 7.70348319e+00, 5.20207659e+01],
                                        [2.00000000e+01, 3.29049278e-01, 5.60000000e-01, 0.00000000e+00, 3.92954695e+00, 1.00583671e+02],
                                        [1.10000000e+02, 3.85923832e-02, 2.41538462e-01, 0.00000000e+00, 1.39152611e+01, 2.70960478e+01],
                                        [2.00000000e+02, 7.74662786e-02, 2.36923077e-01, 0.00000000e+00, 1.15110626e+01, 3.30037282e+01],
                                        [1.10000000e+03, -2.05115033e-03, 1.06153846e-01, 0.00000000e+00, 2.32947100e+01, 1.48739439e+01]],
                                       result_2[:10].values, decimal=5)

        # x, slow, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98864364e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.15128487e-04,
                                         -1.85681823e-02],
                                        [1.10000000e-01,
                                         9.68970306e-01,
                                         9.92000000e-01,
                                         0.00000000e+00,
                                         1.80628087e+00,
                                         2.11637920e+02],
                                        [2.00000000e-01,
                                         9.31074625e-01,
                                         9.90000000e-01,
                                         0.00000000e+00,
                                         2.29633139e+00,
                                         1.29879227e+02],
                                        [1.10000000e+00,
                                         8.61434204e-01,
                                         9.77000000e-01,
                                         0.00000000e+00,
                                         2.73127867e+00,
                                         1.37362615e+02],
                                        [2.00000000e+00,
                                         8.28324975e-01,
                                         9.45000000e-01,
                                         0.00000000e+00,
                                         4.11314221e+00,
                                         7.96523310e+01],
                                        [1.10000000e+01,
                                         7.20513119e-01,
                                         8.74000000e-01,
                                         0.00000000e+00,
                                         5.07522943e+00,
                                         6.61672114e+01],
                                        [2.00000000e+01,
                                         6.40953026e-01,
                                         8.07000000e-01,
                                         0.00000000e+00,
                                         3.25629978e+00,
                                         9.56940464e+01],
                                        [1.10000000e+02,
                                         3.49542583e-01,
                                         5.41000000e-01,
                                         0.00000000e+00,
                                         1.03848144e+01,
                                         2.64342521e+01],
                                        [2.00000000e+02,
                                         3.58691271e-01,
                                         5.54000000e-01,
                                         0.00000000e+00,
                                         9.42305841e+00,
                                         2.82717939e+01],
                                        [1.10000000e+03,
                                         1.69131005e-01,
                                         3.10000000e-01,
                                         0.00000000e+00,
                                         1.39070530e+01,
                                         1.47264225e+01]],
                                       result_1[:10].values,
                                       decimal=5)
        # np.testing.assert_almost_equal([],result_2[:10].values,decimal=5)

        # x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.99004127e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.00951937e-04,
                                         -1.66244715e-02],
                                        [1.10000000e-01,
                                         9.73407171e-01,
                                         2.00000000e-03,
                                         0.00000000e+00,
                                         1.66307070e+00,
                                         2.48199661e+02],
                                        [2.00000000e-01,
                                         9.35825056e-01,
                                         3.00000000e-03,
                                         0.00000000e+00,
                                         2.49616085e+00,
                                         1.64784970e+02],
                                        [1.10000000e+00,
                                         7.90936761e-01,
                                         3.00000000e-02,
                                         0.00000000e+00,
                                         2.51239315e+00,
                                         1.62529909e+02],
                                        [2.00000000e+00,
                                         6.88625594e-01,
                                         1.05000000e-01,
                                         0.00000000e+00,
                                         4.15913517e+00,
                                         9.71594000e+01],
                                        [1.10000000e+01,
                                         4.54025701e-01,
                                         3.29000000e-01,
                                         0.00000000e+00,
                                         5.04226199e+00,
                                         7.94421760e+01],
                                        [2.00000000e+01,
                                         3.30174788e-01,
                                         4.47000000e-01,
                                         0.00000000e+00,
                                         3.43019255e+00,
                                         1.14669853e+02],
                                        [1.10000000e+02,
                                         2.58498614e-02,
                                         7.67000000e-01,
                                         0.00000000e+00,
                                         1.14907876e+01,
                                         3.25131387e+01],
                                        [2.00000000e+02,
                                         5.54434609e-02,
                                         7.67000000e-01,
                                         0.00000000e+00,
                                         1.07529011e+01,
                                         3.48872588e+01],
                                        [1.10000000e+03,
                                         3.91603498e-03,
                                         8.88000000e-01,
                                         0.00000000e+00,
                                         2.15698479e+01,
                                         1.60366508e+01]],
                                       result_1[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.99033173e-01, 0.00000000e+00, 0.00000000e+00, 9.80036933e-05, -7.26690771e-02],
                                        [1.10000000e-01, 9.73383227e-01, 3.07692308e-03, 0.00000000e+00, 2.55710163e+00, 1.61165899e+02],
                                        [2.00000000e-01, 9.34493858e-01, 4.61538462e-03, 0.00000000e+00, 3.83677352e+00, 1.06955457e+02],
                                        [1.10000000e+00, 7.91708997e-01, 2.92307692e-02, 0.00000000e+00, 3.85211907e+00, 1.06018666e+02],
                                        [2.00000000e+00, 7.02398006e-01, 9.84615385e-02, 0.00000000e+00, 5.12252716e+00, 7.90971317e+01],
                                        [1.10000000e+01, 4.73089678e-01, 3.09230769e-01, 0.00000000e+00, 7.70348319e+00, 5.20207659e+01],
                                        [2.00000000e+01, 3.29049278e-01, 4.40000000e-01, 0.00000000e+00, 3.92954695e+00, 1.00583671e+02],
                                        [1.10000000e+02, 3.85923832e-02, 7.58461538e-01, 0.00000000e+00, 1.39152611e+01, 2.70960478e+01],
                                        [2.00000000e+02, 7.74662786e-02, 7.63076923e-01, 0.00000000e+00, 1.15110626e+01, 3.30037282e+01],
                                        [1.10000000e+03, -2.05115033e-03, 8.93846154e-01, 0.00000000e+00, 2.32947100e+01, 1.48739439e+01]],
                                       result_2[:10].values, decimal=5)
        # x, fast, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        # log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        # result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        # os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98864364e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.15128487e-04,
                                         -1.85681823e-02],
                                        [1.10000000e-01,
                                         9.68970306e-01,
                                         8.00000000e-03,
                                         0.00000000e+00,
                                         1.80628087e+00,
                                         2.11637920e+02],
                                        [2.00000000e-01,
                                         9.31074625e-01,
                                         1.00000000e-02,
                                         0.00000000e+00,
                                         2.29633139e+00,
                                         1.29879227e+02],
                                        [1.10000000e+00,
                                         8.61434204e-01,
                                         2.30000000e-02,
                                         0.00000000e+00,
                                         2.73127867e+00,
                                         1.37362615e+02],
                                        [2.00000000e+00,
                                         8.28324975e-01,
                                         5.50000000e-02,
                                         0.00000000e+00,
                                         4.11314221e+00,
                                         7.96523310e+01],
                                        [1.10000000e+01,
                                         7.20513119e-01,
                                         1.26000000e-01,
                                         0.00000000e+00,
                                         5.07522943e+00,
                                         6.61672114e+01],
                                        [2.00000000e+01,
                                         6.40953026e-01,
                                         1.93000000e-01,
                                         0.00000000e+00,
                                         3.25629978e+00,
                                         9.56940464e+01],
                                        [1.10000000e+02,
                                         3.49542583e-01,
                                         4.59000000e-01,
                                         0.00000000e+00,
                                         1.03848144e+01,
                                         2.64342521e+01],
                                        [2.00000000e+02,
                                         3.58691271e-01,
                                         4.46000000e-01,
                                         0.00000000e+00,
                                         9.42305841e+00,
                                         2.82717939e+01],
                                        [1.10000000e+03,
                                         1.69131005e-01,
                                         6.90000000e-01,
                                         0.00000000e+00,
                                         1.39070530e+01,
                                         1.47264225e+01]],
                                       result_1[:10].values,
                                       decimal=5)
        # np.testing.assert_almost_equal([],result_2[:10].values,decimal=5)
        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_log_x}...")

    def test_LogDynamics_2d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_log_x, self.test_file_2d_log_xu}...")
        xu_snapshots = self.dump_2d_log_xu.snapshots
        x_snapshots = None
        ppp = np.array([0, 0])
        condition = (xu_snapshots.snapshots[0].particle_type == 1)

        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=np.array([0, 0]))
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=np.array([0, 0]))
        neighborfile = 'neighborlist.dat'

        # xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.99004127e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.00951932e-04,
                                         -1.65671614e-02],
                                        [1.10000000e-01,
                                         9.73961692e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.67278761e-03,
                                         2.47922998e-03],
                                        [2.00000000e-01,
                                         9.36689188e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         6.62964957e-03,
                                         2.08616903e-02],
                                        [1.10000000e+00,
                                         7.91826907e-01,
                                         9.73000000e-01,
                                         0.00000000e+00,
                                         2.38147105e-02,
                                         7.30150947e-02],
                                        [2.00000000e+00,
                                         6.90313836e-01,
                                         8.99000000e-01,
                                         0.00000000e+00,
                                         3.87689761e-02,
                                         1.79901429e-01],
                                        [1.10000000e+01,
                                         4.56127411e-01,
                                         6.74000000e-01,
                                         0.00000000e+00,
                                         8.87644276e-02,
                                         3.47244235e-01],
                                        [2.00000000e+01,
                                         3.31481504e-01,
                                         5.56000000e-01,
                                         0.00000000e+00,
                                         1.30796577e-01,
                                         2.59409224e-01],
                                        [1.10000000e+02,
                                         2.73535257e-02,
                                         2.34000000e-01,
                                         0.00000000e+00,
                                         3.78405353e-01,
                                         2.02337029e-01],
                                        [2.00000000e+02,
                                         5.50599396e-02,
                                         2.33000000e-01,
                                         0.00000000e+00,
                                         3.81775819e-01,
                                         2.68392690e-01],
                                        [1.10000000e+03,
                                         4.32414155e-03,
                                         1.12000000e-01,
                                         0.00000000e+00,
                                         9.71862967e-01,
                                         1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.99033164e-01, 1.00000000e+00, 0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [1.10000000e-01, 9.74236370e-01, 1.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [2.00000000e-01, 9.35823089e-01, 1.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [1.10000000e+00, 7.93078510e-01, 9.75384615e-01, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [2.00000000e+00, 7.04602830e-01, 9.07692308e-01, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [1.10000000e+01, 4.76322541e-01, 6.95384615e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [2.00000000e+01, 3.30483359e-01, 5.63076923e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [1.10000000e+02, 4.01591949e-02, 2.41538462e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [2.00000000e+02, 7.67369390e-02, 2.36923077e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [1.10000000e+03, -2.04444645e-03, 1.06153846e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)
        # xu, x, slow, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )

        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        # log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        # result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        # os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02, 9.98862702e-01, 1.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [1.10000000e-01, 9.70926555e-01, 1.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [2.00000000e-01, 9.33593192e-01, 1.00000000e+00, 0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [1.10000000e+00, 8.61662385e-01, 9.91000000e-01, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [2.00000000e+00, 8.27684416e-01, 9.63000000e-01, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [1.10000000e+01, 7.17897230e-01, 8.92000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [2.00000000e+01, 6.38977561e-01, 8.19000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [1.10000000e+02, 3.57359912e-01, 5.69000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [2.00000000e+02, 3.64432615e-01, 5.73000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [1.10000000e+03, 1.65484175e-01, 3.14000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        # np.testing.assert_almost_equal([],result_2[:10].values)

        # xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=np.array([0, 0]),
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )

        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.99004127e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.00951932e-04,
                                         -1.65671614e-02],
                                        [1.10000000e-01,
                                         9.73961692e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.67278761e-03,
                                         2.47922998e-03],
                                        [2.00000000e-01,
                                         9.36689188e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         6.62964957e-03,
                                         2.08616903e-02],
                                        [1.10000000e+00,
                                         7.91826907e-01,
                                         2.70000000e-02,
                                         0.00000000e+00,
                                         2.38147105e-02,
                                         7.30150947e-02],
                                        [2.00000000e+00,
                                         6.90313836e-01,
                                         1.01000000e-01,
                                         0.00000000e+00,
                                         3.87689761e-02,
                                         1.79901429e-01],
                                        [1.10000000e+01,
                                         4.56127411e-01,
                                         3.26000000e-01,
                                         0.00000000e+00,
                                         8.87644276e-02,
                                         3.47244235e-01],
                                        [2.00000000e+01,
                                         3.31481504e-01,
                                         4.44000000e-01,
                                         0.00000000e+00,
                                         1.30796577e-01,
                                         2.59409224e-01],
                                        [1.10000000e+02,
                                         2.73535257e-02,
                                         7.66000000e-01,
                                         0.00000000e+00,
                                         3.78405353e-01,
                                         2.02337029e-01],
                                        [2.00000000e+02,
                                         5.50599396e-02,
                                         7.67000000e-01,
                                         0.00000000e+00,
                                         3.81775819e-01,
                                         2.68392690e-01],
                                        [1.10000000e+03,
                                         4.32414155e-03,
                                         8.88000000e-01,
                                         0.00000000e+00,
                                         9.71862967e-01,
                                         1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.99033164e-01, 0.00000000e+00, 0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [1.10000000e-01, 9.74236370e-01, 0.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [2.00000000e-01, 9.35823089e-01, 0.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [1.10000000e+00, 7.93078510e-01, 2.46153846e-02, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [2.00000000e+00, 7.04602830e-01, 9.23076923e-02, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [1.10000000e+01, 4.76322541e-01, 3.04615385e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [2.00000000e+01, 3.30483359e-01, 4.36923077e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [1.10000000e+02, 4.01591949e-02, 7.58461538e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [2.00000000e+02, 7.67369390e-02, 7.63076923e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [1.10000000e+03, -2.04444645e-03, 8.93846154e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)

        # xu, x, fast, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        # log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        # result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        # os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02, 9.98862702e-01, 0.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [1.10000000e-01, 9.70926555e-01, 0.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [2.00000000e-01, 9.33593192e-01, 0.00000000e+00, 0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [1.10000000e+00, 8.61662385e-01, 9.00000000e-03, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [2.00000000e+00, 8.27684416e-01, 3.70000000e-02, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [1.10000000e+01, 7.17897230e-01, 1.08000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [2.00000000e+01, 6.38977561e-01, 1.81000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [1.10000000e+02, 3.57359912e-01, 4.31000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [2.00000000e+02, 3.64432615e-01, 4.27000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [1.10000000e+03, 1.65484175e-01, 6.86000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        # np.testing.assert_almost_equal([],result_2[:10].values)

        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_log_xu}...")

    def test_LogDynamics_3d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_log_x, self.test_file_3d_log_xu}...")
        xu_snapshots = self.dump_3d_log_xu.snapshots
        x_snapshots = self.dump_3d_log_x.snapshots
        condition = (xu_snapshots.snapshots[0].particle_type == 1)
        ppp = np.array([0, 0, 0])

        # xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=None,
                                  max_neighbors=30)
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98794986e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.83249245e-04,
                                         7.62729112e-03],
                                        [8.00000000e-02,
                                         9.81788619e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.79353312e-03,
                                         8.83924401e-03],
                                        [1.40000000e-01,
                                         9.51758118e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         7.51534401e-03,
                                         3.27266536e-03],
                                        [2.00000000e-01,
                                         9.20215225e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.26307715e-02,
                                         -7.52556624e-04],
                                        [8.00000000e-01,
                                         7.63523409e-01,
                                         8.87000000e-01,
                                         0.00000000e+00,
                                         4.16240792e-02,
                                         9.39923902e-02],
                                        [1.40000000e+00,
                                         7.06977773e-01,
                                         8.35000000e-01,
                                         0.00000000e+00,
                                         5.56373684e-02,
                                         3.69059435e-01],
                                        [2.00000000e+00,
                                         6.74200527e-01,
                                         7.87000000e-01,
                                         0.00000000e+00,
                                         6.28006979e-02,
                                         2.98372497e-01],
                                        [8.00000000e+00,
                                         4.89460164e-01,
                                         5.23000000e-01,
                                         0.00000000e+00,
                                         1.24775241e-01,
                                         3.90773736e-01],
                                        [1.40000000e+01,
                                         3.33250067e-01,
                                         3.65000000e-01,
                                         0.00000000e+00,
                                         2.02048496e-01,
                                         3.42854683e-01],
                                        [2.00000000e+01,
                                         3.01752850e-01,
                                         3.47000000e-01,
                                         0.00000000e+00,
                                         2.36770363e-01,
                                         4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98797156e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.82916144e-04,
                                         5.08578384e-04],
                                        [8.00000000e-02,
                                         9.81866650e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.78100867e-03,
                                         1.33905635e-02],
                                        [1.40000000e-01,
                                         9.52395484e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         7.40924729e-03,
                                         9.07554895e-03],
                                        [2.00000000e-01,
                                         9.22445634e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.22484525e-02,
                                         2.58585214e-03],
                                        [8.00000000e-01,
                                         7.80250930e-01,
                                         9.10000000e-01,
                                         0.00000000e+00,
                                         3.79490640e-02,
                                         6.10574946e-02],
                                        [1.40000000e+00,
                                         7.33305660e-01,
                                         8.72500000e-01,
                                         0.00000000e+00,
                                         4.87008876e-02,
                                         2.97100024e-01],
                                        [2.00000000e+00,
                                         7.01825530e-01,
                                         8.23750000e-01,
                                         0.00000000e+00,
                                         5.54054537e-02,
                                         1.94253070e-01],
                                        [8.00000000e+00,
                                         5.12756184e-01,
                                         5.51250000e-01,
                                         0.00000000e+00,
                                         1.10604697e-01,
                                         2.42693270e-01],
                                        [1.40000000e+01,
                                         3.56859577e-01,
                                         3.90000000e-01,
                                         0.00000000e+00,
                                         1.75137652e-01,
                                         1.53847276e-01],
                                        [2.00000000e+01,
                                         3.29340145e-01,
                                         3.77500000e-01,
                                         0.00000000e+00,
                                         2.04353337e-01,
                                         2.83847908e-01]],
                                       result_2[:10].values)

        # xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=None,
                                  max_neighbors=30)
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98794986e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.83249245e-04,
                                         7.62729112e-03],
                                        [8.00000000e-02,
                                         9.81788619e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.79353312e-03,
                                         8.83924401e-03],
                                        [1.40000000e-01,
                                         9.51758118e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         7.51534401e-03,
                                         3.27266536e-03],
                                        [2.00000000e-01,
                                         9.20215225e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.26307715e-02,
                                         -7.52556624e-04],
                                        [8.00000000e-01,
                                         7.63523409e-01,
                                         1.13000000e-01,
                                         0.00000000e+00,
                                         4.16240792e-02,
                                         9.39923902e-02],
                                        [1.40000000e+00,
                                         7.06977773e-01,
                                         1.65000000e-01,
                                         0.00000000e+00,
                                         5.56373684e-02,
                                         3.69059435e-01],
                                        [2.00000000e+00,
                                         6.74200527e-01,
                                         2.13000000e-01,
                                         0.00000000e+00,
                                         6.28006979e-02,
                                         2.98372497e-01],
                                        [8.00000000e+00,
                                         4.89460164e-01,
                                         4.77000000e-01,
                                         0.00000000e+00,
                                         1.24775241e-01,
                                         3.90773736e-01],
                                        [1.40000000e+01,
                                         3.33250067e-01,
                                         6.35000000e-01,
                                         0.00000000e+00,
                                         2.02048496e-01,
                                         3.42854683e-01],
                                        [2.00000000e+01,
                                         3.01752850e-01,
                                         6.53000000e-01,
                                         0.00000000e+00,
                                         2.36770363e-01,
                                         4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98797156e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.82916144e-04,
                                         5.08578384e-04],
                                        [8.00000000e-02,
                                         9.81866650e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.78100867e-03,
                                         1.33905635e-02],
                                        [1.40000000e-01,
                                         9.52395484e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         7.40924729e-03,
                                         9.07554895e-03],
                                        [2.00000000e-01,
                                         9.22445634e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.22484525e-02,
                                         2.58585214e-03],
                                        [8.00000000e-01,
                                         7.80250930e-01,
                                         9.00000000e-02,
                                         0.00000000e+00,
                                         3.79490640e-02,
                                         6.10574946e-02],
                                        [1.40000000e+00,
                                         7.33305660e-01,
                                         1.27500000e-01,
                                         0.00000000e+00,
                                         4.87008876e-02,
                                         2.97100024e-01],
                                        [2.00000000e+00,
                                         7.01825530e-01,
                                         1.76250000e-01,
                                         0.00000000e+00,
                                         5.54054537e-02,
                                         1.94253070e-01],
                                        [8.00000000e+00,
                                         5.12756184e-01,
                                         4.48750000e-01,
                                         0.00000000e+00,
                                         1.10604697e-01,
                                         2.42693270e-01],
                                        [1.40000000e+01,
                                         3.56859577e-01,
                                         6.10000000e-01,
                                         0.00000000e+00,
                                         1.75137652e-01,
                                         1.53847276e-01],
                                        [2.00000000e+01,
                                         3.29340145e-01,
                                         6.22500000e-01,
                                         0.00000000e+00,
                                         2.04353337e-01,
                                         2.83847908e-01]],
                                       result_2[:10].values)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_log_xu,self.test_file_3d_log_x}...")

    def test_LogDynamics_3d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_log_x}...")
        xu_snapshots = None
        x_snapshots = self.dump_3d_log_x.snapshots
        condition = (x_snapshots.snapshots[0].particle_type == 1)
        ppp = np.array([1, 0, 1])

        # xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=None,
                                  max_neighbors=30)
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98794938e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.83256494e-04,
                                         7.63917403e-03],
                                        [8.00000000e-02,
                                         9.81788442e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.79356020e-03,
                                         8.83438470e-03],
                                        [1.40000000e-01,
                                         9.50172480e-01,
                                         9.97000000e-01,
                                         0.00000000e+00,
                                         2.70337159e-01,
                                         1.88077806e+02],
                                        [2.00000000e-01,
                                         9.18160727e-01,
                                         9.95000000e-01,
                                         0.00000000e+00,
                                         4.47054811e-01,
                                         1.12380687e+02],
                                        [8.00000000e-01,
                                         7.62698739e-01,
                                         8.83000000e-01,
                                         0.00000000e+00,
                                         7.23713200e-01,
                                         6.57568728e+01],
                                        [1.40000000e+00,
                                         7.07031033e-01,
                                         8.29000000e-01,
                                         0.00000000e+00,
                                         8.17615823e-01,
                                         5.70228377e+01],
                                        [2.00000000e+00,
                                         6.73944627e-01,
                                         7.86000000e-01,
                                         0.00000000e+00,
                                         4.87666214e-01,
                                         9.04120365e+01],
                                        [8.00000000e+00,
                                         4.90589212e-01,
                                         5.19000000e-01,
                                         0.00000000e+00,
                                         1.21864955e+00,
                                         3.63950212e+01],
                                        [1.40000000e+01,
                                         3.38527850e-01,
                                         3.64000000e-01,
                                         0.00000000e+00,
                                         1.83412839e+00,
                                         2.29917076e+01],
                                        [2.00000000e+01,
                                         3.04131179e-01,
                                         3.40000000e-01,
                                         0.00000000e+00,
                                         2.20950081e+00,
                                         1.91470186e+01]],
                                       result_1[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98797109e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.82923340e-04,
                                         5.20394664e-04],
                                        [8.00000000e-02,
                                         9.81866477e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.78103508e-03,
                                         1.33854969e-02],
                                        [1.40000000e-01,
                                         9.51106637e-01,
                                         9.97500000e-01,
                                         0.00000000e+00,
                                         2.26149609e-01,
                                         2.23576188e+02],
                                        [2.00000000e-01,
                                         9.21240570e-01,
                                         9.96250000e-01,
                                         0.00000000e+00,
                                         3.35917668e-01,
                                         1.47628949e+02],
                                        [8.00000000e-01,
                                         7.79545330e-01,
                                         9.06250000e-01,
                                         0.00000000e+00,
                                         5.71049476e-01,
                                         8.28087884e+01],
                                        [1.40000000e+00,
                                         7.32988949e-01,
                                         8.67500000e-01,
                                         0.00000000e+00,
                                         5.79522242e-01,
                                         7.97066414e+01],
                                        [2.00000000e+00,
                                         7.01518955e-01,
                                         8.23750000e-01,
                                         0.00000000e+00,
                                         2.68613346e-01,
                                         1.50800767e+02],
                                        [8.00000000e+00,
                                         5.13184259e-01,
                                         5.46250000e-01,
                                         0.00000000e+00,
                                         1.05950477e+00,
                                         4.20002154e+01],
                                        [1.40000000e+01,
                                         3.61867323e-01,
                                         3.88750000e-01,
                                         0.00000000e+00,
                                         1.72157260e+00,
                                         2.50070263e+01],
                                        [2.00000000e+01,
                                         3.31157335e-01,
                                         3.70000000e-01,
                                         0.00000000e+00,
                                         1.95158053e+00,
                                         2.18582771e+01]],
                                       result_2[:10].values,
                                       decimal=5)

        # xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=None,
                                  max_neighbors=30)
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98794938e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.83256494e-04,
                                         7.63917403e-03],
                                        [8.00000000e-02,
                                         9.81788442e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.79356020e-03,
                                         8.83438470e-03],
                                        [1.40000000e-01,
                                         9.50172480e-01,
                                         3.00000000e-03,
                                         0.00000000e+00,
                                         2.70337159e-01,
                                         1.88077806e+02],
                                        [2.00000000e-01,
                                         9.18160727e-01,
                                         5.00000000e-03,
                                         0.00000000e+00,
                                         4.47054811e-01,
                                         1.12380687e+02],
                                        [8.00000000e-01,
                                         7.62698739e-01,
                                         1.17000000e-01,
                                         0.00000000e+00,
                                         7.23713200e-01,
                                         6.57568728e+01],
                                        [1.40000000e+00,
                                         7.07031033e-01,
                                         1.71000000e-01,
                                         0.00000000e+00,
                                         8.17615823e-01,
                                         5.70228377e+01],
                                        [2.00000000e+00,
                                         6.73944627e-01,
                                         2.14000000e-01,
                                         0.00000000e+00,
                                         4.87666214e-01,
                                         9.04120365e+01],
                                        [8.00000000e+00,
                                         4.90589212e-01,
                                         4.81000000e-01,
                                         0.00000000e+00,
                                         1.21864955e+00,
                                         3.63950212e+01],
                                        [1.40000000e+01,
                                         3.38527850e-01,
                                         6.36000000e-01,
                                         0.00000000e+00,
                                         1.83412839e+00,
                                         2.29917076e+01],
                                        [2.00000000e+01,
                                         3.04131179e-01,
                                         6.60000000e-01,
                                         0.00000000e+00,
                                         2.20950081e+00,
                                         1.91470186e+01]],
                                       result_1[:10].values,
                                       decimal=5)
        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98797109e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.82923340e-04,
                                         5.20394664e-04],
                                        [8.00000000e-02,
                                         9.81866477e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.78103508e-03,
                                         1.33854969e-02],
                                        [1.40000000e-01,
                                         9.51106637e-01,
                                         2.50000000e-03,
                                         0.00000000e+00,
                                         2.26149609e-01,
                                         2.23576188e+02],
                                        [2.00000000e-01,
                                         9.21240570e-01,
                                         3.75000000e-03,
                                         0.00000000e+00,
                                         3.35917668e-01,
                                         1.47628949e+02],
                                        [8.00000000e-01,
                                         7.79545330e-01,
                                         9.37500000e-02,
                                         0.00000000e+00,
                                         5.71049476e-01,
                                         8.28087884e+01],
                                        [1.40000000e+00,
                                         7.32988949e-01,
                                         1.32500000e-01,
                                         0.00000000e+00,
                                         5.79522242e-01,
                                         7.97066414e+01],
                                        [2.00000000e+00,
                                         7.01518955e-01,
                                         1.76250000e-01,
                                         0.00000000e+00,
                                         2.68613346e-01,
                                         1.50800767e+02],
                                        [8.00000000e+00,
                                         5.13184259e-01,
                                         4.53750000e-01,
                                         0.00000000e+00,
                                         1.05950477e+00,
                                         4.20002154e+01],
                                        [1.40000000e+01,
                                         3.61867323e-01,
                                         6.11250000e-01,
                                         0.00000000e+00,
                                         1.72157260e+00,
                                         2.50070263e+01],
                                        [2.00000000e+01,
                                         3.31157335e-01,
                                         6.30000000e-01,
                                         0.00000000e+00,
                                         1.95158053e+00,
                                         2.18582771e+01]],
                                       result_2[:10].values,
                                       decimal=5)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_log_x}...")

    def test_LogDynamics_3d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_log_xu}...")
        xu_snapshots = self.dump_3d_log_xu.snapshots
        x_snapshots = None
        condition = (xu_snapshots.snapshots[0].particle_type == 1)
        ppp = np.array([0, 0, 0])

        # xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="slow",
                                  neighborfile=None,
                                  max_neighbors=30)
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98794986e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.83249245e-04,
                                         7.62729112e-03],
                                        [8.00000000e-02,
                                         9.81788619e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.79353312e-03,
                                         8.83924401e-03],
                                        [1.40000000e-01,
                                         9.51758118e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         7.51534401e-03,
                                         3.27266536e-03],
                                        [2.00000000e-01,
                                         9.20215225e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.26307715e-02,
                                         -7.52556624e-04],
                                        [8.00000000e-01,
                                         7.63523409e-01,
                                         8.87000000e-01,
                                         0.00000000e+00,
                                         4.16240792e-02,
                                         9.39923902e-02],
                                        [1.40000000e+00,
                                         7.06977773e-01,
                                         8.35000000e-01,
                                         0.00000000e+00,
                                         5.56373684e-02,
                                         3.69059435e-01],
                                        [2.00000000e+00,
                                         6.74200527e-01,
                                         7.87000000e-01,
                                         0.00000000e+00,
                                         6.28006979e-02,
                                         2.98372497e-01],
                                        [8.00000000e+00,
                                         4.89460164e-01,
                                         5.23000000e-01,
                                         0.00000000e+00,
                                         1.24775241e-01,
                                         3.90773736e-01],
                                        [1.40000000e+01,
                                         3.33250067e-01,
                                         3.65000000e-01,
                                         0.00000000e+00,
                                         2.02048496e-01,
                                         3.42854683e-01],
                                        [2.00000000e+01,
                                         3.01752850e-01,
                                         3.47000000e-01,
                                         0.00000000e+00,
                                         2.36770363e-01,
                                         4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98797156e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.82916144e-04,
                                         5.08578384e-04],
                                        [8.00000000e-02,
                                         9.81866650e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         2.78100867e-03,
                                         1.33905635e-02],
                                        [1.40000000e-01,
                                         9.52395484e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         7.40924729e-03,
                                         9.07554895e-03],
                                        [2.00000000e-01,
                                         9.22445634e-01,
                                         1.00000000e+00,
                                         0.00000000e+00,
                                         1.22484525e-02,
                                         2.58585214e-03],
                                        [8.00000000e-01,
                                         7.80250930e-01,
                                         9.10000000e-01,
                                         0.00000000e+00,
                                         3.79490640e-02,
                                         6.10574946e-02],
                                        [1.40000000e+00,
                                         7.33305660e-01,
                                         8.72500000e-01,
                                         0.00000000e+00,
                                         4.87008876e-02,
                                         2.97100024e-01],
                                        [2.00000000e+00,
                                         7.01825530e-01,
                                         8.23750000e-01,
                                         0.00000000e+00,
                                         5.54054537e-02,
                                         1.94253070e-01],
                                        [8.00000000e+00,
                                         5.12756184e-01,
                                         5.51250000e-01,
                                         0.00000000e+00,
                                         1.10604697e-01,
                                         2.42693270e-01],
                                        [1.40000000e+01,
                                         3.56859577e-01,
                                         3.90000000e-01,
                                         0.00000000e+00,
                                         1.75137652e-01,
                                         1.53847276e-01],
                                        [2.00000000e+01,
                                         3.29340145e-01,
                                         3.77500000e-01,
                                         0.00000000e+00,
                                         2.04353337e-01,
                                         2.83847908e-01]],
                                       result_2[:10].values)

        # xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002,
                                  ppp=ppp,
                                  diameters={1: 1.0, 2: 1.0},
                                  a=0.3,
                                  cal_type="fast",
                                  neighborfile=None,
                                  max_neighbors=30)
        # no condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=None,
            outputfile="test_no_condition.csv")
        # condition
        log_dynamic.relaxation(
            qconst=2 * np.pi,
            condition=condition,
            outputfile="test_w_condition.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98794986e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.83249245e-04,
                                         7.62729112e-03],
                                        [8.00000000e-02,
                                         9.81788619e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.79353312e-03,
                                         8.83924401e-03],
                                        [1.40000000e-01,
                                         9.51758118e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         7.51534401e-03,
                                         3.27266536e-03],
                                        [2.00000000e-01,
                                         9.20215225e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.26307715e-02,
                                         -7.52556624e-04],
                                        [8.00000000e-01,
                                         7.63523409e-01,
                                         1.13000000e-01,
                                         0.00000000e+00,
                                         4.16240792e-02,
                                         9.39923902e-02],
                                        [1.40000000e+00,
                                         7.06977773e-01,
                                         1.65000000e-01,
                                         0.00000000e+00,
                                         5.56373684e-02,
                                         3.69059435e-01],
                                        [2.00000000e+00,
                                         6.74200527e-01,
                                         2.13000000e-01,
                                         0.00000000e+00,
                                         6.28006979e-02,
                                         2.98372497e-01],
                                        [8.00000000e+00,
                                         4.89460164e-01,
                                         4.77000000e-01,
                                         0.00000000e+00,
                                         1.24775241e-01,
                                         3.90773736e-01],
                                        [1.40000000e+01,
                                         3.33250067e-01,
                                         6.35000000e-01,
                                         0.00000000e+00,
                                         2.02048496e-01,
                                         3.42854683e-01],
                                        [2.00000000e+01,
                                         3.01752850e-01,
                                         6.53000000e-01,
                                         0.00000000e+00,
                                         2.36770363e-01,
                                         4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02,
                                         9.98797156e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.82916144e-04,
                                         5.08578384e-04],
                                        [8.00000000e-02,
                                         9.81866650e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         2.78100867e-03,
                                         1.33905635e-02],
                                        [1.40000000e-01,
                                         9.52395484e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         7.40924729e-03,
                                         9.07554895e-03],
                                        [2.00000000e-01,
                                         9.22445634e-01,
                                         0.00000000e+00,
                                         0.00000000e+00,
                                         1.22484525e-02,
                                         2.58585214e-03],
                                        [8.00000000e-01,
                                         7.80250930e-01,
                                         9.00000000e-02,
                                         0.00000000e+00,
                                         3.79490640e-02,
                                         6.10574946e-02],
                                        [1.40000000e+00,
                                         7.33305660e-01,
                                         1.27500000e-01,
                                         0.00000000e+00,
                                         4.87008876e-02,
                                         2.97100024e-01],
                                        [2.00000000e+00,
                                         7.01825530e-01,
                                         1.76250000e-01,
                                         0.00000000e+00,
                                         5.54054537e-02,
                                         1.94253070e-01],
                                        [8.00000000e+00,
                                         5.12756184e-01,
                                         4.48750000e-01,
                                         0.00000000e+00,
                                         1.10604697e-01,
                                         2.42693270e-01],
                                        [1.40000000e+01,
                                         3.56859577e-01,
                                         6.10000000e-01,
                                         0.00000000e+00,
                                         1.75137652e-01,
                                         1.53847276e-01],
                                        [2.00000000e+01,
                                         3.29340145e-01,
                                         6.22500000e-01,
                                         0.00000000e+00,
                                         2.04353337e-01,
                                         2.83847908e-01]],
                                       result_2[:10].values)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_log_xu}...")
