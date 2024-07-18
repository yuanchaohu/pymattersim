# coding = utf-8
import os
import unittest
import numpy as np
import pandas as pd
from reader.dump_reader import DumpReader
from dynamic.dynamics import Dynamics, LogDynamics

from utils.logging import get_logger_handle
from neighbors.calculate_neighbors import Nnearests

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
        #2d files
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
        
        #3d files
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
        logger.info(f"Starting test using {self.test_file_2d_x,self.test_file_2d_xu}...")
        xu_snapshots = self.dump_2d_xu.snapshots
        x_snapshots = self.dump_2d_x.snapshots
        ppp = np.array([0,0])
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=ppp)
        neighborfile = 'neighborlist.dat'
        t = 10
        qrange = 10.0
        
        condition=[]
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type==1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)
        
        #xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=None,
                            max_neighbors=100
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.79301118e-01, 8.01120000e-01, 5.40014560e+00, 6.02823045e-02, 3.65974275e-01],
                                        [2.00000000e+01, 5.28087679e-01, 7.46489796e-01, 8.12482132e+00, 7.48699506e-02, 5.86668916e-01],
                                        [3.00000000e+01, 4.95001875e-01, 7.14770833e-01, 1.00640100e+01, 8.75970166e-02, 7.49057501e-01],
                                        [4.00000000e+01, 4.54857244e-01, 6.72255319e-01, 1.29034667e+01, 1.01064301e-01, 7.86468448e-01],
                                        [5.00000000e+01, 4.19751294e-01, 6.41717391e-01, 1.27491593e+01, 1.13082609e-01, 7.96229795e-01],
                                        [6.00000000e+01, 3.99037592e-01, 6.16800000e-01, 1.35556711e+01, 1.22840973e-01, 7.97500434e-01],
                                        [7.00000000e+01, 3.73314917e-01, 5.91704545e-01, 1.31064809e+01, 1.35131361e-01, 8.01051815e-01],
                                        [8.00000000e+01, 3.62810704e-01, 5.82046512e-01, 1.15571141e+01, 1.40822508e-01, 7.91098345e-01],
                                        [9.00000000e+01, 3.51982669e-01, 5.67190476e-01, 1.25491066e+01, 1.49266121e-01, 8.06072881e-01],
                                        [1.00000000e+02, 3.37590282e-01, 5.51878049e-01, 1.10304973e+01, 1.56938955e-01, 7.80859428e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 2.44806887],
                                        [0.43531205, 1.71277516],
                                        [0.65296808, 0.87875058],
                                        [0.8706241 , 0.59557265],
                                        [1.08828013, 0.43417031],
                                        [1.30593615, 0.30121787],
                                        [1.52359218, 0.23364272],
                                        [1.7412482 , 0.22584749],
                                        [1.95890423, 0.1902205 ],
                                        [2.17656025, 0.16829775]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[1.00000000e+01, 5.86630073e-01, 8.08646154e-01, 3.72959323e+00, 5.78972320e-02, 2.92147721e-01],
                                        [2.00000000e+01, 5.35170794e-01, 7.54254317e-01, 5.49555890e+00, 7.14793512e-02, 4.66850613e-01],
                                        [3.00000000e+01, 5.00770078e-01, 7.21891026e-01, 6.93700254e+00, 8.32601468e-02, 6.00347399e-01],
                                        [4.00000000e+01, 4.60693447e-01, 6.79410802e-01, 8.54314587e+00, 9.59008909e-02, 6.27139582e-01],
                                        [5.00000000e+01, 4.24497457e-01, 6.46755853e-01, 8.64270758e+00, 1.07264601e-01, 6.26570118e-01],
                                        [6.00000000e+01, 4.03841190e-01, 6.22803419e-01, 9.11564368e+00, 1.16221706e-01, 6.20895562e-01],
                                        [7.00000000e+01, 3.77788110e-01, 5.97727273e-01, 8.80807613e+00, 1.27807516e-01, 6.36211432e-01],
                                        [8.00000000e+01, 3.65741639e-01, 5.86583184e-01, 7.52760827e+00, 1.33126809e-01, 6.13757514e-01],
                                        [9.00000000e+01, 3.54263030e-01, 5.70915751e-01, 8.16071865e+00, 1.41146236e-01, 6.48107028e-01],
                                        [1.00000000e+02, 3.40293459e-01, 5.55459662e-01, 7.27129090e+00, 1.48284433e-01, 6.30128139e-01]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 2.44806887],
                                        [0.43531205, 1.71277516],
                                        [0.65296808, 0.87875058],
                                        [0.8706241 , 0.59557265],
                                        [1.08828013, 0.43417031],
                                        [1.30593615, 0.30121787],
                                        [1.52359218, 0.23364272],
                                        [1.7412482 , 0.22584749],
                                        [1.95890423, 0.1902205 ],
                                        [2.17656025, 0.16829775]],
                                        result_4[:10].values)
        
        
        #xu, x, slow, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=neighborfile,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 7.95905067e-01, 9.42280000e-01, 4.17801600e-01, 2.65776223e-02, 1.38300457e+00],
                                        [2.00000000e+01, 7.60207846e-01, 9.15571429e-01, 9.57836735e-01, 3.45401845e-02, 1.86021452e+00],
                                        [3.00000000e+01, 7.34798819e-01, 8.94583333e-01, 1.72645139e+00, 4.15105511e-02, 2.23427692e+00],
                                        [4.00000000e+01, 7.10658948e-01, 8.75212766e-01, 2.47625260e+00, 4.84539481e-02, 2.34189058e+00],
                                        [5.00000000e+01, 6.90113764e-01, 8.56304348e-01, 2.86734216e+00, 5.50098303e-02, 2.38146648e+00],
                                        [6.00000000e+01, 6.74916728e-01, 8.43644444e-01, 3.06747358e+00, 6.04883543e-02, 2.39154309e+00],
                                        [7.00000000e+01, 6.58257450e-01, 8.27863636e-01, 3.26461777e+00, 6.63888489e-02, 2.34783602e+00],
                                        [8.00000000e+01, 6.43501971e-01, 8.13930233e-01, 3.29332071e+00, 7.18171176e-02, 2.28602940e+00],
                                        [9.00000000e+01, 6.30701793e-01, 8.03238095e-01, 3.72899093e+00, 7.71761940e-02, 2.24596868e+00],
                                        [1.00000000e+02, 6.20530205e-01, 7.94000000e-01, 3.31995122e+00, 8.20006651e-02, 2.16664493e+00]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 0.66270335],
                                        [0.43531205, 0.38691019],
                                        [0.65296808, 0.27938051],
                                        [0.8706241 , 0.24616192],
                                        [1.08828013, 0.17384301],
                                        [1.30593615, 0.10905469],
                                        [1.52359218, 0.1005385 ],
                                        [1.7412482 , 0.08838945],
                                        [1.95890423, 0.0761088 ],
                                        [2.17656025, 0.07751973]],
                                       result_2[:10].values,
                                       decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 8.10014188e-01, 9.51323077e-01, 2.63369846e-01, 2.37700634e-02, 1.10629443e+00],
                                        [2.00000000e+01, 7.74311237e-01, 9.25651491e-01, 6.21437222e-01, 3.07856274e-02, 1.56609856e+00],
                                        [3.00000000e+01, 7.49439135e-01, 9.05512821e-01, 1.03813034e+00, 3.66358121e-02, 1.86881904e+00],
                                        [4.00000000e+01, 7.24543844e-01, 8.85237316e-01, 1.59179441e+00, 4.29106238e-02, 1.99617825e+00],
                                        [5.00000000e+01, 7.03228729e-01, 8.67257525e-01, 1.77255271e+00, 4.85785173e-02, 2.00891589e+00],
                                        [6.00000000e+01, 6.86983496e-01, 8.52820513e-01, 1.97039316e+00, 5.35090286e-02, 1.98183646e+00],
                                        [7.00000000e+01, 6.69782568e-01, 8.38286713e-01, 2.08819692e+00, 5.88158691e-02, 1.95908867e+00],
                                        [8.00000000e+01, 6.54114926e-01, 8.22289803e-01, 2.16482090e+00, 6.37150900e-02, 1.91877890e+00],
                                        [9.00000000e+01, 6.41436799e-01, 8.12161172e-01, 2.45309960e+00, 6.85143494e-02, 1.90485274e+00],
                                        [1.00000000e+02, 6.30480935e-01, 8.02026266e-01, 2.14014552e+00, 7.27836684e-02, 1.87651216e+00]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 0.66270335],
                                        [0.43531205, 0.38691019],
                                        [0.65296808, 0.27938051],
                                        [0.8706241 , 0.24616192],
                                        [1.08828013, 0.17384301],
                                        [1.30593615, 0.10905469],
                                        [1.52359218, 0.1005385 ],
                                        [1.7412482 , 0.08838945],
                                        [1.95890423, 0.0761088 ],
                                        [2.17656025, 0.07751973]],
                                        result_4[:10].values,
                                        decimal=6)
        
        
        #xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=None,
                            max_neighbors=100
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.79301118e-01, 1.98880000e-01, 5.40014560e+00, 6.02823045e-02, 3.65974275e-01],
                                        [2.00000000e+01, 5.28087679e-01, 2.53510204e-01, 8.12482132e+00, 7.48699506e-02, 5.86668916e-01],
                                        [3.00000000e+01, 4.95001875e-01, 2.85229167e-01, 1.00640100e+01, 8.75970166e-02, 7.49057501e-01],
                                        [4.00000000e+01, 4.54857244e-01, 3.27744681e-01, 1.29034667e+01, 1.01064301e-01, 7.86468448e-01],
                                        [5.00000000e+01, 4.19751294e-01, 3.58282609e-01, 1.27491593e+01, 1.13082609e-01, 7.96229795e-01],
                                        [6.00000000e+01, 3.99037592e-01, 3.83200000e-01, 1.35556711e+01, 1.22840973e-01, 7.97500434e-01],
                                        [7.00000000e+01, 3.73314917e-01, 4.08295455e-01, 1.31064809e+01, 1.35131361e-01, 8.01051815e-01],
                                        [8.00000000e+01, 3.62810704e-01, 4.17953488e-01, 1.15571141e+01, 1.40822508e-01, 7.91098345e-01],
                                        [9.00000000e+01, 3.51982669e-01, 4.32809524e-01, 1.25491066e+01, 1.49266121e-01, 8.06072881e-01],
                                        [1.00000000e+02, 3.37590282e-01, 4.48121951e-01, 1.10304973e+01, 1.56938955e-01, 7.80859428e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.48347404],
                                        [0.43531205, 5.75320892],
                                        [0.65296808, 3.16703736],
                                        [0.8706241 , 2.1943363 ],
                                        [1.08828013, 1.55936207],
                                        [1.30593615, 1.04300192],
                                        [1.52359218, 0.93432971],
                                        [1.7412482 , 0.89077471],
                                        [1.95890423, 0.71495682],
                                        [2.17656025, 0.65568832]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[1.00000000e+01, 5.86630073e-01, 1.91353846e-01, 3.72959323e+00, 5.78972320e-02, 2.92147721e-01],
                                        [2.00000000e+01, 5.35170794e-01, 2.45745683e-01, 5.49555890e+00, 7.14793512e-02, 4.66850613e-01],
                                        [3.00000000e+01, 5.00770078e-01, 2.78108974e-01, 6.93700254e+00, 8.32601468e-02, 6.00347399e-01],
                                        [4.00000000e+01, 4.60693447e-01, 3.20589198e-01, 8.54314587e+00, 9.59008909e-02, 6.27139582e-01],
                                        [5.00000000e+01, 4.24497457e-01, 3.53244147e-01, 8.64270758e+00, 1.07264601e-01, 6.26570118e-01],
                                        [6.00000000e+01, 4.03841190e-01, 3.77196581e-01, 9.11564368e+00, 1.16221706e-01, 6.20895562e-01],
                                        [7.00000000e+01, 3.77788110e-01, 4.02272727e-01, 8.80807613e+00, 1.27807516e-01, 6.36211432e-01],
                                        [8.00000000e+01, 3.65741639e-01, 4.13416816e-01, 7.52760827e+00, 1.33126809e-01, 6.13757514e-01],
                                        [9.00000000e+01, 3.54263030e-01, 4.29084249e-01, 8.16071865e+00, 1.41146236e-01, 6.48107028e-01],
                                        [1.00000000e+02, 3.40293459e-01, 4.44540338e-01, 7.27129090e+00, 1.48284433e-01, 6.30128139e-01]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.48347404],
                                        [0.43531205, 5.75320892],
                                        [0.65296808, 3.16703736],
                                        [0.8706241 , 2.1943363 ],
                                        [1.08828013, 1.55936207],
                                        [1.30593615, 1.04300192],
                                        [1.52359218, 0.93432971],
                                        [1.7412482 , 0.89077471],
                                        [1.95890423, 0.71495682],
                                        [2.17656025, 0.65568832]],
                                        result_4[:10].values)
                                            
        
        #xu, x, fast, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=neighborfile,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 7.95905067e-01, 5.77200000e-02, 4.17801600e-01, 2.65776223e-02, 1.38300457e+00],
                                        [2.00000000e+01, 7.60207846e-01, 8.44285714e-02, 9.57836735e-01, 3.45401845e-02, 1.86021452e+00],
                                        [3.00000000e+01, 7.34798819e-01, 1.05416667e-01, 1.72645139e+00, 4.15105511e-02, 2.23427692e+00],
                                        [4.00000000e+01, 7.10658948e-01, 1.24787234e-01, 2.47625260e+00, 4.84539481e-02, 2.34189058e+00],
                                        [5.00000000e+01, 6.90113764e-01, 1.43695652e-01, 2.86734216e+00, 5.50098303e-02, 2.38146648e+00],
                                        [6.00000000e+01, 6.74916728e-01, 1.56355556e-01, 3.06747358e+00, 6.04883543e-02, 2.39154309e+00],
                                        [7.00000000e+01, 6.58257450e-01, 1.72136364e-01, 3.26461777e+00, 6.63888489e-02, 2.34783602e+00],
                                        [8.00000000e+01, 6.43501971e-01, 1.86069767e-01, 3.29332071e+00, 7.18171176e-02, 2.28602940e+00],
                                        [9.00000000e+01, 6.30701793e-01, 1.96761905e-01, 3.72899093e+00, 7.71761940e-02, 2.24596868e+00],
                                        [1.00000000e+02, 6.20530205e-01, 2.06000000e-01, 3.31995122e+00, 8.20006651e-02, 2.16664493e+00]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 6.78891443],
                                        [0.43531205, 4.30338972],
                                        [0.65296808, 3.52047085],
                                        [0.8706241 , 2.7054433 ],
                                        [1.08828013, 2.10292473],
                                        [1.30593615, 1.44076635],
                                        [1.52359218, 1.44602222],
                                        [1.7412482 , 1.11847775],
                                        [1.95890423, 0.89408065],
                                        [2.17656025, 0.97644806]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 8.10014188e-01, 4.86769231e-02, 2.63369846e-01, 2.37700634e-02, 1.10629443e+00],
                                        [2.00000000e+01, 7.74311237e-01, 7.43485086e-02, 6.21437222e-01, 3.07856274e-02, 1.56609856e+00],
                                        [3.00000000e+01, 7.49439135e-01, 9.44871795e-02, 1.03813034e+00, 3.66358121e-02, 1.86881904e+00],
                                        [4.00000000e+01, 7.24543844e-01, 1.14762684e-01, 1.59179441e+00, 4.29106238e-02, 1.99617825e+00],
                                        [5.00000000e+01, 7.03228729e-01, 1.32742475e-01, 1.77255271e+00, 4.85785173e-02, 2.00891589e+00],
                                        [6.00000000e+01, 6.86983496e-01, 1.47179487e-01, 1.97039316e+00, 5.35090286e-02, 1.98183646e+00],
                                        [7.00000000e+01, 6.69782568e-01, 1.61713287e-01, 2.08819692e+00, 5.88158691e-02, 1.95908867e+00],
                                        [8.00000000e+01, 6.54114926e-01, 1.77710197e-01, 2.16482090e+00, 6.37150900e-02, 1.91877890e+00],
                                        [9.00000000e+01, 6.41436799e-01, 1.87838828e-01, 2.45309960e+00, 6.85143494e-02, 1.90485274e+00],
                                        [1.00000000e+02, 6.30480935e-01, 1.97973734e-01, 2.14014552e+00, 7.27836684e-02, 1.87651216e+00]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 6.78891443],
                                        [0.43531205, 4.30338972],
                                        [0.65296808, 3.52047085],
                                        [0.8706241 , 2.7054433 ],
                                        [1.08828013, 2.10292473],
                                        [1.30593615, 1.44076635],
                                        [1.52359218, 1.44602222],
                                        [1.7412482 , 1.11847775],
                                        [1.95890423, 0.89408065],
                                        [2.17656025, 0.97644806]],
                                        result_4[:10].values,
                                        decimal=6)
        
        os.remove("neighborlist.dat")
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_xu, self.test_file_2d_x}...")
        
        
    def test_Dynamics_2d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_x,}...")
        xu_snapshots = None
        x_snapshots = self.dump_2d_x.snapshots
        ppp = np.array([1,0])
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=ppp)
        neighborfile = 'neighborlist.dat'
        t = 10
        qrange = 10.0
        
        condition=[]
        for snapshot in x_snapshots.snapshots:
            condition.append(snapshot.particle_type==1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)
        
        #xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=None,
                            max_neighbors=100
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 10.        , 0.57810962, 0.79866   , 5.4453444 , 3.42303815, 116.73479551],
                                        [ 20.        , 0.52682895, 0.74389796, 8.12425489, 3.96910686, 100.27412499],
                                        [ 30.        , 0.4938267 , 0.71235417, 10.0696454 , 4.44293574, 89.15175678],
                                        [ 40.        , 0.45379585, 0.66978723, 12.91659303, 4.71079006, 83.6742147 ],
                                        [ 50.        , 0.41877798, 0.63984783, 12.81243336, 4.89164235, 80.09108014],
                                        [ 60.        , 0.39802188, 0.61482222, 13.58303506, 5.36276752, 72.9171652 ],
                                        [ 70.        , 0.37241113, 0.59020455, 13.1125718 , 5.40499485, 71.98421935],
                                        [ 80.        , 0.36186837, 0.58034884, 11.52734343, 5.76007449, 67.53822092],
                                        [ 90.        , 0.35113304, 0.56552381, 12.53529705, 6.20628242, 62.59937219],
                                        [100.        , 0.33696599, 0.55065854, 11.01524926, 6.30979152, 61.38117041]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 2.4771955 ],
                                        [0.43531205, 1.71002943],
                                        [0.65296808, 0.89027071],
                                        [0.8706241 , 0.59668645],
                                        [1.08828013, 0.43658   ],
                                        [1.30593615, 0.30796716],
                                        [1.52359218, 0.25013054],
                                        [1.7412482 , 0.23567053],
                                        [1.95890423, 0.20377613],
                                        [2.17656025, 0.16811808]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[ 10.        , 0.58529405, 0.80584615, 3.73433846, 3.74395974, 106.92229813],
                                        [ 20.        , 0.53374339, 0.75145997, 5.45852818, 4.21313026, 94.63430193],
                                        [ 30.        , 0.49945466, 0.71919872, 6.94993523, 4.79677819, 82.73885648],
                                        [ 40.        , 0.45940081, 0.67656301, 8.54449838, 5.11672103, 77.29345154],
                                        [ 50.        , 0.42335209, 0.64478261, 8.67051694, 5.24245557, 74.97318041],
                                        [ 60.        , 0.40260099, 0.62064957, 9.1343924 , 5.69533562, 68.89934215],
                                        [ 70.        , 0.37667321, 0.59597902, 8.79574936, 5.81838906, 67.14338084],
                                        [ 80.        , 0.36446821, 0.58457961, 7.48897949, 6.36542254, 61.38722079],
                                        [ 90.        , 0.3532864 , 0.56915751, 8.13362289, 6.60127954, 59.0470466 ],
                                        [100.        , 0.33953185, 0.55429644, 7.26155677, 6.78407966, 57.29722271]],
                                       result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 2.4771955 ],
                                        [0.43531205, 1.71002943],
                                        [0.65296808, 0.89027071],
                                        [0.8706241 , 0.59668645],
                                        [1.08828013, 0.43658   ],
                                        [1.30593615, 0.30796716],
                                        [1.52359218, 0.25013054],
                                        [1.7412482 , 0.23567053],
                                        [1.95890423, 0.20377613],
                                        [2.17656025, 0.16811808]],
                                        result_4[:10].values)
        
        #xu, x, slow, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=neighborfile,
                            max_neighbors=100
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 10.        , 0.79716579, 0.92948   , 0.5276896 , 3.54678569, 97.75399651],
                                        [ 20.        , 0.76186741, 0.90210204, 1.09013244, 4.02370136, 83.77864362],
                                        [ 30.        , 0.73652977, 0.87964583, 1.95097873, 4.56637803, 74.3990835 ],
                                        [ 40.        , 0.71268001, 0.86148936, 2.58680308, 4.74204362, 69.8374037 ],
                                        [ 50.        , 0.69244793, 0.84402174, 3.08249953, 4.82004917, 66.56142621],
                                        [ 60.        , 0.67713659, 0.82973333, 3.22757333, 5.31907999, 60.80236915],
                                        [ 70.        , 0.66134685, 0.81675   , 3.44264205, 5.26025975, 59.77824976],
                                        [ 80.        , 0.64693116, 0.80188372, 3.55949811, 5.59241631, 56.44029808],
                                        [ 90.        , 0.63395028, 0.79095238, 3.96271202, 5.97889842, 52.02174209],
                                        [100.        , 0.62416669, 0.78134146, 3.83490779, 6.00452684, 50.74705984]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 1.05458356],
                                        [0.43531205, 0.59516062],
                                        [0.65296808, 0.49683801],
                                        [0.8706241 , 0.27438462],
                                        [1.08828013, 0.25725812],
                                        [1.30593615, 0.15254601],
                                        [1.52359218, 0.21468353],
                                        [1.7412482 , 0.14328172],
                                        [1.95890423, 0.11917867],
                                        [2.17656025, 0.07932298]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[ 10.        , 0.81197716, 0.93793846, 0.31652985, 3.85550734, 89.92422547],
                                        [ 20.        , 0.77687903, 0.91136578, 0.6884657 , 4.24356613, 78.86766396],
                                        [ 30.        , 0.75194713, 0.88987179, 1.22332265, 4.88559586, 69.37269505],
                                        [ 40.        , 0.72743916, 0.87126023, 1.67562071, 5.10020516, 64.76400523],
                                        [ 50.        , 0.70666836, 0.85454849, 1.96660244, 5.11268375, 62.63821414],
                                        [ 60.        , 0.69001968, 0.83876923, 2.07621197, 5.60660807, 57.67206413],
                                        [ 70.        , 0.67366562, 0.82660839, 2.18423633, 5.61917318, 56.08186453],
                                        [ 80.        , 0.65860989, 0.81033989, 2.44653493, 6.09167908, 52.02494064],
                                        [ 90.        , 0.64587742, 0.80007326, 2.69149834, 6.25296165, 49.2559143 ],
                                        [100.        , 0.63540684, 0.78923077, 2.53388368, 6.37957018, 47.61085625]],
                                       result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 1.05458356],
                                        [0.43531205, 0.59516062],
                                        [0.65296808, 0.49683801],
                                        [0.8706241 , 0.27438462],
                                        [1.08828013, 0.25725812],
                                        [1.30593615, 0.15254601],
                                        [1.52359218, 0.21468353],
                                        [1.7412482 , 0.14328172],
                                        [1.95890423, 0.11917867],
                                        [2.17656025, 0.07932298]],
                                        result_4[:10].values)
        
        #xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=None,
                            max_neighbors=100
                            )
        
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 10.        , 0.57810962, 0.20134   , 5.4453444 , 3.42303815, 116.73479551],
                                        [ 20.        , 0.52682895, 0.25610204, 8.12425489, 3.96910686, 100.27412499],
                                        [ 30.        , 0.4938267 , 0.28764583, 10.0696454 , 4.44293574, 89.15175678],
                                        [ 40.        , 0.45379585, 0.33021277, 12.91659303, 4.71079006, 83.6742147 ],
                                        [ 50.        , 0.41877798, 0.36015217, 12.81243336, 4.89164235, 80.09108014],
                                        [ 60.        , 0.39802188, 0.38517778, 13.58303506, 5.36276752, 72.9171652 ],
                                        [ 70.        , 0.37241113, 0.40979545, 13.1125718 , 5.40499485, 71.98421935],
                                        [ 80.        , 0.36186837, 0.41965116, 11.52734343, 5.76007449, 67.53822092],
                                        [ 90.        , 0.35113304, 0.43447619, 12.53529705, 6.20628242, 62.59937219],
                                        [100.        , 0.33696599, 0.44934146, 11.01524926, 6.30979152, 61.38117041]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.38176759],
                                        [0.43531205, 5.65847144],
                                        [0.65296808, 3.18354758],
                                        [0.8706241 , 2.23286466],
                                        [1.08828013, 1.53502364],
                                        [1.30593615, 1.11068991],
                                        [1.52359218, 0.98594114],
                                        [1.7412482 , 0.87277187],
                                        [1.95890423, 0.7716178 ],
                                        [2.17656025, 0.65209741]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[ 10.        , 0.58529405, 0.19415385, 3.73433846, 3.74395974, 106.92229813],
                                        [ 20.        , 0.53374339, 0.24854003, 5.45852818, 4.21313026, 94.63430193],
                                        [ 30.        , 0.49945466, 0.28080128, 6.94993523, 4.79677819, 82.73885648],
                                        [ 40.        , 0.45940081, 0.32343699, 8.54449838, 5.11672103, 77.29345154],
                                        [ 50.        , 0.42335209, 0.35521739, 8.67051694, 5.24245557, 74.97318041],
                                        [ 60.        , 0.40260099, 0.37935043, 9.1343924 , 5.69533562, 68.89934215],
                                        [ 70.        , 0.37667321, 0.40402098, 8.79574936, 5.81838906, 67.14338084],
                                        [ 80.        , 0.36446821, 0.41542039, 7.48897949, 6.36542254, 61.38722079],
                                        [ 90.        , 0.3532864 , 0.43084249, 8.13362289, 6.60127954, 59.0470466 ],
                                        [100.        , 0.33953185, 0.44570356, 7.26155677, 6.78407966, 57.29722271]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.38176759],
                                        [0.43531205, 5.65847144],
                                        [0.65296808, 3.18354758],
                                        [0.8706241 , 2.23286466],
                                        [1.08828013, 1.53502364],
                                        [1.30593615, 1.11068991],
                                        [1.52359218, 0.98594114],
                                        [1.7412482 , 0.87277187],
                                        [1.95890423, 0.7716178 ],
                                        [2.17656025, 0.65209741]],
                                        result_4[:10].values)
        
        
        #xu, x, fast, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=neighborfile,
                            max_neighbors=100
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[1.00000000e+01, 7.97165792e-01, 7.05200000e-02, 5.27689600e-01, 3.54678569e+00, 9.77539965e+01],
                                        [2.00000000e+01, 7.61867412e-01, 9.78979592e-02, 1.09013244e+00, 4.02370136e+00, 8.37786436e+01],
                                        [3.00000000e+01, 7.36529775e-01, 1.20354167e-01, 1.95097873e+00, 4.56637803e+00, 7.43990835e+01],
                                        [4.00000000e+01, 7.12680012e-01, 1.38510638e-01, 2.58680308e+00, 4.74204362e+00, 6.98374037e+01],
                                        [5.00000000e+01, 6.92447927e-01, 1.55978261e-01, 3.08249953e+00, 4.82004917e+00, 6.65614262e+01],
                                        [6.00000000e+01, 6.77136592e-01, 1.70266667e-01, 3.22757333e+00, 5.31907999e+00, 6.08023691e+01],
                                        [7.00000000e+01, 6.61346851e-01, 1.83250000e-01, 3.44264205e+00, 5.26025975e+00, 5.97782498e+01],
                                        [8.00000000e+01, 6.46931159e-01, 1.98116279e-01, 3.55949811e+00, 5.59241631e+00, 5.64402981e+01],
                                        [9.00000000e+01, 6.33950277e-01, 2.09047619e-01, 3.96271202e+00, 5.97889842e+00, 5.20217421e+01],
                                        [1.00000000e+02, 6.24166692e-01, 2.18658537e-01, 3.83490779e+00, 6.00452684e+00, 5.07470598e+01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.87464874],
                                        [0.43531205, 5.30453547],
                                        [0.65296808, 5.37287212],
                                        [0.8706241 , 3.41037673],
                                        [1.08828013, 2.46999568],
                                        [1.30593615, 2.58312528],
                                        [1.52359218, 2.37351669],
                                        [1.7412482 , 1.25127938],
                                        [1.95890423, 1.33206526],
                                        [2.17656025, 0.89113842]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[1.00000000e+01, 8.11977158e-01, 6.20615385e-02, 3.16529846e-01, 3.85550734e+00, 8.99242255e+01],
                                        [2.00000000e+01, 7.76879032e-01, 8.86342229e-02, 6.88465703e-01, 4.24356613e+00, 7.88676640e+01],
                                        [3.00000000e+01, 7.51947126e-01, 1.10128205e-01, 1.22332265e+00, 4.88559586e+00, 6.93726950e+01],
                                        [4.00000000e+01, 7.27439156e-01, 1.28739771e-01, 1.67562071e+00, 5.10020516e+00, 6.47640052e+01],
                                        [5.00000000e+01, 7.06668361e-01, 1.45451505e-01, 1.96660244e+00, 5.11268375e+00, 6.26382141e+01],
                                        [6.00000000e+01, 6.90019682e-01, 1.61230769e-01, 2.07621197e+00, 5.60660807e+00, 5.76720641e+01],
                                        [7.00000000e+01, 6.73665620e-01, 1.73391608e-01, 2.18423633e+00, 5.61917318e+00, 5.60818645e+01],
                                        [8.00000000e+01, 6.58609894e-01, 1.89660107e-01, 2.44653493e+00, 6.09167908e+00, 5.20249406e+01],
                                        [9.00000000e+01, 6.45877416e-01, 1.99926740e-01, 2.69149834e+00, 6.25296165e+00, 4.92559143e+01],
                                        [1.00000000e+02, 6.35406835e-01, 2.10769231e-01, 2.53388368e+00, 6.37957018e+00, 4.76108562e+01]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.87464874],
                                        [0.43531205, 5.30453547],
                                        [0.65296808, 5.37287212],
                                        [0.8706241 , 3.41037673],
                                        [1.08828013, 2.46999568],
                                        [1.30593615, 2.58312528],
                                        [1.52359218, 2.37351669],
                                        [1.7412482 , 1.25127938],
                                        [1.95890423, 1.33206526],
                                        [2.17656025, 0.89113842]],
                                        result_4[:10].values)
        
        os.remove("neighborlist.dat")
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_x}...")
        
        
    def test_Dynamics_2d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_xu}...")
        xu_snapshots = self.dump_2d_xu.snapshots
        x_snapshots = None
        ppp = np.array([0,0])
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6, ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6, ppp=ppp)
        neighborfile = 'neighborlist.dat'
        t = 10
        qrange = 10.0
        
        condition=[]
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type==1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)
        
        #xu, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=None,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.79301118e-01, 8.01120000e-01, 5.40014560e+00, 6.02823045e-02, 3.65974275e-01],
                                        [2.00000000e+01, 5.28087679e-01, 7.46489796e-01, 8.12482132e+00, 7.48699506e-02, 5.86668916e-01],
                                        [3.00000000e+01, 4.95001875e-01, 7.14770833e-01, 1.00640100e+01, 8.75970166e-02, 7.49057501e-01],
                                        [4.00000000e+01, 4.54857244e-01, 6.72255319e-01, 1.29034667e+01, 1.01064301e-01, 7.86468448e-01],
                                        [5.00000000e+01, 4.19751294e-01, 6.41717391e-01, 1.27491593e+01, 1.13082609e-01, 7.96229795e-01],
                                        [6.00000000e+01, 3.99037592e-01, 6.16800000e-01, 1.35556711e+01, 1.22840973e-01, 7.97500434e-01],
                                        [7.00000000e+01, 3.73314917e-01, 5.91704545e-01, 1.31064809e+01, 1.35131361e-01, 8.01051815e-01],
                                        [8.00000000e+01, 3.62810704e-01, 5.82046512e-01, 1.15571141e+01, 1.40822508e-01, 7.91098345e-01],
                                        [9.00000000e+01, 3.51982669e-01, 5.67190476e-01, 1.25491066e+01, 1.49266121e-01, 8.06072881e-01],
                                        [1.00000000e+02, 3.37590282e-01, 5.51878049e-01, 1.10304973e+01, 1.56938955e-01, 7.80859428e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 2.44806828],
                                        [0.43531205, 1.71277529],
                                        [0.65296808, 0.87875024],
                                        [0.8706241 , 0.5955718 ],
                                        [1.08828013, 0.43417012],
                                        [1.30593615, 0.30121784],
                                        [1.52359218, 0.23364366],
                                        [1.7412482 , 0.22584802],
                                        [1.95890423, 0.19021991],
                                        [2.17656025, 0.16829796]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 5.86630073e-01, 8.08646154e-01, 3.72959323e+00, 5.78972320e-02, 2.92147721e-01],
                                        [2.00000000e+01, 5.35170794e-01, 7.54254317e-01, 5.49555890e+00, 7.14793512e-02, 4.66850613e-01],
                                        [3.00000000e+01, 5.00770078e-01, 7.21891026e-01, 6.93700254e+00, 8.32601468e-02, 6.00347399e-01],
                                        [4.00000000e+01, 4.60693447e-01, 6.79410802e-01, 8.54314587e+00, 9.59008909e-02, 6.27139582e-01],
                                        [5.00000000e+01, 4.24497457e-01, 6.46755853e-01, 8.64270758e+00, 1.07264601e-01, 6.26570118e-01],
                                        [6.00000000e+01, 4.03841190e-01, 6.22803419e-01, 9.11564368e+00, 1.16221706e-01, 6.20895562e-01],
                                        [7.00000000e+01, 3.77788110e-01, 5.97727273e-01, 8.80807613e+00, 1.27807516e-01, 6.36211432e-01],
                                        [8.00000000e+01, 3.65741639e-01, 5.86583184e-01, 7.52760827e+00, 1.33126809e-01, 6.13757514e-01],
                                        [9.00000000e+01, 3.54263030e-01, 5.70915751e-01, 8.16071865e+00, 1.41146236e-01, 6.48107028e-01],
                                        [1.00000000e+02, 3.40293459e-01, 5.55459662e-01, 7.27129090e+00, 1.48284433e-01, 6.30128139e-01]],
                                        result_3[:10].values,)
        np.testing.assert_almost_equal([[0.21765603, 2.44806828],
                                        [0.43531205, 1.71277529],
                                        [0.65296808, 0.87875024],
                                        [0.8706241 , 0.5955718 ],
                                        [1.08828013, 0.43417012],
                                        [1.30593615, 0.30121784],
                                        [1.52359218, 0.23364366],
                                        [1.7412482 , 0.22584802],
                                        [1.95890423, 0.19021991],
                                        [2.17656025, 0.16829796]],
                                        result_4[:10].values,
                                        decimal=6)
                
        #xu, slow, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=neighborfile,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 7.95905067e-01, 9.42280000e-01, 4.17801600e-01, 2.65776223e-02, 1.38300457e+00],
                                        [2.00000000e+01, 7.60207846e-01, 9.15571429e-01, 9.57836735e-01, 3.45401845e-02, 1.86021452e+00],
                                        [3.00000000e+01, 7.34798819e-01, 8.94583333e-01, 1.72645139e+00, 4.15105511e-02, 2.23427692e+00],
                                        [4.00000000e+01, 7.10658948e-01, 8.75212766e-01, 2.47625260e+00, 4.84539481e-02, 2.34189058e+00],
                                        [5.00000000e+01, 6.90113764e-01, 8.56304348e-01, 2.86734216e+00, 5.50098303e-02, 2.38146648e+00],
                                        [6.00000000e+01, 6.74916728e-01, 8.43644444e-01, 3.06747358e+00, 6.04883543e-02, 2.39154309e+00],
                                        [7.00000000e+01, 6.58257450e-01, 8.27863636e-01, 3.26461777e+00, 6.63888489e-02, 2.34783602e+00],
                                        [8.00000000e+01, 6.43501971e-01, 8.13930233e-01, 3.29332071e+00, 7.18171176e-02, 2.28602940e+00],
                                        [9.00000000e+01, 6.30701793e-01, 8.03238095e-01, 3.72899093e+00, 7.71761940e-02, 2.24596868e+00],
                                        [1.00000000e+02, 6.20530205e-01, 7.94000000e-01, 3.31995122e+00, 8.20006651e-02, 2.16664493e+00]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 0.66270335],
                                        [0.43531205, 0.38691018],
                                        [0.65296808, 0.27938051],
                                        [0.8706241 , 0.24616191],
                                        [1.08828013, 0.17384301],
                                        [1.30593615, 0.10905469],
                                        [1.52359218, 0.10053849],
                                        [1.7412482 , 0.08838945],
                                        [1.95890423, 0.0761088 ],
                                        [2.17656025, 0.07751973]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 8.10014188e-01, 9.51323077e-01, 2.63369846e-01, 2.37700634e-02, 1.10629443e+00],
                                        [2.00000000e+01, 7.74311237e-01, 9.25651491e-01, 6.21437222e-01, 3.07856274e-02, 1.56609856e+00],
                                        [3.00000000e+01, 7.49439135e-01, 9.05512821e-01, 1.03813034e+00, 3.66358121e-02, 1.86881904e+00],
                                        [4.00000000e+01, 7.24543844e-01, 8.85237316e-01, 1.59179441e+00, 4.29106238e-02, 1.99617825e+00],
                                        [5.00000000e+01, 7.03228729e-01, 8.67257525e-01, 1.77255271e+00, 4.85785173e-02, 2.00891589e+00],
                                        [6.00000000e+01, 6.86983496e-01, 8.52820513e-01, 1.97039316e+00, 5.35090286e-02, 1.98183646e+00],
                                        [7.00000000e+01, 6.69782568e-01, 8.38286713e-01, 2.08819692e+00, 5.88158691e-02, 1.95908867e+00],
                                        [8.00000000e+01, 6.54114926e-01, 8.22289803e-01, 2.16482090e+00, 6.37150900e-02, 1.91877890e+00],
                                        [9.00000000e+01, 6.41436799e-01, 8.12161172e-01, 2.45309960e+00, 6.85143494e-02, 1.90485274e+00],
                                        [1.00000000e+02, 6.30480935e-01, 8.02026266e-01, 2.14014552e+00, 7.27836684e-02, 1.87651216e+00]],
                                        result_3[:10].values,)
        np.testing.assert_almost_equal([[0.21765603, 0.66270335],
                                        [0.43531205, 0.38691018],
                                        [0.65296808, 0.27938051],
                                        [0.8706241 , 0.24616191],
                                        [1.08828013, 0.17384301],
                                        [1.30593615, 0.10905469],
                                        [1.52359218, 0.10053849],
                                        [1.7412482 , 0.08838945],
                                        [1.95890423, 0.0761088 ],
                                        [2.17656025, 0.07751973]],
                                        result_4[:10].values,
                                        decimal=6)
        
        #xu, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=None,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.79301118e-01, 1.98880000e-01, 5.40014560e+00, 6.02823045e-02, 3.65974275e-01],
                                        [2.00000000e+01, 5.28087679e-01, 2.53510204e-01, 8.12482132e+00, 7.48699506e-02, 5.86668916e-01],
                                        [3.00000000e+01, 4.95001875e-01, 2.85229167e-01, 1.00640100e+01, 8.75970166e-02, 7.49057501e-01],
                                        [4.00000000e+01, 4.54857244e-01, 3.27744681e-01, 1.29034667e+01, 1.01064301e-01, 7.86468448e-01],
                                        [5.00000000e+01, 4.19751294e-01, 3.58282609e-01, 1.27491593e+01, 1.13082609e-01, 7.96229795e-01],
                                        [6.00000000e+01, 3.99037592e-01, 3.83200000e-01, 1.35556711e+01, 1.22840973e-01, 7.97500434e-01],
                                        [7.00000000e+01, 3.73314917e-01, 4.08295455e-01, 1.31064809e+01, 1.35131361e-01, 8.01051815e-01],
                                        [8.00000000e+01, 3.62810704e-01, 4.17953488e-01, 1.15571141e+01, 1.40822508e-01, 7.91098345e-01],
                                        [9.00000000e+01, 3.51982669e-01, 4.32809524e-01, 1.25491066e+01, 1.49266121e-01, 8.06072881e-01],
                                        [1.00000000e+02, 3.37590282e-01, 4.48121951e-01, 1.10304973e+01, 1.56938955e-01, 7.80859428e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 8.4834742 ],
                                        [0.43531205, 5.75320869],
                                        [0.65296808, 3.16703796],
                                        [0.8706241 , 2.1943372 ],
                                        [1.08828013, 1.55936301],
                                        [1.30593615, 1.04300149],
                                        [1.52359218, 0.93433027],
                                        [1.7412482 , 0.89077477],
                                        [1.95890423, 0.71495529],
                                        [2.17656025, 0.65568903]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 5.86630073e-01, 1.91353846e-01, 3.72959323e+00, 5.78972320e-02, 2.92147721e-01],
                                        [2.00000000e+01, 5.35170794e-01, 2.45745683e-01, 5.49555890e+00, 7.14793512e-02, 4.66850613e-01],
                                        [3.00000000e+01, 5.00770078e-01, 2.78108974e-01, 6.93700254e+00, 8.32601468e-02, 6.00347399e-01],
                                        [4.00000000e+01, 4.60693447e-01, 3.20589198e-01, 8.54314587e+00, 9.59008909e-02, 6.27139582e-01],
                                        [5.00000000e+01, 4.24497457e-01, 3.53244147e-01, 8.64270758e+00, 1.07264601e-01, 6.26570118e-01],
                                        [6.00000000e+01, 4.03841190e-01, 3.77196581e-01, 9.11564368e+00, 1.16221706e-01, 6.20895562e-01],
                                        [7.00000000e+01, 3.77788110e-01, 4.02272727e-01, 8.80807613e+00, 1.27807516e-01, 6.36211432e-01],
                                        [8.00000000e+01, 3.65741639e-01, 4.13416816e-01, 7.52760827e+00, 1.33126809e-01, 6.13757514e-01],
                                        [9.00000000e+01, 3.54263030e-01, 4.29084249e-01, 8.16071865e+00, 1.41146236e-01, 6.48107028e-01],
                                        [1.00000000e+02, 3.40293459e-01, 4.44540338e-01, 7.27129090e+00, 1.48284433e-01, 6.30128139e-01]],
                                        result_3[:10].values,)
        np.testing.assert_almost_equal([[0.21765603, 8.4834742 ],
                                        [0.43531205, 5.75320869],
                                        [0.65296808, 3.16703796],
                                        [0.8706241 , 2.1943372 ],
                                        [1.08828013, 1.55936301],
                                        [1.30593615, 1.04300149],
                                        [1.52359218, 0.93433027],
                                        [1.7412482 , 0.89077477],
                                        [1.95890423, 0.71495529],
                                        [2.17656025, 0.65568903]],
                                        result_4[:10].values,
                                        decimal=6)
        
        #xu, fast, neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=neighborfile,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 7.95905067e-01, 5.77200000e-02, 4.17801600e-01, 2.65776223e-02, 1.38300457e+00],
                                        [2.00000000e+01, 7.60207846e-01, 8.44285714e-02, 9.57836735e-01, 3.45401845e-02, 1.86021452e+00],
                                        [3.00000000e+01, 7.34798819e-01, 1.05416667e-01, 1.72645139e+00, 4.15105511e-02, 2.23427692e+00],
                                        [4.00000000e+01, 7.10658948e-01, 1.24787234e-01, 2.47625260e+00, 4.84539481e-02, 2.34189058e+00],
                                        [5.00000000e+01, 6.90113764e-01, 1.43695652e-01, 2.86734216e+00, 5.50098303e-02, 2.38146648e+00],
                                        [6.00000000e+01, 6.74916728e-01, 1.56355556e-01, 3.06747358e+00, 6.04883543e-02, 2.39154309e+00],
                                        [7.00000000e+01, 6.58257450e-01, 1.72136364e-01, 3.26461777e+00, 6.63888489e-02, 2.34783602e+00],
                                        [8.00000000e+01, 6.43501971e-01, 1.86069767e-01, 3.29332071e+00, 7.18171176e-02, 2.28602940e+00],
                                        [9.00000000e+01, 6.30701793e-01, 1.96761905e-01, 3.72899093e+00, 7.71761940e-02, 2.24596868e+00],
                                        [1.00000000e+02, 6.20530205e-01, 2.06000000e-01, 3.31995122e+00, 8.20006651e-02, 2.16664493e+00]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.21765603, 6.78891443],
                                        [0.43531205, 4.30338972],
                                        [0.65296808, 3.52047085],
                                        [0.8706241 , 2.7054433 ],
                                        [1.08828013, 2.10292473],
                                        [1.30593615, 1.44076635],
                                        [1.52359218, 1.44602222],
                                        [1.7412482 , 1.11847775],
                                        [1.95890423, 0.89408065],
                                        [2.17656025, 0.97644806]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 8.10014188e-01, 4.86769231e-02, 2.63369846e-01, 2.37700634e-02, 1.10629443e+00],
                                        [2.00000000e+01, 7.74311237e-01, 7.43485086e-02, 6.21437222e-01, 3.07856274e-02, 1.56609856e+00],
                                        [3.00000000e+01, 7.49439135e-01, 9.44871795e-02, 1.03813034e+00, 3.66358121e-02, 1.86881904e+00],
                                        [4.00000000e+01, 7.24543844e-01, 1.14762684e-01, 1.59179441e+00, 4.29106238e-02, 1.99617825e+00],
                                        [5.00000000e+01, 7.03228729e-01, 1.32742475e-01, 1.77255271e+00, 4.85785173e-02, 2.00891589e+00],
                                        [6.00000000e+01, 6.86983496e-01, 1.47179487e-01, 1.97039316e+00, 5.35090286e-02, 1.98183646e+00],
                                        [7.00000000e+01, 6.69782568e-01, 1.61713287e-01, 2.08819692e+00, 5.88158691e-02, 1.95908867e+00],
                                        [8.00000000e+01, 6.54114926e-01, 1.77710197e-01, 2.16482090e+00, 6.37150900e-02, 1.91877890e+00],
                                        [9.00000000e+01, 6.41436799e-01, 1.87838828e-01, 2.45309960e+00, 6.85143494e-02, 1.90485274e+00],
                                        [1.00000000e+02, 6.30480935e-01, 1.97973734e-01, 2.14014552e+00, 7.27836684e-02, 1.87651216e+00]],
                                        result_3[:10].values,)
        np.testing.assert_almost_equal([[0.21765603, 6.78891443],
                                        [0.43531205, 4.30338972],
                                        [0.65296808, 3.52047085],
                                        [0.8706241 , 2.7054433 ],
                                        [1.08828013, 2.10292473],
                                        [1.30593615, 1.44076635],
                                        [1.52359218, 1.44602222],
                                        [1.7412482 , 1.11847775],
                                        [1.95890423, 0.89408065],
                                        [2.17656025, 0.97644806]],
                                        result_4[:10].values,
                                        decimal=6)
        
        os.remove("neighborlist.dat")
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_xu}...")
        

    def test_Dynamics_3d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_x,self.test_file_3d_xu}...")
        xu_snapshots = self.dump_3d_xu.snapshots
        x_snapshots = self.dump_3d_x.snapshots
        ppp = np.array([0, 0, 0])
        t = 10
        qrange = 10.0
        condition=[]
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type==1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)
        
        #xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=None,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.85376432e-01, 6.74640000e-01, 2.93295040e+00, 9.52096594e-02, 6.98532609e-01],
                                        [2.00000000e+01, 4.90235700e-01, 5.58959184e-01, 4.37803915e+00, 1.38404305e-01, 8.25369390e-01],
                                        [3.00000000e+01, 4.21873302e-01, 4.73791667e-01, 4.39766493e+00, 1.77096679e-01, 8.10179925e-01],
                                        [4.00000000e+01, 3.61045399e-01, 4.06914894e-01, 3.48684382e+00, 2.15914607e-01, 7.78429407e-01],
                                        [5.00000000e+01, 3.15219540e-01, 3.56065217e-01, 3.07197401e+00, 2.52605529e-01, 7.54403863e-01],
                                        [6.00000000e+01, 2.71579334e-01, 3.10200000e-01, 2.87918222e+00, 2.89945134e-01, 7.21368012e-01],
                                        [7.00000000e+01, 2.36628680e-01, 2.75636364e-01, 2.30268595e+00, 3.26200049e-01, 7.07334431e-01],
                                        [8.00000000e+01, 2.13471088e-01, 2.48813953e-01, 1.97657004e+00, 3.61630740e-01, 6.91667434e-01],
                                        [9.00000000e+01, 1.89508424e-01, 2.25142857e-01, 1.81088435e+00, 3.97295403e-01, 6.70017169e-01],
                                        [1.00000000e+02, 1.69940900e-01, 2.04975610e-01, 1.54163355e+00, 4.32984486e-01, 6.59707455e-01]],
                                        result_1[:10].values)

        np.testing.assert_almost_equal([[0.66768809, 2.56367503],
                                        [1.33537618, 0.91257918],
                                        [2.00306427, 0.46321748],
                                        [2.67075237, 0.351698  ],
                                        [3.33844046, 0.30670727],
                                        [4.00612855, 0.27797148],
                                        [4.67381664, 0.28517525],
                                        [5.34150473, 0.25558116],
                                        [6.00919282, 0.43465358],
                                        [6.67688091, 1.17440066]],
                                        result_2[:10].values,
                                        decimal=6)

        np.testing.assert_almost_equal([[1.00000000e+01, 6.10009646e-01, 7.03175000e-01, 2.43406050e+00, 8.27417520e-02, 4.16311974e-01],
                                        [2.00000000e+01, 5.15380985e-01, 5.86581633e-01, 3.81830487e+00, 1.18258784e-01, 5.04832768e-01],
                                        [3.00000000e+01, 4.46227174e-01, 4.98567708e-01, 3.82395779e+00, 1.50123309e-01, 5.10452341e-01],
                                        [4.00000000e+01, 3.83955414e-01, 4.30186170e-01, 2.90265844e+00, 1.82651929e-01, 4.80191842e-01],
                                        [5.00000000e+01, 3.36766601e-01, 3.76385870e-01, 2.51859936e+00, 2.12528212e-01, 4.61088863e-01],
                                        [6.00000000e+01, 2.91701902e-01, 3.30722222e-01, 2.38147160e+00, 2.43524094e-01, 4.29056716e-01],
                                        [7.00000000e+01, 2.56238007e-01, 2.95539773e-01, 2.00235214e+00, 2.73147646e-01, 4.08965005e-01],
                                        [8.00000000e+01, 2.30268092e-01, 2.66715116e-01, 1.74837345e+00, 3.02306793e-01, 3.91023380e-01],
                                        [9.00000000e+01, 2.05733913e-01, 2.41309524e-01, 1.60862812e+00, 3.31708111e-01, 3.75629056e-01],
                                        [1.00000000e+02, 1.84639944e-01, 2.20487805e-01, 1.38749256e+00, 3.60816306e-01, 3.58707726e-01]],
                                        result_3[:10].values,)

        np.testing.assert_almost_equal([[0.66768809, 2.56367503],
                                        [1.33537618, 0.91257918],
                                        [2.00306427, 0.46321748],
                                        [2.67075237, 0.351698  ],
                                        [3.33844046, 0.30670727],
                                        [4.00612855, 0.27797148],
                                        [4.67381664, 0.28517525],
                                        [5.34150473, 0.25558116],
                                        [6.00919282, 0.43465358],
                                        [6.67688091, 1.17440066]],
                                        result_4[:10].values,
                                        decimal=6)
        
        #xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1:1.0, 2:1.0},
                           a=0.3, 
                           cal_type = "fast", 
                           neighborfile=None,
                           max_neighbors=100
                           )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.85376432e-01, 3.25360000e-01, 2.93295040e+00, 9.52096594e-02, 6.98532609e-01],
                                        [2.00000000e+01, 4.90235700e-01, 4.41040816e-01, 4.37803915e+00, 1.38404305e-01, 8.25369390e-01],
                                        [3.00000000e+01, 4.21873302e-01, 5.26208333e-01, 4.39766493e+00, 1.77096679e-01, 8.10179925e-01],
                                        [4.00000000e+01, 3.61045399e-01, 5.93085106e-01, 3.48684382e+00, 2.15914607e-01, 7.78429407e-01],
                                        [5.00000000e+01, 3.15219540e-01, 6.43934783e-01, 3.07197401e+00, 2.52605529e-01, 7.54403863e-01],
                                        [6.00000000e+01, 2.71579334e-01, 6.89800000e-01, 2.87918222e+00, 2.89945134e-01, 7.21368012e-01],
                                        [7.00000000e+01, 2.36628680e-01, 7.24363636e-01, 2.30268595e+00, 3.26200049e-01, 7.07334431e-01],
                                        [8.00000000e+01, 2.13471088e-01, 7.51186047e-01, 1.97657004e+00, 3.61630740e-01, 6.91667434e-01],
                                        [9.00000000e+01, 1.89508424e-01, 7.74857143e-01, 1.81088435e+00, 3.97295403e-01, 6.70017169e-01],
                                        [1.00000000e+02, 1.69940900e-01, 7.95024390e-01, 1.54163355e+00, 4.32984486e-01, 6.59707455e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 4.78008808],
                                        [1.33537618, 1.61767014],
                                        [2.00306427, 0.9066165 ],
                                        [2.67075237, 0.65698861],
                                        [3.33844046, 0.62705741],
                                        [4.00612855, 0.60498341],
                                        [4.67381664, 0.56973604],
                                        [5.34150473, 0.51854836],
                                        [6.00919282, 0.68482547],
                                        [6.67688091, 1.0788529 ]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 6.10009646e-01, 2.96825000e-01, 2.43406050e+00, 8.27417520e-02, 4.16311974e-01],
                                        [2.00000000e+01, 5.15380985e-01, 4.13418367e-01, 3.81830487e+00, 1.18258784e-01, 5.04832768e-01],
                                        [3.00000000e+01, 4.46227174e-01, 5.01432292e-01, 3.82395779e+00, 1.50123309e-01, 5.10452341e-01],
                                        [4.00000000e+01, 3.83955414e-01, 5.69813830e-01, 2.90265844e+00, 1.82651929e-01, 4.80191842e-01],
                                        [5.00000000e+01, 3.36766601e-01, 6.23614130e-01, 2.51859936e+00, 2.12528212e-01, 4.61088863e-01],
                                        [6.00000000e+01, 2.91701902e-01, 6.69277778e-01, 2.38147160e+00, 2.43524094e-01, 4.29056716e-01],
                                        [7.00000000e+01, 2.56238007e-01, 7.04460227e-01, 2.00235214e+00, 2.73147646e-01, 4.08965005e-01],
                                        [8.00000000e+01, 2.30268092e-01, 7.33284884e-01, 1.74837345e+00, 3.02306793e-01, 3.91023380e-01],
                                        [9.00000000e+01, 2.05733913e-01, 7.58690476e-01, 1.60862812e+00, 3.31708111e-01, 3.75629056e-01],
                                        [1.00000000e+02, 1.84639944e-01, 7.79512195e-01, 1.38749256e+00, 3.60816306e-01, 3.58707726e-01]],
                                    result_3[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 4.78008808],
                                        [1.33537618, 1.61767014],
                                        [2.00306427, 0.9066165 ],
                                        [2.67075237, 0.65698861],
                                        [3.33844046, 0.62705741],
                                        [4.00612855, 0.60498341],
                                        [4.67381664, 0.56973604],
                                        [5.34150473, 0.51854836],
                                        [6.00919282, 0.68482547],
                                        [6.67688091, 1.0788529 ]],
                                        result_4[:10].values,
                                        decimal=6)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_x,self.test_file_3d_xu}...")
        
        
    def test_Dynamics_3d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_x}...")
        xu_snapshots = None
        x_snapshots = self.dump_3d_x.snapshots
        ppp = np.array([1, 0, 1])
        t = 10
        qrange = 10.0
        condition=[]
        for snapshot in x_snapshots.snapshots:
            condition.append(snapshot.particle_type==1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)
        
        #xu, x, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "slow", 
                            neighborfile=None,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 10.        , 0.58562118, 0.669     , 2.90804   , 1.14867053, 39.46810003],
                                        [ 20.        , 0.49120256, 0.55357143, 4.362     , 1.39083323, 31.50783097],
                                        [ 30.        , 0.42374376, 0.4691875 , 4.35440234, 1.55559195, 27.44939264],
                                        [ 40.        , 0.36364126, 0.40308511, 3.45441829, 1.70880412, 24.39118919],
                                        [ 50.        , 0.31828923, 0.35284783, 3.02908554, 1.84858347, 22.08742322],
                                        [ 60.        , 0.27498272, 0.30708889, 2.84234765, 2.0322594 , 19.80923091],
                                        [ 70.        , 0.2405527 , 0.27288636, 2.26605527, 2.13716318, 18.47472918],
                                        [ 80.        , 0.21749221, 0.24634884, 1.94427366, 2.25244128, 17.24177308],
                                        [ 90.        , 0.19374938, 0.2232381 , 1.78360998, 2.36594246, 16.17150286],
                                        [100.        , 0.17412473, 0.203     , 1.5122439 , 2.4846209 , 15.19692481]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 2.53579305],
                                        [1.33537618, 0.97599021],
                                        [2.00306427, 0.46939176],
                                        [2.67075237, 0.40131863],
                                        [3.33844046, 0.31538759],
                                        [4.00612855, 0.28909345],
                                        [4.67381664, 0.29384433],
                                        [5.34150473, 0.27192163],
                                        [6.00919282, 0.43898384],
                                        [6.67688091, 1.17581095]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[ 10.        , 0.60984638, 0.697475  , 2.4255745 , 1.06660957, 43.00838729],
                                        [ 20.        , 0.51595046, 0.58109694, 3.81697105, 1.27206551, 34.99739507],
                                        [ 30.        , 0.4477483 , 0.49372396, 3.78643175, 1.4512896 , 30.01489827],
                                        [ 40.        , 0.38620829, 0.42643617, 2.8966478 , 1.56666823, 27.13826655],
                                        [ 50.        , 0.33958615, 0.37326087, 2.51584121, 1.70669883, 24.48119192],
                                        [ 60.        , 0.29471306, 0.32738889, 2.36421235, 1.87265578, 22.02864656],
                                        [ 70.        , 0.2599044 , 0.29264205, 1.98427363, 1.96431499, 20.63613338],
                                        [ 80.        , 0.23411386, 0.26415698, 1.71568145, 2.08264386, 19.18861779],
                                        [ 90.        , 0.20976091, 0.23934524, 1.58185941, 2.20480009, 17.90780372],
                                        [100.        , 0.18864996, 0.21841463, 1.34884295, 2.32347205, 16.79386344]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 2.53579305],
                                        [1.33537618, 0.97599021],
                                        [2.00306427, 0.46939176],
                                        [2.67075237, 0.40131863],
                                        [3.33844046, 0.31538759],
                                        [4.00612855, 0.28909345],
                                        [4.67381664, 0.29384433],
                                        [5.34150473, 0.27192163],
                                        [6.00919282, 0.43898384],
                                        [6.67688091, 1.17581095]],
                                        result_4[:10].values)
        
        #xu, x, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=None,
                            max_neighbors=30
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 10.        , 0.58562118, 0.331     , 2.90804   , 1.14867053, 39.46810003],
                                        [ 20.        , 0.49120256, 0.44642857, 4.362     , 1.39083323, 31.50783097],
                                        [ 30.        , 0.42374376, 0.5308125 , 4.35440234, 1.55559195, 27.44939264],
                                        [ 40.        , 0.36364126, 0.59691489, 3.45441829, 1.70880412, 24.39118919],
                                        [ 50.        , 0.31828923, 0.64715217, 3.02908554, 1.84858347, 22.08742322],
                                        [ 60.        , 0.27498272, 0.69291111, 2.84234765, 2.0322594 , 19.80923091],
                                        [ 70.        , 0.2405527 , 0.72711364, 2.26605527, 2.13716318, 18.47472918],
                                        [ 80.        , 0.21749221, 0.75365116, 1.94427366, 2.25244128, 17.24177308],
                                        [ 90.        , 0.19374938, 0.7767619 , 1.78360998, 2.36594246, 16.17150286],
                                        [100.        , 0.17412473, 0.797     , 1.5122439 , 2.4846209 , 15.19692481]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 4.63251721],
                                        [1.33537618, 1.6924979 ],
                                        [2.00306427, 0.89100367],
                                        [2.67075237, 0.6695242 ],
                                        [3.33844046, 0.62921307],
                                        [4.00612855, 0.61722382],
                                        [4.67381664, 0.57443635],
                                        [5.34150473, 0.55610709],
                                        [6.00919282, 0.68165549],
                                        [6.67688091, 1.07903518]],
                                        result_2[:10].values)
        np.testing.assert_almost_equal([[ 10.        , 0.60984638, 0.302525  , 2.4255745 ,1.06660957, 43.00838729],
                                        [ 20.        , 0.51595046, 0.41890306, 3.81697105,1.27206551, 34.99739507],
                                        [ 30.        , 0.4477483 , 0.50627604, 3.78643175,1.4512896 , 30.01489827],
                                        [ 40.        , 0.38620829, 0.57356383, 2.8966478 ,1.56666823, 27.13826655],
                                        [ 50.        , 0.33958615, 0.62673913, 2.51584121,1.70669883, 24.48119192],
                                        [ 60.        , 0.29471306, 0.67261111, 2.36421235,1.87265578, 22.02864656],
                                        [ 70.        , 0.2599044 , 0.70735795, 1.98427363,1.96431499, 20.63613338],
                                        [ 80.        , 0.23411386, 0.73584302, 1.71568145,2.08264386, 19.18861779],
                                        [ 90.        , 0.20976091, 0.76065476, 1.58185941,2.20480009, 17.90780372],
                                        [100.        , 0.18864996, 0.78158537, 1.34884295,2.32347205, 16.79386344]],
                                                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 4.63251721],
                                        [1.33537618, 1.6924979 ],
                                        [2.00306427, 0.89100367],
                                        [2.67075237, 0.6695242 ],
                                        [3.33844046, 0.62921307],
                                        [4.00612855, 0.61722382],
                                        [4.67381664, 0.57443635],
                                        [5.34150473, 0.55610709],
                                        [6.00919282, 0.68165549],
                                        [6.67688091, 1.07903518]],
                                        result_4[:10].values)
        
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_x}...")
        
        
    def test_Dynamics_3d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_xu}...")
        xu_snapshots = self.dump_3d_xu.snapshots
        x_snapshots = None
        ppp = np.array([0, 0, 0])
        t = 10
        qrange = 10.0
        condition=[]
        for snapshot in xu_snapshots.snapshots:
            condition.append(snapshot.particle_type==1)
        condition = np.array(condition)
        condition_sq4 = np.ones(condition.shape)
        
        #xu, slow, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                           x_snapshots=x_snapshots,
                           dt=0.002,
                           ppp=ppp,
                           diameters={1:1.0, 2:1.0},
                           a=0.3, 
                           cal_type = "slow", 
                           neighborfile=None,
                           max_neighbors=30
                           )       
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.85376432e-01, 6.74640000e-01, 2.93295040e+00, 9.52096594e-02, 6.98532609e-01],
                                        [2.00000000e+01, 4.90235700e-01, 5.58959184e-01, 4.37803915e+00, 1.38404305e-01, 8.25369390e-01],
                                        [3.00000000e+01, 4.21873302e-01, 4.73791667e-01, 4.39766493e+00, 1.77096679e-01, 8.10179925e-01],
                                        [4.00000000e+01, 3.61045399e-01, 4.06914894e-01, 3.48684382e+00, 2.15914607e-01, 7.78429407e-01],
                                        [5.00000000e+01, 3.15219540e-01, 3.56065217e-01, 3.07197401e+00, 2.52605529e-01, 7.54403863e-01],
                                        [6.00000000e+01, 2.71579334e-01, 3.10200000e-01, 2.87918222e+00, 2.89945134e-01, 7.21368012e-01],
                                        [7.00000000e+01, 2.36628680e-01, 2.75636364e-01, 2.30268595e+00, 3.26200049e-01, 7.07334431e-01],
                                        [8.00000000e+01, 2.13471088e-01, 2.48813953e-01, 1.97657004e+00, 3.61630740e-01, 6.91667434e-01],
                                        [9.00000000e+01, 1.89508424e-01, 2.25142857e-01, 1.81088435e+00, 3.97295403e-01, 6.70017169e-01],
                                        [1.00000000e+02, 1.69940900e-01, 2.04975610e-01, 1.54163355e+00, 4.32984486e-01, 6.59707455e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 2.56367441],
                                        [1.33537618, 0.91257941],
                                        [2.00306427, 0.46321747],
                                        [2.67075237, 0.3516963 ],
                                        [3.33844046, 0.3067072 ],
                                        [4.00612855, 0.27797203],
                                        [4.67381664, 0.28517459],
                                        [5.34150473, 0.25558046],
                                        [6.00919282, 0.43465354],
                                        [6.67688091, 1.17440061]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 6.10009646e-01, 7.03175000e-01, 2.43406050e+00, 8.27417520e-02, 4.16311974e-01],
                                        [2.00000000e+01, 5.15380985e-01, 5.86581633e-01, 3.81830487e+00, 1.18258784e-01, 5.04832768e-01],
                                        [3.00000000e+01, 4.46227174e-01, 4.98567708e-01, 3.82395779e+00, 1.50123309e-01, 5.10452341e-01],
                                        [4.00000000e+01, 3.83955414e-01, 4.30186170e-01, 2.90265844e+00, 1.82651929e-01, 4.80191842e-01],
                                        [5.00000000e+01, 3.36766601e-01, 3.76385870e-01, 2.51859936e+00, 2.12528212e-01, 4.61088863e-01],
                                        [6.00000000e+01, 2.91701902e-01, 3.30722222e-01, 2.38147160e+00, 2.43524094e-01, 4.29056716e-01],
                                        [7.00000000e+01, 2.56238007e-01, 2.95539773e-01, 2.00235214e+00, 2.73147646e-01, 4.08965005e-01],
                                        [8.00000000e+01, 2.30268092e-01, 2.66715116e-01, 1.74837345e+00, 3.02306793e-01, 3.91023380e-01],
                                        [9.00000000e+01, 2.05733913e-01, 2.41309524e-01, 1.60862812e+00, 3.31708111e-01, 3.75629056e-01],
                                        [1.00000000e+02, 1.84639944e-01, 2.20487805e-01, 1.38749256e+00, 3.60816306e-01, 3.58707726e-01]],
                                        result_3[:10].values,)
        np.testing.assert_almost_equal([[0.66768809, 2.56367441],
                                        [1.33537618, 0.91257941],
                                        [2.00306427, 0.46321747],
                                        [2.67075237, 0.3516963 ],
                                        [3.33844046, 0.3067072 ],
                                        [4.00612855, 0.27797203],
                                        [4.67381664, 0.28517459],
                                        [5.34150473, 0.25558046],
                                        [6.00919282, 0.43465354],
                                        [6.67688091, 1.17440061]],
                                        result_4[:10].values,
                                        decimal=6)
        
        #xu, fast, no neighbor
        dynamic = Dynamics(xu_snapshots=xu_snapshots,
                            x_snapshots=x_snapshots,
                            dt=0.002,
                            ppp=ppp,
                            diameters={1:1.0, 2:1.0},
                            a=0.3, 
                            cal_type = "fast", 
                            neighborfile=None,
                            max_neighbors=100
                            )
        #no condition 
        dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=None, outputfile="test_no_condition_sq4.csv")
        #condition
        dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        dynamic.sq4(t=t, qrange=qrange, condition=condition_sq4, outputfile="test_w_condition_sq4.csv")

        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_no_condition_sq4.csv')
        result_3 = pd.read_csv('test_w_condition.csv')
        result_4 = pd.read_csv('test_w_condition_sq4.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_no_condition_sq4.csv")
        os.remove("test_w_condition_sq4.csv")
        os.remove("test_w_condition.csv")

        np.testing.assert_almost_equal([[1.00000000e+01, 5.85376432e-01, 3.25360000e-01, 2.93295040e+00, 9.52096594e-02, 6.98532609e-01],
                                        [2.00000000e+01, 4.90235700e-01, 4.41040816e-01, 4.37803915e+00, 1.38404305e-01, 8.25369390e-01],
                                        [3.00000000e+01, 4.21873302e-01, 5.26208333e-01, 4.39766493e+00, 1.77096679e-01, 8.10179925e-01],
                                        [4.00000000e+01, 3.61045399e-01, 5.93085106e-01, 3.48684382e+00, 2.15914607e-01, 7.78429407e-01],
                                        [5.00000000e+01, 3.15219540e-01, 6.43934783e-01, 3.07197401e+00, 2.52605529e-01, 7.54403863e-01],
                                        [6.00000000e+01, 2.71579334e-01, 6.89800000e-01, 2.87918222e+00, 2.89945134e-01, 7.21368012e-01],
                                        [7.00000000e+01, 2.36628680e-01, 7.24363636e-01, 2.30268595e+00, 3.26200049e-01, 7.07334431e-01],
                                        [8.00000000e+01, 2.13471088e-01, 7.51186047e-01, 1.97657004e+00, 3.61630740e-01, 6.91667434e-01],
                                        [9.00000000e+01, 1.89508424e-01, 7.74857143e-01, 1.81088435e+00, 3.97295403e-01, 6.70017169e-01],
                                        [1.00000000e+02, 1.69940900e-01, 7.95024390e-01, 1.54163355e+00, 4.32984486e-01, 6.59707455e-01]],
                                        result_1[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 4.78008671],
                                        [1.33537618, 1.61766825],
                                        [2.00306427, 0.90661578],
                                        [2.67075237, 0.65698813],
                                        [3.33844046, 0.62705678],
                                        [4.00612855, 0.60498478],
                                        [4.67381664, 0.5697354 ],
                                        [5.34150473, 0.5185442 ],
                                        [6.00919282, 0.68482562],
                                        [6.67688091, 1.07885023]],
                                        result_2[:10].values,
                                        decimal=6)
        np.testing.assert_almost_equal([[1.00000000e+01, 6.10009646e-01, 2.96825000e-01, 2.43406050e+00, 8.27417520e-02, 4.16311974e-01],
                                        [2.00000000e+01, 5.15380985e-01, 4.13418367e-01, 3.81830487e+00, 1.18258784e-01, 5.04832768e-01],
                                        [3.00000000e+01, 4.46227174e-01, 5.01432292e-01, 3.82395779e+00, 1.50123309e-01, 5.10452341e-01],
                                        [4.00000000e+01, 3.83955414e-01, 5.69813830e-01, 2.90265844e+00, 1.82651929e-01, 4.80191842e-01],
                                        [5.00000000e+01, 3.36766601e-01, 6.23614130e-01, 2.51859936e+00, 2.12528212e-01, 4.61088863e-01],
                                        [6.00000000e+01, 2.91701902e-01, 6.69277778e-01, 2.38147160e+00, 2.43524094e-01, 4.29056716e-01],
                                        [7.00000000e+01, 2.56238007e-01, 7.04460227e-01, 2.00235214e+00, 2.73147646e-01, 4.08965005e-01],
                                        [8.00000000e+01, 2.30268092e-01, 7.33284884e-01, 1.74837345e+00, 3.02306793e-01, 3.91023380e-01],
                                        [9.00000000e+01, 2.05733913e-01, 7.58690476e-01, 1.60862812e+00, 3.31708111e-01, 3.75629056e-01],
                                        [1.00000000e+02, 1.84639944e-01, 7.79512195e-01, 1.38749256e+00, 3.60816306e-01, 3.58707726e-01]],
                                        result_3[:10].values)
        np.testing.assert_almost_equal([[0.66768809, 4.78008671],
                                        [1.33537618, 1.61766825],
                                        [2.00306427, 0.90661578],
                                        [2.67075237, 0.65698813],
                                        [3.33844046, 0.62705678],
                                        [4.00612855, 0.60498478],
                                        [4.67381664, 0.5697354 ],
                                        [5.34150473, 0.5185442 ],
                                        [6.00919282, 0.68482562],
                                        [6.67688091, 1.07885023]],
                                        result_4[:10].values,
                                        decimal=6)
        
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_xu}...")
        

    def test_LogDynamics_2d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_log_x,self.test_file_2d_log_xu}...")
        xu_snapshots = self.dump_2d_log_xu.snapshots
        x_snapshots = self.dump_2d_log_x.snapshots
        ppp = np.array([0,0])
        condition=(xu_snapshots.snapshots[0].particle_type==1)
        
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6,ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6,ppp=ppp)
        neighborfile = 'neighborlist.dat'
        
        #xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99004127e-01, 1.00000000e+00, 0.00000000e+00, 1.00951932e-04, -1.65671614e-02],
                                        [ 1.10000000e-01, 9.73961692e-01, 1.00000000e+00, 0.00000000e+00, 2.67278761e-03, 2.47922998e-03],
                                        [ 2.00000000e-01, 9.36689188e-01, 1.00000000e+00, 0.00000000e+00, 6.62964957e-03, 2.08616903e-02],
                                        [ 1.10000000e+00, 7.91826907e-01, 9.73000000e-01, 0.00000000e+00, 2.38147105e-02, 7.30150947e-02],
                                        [ 2.00000000e+00, 6.90313836e-01, 8.99000000e-01, 0.00000000e+00, 3.87689761e-02, 1.79901429e-01],
                                        [ 1.10000000e+01, 4.56127411e-01, 6.74000000e-01, 0.00000000e+00, 8.87644276e-02, 3.47244235e-01],
                                        [ 2.00000000e+01, 3.31481504e-01, 5.56000000e-01, 0.00000000e+00, 1.30796577e-01, 2.59409224e-01],
                                        [ 1.10000000e+02, 2.73535257e-02, 2.34000000e-01, 0.00000000e+00, 3.78405353e-01, 2.02337029e-01],
                                        [ 2.00000000e+02, 5.50599396e-02, 2.33000000e-01, 0.00000000e+00, 3.81775819e-01, 2.68392690e-01],
                                        [ 1.10000000e+03, 4.32414155e-03, 1.12000000e-01, 0.00000000e+00, 9.71862967e-01, 1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99033164e-01, 1.00000000e+00,0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [ 1.10000000e-01, 9.74236370e-01, 1.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [ 2.00000000e-01, 9.35823089e-01, 1.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [ 1.10000000e+00, 7.93078510e-01, 9.75384615e-01, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [ 2.00000000e+00, 7.04602830e-01, 9.07692308e-01, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [ 1.10000000e+01, 4.76322541e-01, 6.95384615e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [ 2.00000000e+01, 3.30483359e-01, 5.63076923e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [ 1.10000000e+02, 4.01591949e-02, 2.41538462e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [ 2.00000000e+02, 7.67369390e-02, 2.36923077e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [ 1.10000000e+03, -2.04444645e-03, 1.06153846e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)
        
        #xu, x, slow, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        #log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        #result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        #os.remove("test_w_condition.csv")
        
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98862702e-01, 1.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [ 1.10000000e-01, 9.70926555e-01, 1.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [ 2.00000000e-01, 9.33593192e-01, 1.00000000e+00,0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [ 1.10000000e+00, 8.61662385e-01, 9.91000000e-01, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [ 2.00000000e+00, 8.27684416e-01, 9.63000000e-01, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [ 1.10000000e+01, 7.17897230e-01, 8.92000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [ 2.00000000e+01, 6.38977561e-01, 8.19000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [ 1.10000000e+02, 3.57359912e-01, 5.69000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [ 2.00000000e+02, 3.64432615e-01, 5.73000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [ 1.10000000e+03, 1.65484175e-01, 3.14000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        #np.testing.assert_almost_equal([],result_2[:10].values)
        
        #xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99004127e-01, 0.00000000e+00, 0.00000000e+00, 1.00951932e-04, -1.65671614e-02],
                                        [ 1.10000000e-01, 9.73961692e-01, 0.00000000e+00, 0.00000000e+00, 2.67278761e-03, 2.47922998e-03],
                                        [ 2.00000000e-01, 9.36689188e-01, 0.00000000e+00, 0.00000000e+00, 6.62964957e-03, 2.08616903e-02],
                                        [ 1.10000000e+00, 7.91826907e-01, 2.70000000e-02, 0.00000000e+00, 2.38147105e-02, 7.30150947e-02],
                                        [ 2.00000000e+00, 6.90313836e-01, 1.01000000e-01, 0.00000000e+00, 3.87689761e-02, 1.79901429e-01],
                                        [ 1.10000000e+01, 4.56127411e-01, 3.26000000e-01, 0.00000000e+00, 8.87644276e-02, 3.47244235e-01],
                                        [ 2.00000000e+01, 3.31481504e-01, 4.44000000e-01, 0.00000000e+00, 1.30796577e-01, 2.59409224e-01],
                                        [ 1.10000000e+02, 2.73535257e-02, 7.66000000e-01, 0.00000000e+00, 3.78405353e-01, 2.02337029e-01],
                                        [ 2.00000000e+02, 5.50599396e-02, 7.67000000e-01, 0.00000000e+00, 3.81775819e-01, 2.68392690e-01],
                                        [ 1.10000000e+03, 4.32414155e-03, 8.88000000e-01, 0.00000000e+00, 9.71862967e-01, 1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99033164e-01, 0.00000000e+00, 0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [ 1.10000000e-01, 9.74236370e-01, 0.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [ 2.00000000e-01, 9.35823089e-01, 0.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [ 1.10000000e+00, 7.93078510e-01, 2.46153846e-02, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [ 2.00000000e+00, 7.04602830e-01, 9.23076923e-02, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [ 1.10000000e+01, 4.76322541e-01, 3.04615385e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [ 2.00000000e+01, 3.30483359e-01, 4.36923077e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [ 1.10000000e+02, 4.01591949e-02, 7.58461538e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [ 2.00000000e+02, 7.67369390e-02, 7.63076923e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [ 1.10000000e+03, -2.04444645e-03, 8.93846154e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)
        
        #xu, x, fast, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        #log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        #result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        #os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98862702e-01, 0.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [ 1.10000000e-01, 9.70926555e-01, 0.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [ 2.00000000e-01, 9.33593192e-01, 0.00000000e+00, 0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [ 1.10000000e+00, 8.61662385e-01, 9.00000000e-03, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [ 2.00000000e+00, 8.27684416e-01, 3.70000000e-02, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [ 1.10000000e+01, 7.17897230e-01, 1.08000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [ 2.00000000e+01, 6.38977561e-01, 1.81000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [ 1.10000000e+02, 3.57359912e-01, 4.31000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [ 2.00000000e+02, 3.64432615e-01, 4.27000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [ 1.10000000e+03, 1.65484175e-01, 6.86000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        #np.testing.assert_almost_equal([],result_2[:10].values)
        
        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_log_x,self.test_file_2d_log_xu}...")


    def test_LogDynamics_2d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_log_x}...")
        xu_snapshots = None
        x_snapshots = self.dump_2d_log_x.snapshots
        condition=(x_snapshots.snapshots[0].particle_type==1)
        ppp = np.array([1,0])
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6,ppp=ppp)
        else:
            Nnearests(snapshots=x_snapshots, N=6,ppp=ppp)
        neighborfile = 'neighborlist.dat'
        
        #x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02,  9.99004127e-01,  1.00000000e+00,  0.00000000e+00,  1.00951937e-04, -1.66244715e-02],
                                        [ 1.10000000e-01,  9.73407171e-01,  9.98000000e-01,  0.00000000e+00,  1.66307070e+00,  2.48199661e+02],
                                        [ 2.00000000e-01,  9.35825056e-01,  9.97000000e-01,  0.00000000e+00,  2.49616085e+00,  1.64784970e+02],
                                        [ 1.10000000e+00,  7.90936761e-01,  9.70000000e-01,  0.00000000e+00,  2.51239315e+00,  1.62529909e+02],
                                        [ 2.00000000e+00,  6.88625594e-01,  8.95000000e-01,  0.00000000e+00,  4.15913517e+00,  9.71594000e+01],
                                        [ 1.10000000e+01,  4.54025701e-01,  6.71000000e-01,  0.00000000e+00,  5.04226199e+00,  7.94421760e+01],
                                        [ 2.00000000e+01,  3.30174788e-01,  5.53000000e-01,  0.00000000e+00,  3.43019255e+00,  1.14669853e+02],
                                        [ 1.10000000e+02,  2.58498614e-02,  2.33000000e-01,  0.00000000e+00,  1.14907876e+01,  3.25131387e+01],
                                        [ 2.00000000e+02,  5.54434609e-02,  2.33000000e-01,  0.00000000e+00,  1.07529011e+01,  3.48872588e+01],
                                        [ 1.10000000e+03,  3.91603498e-03,  1.12000000e-01,  0.00000000e+00,  2.15698479e+01,  1.60366508e+01]],
                                        result_1[:10].values,decimal=6)
        np.testing.assert_almost_equal([[ 2.00000000e-02,  9.99033173e-01,  1.00000000e+00,  0.00000000e+00,  9.80036933e-05, -7.26690771e-02],
                                        [ 1.10000000e-01,  9.73383227e-01,  9.96923077e-01,  0.00000000e+00,  2.55710163e+00,  1.61165899e+02],
                                        [ 2.00000000e-01,  9.34493858e-01,  9.95384615e-01,  0.00000000e+00,  3.83677352e+00,  1.06955457e+02],
                                        [ 1.10000000e+00,  7.91708997e-01,  9.70769231e-01,  0.00000000e+00,  3.85211907e+00,  1.06018666e+02],
                                        [ 2.00000000e+00,  7.02398006e-01,  9.01538462e-01,  0.00000000e+00,  5.12252716e+00,  7.90971317e+01],
                                        [ 1.10000000e+01,  4.73089678e-01,  6.90769231e-01,  0.00000000e+00,  7.70348319e+00,  5.20207659e+01],
                                        [ 2.00000000e+01,  3.29049278e-01,  5.60000000e-01,  0.00000000e+00,  3.92954695e+00,  1.00583671e+02],
                                        [ 1.10000000e+02,  3.85923832e-02,  2.41538462e-01,  0.00000000e+00,  1.39152611e+01,  2.70960478e+01],
                                        [ 2.00000000e+02,  7.74662786e-02,  2.36923077e-01,  0.00000000e+00,  1.15110626e+01,  3.30037282e+01],
                                        [ 1.10000000e+03, -2.05115033e-03,  1.06153846e-01,  0.00000000e+00,  2.32947100e+01,  1.48739439e+01]],
                                        result_2[:10].values,decimal=6)
        
        #x, slow, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02,  9.98864364e-01,  1.00000000e+00,  0.00000000e+00,  1.15128487e-04, -1.85681823e-02],
                                        [ 1.10000000e-01,  9.68970306e-01,  9.92000000e-01,  0.00000000e+00,  1.80628087e+00,  2.11637920e+02],
                                        [ 2.00000000e-01,  9.31074625e-01,  9.90000000e-01,  0.00000000e+00,  2.29633139e+00,  1.29879227e+02],
                                        [ 1.10000000e+00,  8.61434204e-01,  9.77000000e-01,  0.00000000e+00,  2.73127867e+00,  1.37362615e+02],
                                        [ 2.00000000e+00,  8.28324975e-01,  9.45000000e-01,  0.00000000e+00,  4.11314221e+00,  7.96523310e+01],
                                        [ 1.10000000e+01,  7.20513119e-01,  8.74000000e-01,  0.00000000e+00,  5.07522943e+00,  6.61672114e+01],
                                        [ 2.00000000e+01,  6.40953026e-01,  8.07000000e-01,  0.00000000e+00,  3.25629978e+00,  9.56940464e+01],
                                        [ 1.10000000e+02,  3.49542583e-01,  5.41000000e-01,  0.00000000e+00,  1.03848144e+01,  2.64342521e+01],
                                        [ 2.00000000e+02,  3.58691271e-01,  5.54000000e-01,  0.00000000e+00,  9.42305841e+00,  2.82717939e+01],
                                        [ 1.10000000e+03,  1.69131005e-01,  3.10000000e-01,  0.00000000e+00,  1.39070530e+01,  1.47264225e+01]],
                                        result_1[:10].values,decimal=6)
        #np.testing.assert_almost_equal([],result_2[:10].values,decimal=6)
        
        
        #x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02,  9.99004127e-01,  0.00000000e+00,  0.00000000e+00,  1.00951937e-04, -1.66244715e-02],
                                        [ 1.10000000e-01,  9.73407171e-01,  2.00000000e-03,  0.00000000e+00,  1.66307070e+00,  2.48199661e+02],
                                        [ 2.00000000e-01,  9.35825056e-01,  3.00000000e-03,  0.00000000e+00,  2.49616085e+00,  1.64784970e+02],
                                        [ 1.10000000e+00,  7.90936761e-01,  3.00000000e-02,  0.00000000e+00,  2.51239315e+00,  1.62529909e+02],
                                        [ 2.00000000e+00,  6.88625594e-01,  1.05000000e-01,  0.00000000e+00,  4.15913517e+00,  9.71594000e+01],
                                        [ 1.10000000e+01,  4.54025701e-01,  3.29000000e-01,  0.00000000e+00,  5.04226199e+00,  7.94421760e+01],
                                        [ 2.00000000e+01,  3.30174788e-01,  4.47000000e-01,  0.00000000e+00,  3.43019255e+00,  1.14669853e+02],
                                        [ 1.10000000e+02,  2.58498614e-02,  7.67000000e-01,  0.00000000e+00,  1.14907876e+01,  3.25131387e+01],
                                        [ 2.00000000e+02,  5.54434609e-02,  7.67000000e-01,  0.00000000e+00,  1.07529011e+01,  3.48872588e+01],
                                        [ 1.10000000e+03,  3.91603498e-03,  8.88000000e-01,  0.00000000e+00,  2.15698479e+01,  1.60366508e+01]],
                                        result_1[:10].values,decimal=6)
        np.testing.assert_almost_equal([[ 2.00000000e-02,  9.99033173e-01,  0.00000000e+00,  0.00000000e+00,  9.80036933e-05, -7.26690771e-02],
                                        [ 1.10000000e-01,  9.73383227e-01,  3.07692308e-03,  0.00000000e+00,  2.55710163e+00,  1.61165899e+02],
                                        [ 2.00000000e-01,  9.34493858e-01,  4.61538462e-03,  0.00000000e+00,  3.83677352e+00,  1.06955457e+02],
                                        [ 1.10000000e+00,  7.91708997e-01,  2.92307692e-02,  0.00000000e+00,  3.85211907e+00,  1.06018666e+02],
                                        [ 2.00000000e+00,  7.02398006e-01,  9.84615385e-02,  0.00000000e+00,  5.12252716e+00,  7.90971317e+01],
                                        [ 1.10000000e+01,  4.73089678e-01,  3.09230769e-01,  0.00000000e+00,  7.70348319e+00,  5.20207659e+01],
                                        [ 2.00000000e+01,  3.29049278e-01,  4.40000000e-01,  0.00000000e+00,  3.92954695e+00,  1.00583671e+02],
                                        [ 1.10000000e+02,  3.85923832e-02,  7.58461538e-01,  0.00000000e+00,  1.39152611e+01,  2.70960478e+01],
                                        [ 2.00000000e+02,  7.74662786e-02,  7.63076923e-01,  0.00000000e+00,  1.15110626e+01,  3.30037282e+01],
                                        [ 1.10000000e+03, -2.05115033e-03,  8.93846154e-01,  0.00000000e+00,  2.32947100e+01,  1.48739439e+01]],
                                        result_2[:10].values,decimal=6)
        #x, fast, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        #log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        #result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        #os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02,  9.98864364e-01,  0.00000000e+00,  0.00000000e+00,  1.15128487e-04, -1.85681823e-02],
                                        [ 1.10000000e-01,  9.68970306e-01,  8.00000000e-03,  0.00000000e+00,  1.80628087e+00,  2.11637920e+02],
                                        [ 2.00000000e-01,  9.31074625e-01,  1.00000000e-02,  0.00000000e+00,  2.29633139e+00,  1.29879227e+02],
                                        [ 1.10000000e+00,  8.61434204e-01,  2.30000000e-02,  0.00000000e+00,  2.73127867e+00,  1.37362615e+02],
                                        [ 2.00000000e+00,  8.28324975e-01,  5.50000000e-02,  0.00000000e+00,  4.11314221e+00,  7.96523310e+01],
                                        [ 1.10000000e+01,  7.20513119e-01,  1.26000000e-01,  0.00000000e+00,  5.07522943e+00,  6.61672114e+01],
                                        [ 2.00000000e+01,  6.40953026e-01,  1.93000000e-01,  0.00000000e+00,  3.25629978e+00,  9.56940464e+01],
                                        [ 1.10000000e+02,  3.49542583e-01,  4.59000000e-01,  0.00000000e+00,  1.03848144e+01,  2.64342521e+01],
                                        [ 2.00000000e+02,  3.58691271e-01,  4.46000000e-01,  0.00000000e+00,  9.42305841e+00,  2.82717939e+01],
                                        [ 1.10000000e+03,  1.69131005e-01,  6.90000000e-01,  0.00000000e+00,  1.39070530e+01,  1.47264225e+01]],
                                        result_1[:10].values,decimal=6)
        #np.testing.assert_almost_equal([],result_2[:10].values,decimal=6)
        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_log_x}...")


    def test_LogDynamics_2d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_2d_log_x,self.test_file_2d_log_xu}...")
        xu_snapshots = self.dump_2d_log_xu.snapshots
        x_snapshots = None
        ppp = np.array([0,0])
        condition=(xu_snapshots.snapshots[0].particle_type==1)
        
        if xu_snapshots:
            Nnearests(snapshots=xu_snapshots, N=6,ppp=np.array([0,0]))
        else:
            Nnearests(snapshots=x_snapshots, N=6,ppp=np.array([0,0]))
        neighborfile = 'neighborlist.dat'
        
        #xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99004127e-01, 1.00000000e+00, 0.00000000e+00, 1.00951932e-04, -1.65671614e-02],
                                        [ 1.10000000e-01, 9.73961692e-01, 1.00000000e+00, 0.00000000e+00, 2.67278761e-03, 2.47922998e-03],
                                        [ 2.00000000e-01, 9.36689188e-01, 1.00000000e+00, 0.00000000e+00, 6.62964957e-03, 2.08616903e-02],
                                        [ 1.10000000e+00, 7.91826907e-01, 9.73000000e-01, 0.00000000e+00, 2.38147105e-02, 7.30150947e-02],
                                        [ 2.00000000e+00, 6.90313836e-01, 8.99000000e-01, 0.00000000e+00, 3.87689761e-02, 1.79901429e-01],
                                        [ 1.10000000e+01, 4.56127411e-01, 6.74000000e-01, 0.00000000e+00, 8.87644276e-02, 3.47244235e-01],
                                        [ 2.00000000e+01, 3.31481504e-01, 5.56000000e-01, 0.00000000e+00, 1.30796577e-01, 2.59409224e-01],
                                        [ 1.10000000e+02, 2.73535257e-02, 2.34000000e-01, 0.00000000e+00, 3.78405353e-01, 2.02337029e-01],
                                        [ 2.00000000e+02, 5.50599396e-02, 2.33000000e-01, 0.00000000e+00, 3.81775819e-01, 2.68392690e-01],
                                        [ 1.10000000e+03, 4.32414155e-03, 1.12000000e-01, 0.00000000e+00, 9.71862967e-01, 1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99033164e-01, 1.00000000e+00,0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [ 1.10000000e-01, 9.74236370e-01, 1.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [ 2.00000000e-01, 9.35823089e-01, 1.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [ 1.10000000e+00, 7.93078510e-01, 9.75384615e-01, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [ 2.00000000e+00, 7.04602830e-01, 9.07692308e-01, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [ 1.10000000e+01, 4.76322541e-01, 6.95384615e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [ 2.00000000e+01, 3.30483359e-01, 5.63076923e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [ 1.10000000e+02, 4.01591949e-02, 2.41538462e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [ 2.00000000e+02, 7.67369390e-02, 2.36923077e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [ 1.10000000e+03, -2.04444645e-03, 1.06153846e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)
        #xu, x, slow, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        #log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        #result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        #os.remove("test_w_condition.csv")
        
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98862702e-01, 1.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [ 1.10000000e-01, 9.70926555e-01, 1.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [ 2.00000000e-01, 9.33593192e-01, 1.00000000e+00,0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [ 1.10000000e+00, 8.61662385e-01, 9.91000000e-01, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [ 2.00000000e+00, 8.27684416e-01, 9.63000000e-01, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [ 1.10000000e+01, 7.17897230e-01, 8.92000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [ 2.00000000e+01, 6.38977561e-01, 8.19000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [ 1.10000000e+02, 3.57359912e-01, 5.69000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [ 2.00000000e+02, 3.64432615e-01, 5.73000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [ 1.10000000e+03, 1.65484175e-01, 3.14000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        #np.testing.assert_almost_equal([],result_2[:10].values)
        
        #xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=np.array([0,0]),
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=None,
                                  max_neighbors=30
                                  )
        
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99004127e-01, 0.00000000e+00, 0.00000000e+00, 1.00951932e-04, -1.65671614e-02],
                                        [ 1.10000000e-01, 9.73961692e-01, 0.00000000e+00, 0.00000000e+00, 2.67278761e-03, 2.47922998e-03],
                                        [ 2.00000000e-01, 9.36689188e-01, 0.00000000e+00, 0.00000000e+00, 6.62964957e-03, 2.08616903e-02],
                                        [ 1.10000000e+00, 7.91826907e-01, 2.70000000e-02, 0.00000000e+00, 2.38147105e-02, 7.30150947e-02],
                                        [ 2.00000000e+00, 6.90313836e-01, 1.01000000e-01, 0.00000000e+00, 3.87689761e-02, 1.79901429e-01],
                                        [ 1.10000000e+01, 4.56127411e-01, 3.26000000e-01, 0.00000000e+00, 8.87644276e-02, 3.47244235e-01],
                                        [ 2.00000000e+01, 3.31481504e-01, 4.44000000e-01, 0.00000000e+00, 1.30796577e-01, 2.59409224e-01],
                                        [ 1.10000000e+02, 2.73535257e-02, 7.66000000e-01, 0.00000000e+00, 3.78405353e-01, 2.02337029e-01],
                                        [ 2.00000000e+02, 5.50599396e-02, 7.67000000e-01, 0.00000000e+00, 3.81775819e-01, 2.68392690e-01],
                                        [ 1.10000000e+03, 4.32414155e-03, 8.88000000e-01, 0.00000000e+00, 9.71862967e-01, 1.70318717e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.99033164e-01, 0.00000000e+00, 0.00000000e+00, 9.80046337e-05, -7.26288461e-02],
                                        [ 1.10000000e-01, 9.74236370e-01, 0.00000000e+00, 0.00000000e+00, 2.64330335e-03, -2.42806607e-02],
                                        [ 2.00000000e-01, 9.35823089e-01, 0.00000000e+00, 0.00000000e+00, 6.72553754e-03, 1.75158224e-02],
                                        [ 1.10000000e+00, 7.93078510e-01, 2.46153846e-02, 0.00000000e+00, 2.35368409e-02, 2.71217238e-02],
                                        [ 2.00000000e+00, 7.04602830e-01, 9.23076923e-02, 0.00000000e+00, 3.66670379e-02, 1.91736724e-01],
                                        [ 1.10000000e+01, 4.76322541e-01, 3.04615385e-01, 0.00000000e+00, 8.27177567e-02, 3.35150162e-01],
                                        [ 2.00000000e+01, 3.30483359e-01, 4.36923077e-01, 0.00000000e+00, 1.24750158e-01, 1.96497469e-01],
                                        [ 1.10000000e+02, 4.01591949e-02, 7.58461538e-01, 0.00000000e+00, 3.58516724e-01, 1.53724560e-01],
                                        [ 2.00000000e+02, 7.67369390e-02, 7.63076923e-01, 0.00000000e+00, 3.48722000e-01, 1.93871997e-01],
                                        [ 1.10000000e+03, -2.04444645e-03, 8.93846154e-01, 0.00000000e+00, 9.48509092e-01, 9.56529789e-02]],
                                       result_2[:10].values)
        
        #xu, x, fast, neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=neighborfile,
                                  max_neighbors=30
                                  )
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        #log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        #result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        #os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98862702e-01, 0.00000000e+00, 0.00000000e+00, 1.15296675e-04, -2.09453771e-02],
                                        [ 1.10000000e-01, 9.70926555e-01, 0.00000000e+00, 0.00000000e+00, 2.98907023e-03, -9.32537779e-03],
                                        [ 2.00000000e-01, 9.33593192e-01, 0.00000000e+00, 0.00000000e+00, 6.96818873e-03, 3.08393920e-02],
                                        [ 1.10000000e+00, 8.61662385e-01, 9.00000000e-03, 0.00000000e+00, 1.54546261e-02, 3.08217527e-01],
                                        [ 2.00000000e+00, 8.27684416e-01, 3.70000000e-02, 0.00000000e+00, 2.00778571e-02, 5.05324657e-01],
                                        [ 1.10000000e+01, 7.17897230e-01, 1.08000000e-01, 0.00000000e+00, 3.70905286e-02, 6.05517731e-01],
                                        [ 2.00000000e+01, 6.38977561e-01, 1.81000000e-01, 0.00000000e+00, 5.59671895e-02, 8.97893865e-01],
                                        [ 1.10000000e+02, 3.57359912e-01, 4.31000000e-01, 0.00000000e+00, 1.71316735e-01, 7.51699742e-01],
                                        [ 2.00000000e+02, 3.64432615e-01, 4.27000000e-01, 0.00000000e+00, 1.89270247e-01, 9.58237767e-01],
                                        [ 1.10000000e+03, 1.65484175e-01, 6.86000000e-01, 0.00000000e+00, 5.23812787e-01, 6.71161360e-01]],
                                       result_1[:10].values)
        #np.testing.assert_almost_equal([],result_2[:10].values)
        
        os.remove(neighborfile)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_2d_log_xu}...")


    def test_LogDynamics_3d_x_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_log_x,self.test_file_3d_log_xu}...")
        xu_snapshots = self.dump_3d_log_xu.snapshots
        x_snapshots = self.dump_3d_log_x.snapshots
        condition=(xu_snapshots.snapshots[0].particle_type==1)
        ppp = np.array([0,0,0])

        #xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=None,
                                  max_neighbors=30)
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98794986e-01, 1.00000000e+00, 0.00000000e+00, 1.83249245e-04, 7.62729112e-03],
                                        [ 8.00000000e-02, 9.81788619e-01, 1.00000000e+00, 0.00000000e+00, 2.79353312e-03, 8.83924401e-03],
                                        [ 1.40000000e-01, 9.51758118e-01, 1.00000000e+00, 0.00000000e+00, 7.51534401e-03, 3.27266536e-03],
                                        [ 2.00000000e-01, 9.20215225e-01, 1.00000000e+00, 0.00000000e+00, 1.26307715e-02, -7.52556624e-04],
                                        [ 8.00000000e-01, 7.63523409e-01, 8.87000000e-01, 0.00000000e+00, 4.16240792e-02, 9.39923902e-02],
                                        [ 1.40000000e+00, 7.06977773e-01, 8.35000000e-01, 0.00000000e+00, 5.56373684e-02, 3.69059435e-01],
                                        [ 2.00000000e+00, 6.74200527e-01, 7.87000000e-01, 0.00000000e+00, 6.28006979e-02, 2.98372497e-01],
                                        [ 8.00000000e+00, 4.89460164e-01, 5.23000000e-01, 0.00000000e+00, 1.24775241e-01, 3.90773736e-01],
                                        [ 1.40000000e+01, 3.33250067e-01, 3.65000000e-01, 0.00000000e+00, 2.02048496e-01, 3.42854683e-01],
                                        [ 2.00000000e+01, 3.01752850e-01, 3.47000000e-01, 0.00000000e+00, 2.36770363e-01, 4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98797156e-01, 1.00000000e+00, 0.00000000e+00, 1.82916144e-04, 5.08578384e-04],
                                        [8.00000000e-02, 9.81866650e-01, 1.00000000e+00, 0.00000000e+00, 2.78100867e-03, 1.33905635e-02],
                                        [1.40000000e-01, 9.52395484e-01, 1.00000000e+00, 0.00000000e+00, 7.40924729e-03, 9.07554895e-03],
                                        [2.00000000e-01, 9.22445634e-01, 1.00000000e+00, 0.00000000e+00, 1.22484525e-02, 2.58585214e-03],
                                        [8.00000000e-01, 7.80250930e-01, 9.10000000e-01, 0.00000000e+00, 3.79490640e-02, 6.10574946e-02],
                                        [1.40000000e+00, 7.33305660e-01, 8.72500000e-01, 0.00000000e+00, 4.87008876e-02, 2.97100024e-01],
                                        [2.00000000e+00, 7.01825530e-01, 8.23750000e-01, 0.00000000e+00, 5.54054537e-02, 1.94253070e-01],
                                        [8.00000000e+00, 5.12756184e-01, 5.51250000e-01, 0.00000000e+00, 1.10604697e-01, 2.42693270e-01],
                                        [1.40000000e+01, 3.56859577e-01, 3.90000000e-01, 0.00000000e+00, 1.75137652e-01, 1.53847276e-01],
                                        [2.00000000e+01, 3.29340145e-01, 3.77500000e-01, 0.00000000e+00, 2.04353337e-01, 2.83847908e-01]],
                                       result_2[:10].values)
        
        #xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=None,
                                  max_neighbors=30)
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98794986e-01, 0.00000000e+00, 0.00000000e+00, 1.83249245e-04, 7.62729112e-03],
                                        [ 8.00000000e-02, 9.81788619e-01, 0.00000000e+00, 0.00000000e+00, 2.79353312e-03, 8.83924401e-03],
                                        [ 1.40000000e-01, 9.51758118e-01, 0.00000000e+00, 0.00000000e+00, 7.51534401e-03, 3.27266536e-03],
                                        [ 2.00000000e-01, 9.20215225e-01, 0.00000000e+00, 0.00000000e+00, 1.26307715e-02, -7.52556624e-04],
                                        [ 8.00000000e-01, 7.63523409e-01, 1.13000000e-01, 0.00000000e+00, 4.16240792e-02, 9.39923902e-02],
                                        [ 1.40000000e+00, 7.06977773e-01, 1.65000000e-01, 0.00000000e+00, 5.56373684e-02, 3.69059435e-01],
                                        [ 2.00000000e+00, 6.74200527e-01, 2.13000000e-01, 0.00000000e+00, 6.28006979e-02, 2.98372497e-01],
                                        [ 8.00000000e+00, 4.89460164e-01, 4.77000000e-01, 0.00000000e+00, 1.24775241e-01, 3.90773736e-01],
                                        [ 1.40000000e+01, 3.33250067e-01, 6.35000000e-01, 0.00000000e+00, 2.02048496e-01, 3.42854683e-01],
                                        [ 2.00000000e+01, 3.01752850e-01, 6.53000000e-01, 0.00000000e+00, 2.36770363e-01, 4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98797156e-01, 0.00000000e+00, 0.00000000e+00, 1.82916144e-04, 5.08578384e-04],
                                        [8.00000000e-02, 9.81866650e-01, 0.00000000e+00, 0.00000000e+00, 2.78100867e-03, 1.33905635e-02],
                                        [1.40000000e-01, 9.52395484e-01, 0.00000000e+00, 0.00000000e+00, 7.40924729e-03, 9.07554895e-03],
                                        [2.00000000e-01, 9.22445634e-01, 0.00000000e+00, 0.00000000e+00, 1.22484525e-02, 2.58585214e-03],
                                        [8.00000000e-01, 7.80250930e-01, 9.00000000e-02, 0.00000000e+00, 3.79490640e-02, 6.10574946e-02],
                                        [1.40000000e+00, 7.33305660e-01, 1.27500000e-01, 0.00000000e+00, 4.87008876e-02, 2.97100024e-01],
                                        [2.00000000e+00, 7.01825530e-01, 1.76250000e-01, 0.00000000e+00, 5.54054537e-02, 1.94253070e-01],
                                        [8.00000000e+00, 5.12756184e-01, 4.48750000e-01, 0.00000000e+00, 1.10604697e-01, 2.42693270e-01],
                                        [1.40000000e+01, 3.56859577e-01, 6.10000000e-01, 0.00000000e+00, 1.75137652e-01, 1.53847276e-01],
                                        [2.00000000e+01, 3.29340145e-01, 6.22500000e-01, 0.00000000e+00, 2.04353337e-01, 2.83847908e-01]],
                                       result_2[:10].values)

        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_log_xu,self.test_file_3d_log_x}...")

     
    def test_LogDynamics_3d_x(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_log_x}...")
        xu_snapshots = None
        x_snapshots = self.dump_3d_log_x.snapshots
        condition=(x_snapshots.snapshots[0].particle_type==1)
        ppp = np.array([1,0,1])

        #xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=None,
                                  max_neighbors=30)
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98794938e-01, 1.00000000e+00, 0.00000000e+00, 1.83256494e-04, 7.63917403e-03],
                                        [8.00000000e-02, 9.81788442e-01, 1.00000000e+00, 0.00000000e+00, 2.79356020e-03, 8.83438470e-03],
                                        [1.40000000e-01, 9.50172480e-01, 9.97000000e-01, 0.00000000e+00, 2.70337159e-01, 1.88077806e+02],
                                        [2.00000000e-01, 9.18160727e-01, 9.95000000e-01, 0.00000000e+00, 4.47054811e-01, 1.12380687e+02],
                                        [8.00000000e-01, 7.62698739e-01, 8.83000000e-01, 0.00000000e+00, 7.23713200e-01, 6.57568728e+01],
                                        [1.40000000e+00, 7.07031033e-01, 8.29000000e-01, 0.00000000e+00, 8.17615823e-01, 5.70228377e+01],
                                        [2.00000000e+00, 6.73944627e-01, 7.86000000e-01, 0.00000000e+00, 4.87666214e-01, 9.04120365e+01],
                                        [8.00000000e+00, 4.90589212e-01, 5.19000000e-01, 0.00000000e+00, 1.21864955e+00, 3.63950212e+01],
                                        [1.40000000e+01, 3.38527850e-01, 3.64000000e-01, 0.00000000e+00, 1.83412839e+00, 2.29917076e+01],
                                        [2.00000000e+01, 3.04131179e-01, 3.40000000e-01, 0.00000000e+00, 2.20950081e+00, 1.91470186e+01]],
                                        result_1[:10].values,decimal=6)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98797109e-01, 1.00000000e+00, 0.00000000e+00, 1.82923340e-04, 5.20394664e-04],
                                        [8.00000000e-02, 9.81866477e-01, 1.00000000e+00, 0.00000000e+00, 2.78103508e-03, 1.33854969e-02],
                                        [1.40000000e-01, 9.51106637e-01, 9.97500000e-01, 0.00000000e+00, 2.26149609e-01, 2.23576188e+02],
                                        [2.00000000e-01, 9.21240570e-01, 9.96250000e-01, 0.00000000e+00, 3.35917668e-01, 1.47628949e+02],
                                        [8.00000000e-01, 7.79545330e-01, 9.06250000e-01, 0.00000000e+00, 5.71049476e-01, 8.28087884e+01],
                                        [1.40000000e+00, 7.32988949e-01, 8.67500000e-01, 0.00000000e+00, 5.79522242e-01, 7.97066414e+01],
                                        [2.00000000e+00, 7.01518955e-01, 8.23750000e-01, 0.00000000e+00, 2.68613346e-01, 1.50800767e+02],
                                        [8.00000000e+00, 5.13184259e-01, 5.46250000e-01, 0.00000000e+00, 1.05950477e+00, 4.20002154e+01],
                                        [1.40000000e+01, 3.61867323e-01, 3.88750000e-01, 0.00000000e+00, 1.72157260e+00, 2.50070263e+01],
                                        [2.00000000e+01, 3.31157335e-01, 3.70000000e-01, 0.00000000e+00, 1.95158053e+00, 2.18582771e+01]],
                                        result_2[:10].values,decimal=6)
        
        #xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=None,
                                  max_neighbors=30)
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98794938e-01, 0.00000000e+00, 0.00000000e+00, 1.83256494e-04, 7.63917403e-03],
                                        [8.00000000e-02, 9.81788442e-01, 0.00000000e+00, 0.00000000e+00, 2.79356020e-03, 8.83438470e-03],
                                        [1.40000000e-01, 9.50172480e-01, 3.00000000e-03, 0.00000000e+00, 2.70337159e-01, 1.88077806e+02],
                                        [2.00000000e-01, 9.18160727e-01, 5.00000000e-03, 0.00000000e+00, 4.47054811e-01, 1.12380687e+02],
                                        [8.00000000e-01, 7.62698739e-01, 1.17000000e-01, 0.00000000e+00, 7.23713200e-01, 6.57568728e+01],
                                        [1.40000000e+00, 7.07031033e-01, 1.71000000e-01, 0.00000000e+00, 8.17615823e-01, 5.70228377e+01],
                                        [2.00000000e+00, 6.73944627e-01, 2.14000000e-01, 0.00000000e+00, 4.87666214e-01, 9.04120365e+01],
                                        [8.00000000e+00, 4.90589212e-01, 4.81000000e-01, 0.00000000e+00, 1.21864955e+00, 3.63950212e+01],
                                        [1.40000000e+01, 3.38527850e-01, 6.36000000e-01, 0.00000000e+00, 1.83412839e+00, 2.29917076e+01],
                                        [2.00000000e+01, 3.04131179e-01, 6.60000000e-01, 0.00000000e+00, 2.20950081e+00, 1.91470186e+01]],
                                        result_1[:10].values, decimal=6)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98797109e-01, 0.00000000e+00, 0.00000000e+00, 1.82923340e-04, 5.20394664e-04],
                                        [8.00000000e-02, 9.81866477e-01, 0.00000000e+00, 0.00000000e+00, 2.78103508e-03, 1.33854969e-02],
                                        [1.40000000e-01, 9.51106637e-01, 2.50000000e-03, 0.00000000e+00, 2.26149609e-01, 2.23576188e+02],
                                        [2.00000000e-01, 9.21240570e-01, 3.75000000e-03, 0.00000000e+00, 3.35917668e-01, 1.47628949e+02],
                                        [8.00000000e-01, 7.79545330e-01, 9.37500000e-02, 0.00000000e+00, 5.71049476e-01, 8.28087884e+01],
                                        [1.40000000e+00, 7.32988949e-01, 1.32500000e-01, 0.00000000e+00, 5.79522242e-01, 7.97066414e+01],
                                        [2.00000000e+00, 7.01518955e-01, 1.76250000e-01, 0.00000000e+00, 2.68613346e-01, 1.50800767e+02],
                                        [8.00000000e+00, 5.13184259e-01, 4.53750000e-01, 0.00000000e+00, 1.05950477e+00, 4.20002154e+01],
                                        [1.40000000e+01, 3.61867323e-01, 6.11250000e-01, 0.00000000e+00, 1.72157260e+00, 2.50070263e+01],
                                        [2.00000000e+01, 3.31157335e-01, 6.30000000e-01, 0.00000000e+00, 1.95158053e+00, 2.18582771e+01]],
                                       result_2[:10].values, decimal=6)
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_log_x}...")

        
    def test_LogDynamics_3d_xu(self) -> None:
        logger.info(f"Starting test using {self.test_file_3d_log_xu}...")
        xu_snapshots = self.dump_3d_log_xu.snapshots
        x_snapshots = None
        condition=(xu_snapshots.snapshots[0].particle_type==1)
        ppp = np.array([0,0,0])

        #xu, x, slow, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "slow",
                                  neighborfile=None,
                                  max_neighbors=30)
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98794986e-01, 1.00000000e+00, 0.00000000e+00, 1.83249245e-04, 7.62729112e-03],
                                        [ 8.00000000e-02, 9.81788619e-01, 1.00000000e+00, 0.00000000e+00, 2.79353312e-03, 8.83924401e-03],
                                        [ 1.40000000e-01, 9.51758118e-01, 1.00000000e+00, 0.00000000e+00, 7.51534401e-03, 3.27266536e-03],
                                        [ 2.00000000e-01, 9.20215225e-01, 1.00000000e+00, 0.00000000e+00, 1.26307715e-02, -7.52556624e-04],
                                        [ 8.00000000e-01, 7.63523409e-01, 8.87000000e-01, 0.00000000e+00, 4.16240792e-02, 9.39923902e-02],
                                        [ 1.40000000e+00, 7.06977773e-01, 8.35000000e-01, 0.00000000e+00, 5.56373684e-02, 3.69059435e-01],
                                        [ 2.00000000e+00, 6.74200527e-01, 7.87000000e-01, 0.00000000e+00, 6.28006979e-02, 2.98372497e-01],
                                        [ 8.00000000e+00, 4.89460164e-01, 5.23000000e-01, 0.00000000e+00, 1.24775241e-01, 3.90773736e-01],
                                        [ 1.40000000e+01, 3.33250067e-01, 3.65000000e-01, 0.00000000e+00, 2.02048496e-01, 3.42854683e-01],
                                        [ 2.00000000e+01, 3.01752850e-01, 3.47000000e-01, 0.00000000e+00, 2.36770363e-01, 4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98797156e-01, 1.00000000e+00, 0.00000000e+00, 1.82916144e-04, 5.08578384e-04],
                                        [8.00000000e-02, 9.81866650e-01, 1.00000000e+00, 0.00000000e+00, 2.78100867e-03, 1.33905635e-02],
                                        [1.40000000e-01, 9.52395484e-01, 1.00000000e+00, 0.00000000e+00, 7.40924729e-03, 9.07554895e-03],
                                        [2.00000000e-01, 9.22445634e-01, 1.00000000e+00, 0.00000000e+00, 1.22484525e-02, 2.58585214e-03],
                                        [8.00000000e-01, 7.80250930e-01, 9.10000000e-01, 0.00000000e+00, 3.79490640e-02, 6.10574946e-02],
                                        [1.40000000e+00, 7.33305660e-01, 8.72500000e-01, 0.00000000e+00, 4.87008876e-02, 2.97100024e-01],
                                        [2.00000000e+00, 7.01825530e-01, 8.23750000e-01, 0.00000000e+00, 5.54054537e-02, 1.94253070e-01],
                                        [8.00000000e+00, 5.12756184e-01, 5.51250000e-01, 0.00000000e+00, 1.10604697e-01, 2.42693270e-01],
                                        [1.40000000e+01, 3.56859577e-01, 3.90000000e-01, 0.00000000e+00, 1.75137652e-01, 1.53847276e-01],
                                        [2.00000000e+01, 3.29340145e-01, 3.77500000e-01, 0.00000000e+00, 2.04353337e-01, 2.83847908e-01]],
                                       result_2[:10].values)

        
        #xu, x, fast, no neighbor
        log_dynamic = LogDynamics(xu_snapshots=xu_snapshots,
                                  x_snapshots=x_snapshots,
                                  dt=0.002, 
                                  ppp=ppp,
                                  diameters={1:1.0, 2:1.0},
                                  a=0.3,
                                  cal_type = "fast",
                                  neighborfile=None,
                                  max_neighbors=30)
        #no condition 
        log_dynamic.relaxation(qconst=2*np.pi, condition=None, outputfile="test_no_condition.csv")
        #condition
        log_dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="test_w_condition.csv")
        
        result_1 = pd.read_csv('test_no_condition.csv')
        result_2 = pd.read_csv('test_w_condition.csv')
        os.remove("test_no_condition.csv")
        os.remove("test_w_condition.csv")
        
        np.testing.assert_almost_equal([[ 2.00000000e-02, 9.98794986e-01, 0.00000000e+00, 0.00000000e+00, 1.83249245e-04, 7.62729112e-03],
                                        [ 8.00000000e-02, 9.81788619e-01, 0.00000000e+00, 0.00000000e+00, 2.79353312e-03, 8.83924401e-03],
                                        [ 1.40000000e-01, 9.51758118e-01, 0.00000000e+00, 0.00000000e+00, 7.51534401e-03, 3.27266536e-03],
                                        [ 2.00000000e-01, 9.20215225e-01, 0.00000000e+00, 0.00000000e+00, 1.26307715e-02, -7.52556624e-04],
                                        [ 8.00000000e-01, 7.63523409e-01, 1.13000000e-01, 0.00000000e+00, 4.16240792e-02, 9.39923902e-02],
                                        [ 1.40000000e+00, 7.06977773e-01, 1.65000000e-01, 0.00000000e+00, 5.56373684e-02, 3.69059435e-01],
                                        [ 2.00000000e+00, 6.74200527e-01, 2.13000000e-01, 0.00000000e+00, 6.28006979e-02, 2.98372497e-01],
                                        [ 8.00000000e+00, 4.89460164e-01, 4.77000000e-01, 0.00000000e+00, 1.24775241e-01, 3.90773736e-01],
                                        [ 1.40000000e+01, 3.33250067e-01, 6.35000000e-01, 0.00000000e+00, 2.02048496e-01, 3.42854683e-01],
                                        [ 2.00000000e+01, 3.01752850e-01, 6.53000000e-01, 0.00000000e+00, 2.36770363e-01, 4.19683315e-01]],
                                       result_1[:10].values)
        np.testing.assert_almost_equal([[2.00000000e-02, 9.98797156e-01, 0.00000000e+00, 0.00000000e+00, 1.82916144e-04, 5.08578384e-04],
                                        [8.00000000e-02, 9.81866650e-01, 0.00000000e+00, 0.00000000e+00, 2.78100867e-03, 1.33905635e-02],
                                        [1.40000000e-01, 9.52395484e-01, 0.00000000e+00, 0.00000000e+00, 7.40924729e-03, 9.07554895e-03],
                                        [2.00000000e-01, 9.22445634e-01, 0.00000000e+00, 0.00000000e+00, 1.22484525e-02, 2.58585214e-03],
                                        [8.00000000e-01, 7.80250930e-01, 9.00000000e-02, 0.00000000e+00, 3.79490640e-02, 6.10574946e-02],
                                        [1.40000000e+00, 7.33305660e-01, 1.27500000e-01, 0.00000000e+00, 4.87008876e-02, 2.97100024e-01],
                                        [2.00000000e+00, 7.01825530e-01, 1.76250000e-01, 0.00000000e+00, 5.54054537e-02, 1.94253070e-01],
                                        [8.00000000e+00, 5.12756184e-01, 4.48750000e-01, 0.00000000e+00, 1.10604697e-01, 2.42693270e-01],
                                        [1.40000000e+01, 3.56859577e-01, 6.10000000e-01, 0.00000000e+00, 1.75137652e-01, 1.53847276e-01],
                                        [2.00000000e+01, 3.29340145e-01, 6.22500000e-01, 0.00000000e+00, 2.04353337e-01, 2.83847908e-01]],
                                       result_2[:10].values)
        
        logger.info(f"Finishing test Dynamic.relaxation using {self.test_file_3d_log_xu}...")

    