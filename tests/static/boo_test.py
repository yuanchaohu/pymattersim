# coding = utf-8

import os
import unittest

import numpy as np

from PyMatterSim.neighbors.freud_neighbors import cal_neighbors
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.static.boo import boo_2d, boo_3d
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestBOO(unittest.TestCase):
    """
    Test class for BOO
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2D = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.atom"
        self.test_file_3D = f"{READ_TEST_FILE_PATH}/test_additional_columns.dump"

    def test_boo_2d(self) -> None:
        """
        Test boo_2d works properly for 2D system
        """
        readdump = DumpReader(self.test_file_2D, ndim=2)
        readdump.read_onefile()

        cal_neighbors(readdump.snapshots, outputfile='test')

        logger.info(
            f"Starting test boo_2d without weight using {self.test_file_2D}...")
        boo2d = boo_2d(
            readdump.snapshots,
            l=6,
            neighborfile='test.neighbor.dat')
        boo2d.lthorder()
        average_quantity = boo2d.time_average(
            time_period=12000 * 0.002,
            dt=0.002,
            average_complex=False
        )[0]
        modulus = np.abs(average_quantity)
        spatial_corr_results = boo2d.spatial_corr()
        time_corr_results = boo2d.time_corr()

        np.testing.assert_almost_equal([0.7600972,
                                        0.45321275,
                                        0.54769806,
                                        0.61331616,
                                        0.29276934,
                                        0.34152073,
                                        0.45443368,
                                        0.52252638],
                                       modulus[:,
                                               0])

        np.testing.assert_almost_equal([np.nan, 0.01449476, -0.01748584, 0.00497529, 0.0020236,
                                        0.00815036, 0.00842069, 0.00435055],
                                       (spatial_corr_results["gA"] / spatial_corr_results["gr"]).values[::200])

        np.testing.assert_almost_equal([1.,
                                        0.54684874,
                                        0.46090936,
                                        0.40649116,
                                        0.34239682,
                                        0.29105545,
                                        0.2643266,
                                        0.211893,
                                        0.16744193,
                                        0.14603285],
                                       time_corr_results["time_corr"].values)
        logger.info(
            f"Finishing test boo_2d without weight using {self.test_file_2D}...")

        logger.info(
            f"Starting test boo_2d with weight using {self.test_file_2D}...")
        boo2d = boo_2d(
            readdump.snapshots,
            l=6,
            neighborfile='test.neighbor.dat',
            weightsfile='test.edgelength.dat'
        )
        boo2d.lthorder()
        average_quantity = boo2d.time_average(
            time_period=12000 * 0.002,
            dt=0.002,
            average_complex=False
        )[0]
        modulus = np.abs(average_quantity)
        spatial_corr_results = boo2d.spatial_corr()
        time_corr_results = boo2d.time_corr()

        np.testing.assert_almost_equal([0.71827345,
                                        0.43844028,
                                        0.55639298,
                                        0.59800655,
                                        0.29661671,
                                        0.33365584,
                                        0.43105367,
                                        0.43547185],
                                       modulus[:,
                                               0])

        np.testing.assert_almost_equal([np.nan, 0.013034, -0.01836656, 0.00384457, 0.00233856,
                                        0.00780522, 0.00840458, 0.00359436],
                                       (spatial_corr_results["gA"] / spatial_corr_results["gr"]).values[::200])

        np.testing.assert_almost_equal([1.,
                                        0.54136574,
                                        0.45428546,
                                        0.39985094,
                                        0.33754559,
                                        0.28445635,
                                        0.25830844,
                                        0.21076049,
                                        0.16584196,
                                        0.14278943],
                                       time_corr_results["time_corr"].values)
        logger.info(
            f"Finishing test boo_2d with weight using {self.test_file_2D}...")

    def test_boo_3d(self) -> None:
        """
        Test boo_3d works properly for 3D system
        """
        readdump = DumpReader(self.test_file_3D, ndim=3)
        readdump.read_onefile()

        cal_neighbors(readdump.snapshots, outputfile='test')

        logger.info(
            f"Starting test boo_3d without weight and cg using {self.test_file_3D}...")
        boo3d = boo_3d(
            readdump.snapshots,
            l=6,
            neighborfile='test.neighbor.dat')
        ql = boo3d.ql_Ql(coarse_graining=False)
        sij_ql = boo3d.sij_ql_Ql(coarse_graining=False)
        w_cap = boo3d.w_W_cap(coarse_graining=False)
        spatial_corr = boo3d.spatial_corr(coarse_graining=False)
        time_corr = boo3d.time_corr(coarse_graining=False)

        np.testing.assert_almost_equal([
            0.43227061, 0.18069028, 0.23708262, 0.2618971, 0.23624358,
            0.27478991, 0.44182122, 0.27323332, 0.27575202, 0.32819106,
            0.28877676, 0.32336161, 0.26660538, 0.31879222, 0.55303859,
            0.30757982, 0.18104789, 0.27169199, 0.28589434, 0.23295691,
            0.30206928, 0.29461928, 0.33530106, 0.17097392, 0.35306389,
            0.2194112, 0.32827385, 0.34786644, 0.38473285, 0.20309584,
            0.33917366, 0.29470184, 0.27552243, 0.43496167, 0.23331706,
            0.3508551, 0.22890008, 0.35777388, 0.29653879, 0.35256505,
            0.2230523],
            ql[0, ::200])

        np.testing.assert_almost_equal([
            0.00169547, -0.21999767, 0.65833819, 0.3530955, -0.33517334,
            0.68435889, 0.30768949, -0.13527076, 0.33147907, 0.01519173,
            -0.39238554, 0.33193493, 0.20613125, 0.32522613, 0.21002388,
            0.33825666, -0.05798512, 0.28935847, 0.21305366, -0.0245117,
            0.31239355, 0.3031809, -0.25585464, 0.25254712, -0.05239578,
            0.10212554, 0.31779668, 0.65089369, -0.29892883, -0.31633988,
            0.30167064, 0.14403546, -0.0162279, 0.2801964, -0.22004063,
            0.49720922, -0.00563788, -0.5096429, 0.2953051, 0.24530679,
            0.0892698],
            sij_ql[0][:, 2][::200])

        np.testing.assert_almost_equal([
            -6.80722626e-03, -2.88831867e-05, 1.47122861e-04, -4.64774798e-04,
            -5.19573143e-04, -4.43810406e-04, -6.54182642e-03, -5.26617532e-04,
            -1.19830030e-03, -9.82009095e-04, -1.94379484e-03, 4.17253249e-04,
            -2.09158239e-04, -1.74024018e-03, -2.50606781e-02, 1.64887930e-03,
            1.88476528e-04, -2.14548053e-04, 1.66543218e-04, 2.73414193e-04,
            -1.37181527e-03, 1.03632810e-03, 2.14617756e-04, -1.07260085e-04,
            -2.21706336e-03, -1.21875254e-04, -1.42625526e-03, -1.12756375e-03,
            -2.88910250e-03, -3.84317700e-05, -1.77894646e-04, -8.29594269e-05,
            -1.90982530e-03, -9.12279492e-03, -1.88412163e-03, -7.71710115e-04,
            -4.59257729e-04, -3.65262444e-03, -1.83958289e-03, -5.09011044e-03,
            7.07639787e-04],
            w_cap[0][0][::200])

        np.testing.assert_almost_equal([np.nan, np.nan, np.nan, 1.81951920e-02, -
                                        4.22891141e-03, 3.55648748e-03, 8.79224348e-05, -
                                        9.16034211e-04, 4.94529607e-05, 1.05041291e-04, 8.36798828e-04, -
                                        6.02471259e-04, -
                                        4.52214231e-04, 7.32897451e-05, -
                                        1.92514087e-04, -
                                        7.60890249e-05, 1.29449725e-04, -
                                        3.22753824e-05, -
                                        9.15137072e-05, -
                                        1.81621041e-04, -
                                        7.27825736e-05, -
                                        4.88106118e-05, 7.29147836e-06, -
                                        1.55694710e-04, -
                                        2.80140681e-05, -
                                        1.54288874e-05, -
                                        7.23105473e-06], (spatial_corr["gA"] /
                                                          spatial_corr["gr"]).values[::100])

        np.testing.assert_almost_equal(
            [1., 0.16006052], time_corr["time_corr"].values)
        logger.info(
            f"Finishing test boo_3d without weight and cg using {self.test_file_3D}...")

        logger.info(
            f"Starting test boo_3d with weight and cg using {self.test_file_3D}...")
        boo3d = boo_3d(
            readdump.snapshots,
            l=6,
            neighborfile='test.neighbor.dat',
            weightsfile='test.facearea.dat'
        )
        ql = boo3d.ql_Ql(coarse_graining=True)
        sij_ql = boo3d.sij_ql_Ql(coarse_graining=True)
        w_cap = boo3d.w_W_cap(coarse_graining=True)
        spatial_corr = boo3d.spatial_corr(coarse_graining=True)
        time_corr = boo3d.time_corr(coarse_graining=True)

        np.testing.assert_almost_equal([
            0.1332692, 0.11445406, 0.14497341, 0.11440318, 0.14618663,
            0.12408545, 0.16568628, 0.12062814, 0.18530042, 0.10790806,
            0.17858977, 0.14394689, 0.13878829, 0.10487729, 0.18884249,
            0.13868595, 0.15597141, 0.11172827, 0.10321681, 0.11282331,
            0.13251701, 0.13324954, 0.12079061, 0.13321651, 0.13515637,
            0.11006736, 0.1470679, 0.19564502, 0.1595031, 0.14489631,
            0.13925261, 0.1534666, 0.10513303, 0.14088865, 0.09198143,
            0.1914702, 0.140152, 0.12714194, 0.18523225, 0.12388073,
            0.14949572],
            ql[0, ::200])

        np.testing.assert_almost_equal([
            0.93825674, 0.67991626, 0.79569107, 0.80296838, 0.67566144,
            0.37462878, 0.84344888, 0.2258824, 0.67704016, 0.65985495,
            0.69218916, 0.53932601, 0.58289444, 0.50239587, 0.61709851,
            0.42270994, 0.82060504, 0.46258748, 0.84134471, 0.60550421,
            0.57930553, 0.53517359, 0.36808708, 0.69409168, 0.59680152,
            0.37274721, 0.82393289, 0.69204068, 0.77417517, 0.59920079,
            0.66030884, 0.63247836, 0.60462254, 0.69966084, 0.53116286,
            0.76456851, 0.56097746, 0.39588556, 0.68361235, 0.79557347,
            0.76986873],
            sij_ql[0][:, 2][::200])

        np.testing.assert_almost_equal([
            3.41472390e-05, 8.25167565e-06, 7.22615304e-05, -9.77617691e-05,
            -4.03883043e-05, 5.59739702e-05, -8.14475186e-06, 5.55019418e-05,
            -2.10768125e-04, 8.62946935e-05, -2.93891601e-04, 4.78454588e-05,
            1.28529160e-04, 6.90560459e-05, -1.31313381e-04, -1.53222299e-04,
            1.44966185e-04, -2.51361126e-05, -6.44947280e-08, -1.32365516e-06,
            2.65037885e-05, 1.47364486e-04, -3.46702364e-05, 3.48748210e-05,
            -9.65133883e-05, 9.40869532e-05, 1.63699960e-04, -2.62453946e-04,
            -2.76036960e-05, 1.85317079e-04, 1.15345542e-04, 7.37236100e-05,
            -4.23193661e-05, -1.71422826e-04, -2.19133086e-05, -1.72636273e-04,
            -1.39981367e-04, 3.13124838e-05, 4.40035294e-04, -1.80554067e-04,
            -5.27193766e-05],
            w_cap[0][0][::200])

        np.testing.assert_almost_equal([np.nan, np.nan, np.nan, 1.22922839e-02,
                                        9.25634640e-03, 5.82102564e-03, 3.44985010e-03, 1.42615999e-03,
                                        5.90987511e-04, 7.29334669e-05, 2.09112321e-04, -6.73407264e-05,
                                        7.50911164e-05, -1.21669473e-04, -1.06372728e-04, -7.11123595e-05,
                                        -6.04376376e-05, -8.47082117e-05, 1.14173930e-05, -8.14977159e-05,
                                        -6.61259243e-05, -5.04622004e-05, -3.26793727e-05, -1.66552724e-05,
                                        -3.70631427e-05, -4.66217634e-05, -1.57823384e-06],
                                       (spatial_corr["gA"] / spatial_corr["gr"]).values[::100])

        np.testing.assert_almost_equal(
            [1., 0.1612271], time_corr["time_corr"].values)
        logger.info(
            f"Finishing test boo_3d with weight and cg using {self.test_file_3D}...")

        os.remove('test.edgelength.dat')
        os.remove('test.facearea.dat')
        os.remove('test.neighbor.dat')
        os.remove('test.overall.dat')
