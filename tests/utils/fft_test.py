# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.utils import fft
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


class TestFFT(unittest.TestCase):
    """
    Test class for FFT
    """

    def setUp(self) -> None:
        super().setUp()

    def test_triangle_area(self) -> None:
        """
        Test triangle_area works properly for 2D lammps
        """
        logger.info(f"Starting test FFT...")

        t = np.arange(1, 101, 1)
        result = fft.Filon_COS(C=np.sin(t), t=t)
        np.testing.assert_almost_equal(
            np.array(
                [
                    0.0,
                    0.06346652,
                    0.12693304,
                    0.19039955,
                    0.25386607,
                    0.31733259,
                    0.38079911,
                    0.44426563,
                    0.50773215,
                    0.57119866,
                    0.63466518,
                    0.6981317,
                    0.76159822,
                    0.82506474,
                    0.88853126,
                    0.95199777,
                    1.01546429,
                    1.07893081,
                    1.14239733,
                    1.20586385,
                    1.26933037,
                    1.33279688,
                    1.3962634,
                    1.45972992,
                    1.52319644,
                    1.58666296,
                    1.65012947,
                    1.71359599,
                    1.77706251,
                    1.84052903,
                    1.90399555,
                    1.96746207,
                    2.03092858,
                    2.0943951,
                    2.15786162,
                    2.22132814,
                    2.28479466,
                    2.34826118,
                    2.41172769,
                    2.47519421,
                    2.53866073,
                    2.60212725,
                    2.66559377,
                    2.72906028,
                    2.7925268,
                    2.85599332,
                    2.91945984,
                    2.98292636,
                    3.04639288,
                    3.10985939,
                    3.17332591,
                    3.23679243,
                    3.30025895,
                    3.36372547,
                    3.42719199,
                    3.4906585,
                    3.55412502,
                    3.61759154,
                    3.68105806,
                    3.74452458,
                    3.8079911,
                    3.87145761,
                    3.93492413,
                    3.99839065,
                    4.06185717,
                    4.12532369,
                    4.1887902,
                    4.25225672,
                    4.31572324,
                    4.37918976,
                    4.44265628,
                    4.5061228,
                    4.56958931,
                    4.63305583,
                    4.69652235,
                    4.75998887,
                    4.82345539,
                    4.88692191,
                    4.95038842,
                    5.01385494,
                    5.07732146,
                    5.14078798,
                    5.2042545,
                    5.26772102,
                    5.33118753,
                    5.39465405,
                    5.45812057,
                    5.52158709,
                    5.58505361,
                    5.64852012,
                    5.71198664,
                    5.77545316,
                    5.83891968,
                    5.9023862,
                    5.96585272,
                    6.02931923,
                    6.09278575,
                    6.15625227,
                    6.21971879,
                ]
            ),
            result["omega"].values,
        )
        np.testing.assert_almost_equal(
            np.array(
                [
                    3.20621830e-01,
                    3.20215334e-01,
                    3.18951136e-01,
                    3.16690136e-01,
                    3.13182694e-01,
                    3.08037192e-01,
                    3.00664484e-01,
                    2.90180492e-01,
                    2.75230289e-01,
                    2.53653174e-01,
                    2.21795998e-01,
                    1.72957602e-01,
                    9.33455524e-02,
                    -5.07721955e-02,
                    -3.69813606e-01,
                    -1.55780690e00,
                    7.08343845e00,
                    1.84317398e00,
                    1.28207113e00,
                    1.07217422e00,
                    9.64931223e-01,
                    9.01360788e-01,
                    8.60254642e-01,
                    8.32121050e-01,
                    8.12100850e-01,
                    7.97493696e-01,
                    7.86753538e-01,
                    7.79057447e-01,
                    7.74161357e-01,
                    7.72493975e-01,
                    7.75671948e-01,
                    7.88341223e-01,
                    8.26533205e-01,
                    9.93688958e-01,
                    -2.23221307e-01,
                    4.79720677e-01,
                    5.49195280e-01,
                    5.68274564e-01,
                    5.72068025e-01,
                    5.68977205e-01,
                    5.62050683e-01,
                    5.52679839e-01,
                    5.41607709e-01,
                    5.29283013e-01,
                    5.16006391e-01,
                    5.01998134e-01,
                    4.87432342e-01,
                    4.72455281e-01,
                    4.57195816e-01,
                    4.41771632e-01,
                    4.26293205e-01,
                    4.10866628e-01,
                    3.95595994e-01,
                    3.80585911e-01,
                    3.65944762e-01,
                    3.51789552e-01,
                    3.38253765e-01,
                    3.25500933e-01,
                    3.13749487e-01,
                    3.03321584e-01,
                    2.94747793e-01,
                    2.89019498e-01,
                    2.88306456e-01,
                    2.98582405e-01,
                    3.44789658e-01,
                    8.40949606e-01,
                    -2.58492390e-02,
                    9.03306078e-02,
                    1.15251277e-01,
                    1.22449243e-01,
                    1.23432896e-01,
                    1.21676544e-01,
                    1.18575162e-01,
                    1.14785761e-01,
                    1.10648833e-01,
                    1.06350877e-01,
                    1.01996914e-01,
                    9.76470910e-02,
                    9.33381385e-02,
                    8.91001818e-02,
                    8.49801613e-02,
                    8.11076382e-02,
                    7.80559619e-02,
                    8.60249497e-02,
                    6.11932970e-02,
                    5.95082292e-02,
                    5.57078673e-02,
                    5.14201932e-02,
                    4.68985319e-02,
                    4.22182837e-02,
                    3.74110516e-02,
                    3.24950700e-02,
                    2.74847680e-02,
                    2.23943901e-02,
                    1.72394988e-02,
                    1.20375897e-02,
                    6.80827790e-03,
                    1.57325240e-03,
                    -3.64391071e-03,
                ]
            ),
            result["FFT"].values,
        )
