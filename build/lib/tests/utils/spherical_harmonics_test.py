# coding = utf-8

import unittest

import numpy as np

from PyMatterSim.utils import spherical_harmonics
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


class TestSphericalHarmonics(unittest.TestCase):
    """
    Test class for spherical_harmonics
    """

    def setUp(self) -> None:
        super().setUp()

    def test_SphHarm0(self) -> None:
        """
        Test SphHarm0 works properly
        """
        logger.info(f"Starting test SphHarm0...")
        result = spherical_harmonics.SphHarm0()
        np.testing.assert_almost_equal(0.28209479177387814, result)

    def test_SphHarm1(self) -> None:
        """
        Test SphHarm1 works properly
        """
        logger.info(f"Starting test SphHarm1...")
        result = spherical_harmonics.SphHarm1(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array([0.25912061 - 0.14960336j, 0.24430126 + 0.0j, -0.25912061 - 0.14960336j]),
            result,
        )

    def test_SphHarm2(self) -> None:
        """
        Test SphHarm2 works properly
        """
        logger.info(f"Starting test SphHarm2...")
        result = spherical_harmonics.SphHarm2(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    0.14485283 - 0.25089245j,
                    0.28970565 - 0.16726164j,
                    -0.07884789 + 0.0j,
                    -0.28970565 - 0.16726164j,
                    0.14485283 + 0.25089245j,
                ]
            ),
            result,
        )

    def test_SphHarm3(self) -> None:
        """
        Test SphHarm3 works properly
        """
        logger.info(f"Starting test SphHarm3...")
        result = spherical_harmonics.SphHarm3(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    1.65936471e-17 - 0.27099482j,
                    1.91622277e-01 - 0.33189952j,
                    6.05962845e-02 - 0.03498528j,
                    -3.26529291e-01 + 0.0j,
                    -6.05962845e-02 - 0.03498528j,
                    1.91622277e-01 + 0.33189952j,
                    -1.65936471e-17 - 0.27099482j,
                ]
            ),
            result,
        )

    def test_SphHarm4(self) -> None:
        """
        Test SphHarm4 works properly
        """
        logger.info(f"Starting test SphHarm4...")
        result = spherical_harmonics.SphHarm4(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    -1.24462320e-01 - 0.21557506j,
                    2.48904707e-17 - 0.40649223j,
                    9.40846702e-02 - 0.16295943j,
                    -2.21759694e-01 + 0.12803302j,
                    -2.44629077e-01 + 0.0j,
                    2.21759694e-01 + 0.12803302j,
                    9.40846702e-02 + 0.16295943j,
                    -2.48904707e-17 - 0.40649223j,
                    -1.24462320e-01 + 0.21557506j,
                ]
            ),
            result,
        )

    def test_SphHarm5(self) -> None:
        """
        Test SphHarm5 works properly
        """
        logger.info(f"Starting test SphHarm5...")
        result = spherical_harmonics.SphHarm5(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    -1.95805773e-01 - 0.11304852j,
                    -2.06397408e-01 - 0.3574908j,
                    1.71984067e-17 - 0.2808713j,
                    -7.94423992e-02 + 0.13759827j,
                    -2.85250843e-01 + 0.16468965j,
                    8.40580443e-02 + 0.0j,
                    2.85250843e-01 + 0.16468965j,
                    -7.94423992e-02 - 0.13759827j,
                    -1.71984067e-17 - 0.2808713j,
                    -2.06397408e-01 + 0.3574908j,
                    1.95805773e-01 - 0.11304852j,
                ]
            ),
            result,
        )

    def test_SphHarm6(self) -> None:
        """
        Test SphHarm6 works properly
        """
        logger.info(f"Starting test SphHarm6...")
        result = spherical_harmonics.SphHarm6(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    -2.03801110e-01 - 2.49584378e-17j,
                    -3.52993878e-01 - 2.03801110e-01j,
                    -1.75603278e-01 - 3.04153799e-01j,
                    -3.23835291e-18 + 5.28863164e-02j,
                    -1.75570092e-01 + 3.04096319e-01j,
                    -6.75897331e-02 + 3.90229506e-02j,
                    3.28771968e-01 + 0.00000000e00j,
                    6.75897331e-02 + 3.90229506e-02j,
                    -1.75570092e-01 - 3.04096319e-01j,
                    3.23835291e-18 + 5.28863164e-02j,
                    -1.75603278e-01 + 3.04153799e-01j,
                    3.52993878e-01 - 2.03801110e-01j,
                    -2.03801110e-01 + 2.49584378e-17j,
                ]
            ),
            result,
        )

    def test_SphHarm7(self) -> None:
        """
        Test SphHarm7 works properly
        """
        logger.info(f"Starting test SphHarm7...")
        result = spherical_harmonics.SphHarm7(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    -1.58215643e-01 + 9.13458442e-02j,
                    -3.94659153e-01 - 4.83318069e-17j,
                    -3.48295625e-01 - 2.01088573e-01j,
                    -2.57996759e-02 - 4.46863496e-02j,
                    -2.00752516e-17 + 3.27853739e-01j,
                    -1.04509678e-01 + 1.81016073e-01j,
                    2.16323586e-01 - 1.24894481e-01j,
                    2.43796207e-01 + 0.00000000e00j,
                    -2.16323586e-01 - 1.24894481e-01j,
                    -1.04509678e-01 - 1.81016073e-01j,
                    2.00752516e-17 + 3.27853739e-01j,
                    -2.57996759e-02 + 4.46863496e-02j,
                    3.48295625e-01 - 2.01088573e-01j,
                    -3.94659153e-01 + 4.83318069e-17j,
                    1.58215643e-01 + 9.13458442e-02j,
                ]
            ),
            result,
        )

    def test_SphHarm8(self) -> None:
        """
        Test SphHarm8 works properly
        """
        logger.info(f"Starting test SphHarm8...")
        result = spherical_harmonics.SphHarm8(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    -8.15424760e-02 + 1.41235711e-01j,
                    -3.26169904e-01 + 1.88314282e-01j,
                    -4.36701501e-01 - 5.34805096e-17j,
                    -1.28643145e-01 - 7.42721544e-02j,
                    1.36770224e-01 + 2.36892977e-01j,
                    -1.84551355e-17 + 3.01395236e-01j,
                    6.61476119e-02 - 1.14571025e-01j,
                    2.85073246e-01 - 1.64587115e-01j,
                    -8.56499109e-02 + 0.00000000e00j,
                    -2.85073246e-01 - 1.64587115e-01j,
                    6.61476119e-02 + 1.14571025e-01j,
                    1.84551355e-17 + 3.01395236e-01j,
                    1.36770224e-01 - 2.36892977e-01j,
                    1.28643145e-01 - 7.42721544e-02j,
                    -4.36701501e-01 + 5.34805096e-17j,
                    3.26169904e-01 + 1.88314282e-01j,
                    -8.15424760e-02 - 1.41235711e-01j,
                ]
            ),
            result,
        )

    def test_SphHarm9(self) -> None:
        """
        Test SphHarm9 works properly
        """
        logger.info(f"Starting test SphHarm9...")
        result = spherical_harmonics.SphHarm9(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    -2.66555213e-17 + 1.45105900e-01j,
                    -1.77717706e-01 + 3.07816097e-01j,
                    -3.96218358e-01 + 2.28756775e-01j,
                    -2.34622334e-01 - 2.87329490e-17j,
                    1.72650794e-01 + 9.96799826e-02j,
                    1.77394759e-01 + 3.07256736e-01j,
                    2.02883005e-19 - 3.31333091e-03j,
                    1.71568028e-01 - 2.97164541e-01j,
                    7.03533968e-02 - 4.06185526e-02j,
                    -3.29414147e-01 + 0.00000000e00j,
                    -7.03533968e-02 - 4.06185526e-02j,
                    1.71568028e-01 + 2.97164541e-01j,
                    -2.02883005e-19 - 3.31333091e-03j,
                    1.77394759e-01 - 3.07256736e-01j,
                    -1.72650794e-01 + 9.96799826e-02j,
                    -2.34622334e-01 + 2.87329490e-17j,
                    3.96218358e-01 + 2.28756775e-01j,
                    -1.77717706e-01 - 3.07816097e-01j,
                    2.66555213e-17 + 1.45105900e-01j,
                ]
            ),
            result,
        )

    def test_SphHarm10(self) -> None:
        """
        Test SphHarm10 works properly
        """
        logger.info(f"Starting test SphHarm10...")
        result = spherical_harmonics.SphHarm10(theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    6.43843559e-02 + 1.11516976e-01j,
                    -6.10754721e-17 + 3.32479384e-01j,
                    -2.33546606e-01 + 4.04514587e-01j,
                    -2.66966007e-01 + 1.54132896e-01j,
                    1.14081161e-01 + 1.39709129e-17j,
                    3.22658467e-01 + 1.86286952e-01j,
                    5.95922537e-02 + 1.03216811e-01j,
                    1.84575858e-17 - 3.01435252e-01j,
                    1.09043943e-01 - 1.88869649e-01j,
                    -2.14199672e-01 + 1.23668238e-01j,
                    -2.43327024e-01 + 0.00000000e00j,
                    2.14199672e-01 + 1.23668238e-01j,
                    1.09043943e-01 + 1.88869649e-01j,
                    -1.84575858e-17 - 3.01435252e-01j,
                    5.95922537e-02 - 1.03216811e-01j,
                    -3.22658467e-01 + 1.86286952e-01j,
                    1.14081161e-01 - 1.39709129e-17j,
                    2.66966007e-01 + 1.54132896e-01j,
                    -2.33546606e-01 - 4.04514587e-01j,
                    6.10754721e-17 + 3.32479384e-01j,
                    6.43843559e-02 - 1.11516976e-01j,
                ]
            ),
            result,
        )

    def test_SphHarm12(self) -> None:
        """
        Test SphHarm12 works properly
        """
        logger.info(f"Starting test SphHarm12...")
        result = spherical_harmonics.SphHarm_above(l=12, theta=60 * np.pi / 180, phi=30 * np.pi / 180)
        np.testing.assert_almost_equal(
            np.array(
                [
                    1.00783300e-01 + 2.46847892e-17j,
                    2.46867660e-01 + 1.42529110e-01j,
                    2.30524788e-01 + 3.99280645e-01j,
                    -7.66617815e-17 + 4.17327301e-01j,
                    -3.22639555e-02 + 5.58828102e-02j,
                    2.79620948e-01 - 1.61439229e-01j,
                    3.04865670e-01 + 3.73352768e-17j,
                    -1.03015019e-01 - 5.94757487e-02j,
                    -1.76166582e-01 - 3.05129470e-01j,
                    1.22544413e-18 - 2.00130214e-02j,
                    -1.69777025e-01 + 2.94062434e-01j,
                    -7.18293062e-02 + 4.14706693e-02j,
                    3.29702471e-01 + 0.00000000e00j,
                    7.18293062e-02 + 4.14706693e-02j,
                    -1.69777025e-01 - 2.94062434e-01j,
                    -1.22544413e-18 - 2.00130214e-02j,
                    -1.76166582e-01 + 3.05129470e-01j,
                    1.03015019e-01 - 5.94757487e-02j,
                    3.04865670e-01 - 3.73352768e-17j,
                    -2.79620948e-01 - 1.61439229e-01j,
                    -3.22639555e-02 - 5.58828102e-02j,
                    7.66617815e-17 + 4.17327301e-01j,
                    2.30524788e-01 - 3.99280645e-01j,
                    -2.46867660e-01 + 1.42529110e-01j,
                    1.00783300e-01 - 2.46847892e-17j,
                ]
            ),
            result,
        )
