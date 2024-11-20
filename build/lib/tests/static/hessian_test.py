# coding = utf-8

import os
import unittest

import numpy as np
import pandas as pd

from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.static.hessians import (HessianMatrix, InteractionParams,
                                         ModelName, PairInteractions)
from PyMatterSim.utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestShape(unittest.TestCase):
    """
    Test class for Structure factor
    """

    def setUp(self) -> None:
        super().setUp()
        self.test_file_IPL_2D = f"{READ_TEST_FILE_PATH}/IS.2DIPL.atom"

    def test_hessian_matrix(self) -> None:
        """
        Test hessian matrix
        """
        logger.info(f"Starting test using {self.test_file_IPL_2D}...")
        readdump = DumpReader(self.test_file_IPL_2D, ndim=2)
        readdump.read_onefile()

        masses = {1: 1.0, 2: 1.0}
        epsilons = np.ones((2, 2))

        sigmas = np.zeros((2, 2))
        sigmas[0, 0] = 1.00
        sigmas[0, 1] = 1.18
        sigmas[1, 0] = 1.18
        sigmas[1, 1] = 1.40

        r_cuts = np.zeros((2, 2))
        r_cuts[0, 0] = 1.48
        r_cuts[0, 1] = 1.7464
        r_cuts[1, 0] = 1.7464
        r_cuts[1, 1] = 2.072

        interaction_params = InteractionParams(
            model_name=ModelName.inverse_power_law,
            ipl_n=10,
            ipl_A=1.0
        )

        h = HessianMatrix(
            snapshot=readdump.snapshots.snapshots[0],
            masses=masses,
            epsilons=epsilons,
            sigmas=sigmas,
            r_cuts=r_cuts,
            ppp=np.array([1, 1]),
            shiftpotential=True
        )
        h.diagonalize_hessian(
            interaction_params=interaction_params,
            saveevecs=False,
            savehessian=False,
            outputfile="new"
        )
        calculated = pd.read_csv("new.omega_PR.csv")["PR"][:100:10]
        expected = np.array([
            1., 0.60509595, 0.4184461, 0.47013442, 0.39952026,
            0.48376086, 0.37543589, 0.4782805, 0.37811075, 0.28011895
        ])
        np.testing.assert_array_almost_equal(calculated, expected)

        os.remove("new.omega_PR.csv")
        logger.info(
            f"Finishing test hessian matrix class using {self.test_file_IPL_2D}...")

    def test_pair_interactions(self):
        """test pair interactions"""

        r = 1.05
        epsilon = 1.0
        sigma = 1.1
        r_c = 2.5
        pair_interaction = PairInteractions(r, epsilon, sigma, r_c, shift=True)

        # lennard-jones potential
        interaction_params = InteractionParams(
            model_name=ModelName.lennard_jones
        )
        calculated = pair_interaction.caller(
            interaction_params=interaction_params)
        expected = np.array(
            [-49.673659761738136, 0.06864965447468592, 787.6721753507363])
        np.testing.assert_array_almost_equal(
            calculated,
            expected
        )

        # inverse-power law potential
        interaction_params = InteractionParams(
            model_name=ModelName.inverse_power_law,
            ipl_n=10,
            ipl_A=1.0
        )
        calculated = pair_interaction.caller(
            interaction_params=interaction_params)
        expected = np.array(
            [-15.165074976445762, -0.0010878944375367285, 158.8722140389556])
        np.testing.assert_array_almost_equal(
            calculated,
            expected
        )

        # harmonic potential
        interaction_params = InteractionParams(
            model_name=ModelName.harmonic_hertz,
            harmonic_hertz_alpha=2.0
        )
        calculated = pair_interaction.caller(
            interaction_params=interaction_params)
        expected = np.array([-0.04132231404958684, 0, 0.8264462809917354])
        np.testing.assert_array_almost_equal(
            calculated,
            expected
        )
