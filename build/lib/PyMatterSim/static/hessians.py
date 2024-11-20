# coding = utf-8

"""see documentation @ ../../docs/hessians.md"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..reader.reader_utils import SingleSnapshot
from ..static.vector import participation_ratio
from ..utils.logging import get_logger_handle
from ..utils.pbc import remove_pbc

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=line-too-long


class ModelName(Enum):
    """Define model name class for each pair potential"""
    lennard_jones = 1
    inverse_power_law = 2
    harmonic_hertz = 3


@dataclass(frozen=True)
class InteractionParams:
    """Define model name and its associated parameters"""
    model_name: ModelName
    ipl_n: float = 0
    ipl_A: float = 0
    harmonic_hertz_alpha: float = 0


class PairInteractions:
    """
    Calculate pair (shifted) potential energy and force for different pair potentials
    """

    def __init__(self, r: float, epsilon: float, sigma: float,
                 r_c: float, shift: bool = True) -> None:
        """"
        initialize model input parameters for a pair i-j

        Inputs:
            1. r (float): pair distance
            2. epsilon (float): cohesive enerrgy between the pair
            3. sigma (float): diameter or reference length between the pair
            4. r_c (float): cutoff distance where the potential energy is cut to 0
            5. shift (bool): whether shift the potential energy at r_c to 0, default True

        Return:
            None
        """
        self.r = r
        self.epsilon = epsilon
        self.sigma = sigma
        self.r_c = r_c
        self.shift = shift

    def caller(self, interaction_params: InteractionParams) -> List[float]:
        """
        Calculate pair potential and force based on model inputs

        Input:
            interaction_params (InteractionParams): define the name and parameters
                for the pair potential

        Return:
            [s1, s1rc, s2] (list of float)
        """
        if interaction_params.model_name == ModelName.lennard_jones:
            return self.lennard_jones()

        if interaction_params.model_name == ModelName.inverse_power_law:
            return self.inverse_power_law(
                n=interaction_params.ipl_n,
                A=interaction_params.ipl_A
            )
        # interaction_params.model_name == ModelName.harmonic_hertz
        return self.harmonic_hertz(
            alpha=interaction_params.harmonic_hertz_alpha
        )

    def lennard_jones(self) -> List[float]:
        """
        Lennard-Jones interaction

        Return:
            [s1, s1rc, s2] (list of float)
        """
        ratio = (self.sigma / self.r)**6
        ratio_c = (self.sigma / self.r_c)**6
        s1 = -24 * self.epsilon / self.r * (2 * ratio**2 - ratio)
        if self.shift:
            s1rc = -24 * self.epsilon / self.r_c * (2 * ratio_c**2 - ratio_c)
        else:
            s1rc = 0
        s2 = 24 * self.epsilon / self.r**2 * (26 * ratio**2 - 7 * ratio)

        return [s1, s1rc, s2]

    def inverse_power_law(self, n: float, A: float = 1.0) -> List[float]:
        """
        Inverse power-law potential

        Inputs:
            1. n (float): power law exponent for a pair
            2. A (float): power law prefactor for a pair, default 1.0

        Return:
            [s1, s1rc, s2] (list of float)
        """
        ratio = (self.sigma / self.r)**n
        ratio_c = (self.sigma / self.r_c)**n
        s1 = -A * self.epsilon * n / self.r * ratio
        if self.shift:
            s1rc = -A * self.epsilon * n / self.r_c * ratio_c
        else:
            s1rc = 0
        s2 = A * self.epsilon * n * (n + 1) / self.r**2 * ratio

        return [s1, s1rc, s2]

    def harmonic_hertz(self, alpha: float) -> List[float]:
        """
        harmonic or hertz potential

        Inputs:
            1. alpha (float): spring exponent

        Return:
            [s1, s1rc, s2] (list of float)
        """
        ratio = 1 - self.r / self.sigma
        s1 = -self.epsilon / self.sigma * ratio**(alpha - 1)
        s1rc = 0
        s2 = self.epsilon / self.sigma**2 * (alpha - 1) * ratio**(alpha - 2)
        return [s1, s1rc, s2]


class HessianMatrix:
    """Calculate Hessian matrix and diagonalize it"""

    def __init__(self,
                 snapshot: SingleSnapshot,
                 masses: Dict[int, float],
                 epsilons: npt.NDArray,
                 sigmas: npt.NDArray,
                 r_cuts: npt.NDArray,
                 ppp: npt.NDArray,
                 shiftpotential: bool = True
                 ) -> None:
        """
        Initializing parameters for hessian calculation and diagonalization

        Inputs:
            1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
            2. masses (dict of int to float): mass for each particle type, example {1:1.0, 2:2.0}
            3. epsilons (npt.NDArray): cohesive energies for all pairs of particle type,
                                    shape as [num_of_particletype, num_of_particletype]
            4. sigmas (npt.NDArray): diameters for all pairs of particle type,
                                    shape as [num_of_particletype, num_of_particletype]
            5. r_cuts (npt.NDArray): cutoff distances of pair potentials for all pairs of particle type,
                                    shape as [num_of_particletype, num_of_particletype]
            6. ppp (npt.NDArray): the periodic boundary conditions,
                       setting 1 for yes and 0 for no, default np.array([1,1,1]),
                       set np.array([1,1]) for two-dimensional systems
            7. shiftpotential (bool): whether shift the potential energy at r_c to 0, default True

        Return:
            None
        """
        self.snapshot = snapshot
        self.masses = masses
        self.epsilons = epsilons
        self.sigmas = sigmas
        self.r_cuts = r_cuts
        self.ppp = ppp
        self.ndim = len(ppp)
        self.shiftpotential = shiftpotential

    def pair_matrix(self, Rji: npt.NDArray,
                    dudrs: List[float]) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Hessian matrix block for a pair

        Inputs:
            1. Rji (npt.NDArray): Ri - Rj, positional vector for i-j pair, shape as [ndim]
            2. dudrs (list of float): a list of [s'(r), s'(r_c), s''(r)],
                named as [s1, s1rc, s2], returned by the PairInteractions class

        Return:
            dudr2i (npt.NDArray): hessian matrix block of pair i-j centered on i
            dudr2j (npt.NDArray): hessian matrix block of pair i-j centered on j
            Both have the shape [ndim, ndim]
        """
        s1, s1rc, s2 = dudrs
        z = 0
        if self.ndim == 2:
            x, y = Rji
        else:
            x, y, z = Rji
        r = np.linalg.norm(Rji)

        # initialization
        dudr2i = np.zeros((self.ndim, self.ndim))

        # hessian matrix block for particle i
        xi_xi = s2 * (x / r)**2 + (s1 - s1rc) * (r**2 - x**2) / r**3
        dudr2i[0, 0] = xi_xi
        xi_yi = s2 * (x * y / r**2) + (s1 - s1rc) * (-x * y / r**3)
        dudr2i[0, 1] = xi_yi
        dudr2i[1, 0] = xi_yi
        yi_yi = s2 * (y / r)**2 + (s1 - s1rc) * (r**2 - y**2) / r**3
        dudr2i[1, 1] = yi_yi

        if self.ndim == 3:
            xi_zi = s2 * (x * z / r**2) + (s1 - s1rc) * (-x * z / r**3)
            dudr2i[0, 2] = xi_zi
            dudr2i[2, 0] = xi_zi
            yi_zi = s2 * (y * z / r**2) + (s1 - s1rc) * (-y * z / r**3)
            dudr2i[1, 2] = yi_zi
            dudr2i[2, 1] = yi_zi
            zi_zi = s2 * (z / r)**2 + (s1 - s1rc) * (r**2 - z**2) / r**3
            dudr2i[2, 2] = zi_zi

        dudr2j = -dudr2i
        return dudr2i, dudr2j

    def diagonalize_hessian(
        self,
        interaction_params: InteractionParams,
        saveevecs: bool = True,
        savehessian: bool = False,
        outputfile: str = ""
    ) -> None:
        """
        Calculate the hessian matrix and obatin the eigenvalues and eigenvectors
        by hessian diagonalization

        Inputs:
            1. interaction_params (InteractionParams): pair interaction parameters
            2. saveevecs (bool): save the eigenvectors or not, default True
            3. savehessian (bool): save the hessian matrix in dN*dN or not, default False
            4. outputfile (str): filename to save the computational results, defult None to use model name

        Return:
            None
        """
        logger.info(
            f"Calculate and diagonalize hessian matrix of d={self.ndim} system")
        nparticle = self.snapshot.nparticle
        positions = self.snapshot.positions

        # define hessian matrix prefactor based on particle type
        prefactor = np.zeros_like(self.epsilons)
        for i in range(self.epsilons.shape[0]):
            for j in range(self.epsilons.shape[1]):
                prefactor[i, j] = 1.0 / \
                    np.sqrt(self.masses[i + 1] * self.masses[j + 1])

        # calcualte hessian matrix
        hessian_matrix = np.zeros(
            (self.ndim * nparticle, self.ndim * nparticle))
        for i in range(nparticle):
            if i % 100 == 0:
                logger.info(f"calculating for particle {i} / {nparticle}")
            itype = int(self.snapshot.particle_type[i] - 1)
            RJI = positions[i] - positions
            RJI = remove_pbc(RJI, self.snapshot.hmatrix, self.ppp)
            distance = np.linalg.norm(RJI, axis=1)
            for j in range(nparticle):
                jtype = int(self.snapshot.particle_type[j] - 1)
                if (j != i) & (distance[j] <= self.r_cuts[itype, jtype]):
                    pair_interaction = PairInteractions(
                        r=distance[j],
                        epsilon=self.epsilons[itype, jtype],
                        sigma=self.sigmas[itype, jtype],
                        r_c=self.r_cuts[itype, jtype],
                        shift=self.shiftpotential
                    )
                    dudrs = pair_interaction.caller(interaction_params)
                    dudr2i, dudr2j = self.pair_matrix(RJI[j], dudrs)
                    index_i_0 = i * self.ndim
                    index_i_1 = index_i_0 + self.ndim
                    index_j_0 = j * self.ndim
                    index_j_1 = index_j_0 + self.ndim
                    # for i-i pair
                    hessian_matrix[index_i_0:index_i_1,
                                   index_i_0:index_i_1] += dudr2i * prefactor[itype,
                                                                              jtype]
                    # for i-j pair
                    hessian_matrix[index_i_0:index_i_1,
                                   index_j_0:index_j_1] = dudr2j * prefactor[itype,
                                                                             jtype]

        if not outputfile:
            outputfile = interaction_params.model_name.name
        if savehessian:
            np.save(outputfile + ".hessianmatrix.npy", hessian_matrix)
        # diagonalize the hessian matrix
        evals, evecs = np.linalg.eigh(hessian_matrix)
        del hessian_matrix
        if saveevecs:
            np.save(outputfile + ".evecs.npy", evecs)
        logger.info("Finishing hessian matrix diagonalization")

        # calculate the participation ratio
        PR = np.zeros_like(evals)
        for i in range(evecs.shape[1]):
            PR[i] = participation_ratio(
                evecs[:, i].reshape(nparticle, self.ndim))
        frequencies = np.where(evals > 0, np.sqrt(evals), evals)
        pd.DataFrame({"omega": frequencies, "PR": PR}).to_csv(
            outputfile + ".omega_PR.csv", index=False)
