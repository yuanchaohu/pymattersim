# coding = utf-8

"""
This module calculates bond orientational order parameters
in two-dimensions and three-dimensions

see documentation @ ../docs/boo_3d.md &
see documentation @ ../docs/boo_2d.md
"""

import re
from typing import Tuple
import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from neighbors.read_neighbors import read_neighbors
from static.gr import conditional_gr
from utils.spherical_harmonics import sph_harm_l
from utils.pbc import remove_pbc
from utils.logging import get_logger_handle
from utils.funcs import Wignerindex
from dynamic.time_corr import time_correlation

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=line-too-long


class boo_3d:
    """
    This module calculates bond orientational orders in three dimensions
    Including original quantatities and weighted ones
    The bond orientational order parameters include
        q_l (local), 
        Q_l (coarse-grained),
        w_l (local),
        W_l (coarse-grained),
        sij (original boo, like q_6),
        Sij (coarse-grained boo, like Q_12)
    Also calculate both time correlation and spatial correlation
    This module accounts for both orthogonal and triclinic cells
    """

    def __init__(
            self,
            snapshots: Snapshots,
            l: int,
            neighborfile: str,
            weightsfile: str=None,
            ppp: np.ndarray=np.array([1,1,1]),
            Nmax: int=30
    ) -> None:
        """
        Initializing class for BOO3D

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                         (returned by reader.dump_reader.DumpReader)
            2. l (int): degree of spherical harmonics
            3. neighborfile (str): file name of particle neighbors (see module neighbors)
            4. weightsfile (str): file name of particle-neighbor weights (see module neighbors)
                                  one typical example is Voronoi face area of the polyhedron;
                                  this file should be consistent with neighborfile, default None
            5. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.array([1,1,1]),
            7. Nmax (int): maximum number for neighbors

        Return:
            None
        """
        self.snapshots = snapshots
        self.l = l
        self.neighborfile = neighborfile
        self.weightsfile = weightsfile
        self.ppp = ppp
        self.Nmax = Nmax

        assert len(set(np.diff(
            [snapshot.timestep for snapshot in self.snapshots.snapshots]
        ))) == 1, "Warning: Dump interval changes during simulation"
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len(
            {snapshot.nparticle for snapshot in self.snapshots.snapshots}
        ) == 1, "Paticle number changes during simulation"
        self.boxlength = snapshots.snapshots[0].boxlength
        assert len(
            {tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}
        ) == 1, "Simulation box length changes during simulation"

        # for easy reuse and extendability
        self.smallqlm, self.largeQlm = self.qlm_Qlm()

    def qlm_Qlm(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        BOO of the l-fold symmetry as a 2l + 1 vector

        Inputs:
            None

        Return:
            BOO of order-l in vector complex number
            shape: [nsnapshot, nparticle, 2l+1]
        """

        logger.info(f"Calculate spherical harmonics for l={self.l}")
        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        if self.weightsfile:
            fweights = open(self.weightsfile, 'r', encoding="utf-8")

        smallqlm = []
        largeQlm = []
        for snapshot in self.snapshots.snapshots:
            Neighborlist = read_neighbors(fneighbor, snapshot.nparticle, self.Nmax)
            Particlesmallqlm = np.zeros((snapshot.nparticle, 2*self.l+1), dtype=np.complex128)
            if not self.weightsfile:
                for i in range(snapshot.nparticle):
                    cnlist = Neighborlist[i, 1:(Neighborlist[i, 0]+1)]
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis, :]
                    RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                    distance = np.linalg.norm(RIJ, axis=1)
                    theta = np.arccos(RIJ[:, 2]/distance)
                    phi = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    for j in range(Neighborlist[i, 0]):
                        Particlesmallqlm[i] += sph_harm_l(self.l, theta[j], phi[j])
                Particlesmallqlm /= (Neighborlist[:, 0])[:, np.newaxis]
                smallqlm.append(Particlesmallqlm)
            else:
                weightslist = read_neighbors(fweights, snapshot.nparticle, self.Nmax)[:, 1:]
                # normalization of center-neighbors weights
                weightsfrac = weightslist / weightslist.sum(axis=1)[:, np.newaxis]
                for i in range(snapshot.nparticle):
                    cnlist = Neighborlist[i, 1:(Neighborlist[i, 0]+1)]
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis, :]
                    RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                    distance = np.linalg.norm(RIJ, axis=1)
                    theta = np.arccos(RIJ[:, 2]/distance)
                    phi = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    for j in range(Neighborlist[i, 0]):
                        Particlesmallqlm[i] += sph_harm_l(self.l, theta[j], phi[j]) * weightsfrac[i, j]
                smallqlm.append(Particlesmallqlm)

            # coarse-graining over the neighbors
            ParticlelargeQlm = np.copy(Particlesmallqlm)
            for i in range(snapshot.nparticle):
                for j in range(Neighborlist[i, 0]):
                    ParticlelargeQlm[i] += Particlesmallqlm[Neighborlist[i, j+1]]
            ParticlelargeQlm = ParticlelargeQlm / (1+Neighborlist[:, 0])[:, np.newaxis]
            largeQlm.append(ParticlelargeQlm)

        fneighbor.close()
        if self.weightsfile:
            fweights.close()

        logger.info(f"Spherical harmonics for l={self.l} is ready for further calculations")

        return np.array(smallqlm), np.array(largeQlm)

    def ql_Ql(self, coarse_graining: bool=False, outputfile: str=None) -> np.ndarray:
        """
        Calculate BOO ql (local) or Ql (coarse-grained)

        Inputs:
            1. coarse_graining (bool): whether use coarse-grained Qlm or qlm or not
                                       default False
            2. outputfile (str): txt file name for ql or Ql results, default None
        
        Return:
            calculated ql or Ql (np.ndarray) 
            shape [nsnapshot, nparticle]
        """
        if coarse_graining:
            cal_qlmQlm = self.largeQlm
            logger.info(f'Start calculating coarse-grained rotational invariants for l={self.l}')
        else:
            cal_qlmQlm = self.smallqlm
            logger.info(f'Start calculating local rotational invariants for l={self.l}')

        ql_Ql = np.sqrt(4*np.pi/(2*self.l+1)*np.square(np.abs(cal_qlmQlm)).sum(axis=2))
        if outputfile:
            np.savetxt(outputfile, ql_Ql, fmt="%.6f", header="", comments="")

        logger.info(f'Finish calculating rotational invariants ql or Ql for l={self.l}')
        return ql_Ql

    def sij_ql_Ql(
        self,
        coarse_graining: bool=False,
        c: float=0.7,
        outputqlQl: str=None,
        outputsij: str=None
    ) -> list[np.ndarray]:
        """
        Calculate orientation correlation of qlm or Qlm, named as sij

        Inputs:
            1. coarse_graining (bool): whether use coarse-grained Qlm or qlm or not
                                       default False
            2. c (float): cutoff defining bond property, such as solid or not, default 0.7
            3. outputqlQl (str): csv file name of ql or Ql, default None
            4. outputsij (str): txt file name for sij of ql or Ql, default None

        Return:
            calculated sij (np.ndarray in a list for each snapshot)
        """
        if coarse_graining:
            cal_qlmQlm = self.largeQlm
            logger.info(f'Start calculating coarse-grained bond property sij for l={self.l}')
        else:
            cal_qlmQlm = self.smallqlm
            logger.info(f'Start calculating local bond property for l={self.l}')

        norm_qlmQlm = np.sqrt(np.square(np.abs(cal_qlmQlm)).sum(axis=2))

        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        # particle-level information
        results = []
        # particle-neighbors information, with number of neighbors
        resultssij = []
        for n, snapshot in enumerate(self.snapshots.snapshots):
            Neighborlist = read_neighbors(fneighbor, snapshot.nparticle, self.Nmax)
            sij = np.zeros((snapshot.nparticle, self.Nmax), dtype=np.float32)
            sijresults = np.zeros((snapshot.nparticle, 3))

            if (Neighborlist[:, 0] > self.Nmax).any():
                raise ValueError(
                    f'increase Nmax to include all neighbors, current is {self.Nmax}'
                )

            for i in range(snapshot.nparticle):
                cnlist = Neighborlist[i, 1:Neighborlist[i, 0]+1]
                sijup = (cal_qlmQlm[n, i][np.newaxis, :] * np.conj(cal_qlmQlm[n, cnlist])).sum(axis=1)
                sijdown = norm_qlmQlm[n, i] * norm_qlmQlm[n, cnlist]
                sij[i, :Neighborlist[i, 0]] = sijup.real / sijdown

            sijresults[:, 0] = np.arange(self.nparticle) + 1
            sijresults[:, 1] = (np.where(sij>c, 1, 0)).sum(axis=1)
            sijresults[:, 2] = Neighborlist[:, 0]
            results.append(sijresults)
            resultssij.append(np.column_stack((sijresults[:, 0], Neighborlist[:, 0], sij)))

        results = np.concatenate(results, axis=0)
        if outputqlQl:
            results = pd.DataFrame(results, columns="id sum_sij num_neighbors".split())
            results.to_csv(outputqlQl, float_format="%d", index=False)

        if outputsij:
            resultssij = np.concatenate(resultssij, axis=0)
            names = 'id CN sij'
            max_neighbors = int(resultssij[:, 1].max())
            resultssij = resultssij[:, :2+max_neighbors]
            data_format = "%d "*2 + "%.6f "*max_neighbors
            np.savetxt(outputsij, resultssij, header=names, fmt=data_format, comments="")

        fneighbor.close()
        logger.info(f'Finish calculating bond property sij for l={self.l}')
        return resultssij

    def w_W_cap(
        self,
        coarse_graining: bool=False,
        outputw: str=None,
        outputwcap: str=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate wigner 3-j symbol boo based on qlm or Qlm

        Inputs:
            1. coarse_graining (bool): whether use coarse-grained Qlm or qlm or not
                                       default False
            2. outputw (str): txt file name for w (original) based on qlm or Qlm
            3. outputwcap (str): txt file name for wcap (normalized) based on qlm or Qlm

        Return:
            calculated w and wcap (np.adarray) or W and Wcap (np.adarray)
            shape [nsnapshot, nparticle]
        """
        if coarse_graining:
            cal_qlmQlm = self.largeQlm
            logger.info(f'Start calculating coarse-grained Wigner 3-j symbol boo for l={self.l}')
        else:
            cal_qlmQlm = self.smallqlm
            logger.info(f'Start calculating local Wigner 3-j symbol boo for l={self.l}')

        w_W = np.zeros((cal_qlmQlm.shape[0], cal_qlmQlm.shape[1]))
        Windex = Wignerindex(self.l)
        w3j = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int64) + self.l
        for n in range(cal_qlmQlm.shape[0]):
            for i in range(cal_qlmQlm.shape[1]):
                w_W[n, i] = (np.real(np.prod(cal_qlmQlm[n, i, Windex], axis=1))*w3j).sum()

        if outputw:
            np.savetxt(outputw, w_W, fmt="%.6f", header="", comments="")

        w_W_cap = np.power(np.square(np.abs(cal_qlmQlm)).sum(axis=2), -3/2) * w_W
        if outputwcap:
            np.savetxt(outputwcap, w_W_cap, fmt="%.6f", header="", comments="")

        logger.info(f'Finish calculating wigner 3-j symbol boo for l={self.l}')
        return w_W, w_W_cap

    def spatial_corr(
        self,
        coarse_graining: bool=False,
        rdelta: float=0.01,
        outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate spatial correlation function of qlm or Qlm

        Inputs:
            1. coarse_graining (bool): whether use coarse-grained Qlm or qlm or not
                                       default False
            2. rdelta (float): bin size in calculating g(r) and Gl(r)
            3. outputfile (str): csv file name for gl
        
        Return:
            calculated Gl(r) based on Qlm or qlm
        """
        if coarse_graining:
            cal_qlmQlm = self.largeQlm
            logger.info(f'Start calculating coarse-grained spatial correlation for l={self.l}')
        else:
            cal_qlmQlm = self.smallqlm
            logger.info(f'Start calculating local spatial correlation for l={self.l}')

        glresults = 0
        for n, snapshot in enumerate(self.snapshots.snapshots):
            glresults += conditional_gr(
                snapshot=snapshot,
                condition=cal_qlmQlm[n],
                conditiontype="vector",
                ppp=self.ppp,
                rdelta=rdelta
            )
        glresults /= self.snapshots.nsnapshots
        if outputfile:
            glresults.to_csv(outputfile, float_format="%.8f", index=False)

        logger.info(f'Finish calculating spatial correlation for l={self.l}')
        return glresults

    def time_corr(
        self,
        coarse_graining: bool=False,
        dt: float=0.002,
        outputfile: str=None
    ) -> pd.DataFrame:
        """Calculate time correlation of qlm or Qlm

        Inputs:
            1. coarse_graining (bool): whether use coarse-grained Qlm or qlm or not
                                       default False
            2. dt (float): timestep used in user simulations, default 0.002
            2. outputfile (str): csv file name for time correlation results, default None
        
        Return:
            time correlation quantity (pd.DataFrame)
        """

        if coarse_graining:
            cal_qlmQlm = self.largeQlm
            logger.info(f'Start calculating coarse-grained time correlation for l={self.l}')
        else:
            cal_qlmQlm = self.smallqlm
            logger.info(f'Start calculating time correlation for l={self.l}')

        gl_time = time_correlation(
            snapshots=self.snapshots,
            condition=cal_qlmQlm,
            dt=dt,
        )

        # normalization
        gl_time["time_corr"] *= 4*np.pi/(2*self.l+1)
        gl_time["time_corr"] /= gl_time.loc[0, "time_corr"]
        if outputfile:
            gl_time.to_csv(outputfile, float_format="%.6f", index=False)
        return gl_time


class boo_2d:
    """
    This module calculates bond orientational orders in two dimensions
    Including original quantatities and weighted ones
    Also calculate both time correlation and spatial correlation
    This module accounts for both orthogonal and triclinic cells
    """

    def __init__(
            self,
            snapshots: Snapshots,
            l: int,
            neighborfile: str,
            weightsfile: str=None,
            ppp: np.ndarray=np.array([1,1]),
            Nmax: int=10
    ) -> None:
        """
        Initializing class for BOO2D

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                         (returned by reader.dump_reader.DumpReader)
            2. l (int): degree of orientational order, like l=6 for hexatic order
            3. neighborfile (str): file name of particle neighbors (see module neighbors)
            4. weightsfile (str): file name of particle-neighbor weights (see module neighbors)
                                  one typical example is Voronoi cell edge length of the polygon;
                                  this file should be consistent with neighborfile, default None
            5. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.array([1,1]),
            7. Nmax (int): maximum number for neighbors

        Return:
            None
        """
        self.snapshots = snapshots
        self.l = l
        self.neighborfile = neighborfile
        self.weightsfile = weightsfile
        self.ppp = ppp
        self.Nmax = Nmax

        assert len(
            set(np.diff([snapshot.timestep for snapshot in self.snapshots.snapshots]))
        ) == 1, "Warning: Dump interval changes during simulation"
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len(
            {snapshot.nparticle for snapshot in self.snapshots.snapshots}
        ) == 1, "Paticle number changes during simulation"
        self.boxlength = snapshots.snapshots[0].boxlength
        assert len(
            {tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}
        ) == 1, "Simulation box length changes during simulation"

        self.ParticlePhi = self.lthorder()

    def lthorder(self) -> np.ndarray:
        """
        Calculate l-th orientational order in 2D, such as hexatic order

        Inputs:
            None

        Return:
            Calculated l-th order (np.ndarray)
            shape: [nsnapshots, nparticle]
        """

        logger.info(f"Calculate {self.l}-th orinentational order in 2D")
        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        if self.weightsfile:
            fweights = open(self.weightsfile, 'r', encoding="utf-8")

        results = np.zeros((self.snapshots.nsnapshots, self.nparticle), dtype=np.complex128)
        for n, snapshot in enumerate(self.snapshots.snapshots):
            Neighborlist = read_neighbors(fneighbor, snapshot.nparticle, self.Nmax)
            if not self.weightsfile:
                for i in range(snapshot.nparticle):
                    cnlist = Neighborlist[i, 1:(Neighborlist[i, 0]+1)]
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis, :]
                    RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                    theta = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    results[n, i] = (np.exp(1j*self.l*theta)).mean()
            else:
                weightslist = read_neighbors(fweights, snapshot.nparticle, self.Nmax)[:, 1:]
                for i in range(snapshot.nparticle):
                    cnlist = Neighborlist[i, 1:(Neighborlist[i, 0]+1)]
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis, :]
                    RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                    theta = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    weights = weightslist[i, 1:Neighborlist[i, 0]+1]
                    weights /= weights.sum()
                    results[n, i] = (weights*np.exp(1j*self.l*theta)).sum()
        fneighbor.close()
        if self.weightsfile:
            fweights.close()
        return results

    def tavephi(
            self,
            outputphi: str=None,
            outputavephi: str=None,
            avet: float=0.0,
            dt: float=0.002
    ) -> np.ndarray:
        """
        Compute PHI value and Time Averaged PHI

        Inputs:
            1. outputphi (str): file name for absolute values of phi, default None
            2. outputavephi (str): file name for time averaged phi, default None
            3. avet (float): time used to average, default 0.0
            4. dt (float): timestep used in user simulations, default 0.002
        Return:
            Calculated phi value or time averaged phi (np.ndarray)
        """
        logger.info(f"Calculate PHI value and Time Averaged PHI for l={self.l}")
        results = np.abs(self.lthorder())

        if avet == 0.0:
            # compute absolute phi
            ParticlePhi = np.column_stack((np.arange(self.nparticle)+1, results.T))
            if outputphi:
                names = 'id   phil=' + str(self.l)
                numformat = '%d ' + '%.6f ' * (len(ParticlePhi[0])-1)
                np.savetxt(outputphi, ParticlePhi, fmt=numformat, header=names, comments='')
            return ParticlePhi
        else:
            # compute time averaged phi
            timestep = np.diff([snapshot.timestep for snapshot in self.snapshots.snapshots])[0]
            avet = int(avet/dt/timestep)
            averesults = np.zeros((self.snapshots.nsnapshots-avet, self.nparticle))
            for n in range(self.snapshots.nsnapshots-avet):
                averesults[n] = results[n:n+avet].mean(axis=0)

            averesults = np.column_stack((np.arange(self.nparticle)+1, averesults.T))
            if outputavephi:
                names = 'id   ave_phil=' + str(self.l)
                numformat = '%d ' + '%.6f ' * (len(averesults[0])-1)
                np.savetxt(outputavephi, averesults, fmt=numformat, header=names, comments='')
            return averesults

    def spatial_corr(
            self,
            rdelta: float=0.01,
            outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate spatial correlation of phi in 2D system

        Inputs:
            1. rdelta (float): bin size in calculating g(r) and Gl(r), default 0.01
            2. outputfile (str): csv file name for gl(r)

        Return:
            calculated gl(r) based on phi

        """
        logger.info(f'Start calculating spatial correlation of phi for l={self.l}')

        ParticlePhi = self.lthorder()
        glresults = 0
        for n, snapshot in enumerate(self.snapshots.snapshots):
            glresults += conditional_gr(
                snapshot=snapshot,
                condition=ParticlePhi[n],
                conditiontype=None,
                ppp=self.ppp,
                rdelta=rdelta
            )
        glresults /= self.snapshots.nsnapshots
        if outputfile:
            glresults.to_csv(outputfile, float_format="%.8f", index=False)

        logger.info(f'Finish calculating spatial correlation of phi for l={self.l}')
        return glresults

    def time_corr(
            self,
            dt: float=0.002,
            outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate time correlation of phi in 2D system

        Inputs:
            1. dt (float): timestep used in user simulations, default 0.002
            2. outputfile (str): csv file name for time correlation results, default None

        Return:
            time correlation quantity (pd.DataFrame)
        """
        logger.info(f'Start calculating time correlation of phi for l={self.l}')

        ParticlePhi = self.lthorder()
        gl_time = time_correlation(
            snapshots=self.snapshots,
            condition=ParticlePhi,
            dt=dt
        )

        # normalization
        gl_time["time_corr"] *= 4*np.pi/(2*self.l+1)
        gl_time["time_corr"] /= gl_time.loc[0, "time_corr"]
        if outputfile:
            gl_time.to_csv(outputfile, float_format="%.6f", index=False)
        return gl_time
