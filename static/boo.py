# coding = utf-8

"""
This module calculates bond orientational order parameters
in two-dimensions and three-dimensions

see documentation @ ../docs/boo_3d.md &
see documentation @ ../docs/boo_2d.md
"""

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
                                 setting 1 for yes and 0 for no, default [1,1,1],
            6. Nmax (int): maximum number for neighbors

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
        )))==1, "Warning: Dump interval changes during simulation"
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len(
            {snapshot.nparticle for snapshot in self.snapshots.snapshots}
        )==1, "Paticle number changes during simulation"
        self.boxlength = snapshots.snapshots[0].boxlength
        assert len(
            {tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}
        )==1, "Simulation box length changes during simulation"

        self.smallqlm, self.largeQlm = self.qlmQlm()

    def qlmQlm(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        BOO of the l-fold symmetry as a 2l + 1 vector

        Inputs:
            None
        
        Return:
            BOO of order-l in vector complex number
            shape: [nsnapshot, nparticle, 2l+1]
        """

        logger.info(f"Calculate the spherical harmonics of l={self.l}")
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
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis,:]
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
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis,:]
                    RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                    distance = np.linalg.norm(RIJ, axis=1)
                    theta = np.arccos(RIJ[:, 2]/distance)
                    phi = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    for j in range(Neighborlist[i, 0]):
                        Particlesmallqlm[i] += sph_harm_l(self.l,theta[j],phi[j])*weightsfrac[i,j]
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

        logger.info(f"The spherical harmonics of l={self.l} is ready for further calculations")
        return np.array(smallqlm), np.array(largeQlm)

    def qlQl(self, outputql: str=None, outputQl: str=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate BOO ql (original) and Ql (coarse-grain)

        Inputs:
            1. outputql (str): file name for ql results, default None
            2. outputQl (str): file name for Ql results, defalut None
        
        Return:
            calculated ql and Ql (np.ndarray) 
            shape [nsnapshot, nparticle]
        """
        logger.info(f'Start calculating the rotational invariants ql & Ql for l={self.l}')

        smallql = np.sqrt(4*np.pi/(2*self.l+1)*np.square(np.abs(self.smallqlm)).sum(axis=2))
        if outputql:
            np.savetxt(outputql, smallql, fmt="%.6f", header="", comments="")

        largeQl = np.sqrt(4*np.pi/(2*self.l+1)*np.square(np.abs(self.largeQlm)).sum(axis=2))
        if outputQl:
            np.savetxt(outputQl, largeQl, fmt="%.6f", header="", comments="")

        logger.info(f'Finish calculating the rotational invariants ql & Ql for l={self.l}')
        return (smallql, largeQl)

    def sijsmallql(self, c: float=0.7, outputql: str=None, outputsij: str=None) -> np.ndarray:
        """
        Calculate orientation correlation of qlm, named as sij

        Inputs:
            1. c (float): cutoff defining bond property, such as solid or not, default 0.7
            2. outputql (str): file name for ql, default None
            3. outputsij (str): file name for sij of ql, default None

        Return:
            calculated sij (np.ndarray)
        """
        logger.info(f'Start calculating bond property sij based on ql for l={self.l}')

        norm_smallqlm = np.sqrt(np.square(np.abs(self.smallqlm)).sum(axis=2))

        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        # particle-level information
        results = np.zeros((1, 3))
        # particle-neighbors information, with number of neighbors
        resultssij = np.zeros((1, self.Nmax+1))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            Neighborlist = read_neighbors(fneighbor, snapshot.nparticle, self.Nmax)
            sij = np.zeros((snapshot.nparticle, self.Nmax))
            sijresults = np.zeros((snapshot.nparticle, 3))
            if (Neighborlist[:, 0] > self.Nmax).any():
                raise ValueError(
                    f'increase Nmax to include all neighbors, current is {self.Nmax}'
                )
            for i in range(snapshot.nparticle):
                cnlist = Neighborlist[i, 1:Neighborlist[i, 0]+1]
                sijup = (self.smallqlm[n, i][np.newaxis,:] * np.conj(self.smallqlm[n, cnlist])).sum(axis=1)
                sijdown = norm_smallqlm[n, i] * norm_smallqlm[n, cnlist]
                sij[i, :Neighborlist[i, 0]] = sijup.real / sijdown

            sijresults[:, 0] = np.arange(self.nparticle) + 1
            sijresults[:, 1] = (np.where(sij>c, 1, 0)).sum(axis=1)
            sijresults[:, 2] = Neighborlist[:, 0]
            results = np.vstack((results, sijresults))
            resultssij = np.vstack((resultssij, np.column_stack((Neighborlist[:, 0], sij))))

        if outputql:
            results = pd.DataFrame(results[1:], columns="id sum_sij num_neighbors".split())
            results.to_csv(outputql, float_format="%d", index=False)

        max_neighbors = resultssij[:, 0].max()
        resultssij = resultssij[1:, :int(1+max_neighbors)]
        if outputsij:
            names = 'CN sij'
            formatsij = '%d ' + '%.6f '*max_neighbors
            np.savetxt(outputsij, resultssij, fmt=formatsij, header=names, comments='')

        fneighbor.close()
        logger.info(f'Finish calculating bond property sij based on ql for l={self.l}')
        return resultssij

    def sijlargeQl(self, c: float=0.7, outputQl: str=None, outputsij: str=None) -> np.ndarray:
        """
        Calculate orientation correlation of Qlm, named as sij

        Inputs:
            1. c (float): cutoff defining bond property, such as solid or not, default 0.7
            2. outputQl (str): file name for Ql, default None
            3. outputsij (str): file name for sij of Ql, default None

        Return:
            calculated sij (np.ndarray)
        """
        logger.info(f'Start calculating bond property sij based on Ql for l={self.l}')

        norm_largeQlm = np.sqrt(np.square(np.abs(self.largeQlm)).sum(axis=2))

        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        # particle-level information
        results = np.zeros((1, 3))
        # particle-neighbors information, with number of neighbors
        resultssij = np.zeros((1, self.Nmax+1))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            Neighborlist = read_neighbors(fneighbor, snapshot.nparticle, self.Nmax)
            sij = np.zeros((snapshot.nparticle, self.Nmax))
            sijresults = np.zeros((snapshot.nparticle, 3))
            if (Neighborlist[:, 0] > self.Nmax).any():
                raise ValueError(
                    f'increase Nmax to include all neighbors, current is {self.Nmax}'
                )
            for i in range(snapshot.nparticle):
                cnlist = Neighborlist[i, 1:Neighborlist[i, 0]+1]
                sijup = (self.largeQlm[n, i][np.newaxis,:] * np.conj(self.largeQlm[n, cnlist])).sum(axis=1)
                sijdown = norm_largeQlm[n, i] * norm_largeQlm[n, cnlist]
                sij[i, :Neighborlist[i, 0]] = sijup.real / sijdown

            sijresults[:, 0] = np.arange(self.nparticle) + 1
            sijresults[:, 1] = (np.where(sij>c, 1, 0)).sum(axis=1)
            sijresults[:, 2] = Neighborlist[:, 0]
            results = np.vstack((results, sijresults))
            resultssij = np.vstack((resultssij, np.column_stack((Neighborlist[:, 0], sij))))

        if outputQl:
            results = pd.DataFrame(results[1:], columns="id sum_sij num_neighbors".split())
            results.to_csv(outputQl, float_format="%d", index=False)
        
        max_neighbors = resultssij[:, 0].max()
        resultssij = resultssij[1:, :int(1+max_neighbors)]
        if outputsij:
            names = 'CN sij'
            formatsij = '%d ' + '%.6f '*max_neighbors
            np.savetxt(outputsij, resultssij, fmt=formatsij, header=names, comments='')
        
        fneighbor.close()
        logger.info(f'Finish calculating bond property sij based on Ql for l={self.l}')
        return resultssij

    def spatial_corr(self, rdelta: float=0.01, cal_type: str="local", outputfile: str=None) -> pd.DataFrame:
        """
        Calculate spatial correlation function of Qlm

        Inputs:
            1. rdelta (float): bin size in calculating g(r) and Gl(r)
            2. cal_type (str): calculation input type, whether qlm (local) or Qlm (coarse-grained)
            2. outputfile (str): file name for gl for Qlm
        
        Return:
            calculated Gl(r) based on Qlm or qlm
        """
        logger.info(f'Start calculating spatial correlation of {cal_type} BOO for l={self.l}')
        if cal_type=="local":
            cal_qlmQlm = self.smallqlm.copy()
        elif cal_type=="coarse-grained":
            cal_qlmQlm = self.largeQlm.copy()
        else:
            raise ValueError(
                f"Error cal_type as {cal_type}, should be 'local' or 'coarse-grained'"
            )

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

        logger.info(f'Finish calculating spatial correlation of {cal_type} BOO for l={self.l}')
        return glresults

    def Glsmallq(self, rdelta: float=0.01, outputgl: str=None) -> pd.DataFrame:
        """
        Calculate spatial correlation function of qlm

        Inputs:
            1. rdelta (float): bin size in calculating g(r) and Gl(r)
            2. outputgl (str): file name of gl for Qlm
        
        Return:
            calculated Gl(r) based on qlm
        """
        logger.info('Start calculating spatial correlation of ql for l={self.l}')

        glresults = 0
        for n, snapshot in enumerate(self.snapshots.snapshots):
            glresults += conditional_gr(
                snapshot=snapshot,
                condition=self.smallqlm[n],
                conditiontype="vector",
                ppp=self.ppp,
                rdelta=rdelta
            )
        glresults /= self.snapshots.nsnapshots
        if outputgl:
            glresults.to_csv(outputgl, float_format="%.8f", index=False)

        logger.info(f'Finish calculating spatial correlation of ql for l={self.l}')
        return glresults

    def smallwcap(self, outputw: str=None, outputwcap: str=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate wigner 3-j symbol boo based on qlm

        Inputs:
            1. outputw (str): file name for w (original)
            2. outputwcap (str): file name for wcap (normalized)

        Return:
            calculated w and wcap (np.adarray)
            shape [nsnapshot, nparticle]
        """
        logger.info(f'Start calculating (normalized) w based on qlm for l={self.l}')

        smallw = np.zeros((self.smallqlm.shape[0], self.smallqlm.shape[1]))
        Windex = Wignerindex(self.l)
        w3j = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int64) + self.l
        for n in range(self.smallqlm.shape[0]):
            for i in range(self.smallqlm.shape[1]):
                smallw[n, i] = (np.real(np.prod(self.smallqlm[n, i, Windex], axis=1))*w3j).sum()

        if outputw:
            np.savetxt(outputw, smallw, fmt="%.6f", header="", comments="")

        smallwcap = np.power(np.square(np.abs(self.smallqlm)).sum(axis=2), -3/2) * smallw
        if outputwcap:
            np.savetxt(outputwcap, smallwcap, fmt="%.6f", header="", comments="")

        logger.info(f'Finish calculating (normalized) w based on qlm for l={self.l}')
        return (smallw, smallwcap)

    def largeWcap(self, outputW: str=None, outputWcap: str=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate wigner 3-j symbol boo based on Qlm

        Inputs:
            1. outputw (str): file name for W (original)
            2. outputwcap (str): the file name for Wcap (normalized)

        Return:
            calculated W and Wcap (np.adarray)
            shape [nsnapshot, nparticle]
        """
        logger.info(f'Start calculating (normalized) W based on Qlm for l={self.l}')

        largeW = np.zeros((self.largeQlm.shape[0], self.largeQlm.shape[1]))
        Windex = Wignerindex(self.l)
        w3j = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int64) + self.l
        for n in range(self.largeQlm.shape[0]):
            for i in range(self.largeQlm.shape[1]):
                largeW[n, i] = (np.real(np.prod(self.largeQlm[n, i, Windex], axis=1))*w3j).sum()

        if outputW:
            np.savetxt(outputW, largeW, fmt="%.6f", header="", comments="")

        largeWcap = np.power(np.square(np.abs(self.largeQlm)).sum(axis=2), -3/2) * largeW
        if outputWcap:
            np.savetxt(outputWcap, largeWcap, fmt="%.6f", header="", comments="")

        logger.info(f'Finish calculating (normalized) W based on Qlm for l={self.l}')
        return (largeW, largeWcap)

    # TODO: Calling time correlation function to implement this function
    def timecorr(self, l, ppp = [1,1,1], AreaR = 0, dt = 0.002, outputfile = ''):
        """Calculate time correlation of qlm and Qlm

            AreaR = 0 indicates calculate traditional ql and Ql
            AreaR = 1 indicates calculate voronoi polyhedron face area weighted ql and Ql
        """

        print ('----Calculate the time correlation of qlm & Qlm----')

        (smallqlm, largeQlm) = self.qlmQlm()
        smallqlm = np.array(smallqlm)
        largeQlm = np.array(largeQlm)
        results = np.zeros((self.snapshots.nsnapshots - 1, 3))
        names = 't   timecorr_q   timecorr_Ql=' + str(l)

        cal_smallq = pd.DataFrame(np.zeros((self.snapshots-1))[np.newaxis, :])
        cal_largeQ = pd.DataFrame(np.zeros((self.snapshots-1))[np.newaxis, :])
        fac_smallq = pd.DataFrame(np.zeros((self.snapshots-1))[np.newaxis, :])
        fac_largeQ = pd.DataFrame(np.zeros((self.snapshots-1))[np.newaxis, :])
        deltat     = np.zeros(((self.snapshots-1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.snapshots-1):
            CIJsmallq  = (np.real((smallqlm[n + 1:] * np.conj(smallqlm[n]))).sum(axis = 2)).sum(axis = 1)
            cal_smallq = pd.concat([cal_smallq, pd.DataFrame(CIJsmallq[np.newaxis, :])])
            CIIsmallq  = np.repeat((np.square(np.abs(smallqlm[n])).sum()), len(CIJsmallq)) #consider initial snapshot
            fac_smallq = pd.concat([fac_smallq, pd.DataFrame(CIIsmallq[np.newaxis, :])])
            CIJlargeQ  = (np.real((largeQlm[n + 1:] * np.conj(largeQlm[n]))).sum(axis = 2)).sum(axis = 1)
            cal_largeQ = pd.concat([cal_largeQ, pd.DataFrame(CIJlargeQ[np.newaxis, :])])
            CIIlargeQ  = np.repeat((np.square(np.abs(largeQlm[n])).sum()), len(CIJlargeQ))
            fac_largeQ = pd.concat([fac_largeQ, pd.DataFrame(CIIlargeQ[np.newaxis, :])])

        cal_smallq = cal_smallq.iloc[1:]
        cal_largeQ = cal_largeQ.iloc[1:]
        fac_smallq = fac_smallq.iloc[1:]
        fac_largeQ = fac_largeQ.iloc[1:]
        deltat[:, 0] = np.array(cal_smallq.columns) + 1 #Timeinterval
        deltat[:, 1] = np.array(cal_smallq.count())     #Timeinterval frequency

        results[:, 0] = deltat[:, 0] * self.snapshots.snapshots[0].timestep * dt 
        results[:, 1] = cal_smallq.mean() * (4 * np.pi / (2 * l + 1)) / fac_smallq.mean()
        results[:, 2] = cal_largeQ.mean() * (4 * np.pi / (2 * l + 1)) / fac_largeQ.mean()

        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        return results