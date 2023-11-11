# coding = utf-8

"""
This module calculates bond orientational order parameters
in two-dimensions or three-dimensions

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
            faceareafile: str=None,
            ppp: list=[1,1,1],
            Nmax: int=30
            ) -> None:
        """
        Initializing class for BOO3D

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                         (returned by reader.dump_reader.DumpReader)
            2. l (int): degree of spherical harmonics
            3. neighborfile (str): file name of particle neighbors (see module neighbors)
            4. faceareafile (str): file name of Voronoi particle faceareas (see module neighbors)
                                   or any other reasonable center-neighbors weights
            4. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
            5. Nmax (int): maximum number for neighbors

        Return:
            None
        """
        self.snapshots = snapshots
        self.l = l
        self.neighborfile = neighborfile
        self.faceareafile = faceareafile
        self.ppp = ppp
        self.Nmax = Nmax

        assert len(set(np.diff([snapshot.timestep for snapshot in self.snapshots.snapshots])))==1,\
            "Warning: Dump interval changes during simulation"
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len({snapshot.nparticle for snapshot in self.snapshots.snapshots})==1,\
            "Paticle number changes during simulation"
        self.boxlength = snapshots.snapshots[0].boxlength
        assert len({tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots})==1,\
            "Simulation box length changes during simulation"

        self.particle_type = [snapshot.particle_type for snapshot in self.snapshots.snapshots]
        self.positions = [snapshot.positions for snapshot in self.snapshots.snapshots]
        self.nsnapshots = self.snapshots.nsnapshots

        self.rhototal = self.nparticle / np.prod(self.boxlength)
        self.hmatrix = [snapshot.hmatrix for snapshot in self.snapshots.snapshots]
        self.typenumber, self.typecount = np.unique(self.particle_type[0], return_counts=True)
        assert np.sum(self.typecount) == self.nparticle,\
            "Sum of Indivdual Types is Not the Total Amount"

    def qlmQlm(self) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        BOO of the l-fold symmetry as a 2l + 1 vector

        Inputs:
            None
        
        Return:
            BOO of order-l in vector complex number
        """

        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        if self.faceareafile:
            ffacearea = open(self.faceareafile, 'r', encoding="utf-8")

        smallqlm = []
        largeQlm = []
        for snapshot in self.snapshots.snapshots:
            Neighborlist = read_neighbors(fneighbor, snapshot.nparticle, self.Nmax)
            Particlesmallqlm = np.zeros((snapshot.nparticle, 2*self.l+1), dtype=np.complex128)
            if not self.faceareafile:
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
                facearealist = read_neighbors(ffacearea, snapshot.nparticle, self.Nmax)
                # normalization of center-neighbors weights
                faceareafrac = facearealist[:, 1:] / facearealist[:, 1:].sum(axis=1)[:, np.newaxis]
                for i in range(snapshot.nparticle):
                    cnlist = Neighborlist[i, 1:(Neighborlist[i, 0]+1)]
                    RIJ = snapshot.positions[cnlist] - snapshot.positions[i][np.newaxis,:]
                    RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                    distance = np.linalg.norm(RIJ, axis=1)
                    theta = np.arccos(RIJ[:, 2]/distance)
                    phi = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    for j in range(Neighborlist[i, 0]):
                        Particlesmallqlm[i] += sph_harm_l(self.l,theta[j],phi[j])*faceareafrac[i,j]
                smallqlm.append(Particlesmallqlm)

            # coarse-graining over the neighbors
            ParticlelargeQlm = np.copy(Particlesmallqlm)
            for i in range(snapshot.nparticle):
                for j in range(Neighborlist[i, 0]):
                    ParticlelargeQlm[i] += Particlesmallqlm[Neighborlist[i, j+1]]
            ParticlelargeQlm = ParticlelargeQlm / (1+Neighborlist[:, 0])[:, np.newaxis]
            largeQlm.append(ParticlelargeQlm)

        fneighbor.close()
        if self.faceareafile:
            ffacearea.close()
        return (smallqlm, largeQlm)

    def qlQl(self, outputql: str=None, outputQl: str=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate BOO ql (original) and Ql (coarse-grain)

        Inputs:
            1. outputql (str): file name for ql results
            2. outputQl (str): file name for Ql results
        
        Return:
            calculated ql and Ql (np.ndarray) {shape [nsnapshot, nparticle]}
        """
        logger.info(f'Start Calculating the rotational invariants ql & Ql for l={self.l}')

        (smallqlm, largeQlm) = self.qlmQlm()
        # shape [nsnapshot, nparticle]
        smallql = np.sqrt(4*np.pi/(2*self.l+1)*np.square(np.abs(smallqlm)).sum(axis=2))
        if outputql:
            np.savetxt(outputql, smallql, fmt="%.6f", header="", comments="")

        largeQl = np.sqrt(4*np.pi/(2*self.l+1)*np.square(np.abs(largeQlm)).sum(axis=2))
        if outputQl:
            np.savetxt(outputQl, largeQl, fmt="%.6f", header="", comments="")

        logger.info(f'Finish Calculating the rotational invariants ql & Ql for l={self.l}')
        return (smallql, largeQl)

    def sijsmallql(self, c: float=0.7, outputql: str=None, outputsij: str=None) -> np.ndarray:
        """
        Calculate Crystal Nuclei Criterion s(i, j) based on ql

        Inputs:
            1. c (float): cutoff demonstrating whether a bond is crystalline or not, default 0.7
            2. outputql (str): the file name to store the results of ql
            3. outputsij (str): the file name to store the results of sij

        Return:
            calculated sij (np.ndarray)
        """
        logger.info('Start Calculating Crystal Nuclei Criterion s(i, j) based on ql')

        MaxNeighbor = self.Nmax
        smallqlm, largeQlm = self.qlmQlm()
        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        results = np.zeros((1, 3))
        resultssij = np.zeros((1,  MaxNeighbor+1))
        for n in range(self.nsnapshots):
            Neighborlist = read_neighbors(fneighbor, self.nparticle, self.Nmax)
            sij = np.zeros((self.nparticle, MaxNeighbor))
            sijresults = np.zeros((self.nparticle, 3))
            if (Neighborlist[:, 0] > MaxNeighbor).any():
                raise ValueError('increase Nmax to include all neighbors')
            for i in range(self.nparticle):
                for j in range(Neighborlist[i, 0]):
                    sijup = (smallqlm[n][i]*np.conj(smallqlm[n][Neighborlist[i, j+1]])).sum()
                    sijdown = np.sqrt(np.square(np.abs(smallqlm[n][i])).sum())*np.sqrt(np.square(np.abs(smallqlm[n][Neighborlist[i, j+1]])).sum())
                    sij[i, j] = np.real(sijup/sijdown)
            sijresults[:, 0] = np.arange(self.nparticle) + 1
            sijresults[:, 1] = (np.where(sij>c, 1, 0)).sum(axis=1)
            sijresults[:, 2] = np.where(sijresults[:, 1]>Neighborlist[:, 0]/2, 1, 0)
            results = np.vstack((results, sijresults))
            resultssij = np.vstack((resultssij, np.column_stack((Neighborlist[:, 0], sij))))

        if outputql:
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(self.l)
            np.savetxt(outputql, results[1:], fmt='%d', header=names, comments='')

        if outputsij:
            names = 'CN s(i, j)  l=' + str(self.l)
            formatsij = '%d ' + '%.6f ' * MaxNeighbor
            np.savetxt(outputsij, resultssij[1:], fmt=formatsij, header=names, comments='')

        fneighbor.close()

        logger.info('Finish Calculating Crystal Nuclei Criterion s(i, j) based on ql')
        return resultssij[1:]

    def sijlargeQl(self, c: float=0.7, outputQl: str=None, outputsij: str=None) -> np.ndarray:
        """
        Calculate Crystal Nuclei Criterion s(i, j) based on Ql

        Inputs:
            1. c (float): cutoff demonstrating whether a bond is crystalline or not, default 0.7
            2. outputQl (str): the file name to store the results of Ql
            3. outputsij (str): the file name to store the results of sij

        Return:
            calculated sij (np.ndarray)
        """
        logger.info('Start Calculating Crystal Nuclei Criterion s(i, j) based on Ql')

        MaxNeighbor = self.Nmax
        smallqlm, largeQlm = self.qlmQlm()
        fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
        results = np.zeros((1, 3))
        resultssij = np.zeros((1,  MaxNeighbor+1))
        for n in range(self.nsnapshots):
            Neighborlist = read_neighbors(fneighbor, self.nparticle, self.Nmax)
            sij = np.zeros((self.nparticle, MaxNeighbor))
            sijresults = np.zeros((self.nparticle, 3))
            if (Neighborlist[:, 0]>MaxNeighbor).any():
                raise ValueError('increase Nmax to include all neighbors')
            for i in range(self.nparticle):
                for j in range(Neighborlist[i, 0]):
                    sijup = (largeQlm[n][i] * np.conj(largeQlm[n][Neighborlist[i, j+1]])).sum()
                    sijdown = np.sqrt(np.square(np.abs(largeQlm[n][i])).sum()) * np.sqrt(np.square(np.abs(largeQlm[n][Neighborlist[i, j+1]])).sum())
                    sij[i, j] = np.real(sijup/sijdown)
            sijresults[:, 0] = np.arange(self.nparticle) + 1
            sijresults[:, 1] = (np.where(sij>c, 1, 0)).sum(axis=1)
            sijresults[:, 2] = np.where(sijresults[:, 1] > Neighborlist[:, 0] / 2, 1, 0)
            results = np.vstack((results, sijresults))
            resultssij = np.vstack((resultssij, np.column_stack((Neighborlist[:, 0], sij))))

        if outputQl:
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(self.l)
            np.savetxt(outputQl, results[1:], fmt='%d', header=names, comments='')
            
        if outputsij:
            names = 'CN  s(i, j)  l=' + str(self.l)
            formatsij = '%d ' + '%.6f ' * MaxNeighbor
            np.savetxt(outputsij, resultssij[1:], fmt=formatsij, header=names, comments='')

        fneighbor.close()
        logger.info('Finish Calculating Crystal Nuclei Criterion s(i, j) based on Ql')
        return resultssij[1:]

    def GllargeQ(self, rdelta: float=0.01) -> pd.DataFrame:
        """
        Calculate bond order spatial correlation function Gl(r) based on Qlm

        Inputs:
            1. rdelta (float): bin size in calculating g(r) and Gl(r)
            2. outputgl (str): the file name to store the results of gl
        
        Return:
            calculated bond order spatial correlation function Gl(r) based on Qlm
        """
        logger.info('Start Calculating bond order correlation Gl based on Ql')

        smallqlm, largeQlm = self.qlmQlm()

        grresults = []
        for n in range(self.nsnapshots):
            grresults.append(conditional_gr(self.snapshots.snapshots[n], condition=largeQlm[n], ppp=self.ppp, rdelta=rdelta))

        logger.info('Finish Calculating bond order correlation Gl based on Ql')
        return grresults

    def Glsmallq(self, rdelta=0.01) -> pd.DataFrame:
        """
        Calculate bond order spatial correlation function Gl(r) based on qlm

        Inputs:
            1. rdelta (float): bin size in calculating g(r) and Gl(r)
            2. outputgl (str): the file name to store the results of gl
        
        Return:
            calculated bond order spatial correlation function Gl(r) based on qlm
        """
        logger.info('Start Calculating bond order correlation Gl based on ql')

        smallqlm, largeQlm = self.qlmQlm()

        grresults = []
        for n in range(self.nsnapshots):
            grresults.append(conditional_gr(self.snapshots.snapshots[n], condition=smallqlm[n], ppp=self.ppp, rdelta=rdelta))

        logger.info('Finish Calculating bond order correlation Gl based on ql')
        return grresults


    def smallwcap(self, outputw: str=None, outputwcap: str=None) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Calculate wigner 3-j symbol boo based on qlm

        Inputs:
            1. outputw (str): the file name to store the results of w
            2. outputwcap (str): the file name to store the results of wcap

        Return:
            calculated w and wcap (np.adarray)
        """
        logger.info('Start Calculating bond Orientational order w (normalized) based on qlm')

        smallqlm, largeQlm = self.qlmQlm()
        smallqlm = np.array(smallqlm)
        smallw = np.zeros((self.nsnapshots, self.nparticle))
        Windex = Wignerindex(self.l)
        w3j = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int) + self.l 
        for n in range(self.nsnapshots):
            for i in range(self.nparticle):
                smallw[n, i] = (np.real(np.prod(smallqlm[n, i, Windex], axis=1))*w3j).sum()
       
        smallw = np.column_stack((np.arange(self.nparticle)+1, smallw.T))
        if outputw:
            names = 'id  wl  l=' + str(self.l)
            numformat = '%d ' + '%.10f ' * (len(smallw[0])-1)
            np.savetxt(outputw, smallw, fmt=numformat, header=names, comments='')
   
        smallwcap = np.power(np.square(np.abs(np.array(smallqlm))).sum(axis = 2), -3/2).T * smallw[:, 1:]
        smallwcap = np.column_stack((np.arange(self.nparticle)+1, smallwcap))
        if outputwcap:
            names = 'id  wlcap  l=' + str(self.l)
            numformat = '%d ' + '%.8f ' * (len(smallwcap[0])-1) 
            np.savetxt(outputwcap, smallwcap, fmt=numformat, header=names, comments='')
        
        logger.info('Finish Calculating bond Orientational order w (normalized) based on qlm')
        return (smallw, smallwcap)


    def largeWcap(self, outputW: str=None, outputWcap: str=None) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Calculate wigner 3-j symbol boo based on qlm

        Inputs:
            1. outputw (str): the file name to store the results of W
            2. outputwcap (str): the file name to store the results of Wcap

        Return:
            calculated w and wcap (np.adarray)
        """
        logger.info('Start Calculating bond Orientational order W (normalized) based on Qlm')

        smallqlm, largeQlm = self.qlmQlm()
        largeQlm = np.array(largeQlm)
        largew = np.zeros((self.nsnapshots, self.nparticle))
        Windex = Wignerindex(self.l)
        w3j = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int) + self.l
        for n in range(self.nsnapshots):
            for i in range(self.nparticle):
                largew[n, i] = (np.real(np.prod(largeQlm[n, i, Windex], axis=1))*w3j).sum()

        largew = np.column_stack((np.arange(self.nparticle)+1, np.real(largew.T)))
        if outputW:
            names = 'id  Wl  l=' + str(self.l)
            numformat = '%d ' + '%.10f ' * (len(largew[0])-1)
            np.savetxt(outputW, largew, fmt=numformat, header=names, comments='')
   
        largewcap = np.power(np.square(np.abs(np.array(largeQlm))).sum(axis=2), -3/2).T * largew[:, 1:]
        largewcap = np.column_stack((np.arange(self.nparticle)+1, largewcap))
        if outputWcap:
            names = 'id  Wlcap  l=' + str(self.l)
            numformat = '%d ' + '%.8f ' * (len(largewcap[0])-1)
            np.savetxt(outputWcap, largewcap, fmt=numformat, header=names, comments='')

        logger.info('Finish Calculating bond Orientational order W (normalized) based on Qlm')
        return (largew, largewcap)


    def timecorr(self, l, ppp = [1,1,1], AreaR = 0, dt = 0.002, outputfile = ''):
        """ Calculate time correlation of qlm and Qlm

            AreaR = 0 indicates calculate traditional ql and Ql
            AreaR = 1 indicates calculate voronoi polyhedron face area weighted ql and Ql
        """
        print ('----Calculate the time correlation of qlm & Qlm----')

        (smallqlm, largeQlm) = self.qlmQlm()
        smallqlm = np.array(smallqlm)
        largeQlm = np.array(largeQlm)
        results = np.zeros((self.nsnapshots - 1, 3))
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

    def cal_multiple(
            self,
            c: float=0.7,
            namestr: str=None,
            cqlQl: bool=False,
            csijsmallql: bool=False,
            csijlargeQl: bool=False,
            csmallwcap: bool=False,
            clargeWcap: bool=False
            ):
        """Calculate multiple order parameters at the same time"""

        logger.info('Start Calculating multiple order parameters together')
        smallqlm, largeQlm = self.qlmQlm()

        if cqlQl:
            logger.info('Start Calculating ql and Ql')
            smallql = np.sqrt(4*np.pi/(2*self.l+1) * np.square(np.abs(smallqlm)).sum(axis=2))
            smallql = np.column_stack((np.arange(self.nparticle)+1, smallql.T))
            outputql = namestr + '.smallq_l%d.dat'%self.l
            names = 'id  ql  l=' + str(self.l)
            numformat = '%d ' + '%.6f ' * (len(smallql[0])-1)
            np.savetxt(outputql, smallql, fmt=numformat, header=names, comments='')

            largeQl = np.sqrt(4*np.pi/(2*self.l+1)*np.square(np.abs(largeQlm)).sum(axis=2))
            largeQl = np.column_stack((np.arange(self.nparticle) + 1, largeQl.T))
            outputQl = namestr + '.largeQ_l%d.dat'%self.l
            names = 'id  Ql  l=' + str(self.l)
            numformat = '%d ' + '%.6f ' * (len(largeQl[0]) - 1)
            np.savetxt(outputQl, largeQl, fmt=numformat, header=names, comments='')

            logger.info('Finish Calculating ql and Ql')

        if csijsmallql:
            logger.info('Start Calculating s(i, j) based on ql')
            MaxNeighbor = self.Nmax
            fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
            results = np.zeros((1, 3))
            for n in range(self.nsnapshots):
                Neighborlist = read_neighbors(fneighbor, self.nparticle, self.Nmax)
                sij = np.zeros((self.nparticle, MaxNeighbor))
                sijresults = np.zeros((self.nparticle, 3))
                if (Neighborlist[:, 0]>MaxNeighbor).any():
                    raise ValueError('increase Nmax to include all neighbors')
                for i in range(self.nparticle):
                    for j in range(Neighborlist[i, 0]):
                        sijup = (smallqlm[n][i] * np.conj(smallqlm[n][Neighborlist[i, j+1]])).sum()
                        sijdown = np.sqrt(np.square(np.abs(smallqlm[n][i])).sum()) * np.sqrt(np.square(np.abs(smallqlm[n][Neighborlist[i, j+1]])).sum())
                        sij[i, j] = np.real(sijup/sijdown)
                sijresults[:, 0] = np.arange(self.nparticle) + 1
                sijresults[:, 1] = (np.where(sij>c, 1, 0)).sum(axis=1)
                sijresults[:, 2] = np.where(sijresults[:, 1]>Neighborlist[:, 0]/2, 1, 0)
                results = np.vstack((results, sijresults))

            outputql = namestr + '.sij.smallq_l%d.dat'%self.l
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(self.l)
            np.savetxt(outputql, results[1:], fmt='%d', header=names, comments='')
            fneighbor.close()
            logger.info('Finish Calculating s(i, j) based on ql')

        if csijlargeQl:
            logger.info('Start Calculating s(i, j) based on Ql')
            MaxNeighbor = self.Nmax
            fneighbor = open(self.neighborfile, 'r', encoding="utf-8")
            results = np.zeros((1, 3))
            for n in range(self.nsnapshots):
                Neighborlist = read_neighbors(fneighbor, self.nparticle, self.Nmax)
                sij = np.zeros((self.nparticle, MaxNeighbor))
                sijresults = np.zeros((self.nparticle, 3))
                if (Neighborlist[:, 0] > MaxNeighbor).any():
                    raise ValueError('increase Nmax to include all neighbors')
                for i in range(self.nparticle):
                    for j in range(Neighborlist[i, 0]):
                        sijup = (largeQlm[n][i] * np.conj(largeQlm[n][Neighborlist[i, j+1]])).sum()
                        sijdown = np.sqrt(np.square(np.abs(largeQlm[n][i])).sum()) * np.sqrt(np.square(np.abs(largeQlm[n][Neighborlist[i, j+1]])).sum())
                        sij[i, j] = np.real(sijup / sijdown)
                sijresults[:, 0] = np.arange(self.nparticle) + 1
                sijresults[:, 1] = (np.where(sij > c, 1, 0)).sum(axis=1)
                sijresults[:, 2] = np.where(sijresults[:, 1] > Neighborlist[:, 0] / 2, 1, 0)
                results = np.vstack((results, sijresults))

            outputQl = namestr + '.sij.largeQ_l%d.dat'%self.l
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(self.l)
            np.savetxt(outputQl, results[1:], fmt='%d', header=names, comments='')
            fneighbor.close()
            logger.info('Finish Calculating s(i, j) based on Ql')

        if csmallwcap:
            logger.info('Start Calculating BOO w and normalized (cap) wcap')
            smallqlm = np.array(smallqlm)
            smallw = np.zeros((self.nsnapshots, self.nparticle))
            Windex = Wignerindex(self.l)
            w3j = Windex[:, 3]
            Windex = Windex[:, :3].astype(np.int) + self.l 
            for n in range(self.nsnapshots):
                for i in range(self.nparticle):
                    smallw[n, i] = (np.real(np.prod(smallqlm[n, i, Windex], axis=1))*w3j).sum()
           
            smallw = np.column_stack((np.arange(self.nparticle)+1, smallw.T))
            outputw = namestr + '.smallw_l%d.dat'%self.l
            names = 'id  wl  l=' + str(self.l)
            numformat = '%d ' + '%.10f ' * (len(smallw[0])-1)
            np.savetxt(outputw, smallw, fmt=numformat, header=names, comments='')
       
            smallwcap = np.power(np.square(np.abs(np.array(smallqlm))).sum(axis = 2), -3/2).T * smallw[:, 1:]
            smallwcap = np.column_stack((np.arange(self.nparticle)+1, smallwcap))
            outputwcap = namestr + '.smallwcap_l%d.dat'%self.l
            names = 'id  wlcap  l=' + str(self.l)
            numformat = '%d ' + '%.8f ' * (len(smallwcap[0])-1)
            np.savetxt(outputwcap, smallwcap, fmt=numformat, header=names, comments='')
            logger.info('Finish Calculating BOO w and normalized (cap) wcap')

        if clargeWcap:
            logger.info('Start Calculating BOO W and normalized (cap) Wcap')
            largeQlm = np.array(largeQlm)
            largew = np.zeros((self.nsnapshots, self.nparticle))
            Windex = Wignerindex(self.l)
            w3j    = Windex[:, 3]
            Windex = Windex[:, :3].astype(np.int) + self.l 
            for n in range(self.nsnapshots):
                for i in range(self.nparticle):
                    largew[n, i] = (np.real(np.prod(largeQlm[n, i, Windex], axis=1))*w3j).sum()

            largew = np.column_stack((np.arange(self.nparticle)+1, np.real(largew.T)))
            outputW = namestr + '.largeW_l%d.dat'%self.l
            names = 'id  Wl  l=' + str(self.l)
            numformat = '%d ' + '%.10f ' * (len(largew[0])-1)
            np.savetxt(outputW, largew, fmt=numformat, header=names, comments='')
       
            largewcap = np.power(np.square(np.abs(np.array(largeQlm))).sum(axis = 2), -3/2).T * largew[:, 1:]
            largewcap = np.column_stack((np.arange(self.nparticle)+1, largewcap))
            outputWcap = namestr + '.largeWcap_l%d.dat'%self.l
            names = 'id  Wlcap  l=' + str(self.l)
            numformat = '%d ' + '%.8f ' * (len(largewcap[0]) - 1)
            np.savetxt(outputWcap, largewcap, fmt=numformat, header=names, comments='')     
            logger.info('Finish Calculating BOO W and normalized (cap) Wcap')

        logger.info('Finish Calculating multiple order parameters together')
