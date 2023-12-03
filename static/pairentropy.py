# coding = utf-8

"""see documentation @ ../docs/pair_entropy.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc
from utils.coarse_graining import time_average
from utils.funcs import nidealfac, areafac, gaussian_smooth
from utils.logging import get_logger_handle
from static.gr import conditional_gr

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=line-too-long

def s2_integral(gr: np.ndarray, gr_bins: np.ndarray)->float:
    y = (gr * np.log(gr) - gr + 1) * np.square(gr_bins)
    return np.trapz(y, gr_bins)


class S2:
    """
    Calculate pair entropy S2 by calculating particle g(r) at single snapshot and integral to get S2,
    and then average S2 over desired time scales.
    The code accounts for both orthogonal and triclinic cells, covering unary to quinary systems.
    """
    def __init__(
            self,
            snapshots: Snapshots,
            sigmas: np.ndarray,
            ppp: np.ndarray=np.array([1,1,1]),
            rdelta: float=0.01,
            rmax: int=10,
            outputfile: str=None
    ) -> None:
        """
        Initializing S2 class

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                         (returned by reader.dump_reader.DumpReader)
            2. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.ndarray=np.array([1,1,1]),
                                 set np.ndarray=np.array([1,1]) for two-dimensional systems
            3. rdelta (float): bin size calculating g(r), the default value is 0.01
            4. outputfile (str): the name of csv file to save the calculated S2

        Return:
            None
        """
        self.snapshots = snapshots
        self.sigmas = sigmas
        self.ppp = ppp
        self.rdelta = rdelta
        self.rmax = rmax
        self.outputfile = outputfile

        self.nsnapshots = self.snapshots.nsnapshots
        self.ndim = self.snapshots.snapshots[0].positions.shape[1]
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len({snapshot.nparticle for snapshot in self.snapshots.snapshots}) == 1,\
            "Paticle Number Changes during simulation"
        self.boxvolume = np.prod(self.snapshots.snapshots[0].boxlength)
        assert len({tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}) == 1,\
            "Simulation Box Length Changes during simulation"
        self.typenumber, self.typecount = np.unique(self.snapshots.snapshots[0].particle_type, return_counts=True)
        logger.info(f'System composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')
        assert np.sum(self.typecount) == self.nparticle,\
            "Sum of Indivdual Types is Not the Total Amount"

        self.rhototal = self.nparticle / self.boxvolume
        self.rhotype = self.typecount / self.boxvolume
        self.s2_results = np.zeros((self.nsnapshots, self.nparticle))

    def particle_s2(self):
        """
        calculate the particle-level g(r) by Gaussian smoothing
        """

        gr_bins = np.arange(int(self.rmax/self.rdelta))[1:]*self.rdelta
        if self.ppp.shape[0]==2:
            norms = 2*np.pi*gr_bins # 2d
        else:
            norms = 4*np.pi*np.square(gr_bins) # 3d

        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                itype = int(snapshot.particle_type[i]-1)
                jtypes = (np.delete(snapshot.particle_type, i)-1).astype(np.int64)
                gr_i = 0
                for j, rij in enumerate(distance):
                    sigma = self.sigmas[itype, jtypes[j]]
                    gr_i += gaussian_smooth(gr_bins, rij, sigma)
                gr_i /= self.rhotype[itype] * norms
                self.s2_results[n, i] = s2_integral(gr_i, gr_bins)
        return self.s2_results

    def timeave(
        self,
        time_period: float=0.0,
        dt: float=0.002,
        outputS2: str=None
    ) -> np.ndarray:
        """
        Calculate Time Averaged S2 or instantaneous ones

        Inputs:
            1. time_period (float): time used to average, default 0.0
            2. dt (float): timestep used in user simulations, default 0.002
            3. outputS2 (str): file name for S2 results, default None
                               To reduce storage size and ensure loading speed, save npy file as default with extension ".npy".
                               If the file extension is ".dat" or ".txt", it is saved as a test file.
        
        Return:
            Calculated time averaged S2 (np.ndarray)
            shape [nsnapshots_updated, nparticles]
        """
        if len(self.typenumber) == 1:
            S2results = self.unary()
        if len(self.typenumber) == 2:
            S2results = self.binary()
        if len(self.typenumber) == 3:
            S2results = self.ternary()
        if len(self.typenumber) == 4:
            S2results = self.quarternary()
        if len(self.typenumber) == 5:
            S2results = self.quinary()
        if len(self.typenumber) > 5:
            S2results = self.unary()

        results = time_average(snapshots=self.snapshots, input_property=S2results, time_period=time_period, dt=dt)[0]

        if outputS2:
            if outputS2[-3:] == 'npy':
                np.save(outputS2, results)
            elif outputS2[-3:] == ('dat' or 'txt'):
                np.savetxt(outputS2, results, fmt='%.6f', header="", comments="")
            else:
                logger.info('The default format of outputfile is binary npy with extension "npy" and also supports text file with extension "dat" or "txt"')
        return results

    def spatial_corr(
        self,
        timeaveS2: np.ndarray,
        outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate Spatial Correlation of Time Averaged or individual S2

        Inputs:
            1. timeaveS2 (np.ndarray): calculated time averaged S2
            2. outputfile (str): csv file name for gl, default None
        """
        glresults = 0
        for n, snapshot in enumerate(self.snapshots.snapshots[:timeaveS2.shape[0]]):
            glresults += conditional_gr(
                snapshot=snapshot,
                condition=timeaveS2[n],
                conditiontype=None,
                ppp=self.ppp,
                rdelta=self.rdelta
            )
        glresults /= self.nsnapshots
        if outputfile:
            glresults.to_csv(outputfile, float_format="%.8f", index=False)

        logger.info('Finish calculating spatial correlation of S2')
        return glresults

    def unary(self) -> np.ndarray:
        """
        Calculating pair entropy S2 for unary system
        
        Inputs:
            None

        Return:
            calculated S2 (np.ndarray)
        """
        logger.info('Start calculating S2 of a unary system')

        S2results = np.zeros((self.nsnapshots, self.nparticle))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particlegr = np.zeros(self.maxbin)
                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                binleft = binedge[:-1]
                binright = binedge[1:]
                nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr = countvalue / nideal / self.rhototal
                integralgr = (particlegr*np.log(particlegr+1e-12)-(particlegr-1)) * self.rhototal
                binright -= 0.5 * self.rdelta
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr*self.rdelta)
        logger.info('Finish calculating S2 of a unary system')
        return S2results

    def binary(self) -> np.ndarray:
        """
        Calculating pair entropy S2 for binary system
        
        Inputs:
            None

        Return:
            calculated S2 (np.ndarray)
        """
        logger.info('Start calculating S2 of a binary system')

        S2results = np.zeros((self.nsnapshots, self.nparticle))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                particlegr = np.zeros((self.maxbin, 3))
                usedrho = np.zeros(3)
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                binleft = binedge[:-1]
                binright = binedge[1:]
                nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[:, 0] = countvalue / nideal / self.rhotype[0]
                usedrho[0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]] - 1
                particlegr[:, 1] = countvalue / nideal / self.rhotype[rho12]
                usedrho[1] = self.rhotype[rho12]
                countvalue, binedge = np.histogram(distance[countsum==4], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 2] = countvalue / nideal / self.rhotype[1]
                usedrho[2] = self.rhotype[1]
                integralgr = (particlegr*np.log(particlegr+1e-12)-(particlegr-1)) * usedrho[np.newaxis, :]
                integralgr = integralgr[:, np.any(particlegr, axis=0)]
                binright -= 0.5 * self.rdelta
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results

    def ternary(self) -> np.ndarray:
        """
        Calculating pair entropy S2 for ternary system
        
        Inputs:
            None

        Return:
            calculated S2 (np.ndarray)
        """
        logger.info('Start calculating S2 of a ternary system')

        S2results = np.zeros((self.nsnapshots, self.nparticle))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0] - TIJ[:, 1])

                particlegr = np.zeros((self.maxbin, 6))
                usedrho = np.zeros(6)
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                binleft = binedge[:-1]
                binright = binedge[1:]
                nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[:, 0] = countvalue / nideal / self.rhotype[0]
                usedrho[0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 1] = countvalue / nideal / self.rhotype[1]
                usedrho[1] = self.rhotype[1]
                countvalue, binedge = np.histogram(distance[countsum==6], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 2] = countvalue / nideal / self.rhotype[2]
                usedrho[2] = self.rhotype[2]

                if snapshot.particle_type[i] != 3:
                    countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]][0] - 1
                    particlegr[:, 3] = countvalue / nideal / self.rhotype[rho12]
                    usedrho[3] = self.rhotype[rho12]

                if snapshot.particle_type[i] != 1:
                    countvalue, binedge = np.histogram(distance[countsum==5], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho23 = self.typenumber[self.typenumber != snapshot.particle_type[i]][1] - 1
                    particlegr[:, 4] = countvalue / nideal / self.rhotype[rho23]
                    usedrho[4] = self.rhotype[rho23]

                if snapshot.particle_type[i] != 2:
                    countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]]
                    rho13 = medium[medium != 2] - 1
                    particlegr[:, 5] = countvalue / nideal / self.rhotype[rho13]
                    usedrho[5] = self.rhotype[rho13]

                integralgr = (particlegr*np.log(particlegr+1e-12)-(particlegr-1)) * usedrho[np.newaxis, :]
                integralgr = integralgr[:, np.any(particlegr, axis=0)]
                binright -= 0.5 * self.rdelta
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results

    def quarternary(self) -> np.ndarray:
        """
        Calculating pair entropy S2 for quarternary system
        
        Inputs:
            None

        Return:
            calculated S2 (np.ndarray)
        """
        logger.info('Start calculating S2 of a quarternary system')

        S2results = np.zeros((self.nsnapshots, self.nparticle))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])

                particlegr = np.zeros((self.maxbin, 10))
                usedrho = np.zeros(10)
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                binleft = binedge[:-1]
                binright = binedge[1:]
                nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[:, 0] = countvalue / nideal / self.rhotype[0]
                usedrho[0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 1] = countvalue / nideal / self.rhotype[1]
                usedrho[1] = self.rhotype[1]
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 2] = countvalue / nideal / self.rhotype[2]
                usedrho[2] = self.rhotype[2]
                countvalue, binedge = np.histogram(distance[countsum==8], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 3] = countvalue / nideal / self.rhotype[3]
                usedrho[3] = self.rhotype[3]

                if snapshot.particle_type[i] in [1, 2]:
                    countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]][0] - 1
                    particlegr[:, 4] = countvalue / nideal / self.rhotype[rho12]
                    usedrho[4] = self.rhotype[rho12]

                if snapshot.particle_type[i] in [1, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][:2]
                    rho13 = medium[medium!=2] - 1
                    particlegr[:, 5] = countvalue / nideal / self.rhotype[rho13]
                    usedrho[5] = self.rhotype[rho13]

                if snapshot.particle_type[i] in [1, 4]:
                    countvalue, binedge = np.histogram(distance[countsub==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]]
                    rho14 = medium[(medium!=2)&(medium!=3)] - 1
                    particlegr[:, 6] = countvalue / nideal / self.rhotype[rho14]
                    usedrho[6] = self.rhotype[rho14]

                if snapshot.particle_type[i] in [2, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho23 = self.typenumber[self.typenumber != snapshot.particle_type[i]][1] - 1
                    particlegr[:, 7] = countvalue / nideal / self.rhotype[rho23]
                    usedrho[7] = self.rhotype[rho23]

                if snapshot.particle_type[i] in [2, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][1:]
                    rho24 = medium[medium!=3] - 1
                    particlegr[:, 8] = countvalue / nideal / self.rhotype[rho24]
                    usedrho[8] = self.rhotype[rho24]

                if snapshot.particle_type[i] in [3, 4]:
                    countvalue, binedge = np.histogram(distance[countsum==7], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho34 = self.typenumber[self.typenumber != snapshot.particle_type[i]][-1] - 1
                    particlegr[:, 9] = countvalue / nideal / self.rhotype[rho34]
                    usedrho[9] = self.rhotype[rho34]

                integralgr = (particlegr*np.log(particlegr+1e-12)-(particlegr-1))*usedrho[np.newaxis, :]
                integralgr = integralgr[:, np.any(particlegr, axis=0)]
                binright -= 0.5 * self.rdelta
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results

    def quinary(self) -> np.ndarray:
        """
        Calculating pair entropy S2 for quinary system
        
        Inputs:
            None
        
        Return:
            calculated S2 (np.ndarray)
        """
        logger.info('Start calculating S2 of a quinary system')

        S2results = np.zeros((self.nsnapshots, self.nparticle))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])

                particlegr = np.zeros((self.maxbin, 15)) 
                usedrho = np.zeros(15)
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                binleft = binedge[:-1]
                binright = binedge[1:]
                nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[:, 0] = countvalue / nideal / self.rhotype[0]
                usedrho[0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 1] = countvalue / nideal / self.rhotype[1]
                usedrho[1] = self.rhotype[1]
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 2] = countvalue / nideal / self.rhotype[2]
                usedrho[2] = self.rhotype[2]
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 3] = countvalue / nideal / self.rhotype[3]
                usedrho[3] = self.rhotype[3]
                countvalue, binedge = np.histogram(distance[countsum==10], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[:, 4] = countvalue / nideal / self.rhotype[4]
                usedrho[4] = self.rhotype[4]

                if snapshot.particle_type[i] in [1, 2]:
                    countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]][0] - 1
                    particlegr[:, 5] = countvalue / nideal / self.rhotype[rho12]
                    usedrho[5] = self.rhotype[rho12]

                if snapshot.particle_type[i] in [1, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][:2]
                    rho13 = medium[medium!=2] - 1
                    particlegr[:, 6] = countvalue / nideal / self.rhotype[rho13]
                    usedrho[6] = self.rhotype[rho13]

                if snapshot.particle_type[i] in [1, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==5) & (countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][:3]
                    rho14 = medium[(medium != 2) & (medium != 3)] - 1
                    particlegr[:, 7] = countvalue / nideal / self.rhotype[rho14]
                    usedrho[7] = self.rhotype[rho14]

                if snapshot.particle_type[i] in [1, 5]:
                    countvalue, binedge = np.histogram(distance[(countsum==6) & (countsub==4)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]]
                    rho15 = medium[(medium != 2) & (medium != 3) & (medium != 4)] - 1
                    particlegr[:, 8] = countvalue / nideal / self.rhotype[rho15]
                    usedrho[8] = self.rhotype[rho15]

                if snapshot.particle_type[i] in [2, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==5) & (countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho23 = self.typenumber[self.typenumber != snapshot.particle_type[i]][1] - 1
                    particlegr[:,9] = countvalue / nideal / self.rhotype[rho23]
                    usedrho[9] = self.rhotype[rho23]

                if snapshot.particle_type[i] in [2, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==6) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][1:3]
                    rho24 = medium[medium != 3] - 1
                    particlegr[:,10] = countvalue / nideal / self.rhotype[rho24]
                    usedrho[10] = self.rhotype[rho24]

                if snapshot.particle_type[i] in [2, 5]:
                    countvalue, binedge = np.histogram(distance[(countsum==7) & (countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][1:]
                    rho25 = medium[(medium != 3) & (medium != 4)] - 1
                    particlegr[:,11] = countvalue / nideal / self.rhotype[rho25]
                    usedrho[11] = self.rhotype[rho25]

                if snapshot.particle_type[i] in [3, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==7) & (countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho34 = self.typenumber[self.typenumber != snapshot.particle_type[i]][2] - 1
                    particlegr[:,12] = countvalue / nideal / self.rhotype[rho34]
                    usedrho[12] = self.rhotype[rho34]

                if snapshot.particle_type[i] in [3, 5]:
                    countvalue, binedge = np.histogram(distance[(countsum==8) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][2:]
                    rho35 = medium[medium != 4] - 1
                    particlegr[:,13] = countvalue / nideal / self.rhotype[rho35]
                    usedrho[13] = self.rhotype[rho35]

                if snapshot.particle_type[i] in [4, 5]:
                    countvalue, binedge = np.histogram(distance[countsum==9], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho45 = self.typenumber[self.typenumber != snapshot.particle_type[i]][-1] - 1
                    particlegr[:,14] = countvalue / nideal / self.rhotype[rho45]
                    usedrho[14] = self.rhotype[rho45]

                integralgr = (particlegr*np.log(particlegr+1e-12)-(particlegr-1)) * usedrho[np.newaxis, :]
                integralgr = integralgr[:, np.any(particlegr, axis=0)]
                binright -= 0.5 * self.rdelta
                S2results[i, n] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results


class S2AVE:
    """
    Calculate pair entropy S2 by averaging particle g(r) over desired time scales
    and then integral to calculate particle S2.
    The code accounts for both orthogonal and triclinic cells, covering unary to quinary systems.
    """

    def __init__(
        self,
        snapshots: Snapshots,
        time_period: float,
        dt: float=0.002,
        ppp: np.ndarray=np.array([1,1,1]),
        rdelta: float=0.01,
        outputfile: str=None
    ) -> None:
        """
        Initializing S2AVE class

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                         (returned by reader.dump_reader.DumpReader)
            2. time_period (float): time used to average, default 0.0
            3. dt (float): timestep used in user simulations, default 0.002
            4. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.ndarray=np.array([1,1,1]),
                                 set np.ndarray=np.array([1,1]) for two-dimensional systems
            5. rdelta (float): bin size calculating g(r), the default value is 0.01
            6. outputfile (str): the name of csv file to save the calculated S2
        
        Return:
            None
        """
        self.snapshots = snapshots
        self.time_period = time_period
        self.dt = dt
        self.ppp = ppp
        self.rdelta = rdelta
        self.outputfile = outputfile

        self.timestep = self.snapshots.snapshots[1].timestep - self.snapshots.snapshots[0].timestep
        self.nsnapshots = self.snapshots.nsnapshots
        self.ndim = self.snapshots.snapshots[0].positions.shape[1]
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len({snapshot.nparticle for snapshot in self.snapshots.snapshots}) == 1,\
            "Paticle Number Changes during simulation"
        self.boxvolume = np.prod(self.snapshots.snapshots[0].boxlength)
        assert len({tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}) == 1,\
            "Simulation Box Length Changes during simulation"
        self.typenumber, self.typecount = np.unique(self.snapshots.snapshots[0].particle_type, return_counts=True)
        logger.info(f'System composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')
        assert np.sum(self.typecount) == self.nparticle,\
            "Sum of Indivdual Types is Not the Total Amount"
        
        self.nidealfac = nidealfac(self.ndim)
        self.areafac = areafac(self.ndim)
        self.rhototal = self.nparticle / self.boxvolume
        self.rhotype = self.typecount / self.boxvolume
        self.maxbin = int(self.snapshots.snapshots[0].boxlength.min()/2.0/self.rdelta)

    def getS2(
        self,
        outputS2: str=None
    ) -> np.ndarray:
        """
        Get Particle-level S2 by averaging particle gr over different snapshots

        Inputs:
            1. outputS2 (str): file name for S2 results, default None
                               To reduce storage size and ensure loading speed, save npy file as default with extension ".npy".
                               If the file extension is ".dat" or ".txt", it is saved as a test file.

        Return:
            calculated S2 (np.ndarray)
            shape [nsnapshots, nparticle]
        """
        if len(self.typenumber) == 1:
            S2results = self.unary()
        if len(self.typenumber) == 2:
            S2results = self.binary()
        if len(self.typenumber) == 3:
            S2results = self.ternary()
        if len(self.typenumber) == 4:
            S2results = self.quarternary()
        if len(self.typenumber) == 5:
            S2results = self.quinary()
        if len(self.typenumber) > 5:
            S2results = self.unary()

        if outputS2:
            if outputS2[-3:] == 'npy':
                np.save(outputS2, S2results)
            elif outputS2[-3:] == ('dat' or 'txt'):
                np.savetxt(outputS2, S2results, fmt='%.6f', header="", comments="")
            else:
                logger.info('The default format of outputfile is binary npy with extension "npy" and also supports text file with extension "dat" or "txt"')
        return S2results
    
    def spatial_corr(
        self,
        timeaveS2: np.ndarray,
        outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate Spatial Correlation of Time Averaged or individual S2

        Inputs:
            1. timeaveS2 (np.ndarray): calculated time averaged S2
            2. outputfile (str): csv file name for gl, default None
        """
        glresults = 0
        for n, snapshot in enumerate(self.snapshots.snapshots[:timeaveS2.shape[0]]):
            glresults += conditional_gr(
                snapshot=snapshot,
                condition=timeaveS2[n],
                conditiontype=None,
                ppp=self.ppp,
                rdelta=self.rdelta
            )
        glresults /= self.nsnapshots
        if outputfile:
            glresults.to_csv(outputfile, float_format="%.8f", index=False)

        logger.info('Finish calculating spatial correlation of S2')
        return glresults

    def unary(self) -> np.ndarray:
        avetime = int(self.time_period/self.dt/self.timestep)
        S2results = np.zeros((self.nsnapshots-avetime, self.nparticle))
        particlegr = np.zeros((self.nsnapshots, self.nparticle, self.maxbin))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                if n == 0 and i == 0:
                    binleft = binedge[:-1]
                    binright = binedge[1:]
                    nideal = nidealfac(self.ndim) * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[n, i] = countvalue / nideal / self.rhototal
        
        binright -= 0.5 * self.rdelta
        for n in range(self.nsnapshots - avetime):
            for i in range(self.nparticle):
                aveparticlegr = particlegr[i, n:n+avetime].mean(axis=0)
                integralgr = (aveparticlegr*np.log(aveparticlegr+1e-12)-(aveparticlegr-1)) * self.rhototal
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr*self.rdelta)

        return S2results

    def binary(self) -> np.ndarray:
        avetime = int(self.time_period/self.dt/self.timestep)
        S2results = np.zeros((self.nsnapshots-avetime, self.nparticle))
        particlegr = np.zeros((self.nsnapshots, self.nparticle, self.maxbin, 3))
        usedrho = np.zeros((self.nsnapshots, self.nparticle, 3))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                if n == 0 and i == 0:
                    binleft = binedge[:-1]
                    binright = binedge[1:]
                    nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[n, i, :, 0]  = countvalue / nideal / self.rhotype[0]
                usedrho[n, i, 0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]] - 1
                particlegr[n, i, :, 1]  = countvalue / nideal / self.rhotype[rho12]
                usedrho[n, i, 1] = self.rhotype[rho12]
                countvalue, binedge = np.histogram(distance[countsum==4], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 2] = countvalue / nideal / self.rhotype[1]
                usedrho[n, i, 2] = self.rhotype[1]

        binright -= 0.5 * self.rdelta
        for n in range(self.nsnapshots-avetime):
            for i in range(self.nparticle):
                aveparticlegr = particlegr[n:n+avetime, i].mean(axis=0)
                integralgr = (aveparticlegr*np.log(aveparticlegr+1e-12)-(aveparticlegr-1))*usedrho[n, i][np.newaxis, :]
                integralgr = integralgr[:, np.any(aveparticlegr, axis=0)]
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results

    def ternary(self) -> np.ndarray:
        avetime = int(self.time_period/self.dt/self.timestep)
        S2results = np.zeros((self.nsnapshots-avetime, self.nparticle))
        particlegr = np.zeros((self.nsnapshots, self.nparticle, self.maxbin, 6))
        usedrho = np.zeros((self.nsnapshots, self.nparticle, 6))
        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])

                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                if n == 0 and i == 0:
                    binleft = binedge[:-1]
                    binright = binedge[1:]
                    nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[n, i, :, 0] = countvalue / nideal / self.rhotype[0]
                usedrho[n, i, 0]       = self.rhotype[0]
                countvalue, binedge    = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 1] = countvalue / nideal / self.rhotype[1]
                usedrho[n, i, 1]       = self.rhotype[1]
                countvalue, binedge    = np.histogram(distance[countsum==6], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 2] = countvalue / nideal / self.rhotype[2]
                usedrho[n, i, 2]       = self.rhotype[2]

                if snapshot.particle_type[i] != 3:
                    countvalue, binedge    = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho12                  = self.typenumber[self.typenumber != snapshot.particle_type[i]][0] - 1
                    particlegr[n, i, :, 3] = countvalue / nideal / self.rhotype[rho12]
                    usedrho[n, i, 3]       = self.rhotype[rho12]

                if snapshot.particle_type[i] != 1:
                    countvalue, binedge = np.histogram(distance[countsum==5], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho23 = self.typenumber[self.typenumber != snapshot.particle_type[i]][1] - 1
                    particlegr[n, i, :, 4] = countvalue / nideal / self.rhotype[rho23]
                    usedrho[n, i, 4]       = self.rhotype[rho23]

                if snapshot.particle_type[i] != 2:
                    countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]]
                    rho13 = medium[medium != 2] - 1
                    particlegr[n, i, :, 5] = countvalue / nideal / self.rhotype[rho13]
                    usedrho[n, i, 5] = self.rhotype[rho13]

        binright -= 0.5 * self.rdelta
        for n in range(self.nsnapshots-avetime):
            for i in range(self.nparticle):
                aveparticlegr = particlegr[n:n+avetime, i].mean(axis=0)
                integralgr = (aveparticlegr*np.log(aveparticlegr+1e-12)-(aveparticlegr-1))*usedrho[n, i][np.newaxis, :]
                integralgr = integralgr[:, np.any(aveparticlegr, axis=0)]
                S2results[n, i] = -0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results

    def quarternary(self) -> np.ndarray:
        avetime = int(self.time_period/self.dt/self.timestep)
        S2results = np.zeros((self.nsnapshots-avetime, self.nparticle))
        particlegr = np.zeros((self.nsnapshots, self.nparticle, self.maxbin, 10))
        usedrho = np.zeros((self.nsnapshots, self.nparticle, 10))

        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])

                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                if n == 0 and i == 0:
                    binleft = binedge[:-1]
                    binright = binedge[1:]
                    nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[n, i, :, 0] = countvalue / nideal / self.rhotype[0]
                usedrho[n, i, 0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub == 0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 1] = countvalue / nideal / self.rhotype[1]
                usedrho[n, i, 1] = self.rhotype[1]
                countvalue, binedge    = np.histogram(distance[(countsum==6) & (countsub == 0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 2] = countvalue / nideal / self.rhotype[2]
                usedrho[n, i, 2] = self.rhotype[2]
                countvalue, binedge = np.histogram(distance[countsum==8], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 3] = countvalue / nideal / self.rhotype[3]
                usedrho[n, i, 3] = self.rhotype[3]

                if snapshot.particle_type in [1, 2]:
                    countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]][0] - 1
                    particlegr[n, i, :, 4] = countvalue / nideal / self.rhotype[rho12] #12
                    usedrho[n, i, 4] = self.rhotype[rho12]

                if snapshot.particle_type in [1, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][:2]
                    rho13 = medium[medium!=2] - 1
                    particlegr[n, i, :, 5] = countvalue / nideal / self.rhotype[rho13] #13
                    usedrho[n, i, 5] = self.rhotype[rho13]

                if snapshot.particle_type in [1, 4]:
                    countvalue, binedge = np.histogram(distance[countsub==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]]
                    rho14 = medium[(medium!=2) & (medium!=3)] - 1
                    particlegr[n, i, :, 6] = countvalue / nideal / self.rhotype[rho14] #14
                    usedrho[n, i, 6] = self.rhotype[rho14]

                if snapshot.particle_type in [2, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho23 = self.typenumber[self.typenumber!=snapshot.particle_type[i]][1] - 1
                    particlegr[n, i, :, 7] = countvalue / nideal / self.rhotype[rho23] #23
                    usedrho[n, i, 7] = self.rhotype[rho23]

                if snapshot.particle_type in [2, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber!=snapshot.particle_type[i]][1:]
                    rho24 = medium[medium!=3] - 1
                    particlegr[n, i, :, 8] = countvalue / nideal / self.rhotype[rho24]
                    usedrho[n, i, 8] = self.rhotype[rho24]

                if snapshot.particle_type in [3, 4]:
                    countvalue, binedge = np.histogram(distance[countsum==7], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho34 = self.typenumber[self.typenumber != snapshot.particle_type[i]][-1] - 1
                    particlegr[n, i, :, 9] = countvalue / nideal / self.rhotype[rho34]
                    usedrho[n, i, 9] = self.rhotype[rho34]

        binright -= 0.5 * self.rdelta
        for n in range(self.nsnapshots - avetime):
            for i in range(self.nparticle):
                aveparticlegr = particlegr[i, n:n+avetime].mean(axis = 0)
                integralgr = (aveparticlegr*np.log(aveparticlegr+1e-12)-(aveparticlegr-1))*usedrho[i, n][np.newaxis, :]
                integralgr = integralgr[:, np.any(aveparticlegr, axis=0)]
                S2results[i, n] =-0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results

    def quinary(self) -> np.ndarray:
        avetime = int(self.time_period/self.dt/self.timestep)
        S2results = np.zeros((self.nsnapshots-avetime, self.nparticle))
        particlegr = np.zeros((self.nsnapshots, self.nparticle, self.maxbin, 15))
        usedrho = np.zeros((self.nsnapshots, self.nparticle, 15))

        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                particletype = np.delete(snapshot.particle_type, i)
                TIJ = np.c_[particletype, np.zeros_like(particletype)+snapshot.particle_type[i]]
                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])

                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                if n == 0 and i == 0:
                    binleft = binedge[:-1]
                    binright = binedge[1:]
                    nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
                particlegr[n, i, :, 0] = countvalue / nideal / self.rhotype[0] #11
                usedrho[n, i, 0] = self.rhotype[0]
                countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 1] = countvalue / nideal / self.rhotype[1] #22
                usedrho[n, i, 1] = self.rhotype[1]
                countvalue, binedge = np.histogram(distance[(countsum==6) & (countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 2] = countvalue / nideal / self.rhotype[2] #33
                usedrho[n, i, 2] = self.rhotype[2]
                countvalue, binedge = np.histogram(distance[(countsum==8) & (countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 3] = countvalue / nideal / self.rhotype[3] #44
                usedrho[n, i, 3] = self.rhotype[3]
                countvalue, binedge = np.histogram(distance[countsum==10], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                particlegr[n, i, :, 4] = countvalue / nideal / self.rhotype[4] #55
                usedrho[n, i, 4] = self.rhotype[4]

                if snapshot.particle_type[i] in [1, 2]:
                    countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho12 = self.typenumber[self.typenumber != snapshot.particle_type[i]][0] - 1
                    particlegr[n, i, :, 5] = countvalue / nideal / self.rhotype[rho12] #12
                    usedrho[n, i, 5] = self.rhotype[rho12]

                if snapshot.particle_type[i] in [1, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][:2]
                    rho13 = medium[medium != 2] - 1
                    particlegr[n, i, :, 6] = countvalue / nideal / self.rhotype[rho13] #13
                    usedrho[n, i, 6] = self.rhotype[rho13]

                if snapshot.particle_type[i] in [1, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][:3]
                    rho14 = medium[(medium != 2) & (medium != 3)] - 1
                    particlegr[n, i, :, 7] = countvalue / nideal / self.rhotype[rho14] #14
                    usedrho[n, i, 7] = self.rhotype[rho14]

                if snapshot.particle_type[i] in [1, 5]:
                    countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==4)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]]
                    rho15 = medium[(medium != 2) & (medium != 3) & (medium != 4)] - 1
                    particlegr[n, i, :, 8] = countvalue / nideal / self.rhotype[rho15] #15
                    usedrho[n, i, 8] = self.rhotype[rho15]

                if snapshot.particle_type[i] in [2, 3]:
                    countvalue, binedge = np.histogram(distance[(countsum==5) & (countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho23 = self.typenumber[self.typenumber != snapshot.particle_type[i]][1] - 1
                    particlegr[n, i, :,9] = countvalue / nideal / self.rhotype[rho23] #23
                    usedrho[n, i, 9] = self.rhotype[rho23]

                if snapshot.particle_type[i] in [2, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==6) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][1:3]
                    rho24 = medium[medium != 3] - 1
                    particlegr[n, i, :,10] = countvalue / nideal / self.rhotype[rho24] #24
                    usedrho[n, i, 10] = self.rhotype[rho24]

                if snapshot.particle_type[i] in [2, 5]:
                    countvalue, binedge = np.histogram(distance[(countsum==7) & (countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][1:]
                    rho25 = medium[(medium != 3) & (medium != 4)] - 1
                    particlegr[n, i, :,11] = countvalue / nideal / self.rhotype[rho25] #25
                    usedrho[n, i, 11] = self.rhotype[rho25]

                if snapshot.particle_type[i] in [3, 4]:
                    countvalue, binedge = np.histogram(distance[(countsum==7) & (countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho34 = self.typenumber[self.typenumber != snapshot.particle_type[i]][2] - 1
                    particlegr[n, i, :,12] = countvalue / nideal / self.rhotype[rho34] #34
                    usedrho[n, i, 12] = self.rhotype[rho34]

                if snapshot.particle_type[i] in [3, 5]:
                    countvalue, binedge = np.histogram(distance[(countsum==8) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    medium = self.typenumber[self.typenumber != snapshot.particle_type[i]][2:]
                    rho35 = medium[medium != 4] - 1
                    particlegr[n, i, :,13] = countvalue / nideal / self.rhotype[rho35] #35
                    usedrho[n, i, 13] = self.rhotype[rho35]

                if snapshot.particle_type[i] in [4, 5]:
                    countvalue, binedge   = np.histogram(distance[countsum==9], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                    rho45 = self.typenumber[self.typenumber != snapshot.particle_type[i]][-1] - 1
                    particlegr[n, i, :,14] = countvalue / nideal / self.rhotype[rho45] #45
                    usedrho[n, i, 14] = self.rhotype[rho45]

        binright -= 0.5 * self.rdelta
        for n in range(self.nsnapshots - avetime):
            for i in range(self.nparticle):
                aveparticlegr = particlegr[i, n:n+avetime].mean(axis=0)
                integralgr = (aveparticlegr*np.log(aveparticlegr+1e-12)-(aveparticlegr-1))*usedrho[i, n][np.newaxis, :]
                integralgr = integralgr[:, np.any(aveparticlegr, axis=0)] 
                S2results[n, i] =-0.5 * np.sum(self.areafac*np.pi*binright**(self.ndim-1)*integralgr.sum(axis=1)*self.rdelta)

        return S2results
