# coding = utf-8

"""see documentation @ ../docs/pair_entropy.md"""

import numpy as np
import numexpr as ne
from math import pi
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
        gr_bins = np.arange(0, self.rmax, self.rdelta)[1:] - 0.5*self.rdelta
        
        if self.ppp.shape[0]==2:
            norms = 2*np.pi*gr_bins # 2d
        else:
            norms = 4*np.pi*np.square(gr_bins) # 3d

        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(self.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                distance = distance[distance<self.rmax]
                itype = int(snapshot.particle_type[i]-1)
                jtypes = (np.delete(snapshot.particle_type, i)-1).astype(np.int64)
                gr_i = 0
                for j, rij in enumerate(distance):
                    sigma = self.sigmas[itype, jtypes[j]]
                    gr_i += gaussian_smooth(gr_bins, rij, sigma)
                gr_i /= (self.rhotype[itype] * norms)
                self.s2_results[n, i] = -2 * np.pi * self.rhotype[itype] * s2_integral(gr_i, gr_bins)
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

