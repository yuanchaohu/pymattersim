# coding = utf-8

"""see documentation @ ../docs/orderings.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc
from utils.funcs import gaussian_smooth
from utils.logging import get_logger_handle
from static.gr import conditional_gr
from dynamic.time_corr import time_correlation

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=line-too-long

def s2_integral(gr:np.ndarray, gr_bins:np.ndarray, ndim:int=3)->float:
    """
    spatial integration of derivative g(r) product to get S2

    Inputs:
        1. gr (np.ndarray): g(r) results, supporting both particle-level gr and system gr
        2. gr_bins (np.ndarray): r range
        3. ndim (int): dimensionality of system, default 3

    Return:
        integral results to get S2 (float)
    """
    y = gr * np.log(gr) - gr + 1
    y *= np.power(gr_bins, ndim-1)
    return np.trapz(y, gr_bins)


class S2:
    """
    Calculate pair entropy S2 by calculating particle g(r) at single snapshot and integral to get S2.
    The code accounts for both orthogonal and triclinic cells
    """
    def __init__(
        self,
        snapshots: Snapshots,
        sigmas: np.ndarray,
        ppp: np.ndarray=np.array([1,1,1]),
        rdelta: float=0.02,
        ndelta: int=500,
    ) -> None:
        """
        Initializing S2 instance

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                         (returned by reader.dump_reader.DumpReader)
            2. sigmas (np.ndarray): gaussian standard deviation for each pair particle type
            3. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.ndarray=np.array([1,1,1]),
                                 set np.ndarray=np.array([1,1]) for two-dimensional systems
            4. rdelta (float): bin size calculating g(r), the default value is 0.02
            5. ndelta (int): number of bins for g(r) calculation, ndelta*rdelta determines the range

        Return:
            None
        """
        self.snapshots = snapshots
        self.sigmas = sigmas
        self.ppp = ppp
        self.rdelta = rdelta
        self.ndelta = ndelta

        self.ndim = ppp.shape[0]
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
        self.s2_results = 0

    def particle_s2(
        self,
        savegr: bool=False,
        outputfile: str=None
    ) -> np.ndarray:
        """
        Calculate the particle-level g(r) by Gaussian smoothing
        and then calculate the particle-level S2

        Inputs:
            1. savegr (bool): whether save particle g(r), default false
            2. outputfile (str): the name of csv file to save the calculated S2

        Return:
            particle level S2 in shape [nsnapshots, nparticle]
        """
        logger.info('Start calculating particle S2 in {self.ndim} dimensionality')
        s2_results = np.zeros((self.snapshots.nsnapshots, self.nparticle))
        gr_bins = np.arange(self.ndelta)*self.rdelta + self.rdelta/2
        rmax = gr_bins.max()
        if savegr:
            particle_gr = np.zeros((self.snapshots.nsnapshots, self.nparticle, self.ndelta))

        if self.ppp.shape[0]==2:
            norms = 2*gr_bins*self.rhototal*np.pi
        elif self.ppp.shape[0]==3:
            norms = 4*np.square(gr_bins)*self.rhototal*np.pi
        else:
            raise ValueError("Input dimension is not 2 or 3")

        for n, snapshot in enumerate(self.snapshots.snapshots):
            for i in range(snapshot.nparticle):
                RIJ = np.delete(snapshot.positions, i, axis=0) - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                distance = distance[distance<rmax]
                itype = int(snapshot.particle_type[i]-1)
                jtypes = (np.delete(snapshot.particle_type, i)-1).astype(np.int64)
                gr_i = 0
                for j, rij in enumerate(distance):
                    sigma = self.sigmas[itype, jtypes[j]]
                    gr_i += gaussian_smooth(gr_bins, rij, sigma)
                gr_i /= norms

                if savegr:
                    particle_gr[n, i, :] = gr_i

                s2_results[n, i] = -(self.ndim-1)*np.pi*self.rhototal*s2_integral(gr_i, gr_bins, self.ndim)
        self.s2_results = s2_results
        if outputfile:
            np.save(outputfile, s2_results)
        if savegr:
            np.save('particle_gr.'+outputfile, particle_gr)
            return s2_results, particle_gr
        logger.info('Finish calculating particle S2 in {self.ndim} dimensionality')
        return s2_results

    def spatial_corr(
        self,
        mean_norm: bool=False,
        outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate Spatial Correlation of S2

        Inputs:
            1. mean_norm (bool): whether use mean normalized S2
            2. outputfile (str): csv file name for gl, default None

        Return:
            calculated Gl(r) based on S2 (pd.DataFrame)
        """
        logger.info('Start calculating spatial correlation of S2 in {self.ndim} dimensionality')
        glresults = 0
        
        for n, snapshot in enumerate(self.snapshots.snapshots):
            if mean_norm:
                s2_snapshot = self.s2_results[n] / np.mean(self.s2_results[n])
            else:
                s2_snapshot = self.s2_results[n]
            glresults += conditional_gr(
                snapshot=snapshot,
                condition=s2_snapshot,
                conditiontype=None,
                ppp=self.ppp,
                rdelta=self.rdelta
            )
        glresults /= self.snapshots.nsnapshots
        if outputfile:
            glresults.to_csv(outputfile, float_format="%.8f", index=False)

        logger.info('Finish calculating spatial correlation of S2 in {self.ndim} dimensionality')
        return glresults

    def time_corr(
        self,
        dt: float=0.002,
        outputfile: str=None
    ) -> pd.DataFrame:
        """
        Calculate time correlation of S2

        Inputs:
            1. dt (float): timestep used in user simulations, default 0.002
            2. outputfile (str): csv file name for time correlation results, default None

        Return:
            time correlation of S2 (pd.DataFrame)
        """
        logger.info(f'Start calculating time correlation of S2 in {self.ndim} dimensionality')

        gl_time = time_correlation(
            snapshots=self.snapshots,
            condition=self.s2_results,
            dt=dt
        )

        # normalization
        gl_time["time_corr"] /= gl_time.loc[0, "time_corr"]
        if outputfile:
            gl_time.to_csv(outputfile, float_format="%.6f", index=False)

        logger.info(f'Finish calculating time correlation of S2 in {self.ndim} dimensionality')
        return gl_time
