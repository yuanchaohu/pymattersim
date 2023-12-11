# coding = utf-8

"""see documentation @ ../docs/dynamics.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc
from utils.funcs import alpha2factor
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


class dynamics:
    """
    This module calculates particle-level dynamics.
    Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF.
    Overlap function Qt and its corresponding dynamic susceptibility QtX4.
    Mean-square displacements msd; non-Gaussion parameter alpha2
    four-point dynamic structure factor of fast and slow particles, respectively

    The module also computes corresponding particle type related dynamics by using the function partial()
    The module accounts for systems ranging from unary to senary

    Mean-squared displacements and non-Gaussian parameter should be calculated with the absolute coordinates
    e.g. via LAMMPS (xu, yu, zu)

    Two sets of configurations are suggested to use together, one with (x, y, z) and one with (xu, yu, zu);
    the former is used to calculate structure factor and the latter for dynamics
    """

    def __init__(
        self,
        xu_snapshots: Snapshots=None,
        x_snapshots: Snapshots=None,
        dt: float=0.002,
        ppp: np.ndarray=np.array([0,0,0]),
        neighborfile: str=None
    ) -> None:
        """
        Initializing dynamics class

        Inputs:
            1. xu_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [xu, yu, zu], true coordinates, default None
            2. x_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [x, y, z], coordinates with periodic boundary conditions, default None
            3. dt (float): timestep used in user simulations, default 0.002
            4. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.array([0,0,0]),
                                 set np.array([0,0]) for two-dimensional systems
        
        Return:
            None
        """
        self.ppp = ppp
        self.neighborfile = neighborfile
        self.ndim = len(ppp)
        if self.ndim==2 and self.neighborfile:
            logger.info("Computer cage-relative dynamics for two-dimensional systems")

        if x_snapshots and xu_snapshots:
            logger.info('Use both x and xu coordinates to calculate dynamics')
            self.snapshots = xu_snapshots
            self.x_snapshots = x_snapshots
            if xu_snapshots.nsnapshots != x_snapshots.nsnapshots:
                raise ValueError('----incompatible x and xu format coordinates----')
        elif xu_snapshots and not x_snapshots:
            logger.info('Use xu coordinates to calculate dynamics')
            self.snapshots = xu_snapshots
        elif x_snapshots and not xu_snapshots:
            logger.info('Use x coordinates to calculate dynamics')
            self.snapshots = x_snapshots
        else:
            logger.info("Please provide ")
        
        self.time = [snapshot.timestep for snapshot in self.snapshots.snapshots]
        self.time = (np.array(self.time[1:]) - self.time[0])*dt

    def total(self, outputfile):
        if self.neighborfile:
            self.results, self.particle_msd = self.cagerelative()
        else:
           self.results, self.particle_msd = self.absolute()

    def absolute(
        self,
        qmax: float=6.28,
        a: float=0.3
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. qmax (float): wavenumber corresponding to the first peak of structure factor
            2. a (float): cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            3. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in np.ndarray
        """
        logger.info(f'Start calculating abosulte dynamics in {self.ndim} dimensional system')
        a2 = a**2
        counts = np.zeros(self.snapshots.nsnapshots-1)
        isf = np.zeros_like(counts)
        isf2 = np.zeros_like(counts)
        qt = np.zeros_like(counts)
        qt2 = np.zeros_like(counts)
        r2 = np.zeros_like(counts)
        r4 = np.zeros_like(counts)
        particle_msd = np.zeros((self.snapshots.nsnapshots, self.snapshots.snapshots[0].nparticle))
        for n in range(self.snapshots.nsnapshots):
            for nn in range(1, n+1):
                counts[nn] += 1
                RII = self.snapshots.positions[n] - self.snapshots.positions[n-nn]
                RII = remove_pbc(RII, self.snapshots.hmatrix[n-nn], self.ppp)
                # self-intermediate scattering function
                medium = np.cos(RII*qmax).mean(axis=1).sum()
                isf[nn] += medium
                isf2[nn] += medium**2
                # overlap function
                distance = np.square(RII).sum(axis=1)
                medium =( distance < a2).sum()
                qt[nn] += medium
                qt2[nn] += medium**2
                # mean-squared displacements & non-gaussian parameter
                particle_msd[nn, :] = distance
                r2[nn] += distance.sum()
                r4[nn] += np.square(distance).sum()
        isf /= counts
        isf2 /= counts
        x4_isf = np.square(isf) - isf2

        qt /= counts
        qt2 /= counts
        x4_qt = np.square(qt) - qt2

        alpha2 = alpha2factor(self.ndim) * r2/r4 - 1
        results = np.column_stack((isf, x4_isf, qt, x4_qt, r2, alpha2))
        results /= self.snapshots.snapshots[0].nparticle
        results = np.column_stack((self.time, results))
        results = pd.DataFrame(results, columns='t isf X4_isf Qt X4_Qt msd alpha2'.split())
        return results, particle_msd

    def cagerelative(self):
        # step 1: read neighbors to a list
        # step 2: calculate cag
        logger.info(f'Start calculating cage-relative dynamics in {self.ndim} dimensional system')
        a2 = a**2
        counts = np.zeros(self.snapshots.nsnapshots-1)
        isf = np.zeros_like(counts)
        isf2 = np.zeros_like(counts)
        qt = np.zeros_like(counts)
        qt2 = np.zeros_like(counts)
        r2 = np.zeros_like(counts)
        r4 = np.zeros_like(counts)
        for n in range(self.snapshots.nsnapshots):
            for nn in range(1, n+1):
                counts[nn] += 1
                RII = self.snapshots.positions[n] - self.snapshots.positions[n-nn]
                RII = remove_pbc(RII, self.snapshots.hmatrix[n-nn], self.ppp)
                # self-intermediate scattering function
                medium = np.cos(RII*qmax).mean(axis=1).sum()
                isf[nn] += medium
                isf2[nn] += medium**2
                # overlap function
                distance = np.square(RII).sum(axis=1)
                medium =( distance < a2).sum()
                qt[nn] += medium
                qt2[nn] += medium**2
                # mean-squared displacements & non-gaussian parameter
                r2[nn] += distance.sum()
                r4[nn] += np.square(distance).sum()
        isf /= counts
        isf2 /= counts
        x4_isf = np.square(isf) - isf2

        qt /= counts
        qt2 /= counts
        x4_qt = np.square(qt) - qt2

        alpha2 = alpha2factor(self.ndim) * r2/r4 - 1
        results = np.column_stack((isf, x4_isf, qt, x4_qt, r2, alpha2))
        results /= self.snapshots.snapshots[0].nparticle
        results = np.column_stack((self.time, results))
        results = pd.DataFrame(results, columns='t isf X4_isf Qt X4_Qt msd alpha2'.split())
        return results


    def conditional(self, conditions):
        pass

    def S4(self, conditions):
        pass
    def slowS4(self, X4time, a=1.0, qrange=10, onlypositive=False, outputfile=''):
        """ Compute four-point dynamic structure factor of slow atoms at peak timescale of dynamic susceptibility

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            X4time is the peaktime scale of X4
            Dynamics should be calculated before computing S4
            xu coordinates should be used to identify slow particles
            x  cooridnates should be used to calcualte FFT
        """
        print ('----Compute dynamic S4(q) of slow particles----')

        X4time = int(X4time / self.dt / self.TimeStep)
        
        twopidl = 2 * pi / self.Boxlength
        Numofq = int(qrange*2.0 / twopidl.min())
        qvector = choosewavevector(self.ndim, Numofq, onlypositive)
        qvector = qvector.astype(np.float64) * twopidl[np.newaxis, :] #considering non-cubic box
        qvalues = np.linalg.norm(qvector, axis=1)
        
        sqresults = np.zeros((qvector.shape[0], 2))
        sqresults[:, 0] = qvalues
        for n in range(self.SnapshotNumber - X4time):
            RII = self.Positions[n + X4time] - self.Positions[n]
            if self.PBC:
                #hmatrixinv = np.linalg.inv(self.hmatrix[n])
                #matrixij   = np.dot(RII, hmatrixinv)
                #RII        = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC
                RII = remove_PBC(RII, self.hmatrix[n], self.ppp)

            RII = np.linalg.norm(RII, axis = 1)
            RII = np.where(RII <= a, 1, 0)
            
            sqtotal = np.zeros_like(sqresults)
            for i in range(self.ParticleNumber):
                if RII[i]:
                    thetas = (self.original[n][i][np.newaxis, :] * qvector).sum(axis=1)
                    sqtotal[:, 0] += np.sin(thetas)
                    sqtotal[:, 1] += np.cos(thetas)
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1) / self.ParticleNumber
        sqresults[:, 1] /= (self.SnapshotNumber - X4time)
        
        sqresults = pd.DataFrame(sqresults).round(8)
        results = sqresults.groupby(sqresults[0]).mean().reset_index().values     
        names = 'q  S4'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.8f', header = names, comments = '')
        print ('----Compute S4(q) of slow particles over----')
        return results, names

