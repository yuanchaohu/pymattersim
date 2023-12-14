# coding = utf-8

"""see documentation @ ../docs/dynamics.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc
from utils.funcs import alpha2factor
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


class DynamicsAbs:
    """
    This module calculates particle-level dynamics with orignal coordinates.
    The calculated quantities include:
        1. self-intermediate scattering function at a specific wavenumber
        2. overlap function and its associated dynamical susceptibility
        3. measure-squared displacements
        4. dynamical structure factor based on particle mobility

    A conditional function is implemented to calculate dynamics of specific atoms.
    This module recommends to use absolute coordinates (like xu in LAMMPS) to
    calculate dynamics, while PBC is taken care of as well.
    """

    def __init__(
        self,
        xu_snapshots: Snapshots=None,
        x_snapshots: Snapshots=None,
        dt: float=0.002,
        ppp: np.ndarray=np.array([0,0,0])
    ) -> None:
        """
        Initializing DynamicsAbs class

        Inputs:
            1. xu_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [xu, yu, zu], true coordinates, default None
            2. x_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [x, y, z], coordinates with PBCs, default None
            3. dt (float): timestep used in user simulations, default 0.002
            4. ppp (np.ndarray): the periodic boundary conditions (PBCs),
                                 setting 1 for yes and 0 for no, default np.array([0,0,0]),
                                 set np.array([0,0]) for two-dimensional systems
        
        Return:
            None
        """
        self.ppp = ppp
        self.ndim = len(ppp)
        logger.info(f"Calculating absolute dynamics for a {self.ndim}-dimensional system")

        if x_snapshots and xu_snapshots:
            logger.info('Use xu coordinates to calculate dynamics and x for dynamical Sq')
            self.snapshots = xu_snapshots
            self.x_snapshots = x_snapshots
            if xu_snapshots.nsnapshots != x_snapshots.nsnapshots:
                raise ValueError('----incompatible x and xu format coordinates----')
        elif xu_snapshots and not x_snapshots:
            logger.info('Use xu coordinates to calculate dynamics and for dynamical Sq')
            self.snapshots = xu_snapshots
        elif x_snapshots and not xu_snapshots:
            logger.info('Use x coordinates to calculate dynamics and for dynamical Sq')
            self.snapshots = x_snapshots
        else:
            logger.info("Please provide correct snapshots for dynamics measurement")

        timesteps = [snapshot.timestep for snapshot in self.snapshots.snapshots]
        self.time = (np.array(timesteps) - timesteps[0])*dt

    def slowdynamics(
        self,
        qmax: float=6.28,
        a: float=0.3,
        outputfile: str=None,
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. qmax (float): characteristic wavenumber
            2. a (float): cutoff for the overlap function, should be reduced to particle size
            3. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info("Calculate slow dynamics without differentiating particle types")
        a *= a
        counts = np.zeros(self.snapshots.nsnapshots)
        isf = np.zeros_like(counts)
        qt = np.zeros_like(counts)
        qt2 = np.zeros_like(counts)
        r2 = np.zeros_like(counts)
        r4 = np.zeros_like(counts)
        for n in range(self.snapshots.nsnapshots):
            for nn in range(n+1):
                counts[nn] += 1
                RII = self.snapshots.snapshots[n].positions-self.snapshots.snapshots[n-nn].positions
                RII = remove_pbc(RII, self.snapshots.snapshots[n-nn].hmatrix, self.ppp)
                # self-intermediate scattering function
                isf[nn] += np.cos(RII*qmax).mean()
                # overlap function
                distance = np.square(RII).sum(axis=1)
                medium = (distance<a).mean()
                qt[nn] += medium
                qt2[nn] += medium**2
                # mean-squared displacements & non-gaussian parameter
                r2[nn] += distance.mean()
                r4[nn] += np.square(distance).mean()
        isf /= counts
        qt /= counts
        qt2 /= counts
        x4_qt = (np.square(qt) - qt2) * self.snapshots.snapshots[0].nparticle
        r2 /= counts
        r4 /= counts
        alpha2 = np.where(r2==0, 0, alpha2factor(self.ndim)*r4/np.square(r2)-1)
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results

    def slowdynamics_conditional(
        self,
        conditions: np.ndarray,
        qmax: float=6.28,
        a: float=0.3,
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. conditions (np.ndarray): selecting particles based on certain condition, 
                                        shape as [nsnapshots, nparticles]
            1. qmax (float): characteristic wavenumber
            2. a (float): cutoff for the overlap function, should be reduced to particle size
            3. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info("Calculate slow dynamics based on input particle-level conditions")
        a *= a
        counts = np.zeros(self.snapshots.nsnapshots-1)
        isf = np.zeros_like(counts)
        qt = np.zeros_like(counts)
        qt2 = np.zeros_like(counts)
        r2 = np.zeros_like(counts)
        r4 = np.zeros_like(counts)
        for n in range(1, self.snapshots.nsnapshots):
            for nn in range(1, n+1):
                counts[nn] += 1
                selection = conditions[n-nn][:, np.newaxis]
                pos_end = self.snapshots.snapshots[n].positions[selection]
                pos_init = self.snapshots.snapshots[n-nn].positions[selection]
                RII = pos_end - pos_init
                RII = remove_pbc(RII, self.snapshots.snapshots[n-nn].hmatrix, self.ppp)
                # self-intermediate scattering function
                isf[nn] += np.cos(RII*qmax).mean()
                # overlap function
                distance = np.square(RII).sum(axis=1)
                medium = (distance<a).mean()
                qt[nn] += medium
                qt2[nn] += medium**2
                # mean-squared displacements & non-gaussian parameter
                r2[nn] += distance.mean()
                r4[nn] += np.square(distance).mean()
        isf /= counts
        qt /= counts
        qt2 /= counts
        x4_qt = (np.square(qt) - qt2) * conditions.sum(axis=1).mean()
        r2 /= counts
        r4 /= counts
        alpha2 = alpha2factor(self.ndim) * r4/np.square(r2) - 1
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results


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

