# coding = utf-8

"""see documentation @ ../docs/dynamics.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from static.sq import conditional_sq
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
        3. mean-squared displacements
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
                raise ValueError('incompatible x and xu format coordinates')
        elif xu_snapshots and not x_snapshots:
            logger.info('Use xu coordinates to calculate dynamics and for dynamical Sq')
            self.snapshots = xu_snapshots
            self.x_snapshots = None
        elif x_snapshots and not xu_snapshots:
            logger.info('Use x coordinates to calculate dynamics and for dynamical Sq')
            self.snapshots = x_snapshots
            self.x_snapshots = None
        else:
            logger.info("Please provide correct snapshots for dynamics measurement")

        timesteps = [snapshot.timestep for snapshot in self.snapshots.snapshots]
        self.time = (np.array(timesteps)[1:] - timesteps[0])*dt

    def slowdynamics(
        self,
        qmax: float=6.28,
        a: float=0.3,
        condition: np.ndarray=None,
        outputfile: str=None,
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. qmax (float): characteristic wavenumber, typically first peak of S(q)
            2. a (float): cutoff for the overlap function, should be reduced to particle size
            3. condition (np.ndarray): particle-level condition / property, 
               shape [nsnapshots, nparticles]
            4. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info("Calculate slow dynamics")
        a *= a
        counts = np.zeros(self.snapshots.nsnapshots-1)
        isf = np.zeros_like(counts)
        qt = np.zeros_like(counts)
        qt2 = np.zeros_like(counts)
        r2 = np.zeros_like(counts)
        r4 = np.zeros_like(counts)
        for n in range(1, self.snapshots.nsnapshots):
            for nn in range(1, n+1):
                index = nn-1
                counts[index] += 1
                if condition is not None:
                    selection = condition[n-nn][:, np.newaxis]
                    pos_end = self.snapshots.snapshots[n].positions[selection]
                    pos_init = self.snapshots.snapshots[n-nn].positions[selection]
                else:
                    pos_end = self.snapshots.snapshots[n].positions
                    pos_init = self.snapshots.snapshots[n-nn].positions
                RII = pos_end - pos_init
                RII = remove_pbc(RII, self.snapshots.snapshots[n-nn].hmatrix, self.ppp)
                # self-intermediate scattering function
                isf[index] += np.cos(RII*qmax).mean()
                # overlap function
                distance = np.square(RII).sum(axis=1)
                medium = (distance<a).mean()
                qt[index] += medium
                qt2[index] += medium**2
                # mean-squared displacements & non-gaussian parameter
                r2[index] += distance.mean()
                r4[index] += np.square(distance).mean()
        isf /= counts
        qt /= counts
        qt2 /= counts
        x4_qt = (qt2-np.square(qt)) * self.snapshots.snapshots[0].nparticle
        r2 /= counts
        r4 /= counts
        alpha2 = alpha2factor(self.ndim)*r4/np.square(r2)-1
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results

    def slow_S4(
        self,
        t: float,
        qvector: np.ndarray, # change this to qrange and convert to qvector as input for conditional_sq
        a: float=0.3,
        outputfile: str=None
    ) -> pd.DataFrame:
        """
        Compute four-point dynamic structure factor of slow atoms at characteristic timescale

        Inputs:
            1. t (float): characteristic time for slow dynamics, typically peak time of X4
            2. a (float): cutoff for the overlap function, should be reduced to particle size
            3. qrange (float): the wave number range to be calculated, default 10.0

        Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
        a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
        
        Dynamics should be calculated before computing S4
        xu coordinates should be used to identify slow particles
        x  cooridnates should be used to calcualte FFT
        """
        logger.info('Calculate dynamic S4(q) of slow particles')
        if self.x_snapshots is None:
            snapshots = self.snapshots
        else:
            snapshots = self.x_snapshots

        a *= a
        n_t = int(t/self.time[0])
        for n in range(self.snapshots.nsnapshots-n_t):
            RII = self.snapshots.snapshots[n+n_t].positions - self.snapshots.snapshots[n].positions
            RII = remove_pbc(RII, self.snapshots.snapshots[n].hmatrix, self.ppp)
            RII = np.sqaure(RII).sum(axis=1)
            condition = RII < a
            ave_sqresults += conditional_sq(snapshots.snapshots[n],
                                            qvector=qvector,
                                            condition=condition
                                            )[1]

        ave_sqresults /= self.snapshots.nsnapshots-n_t
        if outputfile:
            ave_sqresults.to_csv(outputfile, index=False)
        return ave_sqresults
