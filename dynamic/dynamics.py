# coding = utf-8

"""see documentation @ ../docs/dynamics.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from neighbors.read_neighbors import read_neighbors
from static.sq import conditional_sq
from utils.pbc import remove_pbc
from utils.funcs import alpha2factor
from utils.logging import get_logger_handle
from utils.wavevector import choosewavevector

logger = get_logger_handle(__name__)

# pylint: disable=dangerous-default-value

def cage_relative(RII:np.ndarray, cnlist:np.ndarray) -> np.ndarray:
    """ 
    get the cage-relative or coarse-grained motion for single configuration
    
    inputs:
        RII (np.ndarray): absolute displacement matrix
        cnlist (np.ndarray): neighbor list of the initial or reference configuration
    
    return:
        np.ndarray: cage-relative displacement matrix
    """
    RII_relative = np.zeros_like(RII)
    for i in range(RII.shape[0]):
        RII_relative[i] = RII[i] - RII[cnlist[i, 1:cnlist[i,0]+1]].mean(axis=0)
    return RII_relative

class Dynamics:
    """
    This module calculates particle-level dynamics with orignal coordinates.
    The calculated quantities include:
        1. self-intermediate scattering function at a specific wavenumber
        2. overlap function and its associated dynamical susceptibility
        3. mean-squared displacements and non-Gaussian parameter
        4. dynamical structure factor based on particle mobility

    A conditional function is implemented to calculate dynamics of specific atoms.
    This module recommends to use absolute coordinates (like xu in LAMMPS) to
    calculate dynamics, while PBC is taken care of as well.

    Linear output configurations are required!
    """

    def __init__(
        self,
        xu_snapshots: Snapshots=None,
        x_snapshots: Snapshots=None,
        dt: float=0.002,
        ppp: np.ndarray=np.array([0,0,0]),
        diameters: dict[int, float]={1:1.0, 2:1.0},
        a: float=0.3,
        cal_type: str="slow",
        neighborfile: str=""
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
            5. diameters (dict): map particle types to particle diameters
            6. a (float): slow mobility cutoff, must be reduced to particle size, default 0.3
            7. cal_type (str): calculation type, can be either slow [default] or fast
            8. neighborfile: neighbor list filename for coarse-graining
        
        Return:
            None
        """
        self.ppp = ppp
        self.ndim = len(ppp)
        self.cal_type = cal_type
        logger.info(f"Calculate {cal_type} dynamics [Linear] for a {self.ndim}-dimensional system")

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

        self.diameters = pd.Series(self.snapshots.snapshots[0].particle_type).map(diameters).values
        self.a_cuts = np.square(self.diameters * a)

        self.neighborlists = []
        if neighborfile:
            fneighbor = open(neighborfile, "r", encoding="utf-8")
            for n in range(self.snapshots.nsnapshots):
                medium = read_neighbors(fneighbor, self.snapshots.snapshots[n].nparticle)
                self.neighborlists.append(medium)
            fneighbor.close()

    def relaxation(
        self,
        qconst: float=6.28,
        condition: np.ndarray=None,
        outputfile: str="",
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. qconst (float): characteristic wavenumber nominator [2pi/sigma], default 2pi
            2. condition (np.ndarray): particle-level condition / property, 
               shape [nsnapshots, nparticles]
            3. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info("Calculate slow dynamics in linear output")
        # define particle type specific cutoffs
        self.q_const = qconst / self.diameters # 2PI/sigma
        q_const = self.q_const.copy()
        a_cuts = self.a_cuts.copy()

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
                    q_const = self.q_const[selection]
                    a_cuts = self.a_cuts[selection]
                else:
                    pos_end = self.snapshots.snapshots[n].positions
                    pos_init = self.snapshots.snapshots[n-nn].positions

                RII = pos_end - pos_init
                RII = remove_pbc(RII, self.snapshots.snapshots[n-nn].hmatrix, self.ppp)

                if self.neighborlists:
                    RII = cage_relative(RII, self.neighborlists[n-nn])

                # self-intermediate scattering function
                isf[index] += np.cos(RII*q_const).mean()

                distance = np.square(RII).sum(axis=1)
                # overlap function
                if self.cal_type=="slow":
                    medium = (distance<a_cuts).mean()
                else: # fast
                    medium = (distance>a_cuts).mean()
                qt[index] += medium
                qt2[index] += medium**2

                # mean-squared displacements & non-gaussian parameter
                r2[index] += distance.mean()
                r4[index] += np.square(distance).mean()
        isf /= counts
        qt /= counts
        qt2 /= counts
        x4_qt = (qt2-np.square(qt)) * len(a_cuts)
        r2 /= counts
        r4 /= counts
        alpha2 = alpha2factor(self.ndim)*r4/np.square(r2)-1
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results

    def sq4(
        self,
        t: float,
        qrange: float=10.0,
        outputfile: str=""
    ) -> pd.DataFrame:
        """
        Compute four-point dynamic structure factor of slow atoms at characteristic timescale

        Inputs:
            1. t (float): characteristic time for slow dynamics, typically peak time of X4
            2. qrange (float): the wave number range to be calculated, default 10.0
            3. outputfile (str): output filename for the calculated dynamical structure factor

        Based on overlap function Qt and its corresponding dynamic susceptibility QtX4        
        Dynamics should be calculated before computing S4
        xu coordinates should be used to identify slow/fast particles based on particle type
        x  cooridnates should be used to calcualte FFT

        Return:
            calculated dynamical structure factor as pandas dataframe
        """
        logger.info(f"Calculate S4(q) of {self.cal_type} particles at the time interval {t}")
        if self.x_snapshots is None:
            snapshots = self.snapshots
        else:
            snapshots = self.x_snapshots

        # define the wavevector based on the wavenumber
        twopidl = 2 * np.pi /snapshots.snapshots[0].boxlength
        numofq = int(qrange*2.0 / twopidl.min())
        qvector = choosewavevector(
            ndim=self.ndim,
            numofq=numofq,
            onlypositive=False
        )

        n_t = int(t/self.time[0])
        ave_sqresults = 0
        for n in range(self.snapshots.nsnapshots-n_t):
            RII = self.snapshots.snapshots[n+n_t].positions - self.snapshots.snapshots[n].positions
            RII = remove_pbc(RII, self.snapshots.snapshots[n].hmatrix, self.ppp)
            if self.neighborlists:
                RII = cage_relative(RII, self.neighborlists[n])

            RII = np.square(RII).sum(axis=1)

            if self.cal_type=="slow":
                condition = RII < self.a_cuts
            else: # fast
                condition = RII > self.a_cuts

            ave_sqresults += conditional_sq(snapshots.snapshots[n],
                                            qvector=qvector,
                                            condition=condition
                                            )[1]

        ave_sqresults /= self.snapshots.nsnapshots-n_t
        if outputfile:
            ave_sqresults.to_csv(outputfile, index=False)
        return ave_sqresults

class LogDynamics:
    """
    This module calculates particle-level dynamics with orignal coordinates.
    The calculated quantities include:
        1. self-intermediate scattering function at a specific wavenumber
        2. overlap function and its associated dynamical susceptibility [default 0]
        3. mean-squared displacements and non-Gaussian parameter

    A conditional function is implemented to calculate dynamics of specific atoms.
    This module recommends to use absolute coordinates (like xu in LAMMPS) to
    calculate dynamics, while PBC is taken care of as well.

    Log output configurations are required!
    """

    def __init__(
        self,
        xu_snapshots: Snapshots=None,
        x_snapshots: Snapshots=None,
        dt: float=0.002,
        ppp: np.ndarray=np.array([0,0,0]),
        diameters: dict[int, float]={1:1.0, 2:1.0},
        cal_type: str="slow",
        neighborfile: str=""
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
            5. diameters (dict): map particle types to particle diameters
            6. cal_type (str): calculation type, can be either slow [default] or fast
            7. neighborfile: neighbor list filename for coarse-graining
        
        Return:
            None
        """
        self.ppp = ppp
        self.ndim = len(ppp)
        self.cal_type = cal_type
        logger.info(f"Calculate {cal_type} dynamics [Log] for a {self.ndim}-dimensional system")

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

        self.diameters = pd.Series(self.snapshots.snapshots[0].particle_type).map(diameters).values

        if neighborfile:
            fneighbor = open(neighborfile, "r", encoding="utf-8")
            self.neighborlists = read_neighbors(fneighbor, self.snapshots.snapshots[0].nparticle)
            fneighbor.close()
        else:
            self.neighborlists = np.zeros(3)

    def relaxation(
        self,
        qconst: float=6.28,
        a: float=0.3,
        condition: np.ndarray=None,
        outputfile: str="",
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. qconst (float): characteristic wavenumber nominator [2pi/sigma], default 2pi
            2. a (float): slow mobility cutoff, must be reduced to particle size, default 0.3
            3. condition (np.ndarray): particle-level condition / property, 
               shape [nsnapshots, nparticles]
            4. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info("Calculate slow dynamics in log-scale output")
        # define particle type specific cutoffs
        q_const = qconst / self.diameters # 2PI/sigma
        a_cuts = np.square(self.diameters * a)

        isf = np.zeros_like(self.time)
        qt = np.zeros_like(self.time)
        r2 = np.zeros_like(self.time)
        r4 = np.zeros_like(self.time)
        for n in range(1, self.snapshots.nsnapshots):
            index = n-1
            if condition is not None:
                selection = condition[:, np.newaxis]
                pos_end = self.snapshots.snapshots[n].positions[selection]
                pos_init = self.snapshots.snapshots[0].positions[selection]
                q_const = q_const[selection]
                a_cuts = a_cuts[selection]
            else:
                pos_end = self.snapshots.snapshots[n].positions
                pos_init = self.snapshots.snapshots[0].positions

            RII = pos_end - pos_init
            RII = remove_pbc(RII, self.snapshots.snapshots[0].hmatrix, self.ppp)

            if self.neighborlists.any():
                RII = cage_relative(RII, self.neighborlists)

            # self-intermediate scattering function
            isf[index] = np.cos(RII*q_const).mean()

            distance = np.square(RII).sum(axis=1)
            # overlap function
            if self.cal_type=="slow":
                medium = (distance<a_cuts).mean()
            else: # fast
                medium = (distance>a_cuts).mean()
            qt[index] = medium

            # mean-squared displacements & non-gaussian parameter
            r2[index] = distance.mean()
            r4[index] = np.square(distance).mean()

        x4_qt = np.zeros_like(qt)
        alpha2 = alpha2factor(self.ndim)*r4/np.square(r2)-1
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results