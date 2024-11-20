# coding = utf-8

"""see documentation @ ../../docs/dynamics.md"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..neighbors.read_neighbors import read_neighbors
from ..reader.reader_utils import Snapshots
from ..static.sq import conditional_sq
from ..utils.funcs import alpha2factor
from ..utils.logging import get_logger_handle
from ..utils.pbc import remove_pbc
from ..utils.wavevector import choosewavevector

logger = get_logger_handle(__name__)

# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def cage_relative(RII: npt.NDArray, cnlist: npt.NDArray) -> npt.NDArray:
    """
    get the cage-relative or coarse-grained motion for single configuration
    The coarse-graining is done based on given neighboring particles

    inputs:
        RII (npt.NDArray): original (absolute) displacement matrix
                          shape [num_of_particles, ndim]
        cnlist (npt.NDArray): neighbor list of the initial or reference configuration
                          shape [num_of_particles, num_of_neighbors]
                          available from the 'neighbors' module

    return:
        npt.NDArray: cage-relative displacement matrix
                    shape [num_of_particles, ndim]
    """
    RII_relative = np.zeros_like(RII)
    for i in range(RII.shape[0]):
        i_neighbors = cnlist[i, 1:cnlist[i, 0] + 1]
        RII_relative[i] = RII[i] - RII[i_neighbors].mean(axis=0)
    return RII_relative


class Dynamics:
    """
    This module calculates particle-level dynamics with orignal coordinates.
    It considers systems at both two-dimension and three-dimension.
    Both absolute and cage-relative dynamics are considered.
    The calculated quantities include:
        1. self-intermediate scattering function at a specific wavenumber
        2. overlap function and its associated dynamical susceptibility
        3. mean-squared displacements and non-Gaussian parameter
        4. dynamical structure factor based on particle mobility

    A conditional function is implemented to calculate dynamics of specific atoms,
    for example, can be used to calculate for a specific atomic type,
    see below functions

    This module recommends to use absolute coordinates (like xu in LAMMPS) to
    calculate dynamics, while PBC is taken care of others as well.

    Linear (constant time interval) output configurations are required!
    """

    def __init__(
        self,
        xu_snapshots: Snapshots = None,
        x_snapshots: Snapshots = None,
        dt: float = 0.002,
        ppp: npt.NDArray = np.array([0, 0, 0]),
        diameters: dict[int, float] = {1: 1.0, 2: 1.0},
        a: float = 0.3,
        cal_type: str = "slow",
        neighborfile: str = "",
        max_neighbors: int = 30,
    ) -> None:
        """
        Initialization

        Inputs:
            1. xu_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [xu, yu, zu], absolute coordinates, default None
            2. x_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [x,y,z] or [xs,ys,zs], coordinates with PBCs, default None
            3. dt (float): timestep used in user simulations, default 0.002
            4. ppp (npt.NDArray): the periodic boundary conditions (PBCs),
                                 setting 1 for yes and 0 for no, default np.array([0,0,0]) for 3D,
                                 default no periodic boundary conditions considered.
                                 Dimensionality is refered from the shape of ppp
            5. diameters (dict): map particle types to particle diameters
            6. a (float): slow mobility cutoff scaling factor, default 0.3,
                          will scale based on particle diameter. When diameter is 1.0,
                          cutoff is 0.3; if diamter 2.0, cutoff will be 0.6 in calculation
            7. cal_type (str): calculation type, can be either slow [default] or fast
                               slow: particles moving shorter than a distance during some time
                               fast: particles moving further than a distance during some time
            8. neighborfile: neighbor list filename for coarse-graining
                             only provided when calculating (cage-)relative displacements
            9. max_neighbors: maximum of particle neighbors considered, default 30

        Return:
            None
        """
        self.ppp = ppp
        self.ndim = len(ppp)
        self.cal_type = cal_type
        logger.info(f"Calculate {cal_type} dynamics[Linear] for a {self.ndim} - dimensional system")

        if x_snapshots and xu_snapshots:
            logger.info(
                'Use xu coordinates to calculate dynamics and x/xs for dynamical Sq')
            self.snapshots = xu_snapshots
            self.x_snapshots = x_snapshots
            self.PBC = False
            if xu_snapshots.nsnapshots != x_snapshots.nsnapshots:
                raise ValueError('incompatible x/xs and xu format coordinates')
        elif xu_snapshots and not x_snapshots:
            logger.info(
                'Use xu coordinates to calculate dynamics and dynamical Sq')
            self.snapshots = xu_snapshots
            self.x_snapshots = None
            self.PBC = False
        elif x_snapshots and not xu_snapshots:
            logger.info(
                'Use x/xs coordinates to calculate dynamics and dynamical Sq')
            self.snapshots = x_snapshots
            self.x_snapshots = None
            self.PBC = True
            if not ppp.any():
                raise ValueError("No periodic boundary conditions provided")
        else:
            logger.info(
                "Please provide correct snapshots for dynamics measurement")

        timesteps = [
            snapshot.timestep for snapshot in self.snapshots.snapshots]
        self.time = (np.array(timesteps)[1:] - timesteps[0]) * dt

        # TODO confirm the values of self.diameters
        self.diameters = pd.Series(
            self.snapshots.snapshots[0].particle_type).map(diameters).values
        self.a2_cuts = np.square(self.diameters * a)

        self.neighborlists = []
        if neighborfile:
            fneighbor = open(neighborfile, "r", encoding="utf-8")
            for n in range(self.snapshots.nsnapshots):
                medium = read_neighbors(
                    f=fneighbor,
                    nparticle=self.snapshots.snapshots[n].nparticle,
                    Nmax=max_neighbors
                )
                self.neighborlists.append(medium)
            fneighbor.close()

    def relaxation(
        self,
        qconst: float = 2 * np.pi,
        condition: npt.NDArray = None,
        outputfile: str = "",
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2

        Default to calculate for all particles;
        Use 'condition' to perform calculations over specific particles

        Inputs:
            1. qconst (float): characteristic wavenumber nominator [2pi/sigma], default 2pi
            2. condition (npt.NDArray): particle-level condition / property, shape [nsnapshots, nparticles]
            3. outputfile (str): file name to save the calculated dynamics results

        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info(f"Calculate {self.cal_type} dynamics in linear output")
        # define particle type specific cutoffs
        self.q_const = qconst / self.diameters  # 2PI/sigma
        q_const = self.q_const.copy()
        a2_cuts = self.a2_cuts.copy()

        counts = np.zeros(self.snapshots.nsnapshots - 1)
        isf = np.zeros_like(counts)
        qt = np.zeros_like(counts)
        qt2 = np.zeros_like(counts)
        r2 = np.zeros_like(counts)
        r4 = np.zeros_like(counts)
        for n in range(1, self.snapshots.nsnapshots):
            for nn in range(1, n + 1):
                index = nn - 1
                counts[index] += 1
                pos_init = self.snapshots.snapshots[n - nn].positions
                pos_end = self.snapshots.snapshots[n].positions
                RII = pos_end - pos_init

                # remove periodic boundary conditions when necessary
                if self.PBC:
                    RII = remove_pbc(
                        RII, self.snapshots.snapshots[n - nn].hmatrix, self.ppp)

                # calculate cage-relative displacements when necessary
                if self.neighborlists:
                    RII = cage_relative(RII, self.neighborlists[n - nn])

                # select results for specific particles
                if condition is not None:
                    selection = condition[n - nn]
                    q_const = self.q_const[selection]
                    a2_cuts = self.a2_cuts[selection]
                    RII = RII[selection]

                # self-intermediate scattering function
                # average over [1,0,0], [0,1,0] & [0,0,1] direction
                isf[index] += np.cos(RII * q_const[:, np.newaxis]).mean()

                # squared scalar displacements
                distance = np.square(RII).sum(axis=1)

                # overlap function
                if self.cal_type == "slow":
                    medium = (distance < a2_cuts).mean()
                else:  # fast
                    medium = (distance > a2_cuts).mean()
                qt[index] += medium
                qt2[index] += medium**2

                # mean-squared displacements & non-gaussian parameter
                r2[index] += distance.mean()
                r4[index] += np.square(distance).mean()
        isf /= counts
        qt /= counts
        qt2 /= counts
        x4_qt = (qt2 - np.square(qt)) * len(a2_cuts)
        r2 /= counts
        r4 /= counts
        alpha2 = alpha2factor(self.ndim) * r4 / np.square(r2) - 1
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(
            results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results

    def sq4(
        self,
        t: float,
        qrange: float = 10.0,
        condition: npt.NDArray = None,
        outputfile: str = ""
    ) -> pd.DataFrame:
        """
        Compute four-point dynamic structure factor of specific atoms at characteristic timescale

        Inputs:
            1. t (float): characteristic time, typically peak time of X4, see self.relaxation()
            2. qrange (float): the wave number range to be calculated, default 10.0
            3. condition (npt.NDArray): particle-level condition / property, shape [nsnapshots, nparticles]
            4. outputfile (str): output filename for the calculated dynamical structure factor

        Based on overlap function Qt and its corresponding dynamic susceptibility X4_Qt
        Dynamics should be calculated before computing self.sq4().
        A general practice is to calcualte X4_Qt by self.relaxation() first and get the
        peak timescale for this function

        Return:
            calculated dynamical structure factor as pandas dataframe
        """
        logger.info(f"Calculate S4(q) of {self.cal_type} particles at the time interval {t}")
        if self.x_snapshots is None:
            logger.info(
                "Use xu coordinates for dynamics and x/xs coordinates for Sq4")
            snapshots = self.snapshots
        else:
            logger.info(
                "Use only xu or x/xs for calculating both dynamics and Sq4")
            snapshots = self.x_snapshots

        # define the wavevector based on the wavenumber large limit
        twopidl = 2 * np.pi / snapshots.snapshots[0].boxlength
        numofq = int(qrange * 2.0 / twopidl.min())
        qvector = choosewavevector(
            ndim=self.ndim,
            numofq=numofq,
            onlypositive=False
        )

        # convert input time to configuration interval
        n_t = round(t / self.time[0])
        ave_sqresults = 0
        for n in range(self.snapshots.nsnapshots - n_t):
            pos_init = self.snapshots.snapshots[n].positions
            pos_end = self.snapshots.snapshots[n + n_t].positions
            RII = pos_end - pos_init

            # remove periodic boundary conditions when necessary
            if self.PBC:
                RII = remove_pbc(
                    RII, self.snapshots.snapshots[n].hmatrix, self.ppp)

            # calculate cage-relative displacements when necessary
            if self.neighborlists:
                RII = cage_relative(RII, self.neighborlists[n])

            RII = np.square(RII).sum(axis=1)

            if self.cal_type == "slow":
                mobility_condition = RII < self.a2_cuts
            else:  # fast
                mobility_condition = RII > self.a2_cuts

            if condition is not None:
                mobility_condition *= condition[n].astype(bool)

            ave_sqresults += conditional_sq(snapshots.snapshots[n],
                                            qvector=qvector,
                                            condition=mobility_condition
                                            )[1]

        ave_sqresults /= self.snapshots.nsnapshots - n_t
        if outputfile:
            ave_sqresults.to_csv(outputfile, index=False)
        return ave_sqresults


class LogDynamics:
    """
    This module calculates particle-level dynamics with orignal coordinates.
    It considers systems at both two-dimension and three-dimension.
    Both absolute and cage-relative dynamics are considered.
    The calculated quantities include:
        1. self-intermediate scattering function at a specific wavenumber
        2. overlap function and its associated dynamical susceptibility [default 0]
        3. mean-squared displacements and non-Gaussian parameter

    A conditional function is implemented to calculate dynamics of specific atoms,
    for example, can be used to calculate for a specific atomic type,
    see below functions

    This module recommends to use absolute coordinates (like xu in LAMMPS) to
    calculate dynamics, while PBC is taken care of others as well.

    Log (changing time interval) output configurations are required!
    Ensemble average is absent compared to the above Dynamics() class!
    """

    def __init__(
        self,
        xu_snapshots: Snapshots = None,
        x_snapshots: Snapshots = None,
        dt: float = 0.002,
        ppp: npt.NDArray = np.array([0, 0, 0]),
        diameters: dict[int, float] = {1: 1.0, 2: 1.0},
        a: float = 0.3,
        cal_type: str = "slow",
        neighborfile: str = "",
        max_neighbors: int = 30,
    ) -> None:
        """
        Initialization

        Inputs:
            1. xu_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [xu, yu, zu], absolute coordinates, default None
            2. x_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [x,y,z] or [xs,ys,zs], coordinates with PBCs, default None
            3. dt (float): timestep used in user simulations, default 0.002
            4. ppp (npt.NDArray): the periodic boundary conditions (PBCs),
                                 setting 1 for yes and 0 for no, default np.array([0,0,0]) for 3D,
                                 default no periodic boundary conditions considered.
                                 Dimensionality is refered from the shape of ppp
            5. diameters (dict): map particle types to particle diameters
            6. a (float): slow mobility cutoff scaling factor, default 0.3,
                          will scale based on particle diameter. When diameter is 1.0,
                          cutoff is 0.3; if diamter 2.0, cutoff will be 0.6 in calculation
            7. cal_type (str): calculation type, can be either slow [default] or fast
                               slow: particles moving shorter than a distance during some time
                               fast: particles moving further than a distance during some time
            8. neighborfile: neighbor list filename for coarse-graining
                             only provided when calculating (cage-)relative displacements
            9. max_neighbors: maximum of particle neighbors considered, default 30

        Return:
            None
        """
        self.ppp = ppp
        self.ndim = len(ppp)
        self.cal_type = cal_type
        logger.info(f"Calculate {cal_type} dynamics[Log] for a {self.ndim} - dimensional system")

        if x_snapshots and xu_snapshots:
            logger.info('Use xu coordinates to calculate dynamics and x/xs for dynamical Sq')
            self.snapshots = xu_snapshots
            self.x_snapshots = x_snapshots
            self.PBC = False
            if xu_snapshots.nsnapshots != x_snapshots.nsnapshots:
                raise ValueError('incompatible x/xs and xu format coordinates')
        elif xu_snapshots and not x_snapshots:
            logger.info(
                'Use xu coordinates to calculate dynamics and for dynamical Sq')
            self.snapshots = xu_snapshots
            self.x_snapshots = None
            self.PBC = False
        elif x_snapshots and not xu_snapshots:
            logger.info(
                'Use x/xs coordinates to calculate dynamics and for dynamical Sq')
            self.snapshots = x_snapshots
            self.x_snapshots = None
            self.PBC = True
            if not ppp.any():
                raise ValueError("No periodic boundary conditions provided")
        else:
            logger.info(
                "Please provide correct snapshots for dynamics measurement")

        timesteps = [
            snapshot.timestep for snapshot in self.snapshots.snapshots]
        self.time = (np.array(timesteps)[1:] - timesteps[0]) * dt

        self.diameters = pd.Series(
            self.snapshots.snapshots[0].particle_type).map(diameters).values
        self.a2_cuts = np.square(self.diameters * a)

        if neighborfile:
            fneighbor = open(neighborfile, "r", encoding="utf-8")
            self.neighborlists = read_neighbors(
                f=fneighbor,
                nparticle=self.snapshots.snapshots[0].nparticle,
                Nmax=max_neighbors
            )
            fneighbor.close()
        else:
            self.neighborlists = np.zeros(3)

    def relaxation(
        self,
        qconst: float = 2 * np.pi,
        condition: npt.NDArray = None,
        outputfile: str = "",
    ) -> pd.DataFrame:
        """
        Compute self-intermediate scattering functions ISF,
        Overlap function Qt and set its corresponding dynamic susceptibility X4_Qt as 0
        Mean-square displacements msd; non-Gaussion parameter alpha2

        Inputs:
            1. qconst (float): characteristic wavenumber nominator [2pi/sigma], default 2pi
            2. condition (npt.NDArray): particle-level condition / property,
                                       shape [nparticles]
            3. outputfile (str): file name to save the calculated dynamic results

        Return:
            Calculated dynamics results in pd.DataFrame
        """
        logger.info(f"Calculate {self.cal_type} dynamics in log-scale output")
        # define particle type specific cutoffs
        self.q_const = qconst / self.diameters  # 2PI/sigma
        q_const = self.q_const.copy()
        a2_cuts = self.a2_cuts.copy()

        isf = np.zeros_like(self.time)
        qt = np.zeros_like(self.time)
        r2 = np.zeros_like(self.time)
        r4 = np.zeros_like(self.time)
        for n in range(1, self.snapshots.nsnapshots):
            index = n - 1
            pos_init = self.snapshots.snapshots[0].positions
            pos_end = self.snapshots.snapshots[n].positions
            RII = pos_end - pos_init

            # remove periodic boundary conditions when necessary
            if self.PBC:
                RII = remove_pbc(
                    RII, self.snapshots.snapshots[0].hmatrix, self.ppp)

            # calculate cage-relative displacements when nencessary
            if self.neighborlists.any():
                RII = cage_relative(RII, self.neighborlists)

            # select results for specific particles
            if condition is not None:
                q_const = self.q_const[condition]
                a2_cuts = self.a2_cuts[condition]
                RII = RII[condition]

            # self-intermediate scattering function
            # average over [1,0,0], [0,1,0] & [0,0,1] direction
            isf[index] = np.cos(RII * q_const[:, np.newaxis]).mean()

            # squared scalar displacements
            distance = np.square(RII).sum(axis=1)

            # overlap function
            if self.cal_type == "slow":
                medium = (distance < a2_cuts).mean()
            else:  # fast
                medium = (distance > a2_cuts).mean()
            qt[index] = medium

            # mean-squared displacements & non-gaussian parameter
            r2[index] = distance.mean()
            r4[index] = np.square(distance).mean()

        x4_qt = np.zeros_like(qt)
        alpha2 = alpha2factor(self.ndim) * r4 / np.square(r2) - 1
        results = np.column_stack((self.time, isf, qt, x4_qt, r2, alpha2))
        results = pd.DataFrame(
            results, columns='t isf Qt X4_Qt msd alpha2'.split())
        if outputfile:
            results.to_csv(outputfile, index=False)
        return results
