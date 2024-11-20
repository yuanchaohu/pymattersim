# coding = utf-8

"""see documentation @ ../../docs/orderings.md"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..dynamic.time_corr import time_correlation
from ..reader.reader_utils import Snapshots
from ..static.gr import conditional_gr
from ..utils.coarse_graining import spatial_average
from ..utils.funcs import kronecker
from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


class NematicOrder():
    """calculate tensor order paramater and its correlation functions"""

    def __init__(
        self,
        snapshots_orientation: Snapshots,
        snapshots_position: Snapshots = None,
    ) -> None:
        """
        Calculate the nematic order parameter of a model system

        Inputs:
            1. snapshots_orientation (reader.reader_utils.Snapshots):
                                snapshot object of input trajectory
                                (returned by reader.dump_reader.DumpReader)
                                !!! DumpFileType=LAMMPSVECTOR
            2. snapshots_position (reader.reader_utils.Snapshots):
                                snapshot object of input trajectory
                                (returned by reader.dump_reader.DumpReader)
                                !!! DumpFileType=LAMMPS or LAMMPSCENTER
                                or any other to provide atom positions
                                Only required for spatial correlation calculation

        Return:
            None
        """
        self.orientations = snapshots_orientation
        self.snapshots = snapshots_position
        self.QIJ = 0

    def tensor(
        self,
        ndim: int = 2,
        neighborfile: str = "",
        Nmax: int = 30,
        eigvals: bool = False,
        outputfile: str = "",
    ) -> npt.NDArray:
        """
        Calculate the nematic order parameter of a model system

        Inputs:
            1. ndim (int): dimensionality of the input configurations
            2. neighborfile (str): file name of particle neighbors (see module neighbors)
            3. Nmax (int): maximum number for neighbors, default 30
            4. eigvals (bool): whether calculate eigenvalue of the Qtensor or not, default False
            5. outputfile (str): file name of the calculation output

        Return:
            Q-tensor or eigenvalue scalar nematic order parameter in numpy ndarray format
            shape as [num_of_snapshots, num_of_particles]
        """
        if neighborfile:
            logger.info(f"Calcualte coarse - grained nematic order for {self.orientations.nsnapshots} configurations")
        else:
            logger.info(f"Calcualte original nematic order for {self.orientations.nsnapshots} configurations")
        # TODO add a function for three dimensioanl systems
        assert ndim == 2, "please set the correction dimensionality"

        QIJ = []
        for snapshot in self.orientations.snapshots:
            medium = np.zeros((snapshot.nparticle, ndim, ndim))
            for i in range(snapshot.nparticle):
                mu = snapshot.positions[i]
                for x in range(ndim):
                    for y in range(ndim):
                        medium[i, x, y] = (
                            ndim * mu[x] * mu[y] - kronecker(x, y)) / 2
            QIJ.append(medium)
        QIJ = np.array(QIJ)

        if neighborfile:
            # coarse-graining over certain volume if neighbor list provided
            QIJ = spatial_average(
                input_property=QIJ,
                neighborfile=neighborfile,
                Nmax=Nmax,
            )
            np.save(outputfile + ".QIJ_cg.npy", QIJ)
        else:
            np.save(outputfile + ".QIJ_raw.npy", QIJ)
        self.QIJ = QIJ

        logger.info("Compute the scalar nematic order parameter")
        if eigvals:
            eigenvalues = np.zeros((QIJ.shape[0], QIJ.shape[1]))
            for n in range(self.orientations.nsnapshots):
                for i in range(self.orientations.snapshots[n].nparticle):
                    eigenvalues[n, i] = np.linalg.eig(QIJ[n, i])[0].max() * 2.0
            np.save(outputfile + ".eigval.npy", eigenvalues)
            return eigenvalues
        # otherwise calculate the trace
        Qtrace = np.zeros((QIJ.shape[0], QIJ.shape[1]))
        for n in range(self.orientations.nsnapshots):
            for i in range(self.orientations.snapshots[n].nparticle):
                Qtrace[n, i] = np.trace(np.matmul(QIJ[n, i], QIJ[n, i]))
        Qtrace *= ndim / (ndim - 1)
        Qtrace = np.sqrt(Qtrace)
        np.save(outputfile + ".Qtrace.npy", Qtrace)
        return Qtrace

    def spatial_corr(self, rdelta: float = 0.01,
                     ppp: npt.NDArray = np.array([1, 1]), outputfile: str = ""):
        """
        Calculate the spatial correlation of the nematic order QIJ

        Inputs:
            1. rdelta (float): bin size in calculating g(r) and G_Q(r), default 0.01
            2. ppp (npt.NDArray): the periodic boundary conditions,
                       setting 1 for yes and 0 for no, default np.array([1,1])
                       for two-dimensional systems
            3. outputfile (str): csv file name for G_Q(r), default None

        Return:
            calculated g_Q(r) based on QIJ tensor
        """
        logger.info("Calculate spatial correlation of tensorial QIJ")
        gQresults = 0
        for n, snapshot in enumerate(self.snapshots.snapshots):
            gQresults += conditional_gr(
                snapshot=snapshot,
                condition=self.QIJ[n],
                conditiontype="tensor",
                ppp=ppp,
                rdelta=rdelta
            )
        gQresults /= self.snapshots.nsnapshots
        if outputfile:
            gQresults.to_csv(outputfile, float_format="%.8f", index=False)
        return gQresults

    def time_corr(
            self,
            dt: float = 0.002,
            outputfile: str = "") -> pd.DataFrame:
        """
        Calculate time correlation of the tensorial nematic order parameter

        Inputs:
            1. dt (float): timestep used in user simulations, default 0.002
            2. outputfile (str): csv file name for time correlation results, default None

        Return:
            time correlation quantity (pd.DataFrame)
        """
        logger.info('Calculate time correlation of tensorial QIJ')
        gQ_time = time_correlation(
            snapshots=self.orientations,
            condition=self.QIJ,
            dt=dt,
            outputfile=outputfile
        )
        return gQ_time
