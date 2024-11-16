# coding = utf-8

"""
see documentation @ ../../docs/utils.md
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt

from ..neighbors.read_neighbors import read_neighbors
from ..reader.reader_utils import Snapshots
from ..utils.funcs import grid_gaussian
from ..utils.logging import get_logger_handle
from ..utils.pbc import remove_pbc

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def time_average(
        snapshots: Snapshots,
        input_property: npt.NDArray,
        time_period: float = 0.0,
        dt: float = 0.002
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculate time average of the input property

    Input:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. input_property (npt.NDArray): the input particle-level property,
                          in npt.NDArray with shape [nsnapshots, nparticle]
        3. time_period (float): time used to average, default 0.0
        4. dt (float): timestep used in user simulations, default 0.002

    Return:
        1. Calculated time averaged input results (npt.NDArray)
           shape [nsnapshots_updated, nparticles]
        2. Corresponding snapshot id of the middle snapshot of each time period
           shape [nsnapshots_updated]
    """
    logger.info(
        f"Time average of the input variables for time_period={time_period}")
    time_interval = snapshots.snapshots[1].timestep - \
        snapshots.snapshots[0].timestep
    time_interval *= dt
    time_nsnapshot = int(time_period / time_interval)
    # save the time averaged results
    results = np.zeros((
        snapshots.nsnapshots - time_nsnapshot,
        snapshots.snapshots[0].nparticle),
        dtype=np.complex128)
    # save the middle snapshot id for each time average period
    results_middle_snapshots = []

    for n in range(results.shape[0]):
        results[n, :] = input_property[n:n + time_nsnapshot].mean(axis=0)
        results_middle_snapshots.append(round(n + time_nsnapshot / 2))
    return results, np.array(results_middle_snapshots)


def spatial_average(
    input_property: npt.NDArray,
    neighborfile: str,
    Nmax: int = 30,
    outputfile: str = "",
) -> npt.NDArray:
    """
    coarse-graining the input variable over certain length scale
    given by the pre-defined neighbor list

    Inputs:
        1. input_property (npt.NDArray): input property to be coarse-grained,
            should be in the shape [num_of_snapshots, num_of_particles, xxx]
            The input property can be scalar or vector or tensor
        2. neighborfile (str): file name of pre-defined neighbor list
        3. Namx (int): maximum number of particle neighbors
        4. outputfile (str): file name of coarse-grained variable

    Return:
        coarse-grained input property in numpy ndarray
    """
    logger.info(f"Performing coarse-graining from {neighborfile}")
    cg_input_property = np.copy(input_property)
    with open(neighborfile, mode="r", encoding="utf-8") as fneighbor:
        for n in range(input_property.shape[0]):
            cnlist = read_neighbors(fneighbor, input_property.shape[1], Nmax)
            for i in range(input_property.shape[1]):
                # in case input_property is multi-dimensional
                for j in cnlist[i, 1:1 + cnlist[i, 0]]:
                    cg_input_property[n, i] += input_property[n, j]
                cg_input_property[n, i] /= (1 + cnlist[i, 0])
    if outputfile:
        np.save(outputfile, cg_input_property)
    return cg_input_property


def gaussian_blurring(
    snapshots: Snapshots,
    condition: npt.NDArray,
    ngrids: npt.NDArray,
    sigma: float = 2.0,
    ppp: npt.NDArray = np.array([1, 1, 1]),
    gaussian_cut: float = 6.0,
    outputfile: str = "",
):
    """
    Project particle-level property into a pre-defined grid,
    this is usually called Gaussian blurring

    Inputs:
        1. snapshots (read.reader_utils.snapshots): multiple trajectories dumped linearly or in logscale
        2. condition (npt.NDArray): particle-level condition / property, type should be float
                                   shape: [num_of_snapshots, num_of_particles, xxx],
                                   The input property can be scalar or vector or tensor, based on
                                   the shape of condition, mapping as
                                   {"scalar": 3, "vector": 4, "tensor": 5}
        3. ngrids (npt.NDArray of int): predefined grid number in the simulation box,
                                    shape as the dimension, for example, [25, 25] for 2D systems
        4. sigma (float): standard deviation of the gaussian distribution function, default 2.0
        5. ppp (npt.NDArray): the periodic boundary conditions (PBCs),
                            setting 1 for yes and 0 for no, default np.array([1,1,1])
        6. gaussian_cut (float): the longest distance to consider the gaussian probability
                            or the contribution from the simulation particles.
                            default 6.0.
        7. outputfile (str): file name to save the grid positions and the corresponding properties

    Return:
        grid_positions (npt.NDArray): Positions of the grids of each snapshot
        grid_property (npt.NDArray): properties of each grid of each snapshot
    """

    ndim = len(ngrids)
    ppp = ppp[:ndim]
    grid_positions = np.zeros((snapshots.nsnapshots, np.prod(ngrids), ndim))

    if len(condition.shape) == 2:
        cal_type = "scalar"
    elif len(condition.shape) == 3:
        cal_type = "vector"
    elif len(condition.shape) == 4:
        cal_type = "tensor"
    else:
        raise ValueError("Wrong input condition variable")
    new_shape = list(condition.shape)
    new_shape[1] = np.prod(ngrids)
    grid_property = np.zeros(tuple(new_shape))

    logger.info(f"Performing Gaussian Blurring of {cal_type} field for a {ndim} - dimensional system")
    for n, snapshot in enumerate(snapshots.snapshots):
        bxobounds = snapshot.boxbounds
        X = np.linspace(bxobounds[0, 0], bxobounds[0, 1], ngrids[0])
        Y = np.linspace(bxobounds[1, 0], bxobounds[1, 1], ngrids[1])
        if ndim == 2:
            for i in range(ngrids[0]):
                for j in range(ngrids[1]):
                    indice = i * ngrids[0] + j
                    grid_positions[n, indice] = [X[i], Y[j]]
        else:
            Z = np.linspace(bxobounds[2, 0], bxobounds[2, 1], ngrids[2])
            for i in range(ngrids[0]):
                for j in range(ngrids[1]):
                    for k in range(ngrids[2]):
                        indice = i * ngrids[0] + j * ngrids[1] + k
                        grid_positions[n, indice] = [X[i], Y[j], Z[k]]

        for i in range(grid_positions.shape[1]):
            RIJ = grid_positions[n, i] - snapshot.positions
            RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp=ppp)
            RIJ = np.linalg.norm(RIJ, axis=1)
            selection = RIJ < gaussian_cut
            probability = grid_gaussian(RIJ[selection], sigma)
            if cal_type == "scalar":
                grid_property[n, i] = (
                    probability * condition[n, selection]).sum()
            elif cal_type == "vector":
                # vector
                grid_property[n, i] = (
                    probability[:, np.newaxis] * condition[n, selection]).sum(axis=0)
            else:
                # tensor
                grid_property[n,
                              i] = (probability[:,
                                                np.newaxis,
                                                np.newaxis] * condition[n,
                                                                        selection]).sum(axis=0)

    if outputfile:
        np.save(outputfile + "_positions.npy", grid_positions)
        np.save(outputfile + "_properties.npy", grid_property)
    return grid_positions, grid_property


def atomic_position_average():
    """
    average particle postions over a time window
    """
    pass
