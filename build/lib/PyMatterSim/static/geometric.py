# coding = utf-8

"""see documentation @ ../../docs/orderings.md"""

from itertools import combinations

import numpy as np
import numpy.typing as npt

from ..neighbors.read_neighbors import read_neighbors
from ..reader.reader_utils import Snapshots
from ..utils.geometry import triangle_angle
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


def packing_capability_2d(
    snapshots: Snapshots,
    sigmas: npt.NDArray,
    neighborfile: str,
    ppp: npt.NDArray = np.array([1, 1]),
    outputfile: str = "",
) -> npt.NDArray:
    """
    Calculate packing capability of a 2D system based on geometry

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. sigmas (npt.NDArray): particle sizes for each pair of particle type (ascending order)
                                in numpy array, shape [particle_type, particle_type]
        3. neighborfile (str): file name of particle neighbors (see module neighbors)
        4. ppp (npt.NDArray): the periodic boundary conditions, setting 1 for yes and 0 for no,
                             default npt.NDArray=np.array([1,1])
        5. outputfile (str): file name to save the calculated packing capability, default None

    Return:
        Calculated packing capability of a 2D system
        shape [nsnapshots, nparticles]
    """
    assert ppp.shape[0] == 2, "This calculation is for a two-dimensional system"
    logger.info("Start calculating packing capability in two dimensions")

    # calcualte reference angles, save to a 3d numpy array
    particle_type = sigmas.shape[0]
    assert particle_type == snapshots.snapshots[0].particle_type.max(), "Error shape of input sigmas (particle diameters)"

    reference_angles = np.zeros((particle_type, particle_type, particle_type))
    for o in range(particle_type):
        for i in range(particle_type):
            for j in range(particle_type):
                reference_angles[o, i, j] = triangle_angle(sigmas[o, i], sigmas[o, j], sigmas[i, j])

    # calculate real angles in trajectory
    results = np.zeros((snapshots.nsnapshots, snapshots.snapshots[0].nparticle))
    fneighbor = open(neighborfile, "r", encoding="utf-8")
    for n, snapshot in enumerate(snapshots.snapshots):
        neighborlist = read_neighbors(fneighbor, snapshot.nparticle, 20)
        for o in range(snapshot.nparticle):
            cnlist = neighborlist[o, 1 : neighborlist[o, 0] + 1]
            theta_o = 0
            for i, j in combinations(cnlist, 2):
                i_cnlist = neighborlist[i, 1 : neighborlist[i, 0] + 1]
                j_cnlist = neighborlist[j, 1 : neighborlist[j, 0] + 1]
                if (j in i_cnlist) & (i in j_cnlist):
                    vectors_oij = snapshot.positions[[i, j]] - snapshot.positions[o][np.newaxis, :]
                    vectors_oij = remove_pbc(vectors_oij, snapshot.hmatrix, ppp)
                    distance = np.linalg.norm(vectors_oij, axis=1)
                    vectors_oij /= distance[:, np.newaxis]
                    theta = np.dot(vectors_oij[0], vectors_oij[1])
                    theta = np.arccos(theta)
                    otype, itype, jtype = snapshot.particle_type[[o, i, j]] - 1
                    theta_o += abs(theta - reference_angles[otype, itype, jtype])
            results[n, o] = theta_o / neighborlist[o, 0]
    fneighbor.close()

    if outputfile:
        np.save(outputfile, results)
    return results


def q8_tetrahedral(
    snapshots: Snapshots,
    ppp: npt.NDArray = np.array([1, 1, 1]),
    outputfile: str = "",
) -> npt.NDArray:
    """
    Calculate local tetrahedral order of the simulation system,
    such as for water-type and silicon-type systems in three dimensions

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. ppp (npt.NDArray): the periodic boundary conditions,
                        setting 1 for yes and 0 for no, default npt.NDArray=np.array([1,1,1])
        3. outputfile (str): file name to save the calculated local tetrahedral order, default None
                             To reduce storage size and ensure loading speed, save npy file as default with extension ".npy".
                             If the file extension is ".dat" or ".txt", also saved a text file.

    Return:
        calculated local tetrahedral order in npt.NDArray with shape [nsnapshots, nparticle]
    """
    logger.info("Start calculating local tetrahedral order q8")

    assert len({snapshot.nparticle for snapshot in snapshots.snapshots}) == 1, "Paticle number changes during simulation"
    assert len({tuple(snapshot.boxlength) for snapshot in snapshots.snapshots}) == 1, "Simulation box length changes during simulation"
    assert ppp.shape[0] == 3, "Simulation box is in three dimensions"

    # only consider the nearest four neighbors
    num_nearest = 4
    results = np.zeros((snapshots.nsnapshots, snapshots.snapshots[0].nparticle))
    for n, snapshot in enumerate(snapshots.snapshots):
        for i in range(snapshot.nparticle):
            RIJ = snapshot.positions - snapshot.positions[i]
            RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
            distance = np.linalg.norm(RIJ, axis=1)
            nearests = np.argpartition(distance, num_nearest + 1)[: num_nearest + 1]
            nearests = [j for j in nearests if j != i]
            for j in range(num_nearest - 1):
                for k in range(j + 1, num_nearest):
                    medium1 = np.dot(RIJ[nearests[j]], RIJ[nearests[k]])
                    medium2 = distance[nearests[j]] * distance[nearests[k]]
                    results[n, i] += (medium1 / medium2 + 1.0 / 3) ** 2
    results = 1.0 - 3.0 / 8 * results / num_nearest
    if outputfile:
        np.save(outputfile, results)
    return results
