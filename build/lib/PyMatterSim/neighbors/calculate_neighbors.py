# coding = utf-8

"""see documentation @ ../../docs/neighbors.md"""

import re

import numpy as np
import numpy.typing as npt

from ..reader.reader_utils import Snapshots
from ..utils.logging import get_logger_handle
from ..utils.pbc import remove_pbc

logger = get_logger_handle(__name__)

# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def Nnearests(
    snapshots: Snapshots,
    N: int = 12,
    ppp: npt.NDArray = np.array([1, 1, 1]),
    fnfile: str = 'neighborlist.dat'
) -> None:
    """
    Get the N nearest neighbors of a particle.

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)

        2. N (int): the number of nearest neighbors, default=12

        3. ppp (npt.NDArray): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default np.array([1,1,1]), that is, PBC is applied in all three dimensions for 3D box.
                       set np.array([1,1]) for two-dimensional systems

        4. fnfile (str): the name of output file that stores the calculated neighborlist
                         default is 'neighborlist.dat'

    Return:
        None [output neighbor list to a document]
    """
    logger.info(f"Calculate {N} nearest neighbors for a {len(ppp)}-dimensional system")

    fneighbor = open(fnfile, 'w', encoding="utf-8")
    for snapshot in snapshots.snapshots:
        hmatrix = snapshot.hmatrix
        positions = snapshot.positions
        nparticle = snapshot.nparticle
        neighbor = np.zeros((nparticle, 2 + N), dtype=np.int32)
        neighbor[:, 0] = np.arange(nparticle) + 1
        neighbor[:, 1] = N
        for i in range(nparticle):
            RIJ = positions - positions[i]
            RIJ = remove_pbc(RIJ, hmatrix, ppp)
            RIJ_norm = np.linalg.norm(RIJ, axis=1)
            nearests = np.argpartition(RIJ_norm, N + 1)[:N + 1]
            # sort nearests based on distance
            nearests = nearests[RIJ_norm[nearests].argsort()]
            # nearests include the centered atom itself, so indexing [1:]
            # the saved particle ID is numbered starting from 1
            neighbor[i, 2:] = nearests[1:] + 1
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # the neighborlist of each snapshot starts with header "id cn
        # neighborlist"
        fneighbor.write('id     cn     neighborlist\n')
        fneighbor.write(
            re.sub(
                r'[\[\]]',
                ' ',
                np.array2string(neighbor) +
                '\n'))

    fneighbor.close()
    logger.info(f"{N}-nearest neighbors saved to {fnfile}")


def cutoffneighbors(
    snapshots: Snapshots,
    r_cut: float,
    ppp: npt.NDArray = np.array([1, 1, 1]),
    fnfile: str = 'neighborlist.dat'
) -> None:
    """
    Get the nearest neighbors around a particle by setting a global cutoff distance r_cut
    This is useful for single-component systems

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): returned by reader.dump_reader.DumpReader

        2. r_cut (float): the global cutoff distance to screen the nearest neighbors

        3. ppp (npt.NDArray): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default np.array([1,1,1]), that is, PBC is applied in all three dimensions for 3D box
                       set np.array([1,1]) for two dimensional systems

        4. fnfile (str): the name of output file that stores the calculated neighborlist
                         default is 'neighborlist.dat'

    Return:
        None [saved to fnfile]
    """

    logger.info( f"Calculate neighbors within {r_cut} for a {len(ppp)} - dimensional system")
    fneighbor = open(fnfile, 'w', encoding="utf-8")
    for snapshot in snapshots.snapshots:
        hmatrix = snapshot.hmatrix
        positions = snapshot.positions
        nparticle = snapshot.nparticle
        neighbor = np.arange(nparticle).astype(np.int32)
        fneighbor.write('id     cn     neighborlist\n')
        for i in range(nparticle):
            RIJ = positions - positions[i]
            RIJ = remove_pbc(RIJ, hmatrix, ppp)
            RIJ_norm = np.linalg.norm(RIJ, axis=1)
            nearests = neighbor[RIJ_norm <= r_cut]
            CN = nearests.shape[0] - 1
            nearests = nearests[RIJ_norm[nearests].argsort()]
            # nearests include the centered atom itself, so indexing [1:]
            nearests = nearests[1:] + 1
            # the saved particle ID is numbered starting from 1
            # the neighborlist of each snapshot starts with header "id cn
            # neighborlist"
            fneighbor.write('%d %d ' % (i + 1, CN))
            fneighbor.write(' '.join(map(str, nearests)))
            fneighbor.write('\n')

    fneighbor.close()
    logger.info(f"Neighbors within {r_cut} saved to {fnfile}")


def cutoffneighbors_particletype(
    snapshots: Snapshots,
    r_cut: npt.NDArray,
    ppp: npt.NDArray = np.array([1, 1, 1]),
    fnfile: str = 'neighborlist.dat'
) -> None:
    """
    Get the nearest neighbors around a particle by setting a cutoff distance r_cut
    for each particle type pair, should be used for multi-component systems

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): returned by reader.dump_reader.DumpReader

        2. r_cut (npt.NDArray): the cutoff distances of each particle pair, for example,
                             for a binary system A-B should be np.array([[A-A, A-B], [B-A, B-B]]))
                             Usually, these cutoff distances can be determined as the position of
                             the first valley in partial pair correlation function of each pair.

        3. ppp (npt.NDArray): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default np.array([1,1,1]), that is, PBC is applied in all three dimensions for 3D box
                       set np.array([1,1]) for two-dimensional system

        4. fnfile (str): the name of output file that stores the calculated neighborlist
                         default is 'neighborlist.dat'
    """

    logger.info(f"Calculate the particle type specific cutoff neighbors for "
                f"{len(ppp)}-dimensional system")

    if not isinstance(r_cut, np.ndarray):
        errorinfo = "input r_cut type error: please give a numpy array over all pairs\n"
        errorinfo += "shape of r_cut input is (atom_type_number, atom_type_number)"
        raise IOError(errorinfo)

    nparticle_type = np.unique(snapshots.snapshots[0].particle_type).shape[0]

    if r_cut.shape[0] != nparticle_type:
        errorinfo = 'Wrong atom_type_number for input r_cut'
        raise IOError(errorinfo)

    # define cutoffs for each pair based on particle type
    cutoffs = np.zeros((nparticle_type, snapshots.snapshots[0].nparticle))
    for i in range(cutoffs.shape[0]):
        for j in range(cutoffs.shape[1]):
            cutoffs[i, j] = r_cut[i, snapshots.snapshots[0].particle_type[j] - 1]

    fneighbor = open(fnfile, 'w', encoding="utf-8")
    for snapshot in snapshots.snapshots:
        hmatrix = snapshot.hmatrix
        positions = snapshot.positions
        nparticle = snapshot.nparticle
        particle_type = snapshot.particle_type
        neighbor = np.arange(nparticle).astype(np.int32)
        fneighbor.write('id     cn     neighborlist\n')
        for i in range(nparticle):
            RIJ = positions - positions[i]
            RIJ = remove_pbc(RIJ, hmatrix, ppp)
            RIJ_norm = np.linalg.norm(RIJ, axis=1)
            i_cutoffs = cutoffs[particle_type[i] - 1]
            nearests = neighbor[(RIJ_norm - i_cutoffs) <= 0]
            CN = nearests.shape[0] - 1
            nearests = nearests[RIJ_norm[nearests].argsort()]
            nearests = nearests[1:] + 1
            fneighbor.write('%d %d ' % (i + 1, CN))
            fneighbor.write(' '.join(map(str, nearests)))
            fneighbor.write('\n')
    fneighbor.close()

    logger.info(f"Particle-type specific neighbors saved to {fnfile}")
