# coding = utf-8

import re
import numpy as np
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc

Docstr = """
         This module is used to calculate the neighbors around a particle.
         The format of the saved neighbor list file (named as 'neighborlist.dat' in default)
         must be identification of the centered particle, coordination number of the centered particle,
         and identification of neighboring particles. In default, the neighbors in the output file is
         sorted by their distances to the centered particle in the ascending order.
         neighbor list of different snapshots is continuous without any gap
         all start with the header "id cn neighborlist",
        """

# pylint: disable=invalid-name

def Nnearests(
        snapshots: Snapshots,
        N: int = 12,
        ppp: list = [1, 1, 1],
        fnfile='neighborlist.dat') -> None:
    """
    Get the N nearest neighbors around a particle.

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): returned by reader.dump_reader.DumpReader

        2. N (int): the number of nearest neighbors, default 12

        3. ppp (list): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box

        4. fnfile (str): the name of output file that stores the calculated neighborlist
                         default is 'neighborlist.dat'
    """
    fneighbor = open(fnfile, 'w')
    for n in range(snapshots.nsnapshots):
        hmatrix = snapshots.snapshots[n].hmatrix
        positions = snapshots.snapshots[n].positions
        nparticle = snapshots.snapshots[n].nparticle
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
        # the neighborlist of each snapshot starts with header "id cn neighborlist"
        fneighbor.write('id     cn     neighborlist\n')
        fneighbor.write(re.sub('[\[\]]', ' ', np.array2string(neighbor) + '\n'))

    fneighbor.close()
    print('---Calculate %d nearest neighbors done---' % N)


def cutoffneighbors(
        snapshots: Snapshots,
        r_cut: float,
        ppp: list = [1, 1, 1],
        fnfile='neighborlist.dat') -> None:
    """
    Get the nearest neighbors around a particle by setting a global cutoff distance r_cut

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): returned by reader.dump_reader.DumpReader

        2. r_cut (float): the global cutoff distance to screen the nearest neighbors

        3. ppp (list): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box

        4. fnfile (str): the name of output file that stores the calculated neighborlist
                         default is 'neighborlist.dat'
    """
    fneighbor = open(fnfile, 'w')
    for n in range(snapshots.nsnapshots):
        hmatrix = snapshots.snapshots[n].hmatrix
        positions = snapshots.snapshots[n].positions
        nparticle = snapshots.snapshots[n].nparticle
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
            # the neighborlist of each snapshot starts with header "id cn neighborlist"
            fneighbor.write('%d %d ' % (i + 1, CN))
            fneighbor.write(' '.join(map(str, nearests)))
            fneighbor.write('\n')

    fneighbor.close()
    print('---Calculate nearest neighbors with r_cut = %.6f done---' % r_cut)


def cutoffneighbors_particletype(
        snapshots: Snapshots,
        r_cut: np.array,
        ppp: list = [1, 1, 1],
        fnfile='neighborlist.dat') -> None:
    """
    Get the nearest neighbors around a particle by setting a cutoff distance r_cut
    for each particle type pair

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): returned by reader.dump_reader.DumpReader

        2. r_cut (np.array): the cutoff distances of each particle pair, for example,
                             for a binary system A-B should be np.array([[A-A, A-B], [B-A, B-B]]))
                             Usually, these cutoff distances can be determined as the position of
                             the first valley in partial pair correlation function of each pair.

        3. ppp (list): the periodic boundary conditions, setting 1 for yes and 0 for no
                       default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box

        4. fnfile (str): the name of output file that stores the calculated neighborlist
                         default is 'neighborlist.dat'
    """
    if type(r_cut) is not np.ndarray:
        errorinfo = "---input r_cut type error: please give a numpy array over all pairs\n"
        errorinfo += "shape of r_cut input is (atom_type_number, atom_type_number)---"
        raise IOError(errorinfo)

    nparticle_type = np.unique(snapshots.snapshots[0].particle_type).shape[0]

    if r_cut.shape[0] != nparticle_type:
        errorinfo = '---Wrong atom_type_number for input r_cut---'
        raise IOError(errorinfo)

    # define cutoffs for each pair based on particle type
    cutoffs = np.zeros((nparticle_type, snapshots.snapshots[0].nparticle))
    for i in range(nparticle_type):
        for j in range(snapshots.snapshots[0].nparticle):
            cutoffs[i, j] = r_cut[i, snapshots.snapshots[0].particle_type[j] - 1]

    fneighbor = open(fnfile, 'w')
    for n in range(snapshots.nsnapshots):
        hmatrix = snapshots.snapshots[n].hmatrix
        positions = snapshots.snapshots[n].positions
        nparticle = snapshots.snapshots[n].nparticle
        particle_type = snapshots.snapshots[n].particle_type
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
    print('---Calculate nearest neighbors for atom-pairs done---')
