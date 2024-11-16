# coding=utf-8

"""see documentation @ ../../docs/neighbors.md"""

from typing import TextIO

import numpy as np
import numpy.typing as npt

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name


def read_neighbors(f: TextIO, nparticle: int, Nmax: int = 200) -> npt.NDArray:
    """
    Read the property of neighboring particles from a saved file,
    as long as the format of file is compatible, as like neighborlist.dat

    Inputs:
        1. f (TextIO): opened file which save the property of neighboring particles,
                       f = open(filename, 'r')

           Opening the file outside of the function ensures that
           when reading a file containing multiple snapshots,
           the file pointer can move to the next snapshot

        2. Nmax (int): the maximum number of neighboring particles to consider
           Setting Nmax to a sufficiently large value is ideal, with the default being 200.
           Nmax is defined to address the varying number of neighboring particles
           among different particles. In this way, we can create a regular
           two-dimensional NumPy array to save the property of neighboring particles.

    Return:
        two-dimensional NumPy array, with shape (nparticle, 1+Nmax_fact)
        Nmax_fact means the maximum coordination number for one particle in the system
        The first column is the coordination numbner (cn), so number of columns plus 1.
        For particles with coordination number less than `Nmax_fact` (which is generally the case),
        the unoccupied positions in `neighborprop` (see source code) are padded with `0`.
    """
    logger.info("Reading neighboring particle properties")

    header = f.readline().split()  # header
    neighborprop = np.zeros((nparticle, Nmax + 1))

    for i in range(nparticle):
        item = f.readline().split()
        atom_index = int(item[0]) - 1

        if "neighborlist" in header:
            if int(item[1]) <= Nmax:
                neighborprop[atom_index, 0] = float(item[1])
                # Be attention to the '-1' after '=', all particle id has been
                # reduced by 1
                neighborprop[atom_index, 1 : (int(item[1]) + 1)] = [float(j) - 1 for j in item[2 : (int(item[1]) + 2)]]
            else:
                neighborprop[atom_index, 0] = Nmax
                neighborprop[atom_index, 1 : Nmax + 1] = [float(j) - 1 for j in item[2 : Nmax + 2]]
                if i == 0:
                    logger.info(f"Too Many neighbors {Nmax}")
                    logger.info("Warning: not for unsorted neighbor list")
        else:
            if int(item[1]) <= Nmax:
                neighborprop[atom_index, 0] = float(item[1])
                neighborprop[atom_index, 1 : (int(item[1]) + 1)] = [float(j) for j in item[2 : (int(item[1]) + 2)]]
            else:
                neighborprop[atom_index, 0] = Nmax
                neighborprop[atom_index, 1 : Nmax + 1] = [float(j) for j in item[2 : Nmax + 2]]
                if i == 0:
                    logger.info(f"Too Many neighbors {Nmax}")
                    logger.info("Warning: not for unsorted neighbor list")

    max_cn = int(neighborprop[:, 0].max())
    if max_cn < Nmax:
        neighborprop = neighborprop[:, : max_cn + 1]  # save storage
    else:
        logger.info("Warning: increase 'Nmax' to include all the neighbors")

    if "neighborlist" in header:  # neighbor list should be integer
        neighborprop = neighborprop.astype(np.int32)

    return neighborprop
