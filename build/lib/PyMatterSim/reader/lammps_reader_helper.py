# coding = utf-8
"""This module provide helper functions to read lammps files"""

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..reader.reader_utils import SingleSnapshot, Snapshots
from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def read_lammps_wrapper(file_name: str, ndim: int) -> Snapshots:
    """
    Wrapper function around read lammps atomistic system

    input:
        1. filename (str): the name of dump file

        2. ndim (int): dimensionality

    Return:
        Snapshots ([SingleSnapshot]): list of snapshots for the input file_name
    """
    logger.info('--------Start Reading LAMMPS Atomic Dump---------')
    snapshots = []
    nsnapshots = 0
    with open(file_name, "r", encoding="utf-8") as f:
        while True:
            snapshot = read_lammps(f, ndim)
            if not snapshot:
                break
            snapshots.append(snapshot)
            nsnapshots += 1
    logger.info('-------LAMMPS Atomic Dump Reading Completed--------')
    return Snapshots(nsnapshots=nsnapshots, snapshots=snapshots)


def read_lammps_centertype_wrapper(
        file_name: str, ndim: int, moltypes: Dict[int, int]) -> Snapshots:
    """
    Wrapper function around read lammps molecular system

    Inputs:
        1. filename (str): the name of dump file

        2. ndim (int): dimensionality

        3. moltypes (dict, optional): only used for molecular system in LAMMPS, default is None.
            To specify, for example, if the system has 5 types of atoms in which 1-3 is
            one type of molecules and 4-5 is the other, and type 3 and 5 are the center of mass.
            Then moltypes should be {3:1, 5:2}. The keys ([3, 5]) of the dict (moltypes)
            are used to select specific atoms to present the corresponding molecules.
            The values ([1, 2]) is used to record the type of molecules.

    Return:
        Snapshots ([SingleSnapshot]): list of snapshots for the input file_name
    """
    logger.info('-----Start Reading LAMMPS Molecule Center Dump-----')
    snapshots = []
    nsnapshots = 0
    with open(file_name, 'r', encoding="utf-8") as f:
        while True:
            snapshot = read_lammps_centertype(f, ndim, moltypes)
            if not snapshot:
                break
            snapshots.append(snapshot)
            nsnapshots += 1
    logger.info('---LAMMPS Molecule Center Dump Reading Completed---')
    return Snapshots(nsnapshots=nsnapshots, snapshots=snapshots)


def read_lammps_vector_wrapper(
    file_name: str, ndim: int, columnsids: List[int]
) -> Snapshots:
    """
    Wrapper function for reading additional information from a lammps configuration
    For example, the dumped config is "id type x y vx vy", getting the information "vx vy"
    Note that `Positions` is used to take these additional column(s).
    The H-matrix only considers the diagonal elements from simulation box length

    Inputs:
        1. filename (str): the name of dump file
        2. ndim (int): dimensionality
        3. columnsids (list of int): column id for additional information,
                for example, [5, 6] for "vx vy" above

    Return:
        Snapshots ([SingleSnapshot]): list of snapshots for the input file_name
    """
    logger.info('-----Start Reading LAMMPS Dump with Additional Column(s)-----')
    if len(columnsids) == 1:
        logger.info("LAMMPS additional information read is a scalar")
    elif len(columnsids) > 1:
        logger.info(f"LAMMPS additional information read is a vector of length {len(columnsids)}")
    else:
        raise ValueError("Empty input variable columnsids")

    snapshots = []
    nsnapshots = 0
    with open(file_name, 'r', encoding="utf-8") as f:
        while True:
            snapshot = read_lammps_vector(f, ndim, columnsids)
            if not snapshot:
                break
            snapshots.append(snapshot)
            nsnapshots += 1
    logger.info('---LAMMPS Dump with Additional Column(s) Reading Completed---')
    return Snapshots(nsnapshots=nsnapshots, snapshots=snapshots)


def read_lammps(f: Any, ndim: int) -> SingleSnapshot:
    """
    Read a snapshot at one time from LAMMPS atomistic system

    Inputs:
        1. f: file open by python for the input dump file

        2. ndim: dimensionality

    Return:
        Single snapshot information
    """
    item = f.readline()
    # End of file:
    if not item:
        logger.info("Reach end of file.")
        return None
    timestep = int(f.readline())
    item = f.readline()
    particle_number = int(f.readline())

    item = f.readline().split()
    # -------Read Orthogonal Boxes---------
    if 'xy' not in item:
        # box boundaries of (x y z)
        boxbounds = np.zeros((ndim, 2))
        boxlength = np.zeros(ndim)  # box length along (x y z)
        for i in range(ndim):
            item = f.readline().split()
            boxbounds[i, :] = item[:2]

        boxlength = boxbounds[:, 1] - boxbounds[:, 0]
        if ndim < 3:
            for i in range(3 - ndim):
                f.readline()
        # shiftfactors = (boxbounds[:, 0] + boxlength / 2)
        # self.Boxbounds.append(boxbounds)
        hmatrix = np.diag(boxlength)

        item = f.readline().split()
        names = item[2:]
        positions = np.zeros((particle_number, ndim))
        particle_type = np.zeros(particle_number, dtype=int)

        if 'xu' in names or 'x' in names:
            for i in range(particle_number):
                item = f.readline().split()
                atom_index = int(item[0]) - 1
                particle_type[atom_index] = int(item[1])
                positions[atom_index] = [float(j) for j in item[2: ndim + 2]]

            if 'x' in names:
                positions = np.where(
                    positions < boxbounds[:, 0], positions + boxlength, positions)
                positions = np.where(
                    positions > boxbounds[:, 1], positions - boxlength, positions)

        elif 'xs' in names:
            for i in range(particle_number):
                item = f.readline().split()
                atom_index = int(item[0]) - 1
                particle_type[atom_index] = int(item[1])
                positions[atom_index] = [
                    float(j) for j in item[2: ndim + 2]] * boxlength

        snapshot = SingleSnapshot(
            timestep=timestep,
            nparticle=particle_number,
            particle_type=particle_type,
            positions=positions,
            boxlength=boxlength,
            boxbounds=boxbounds,
            realbounds=None,
            hmatrix=hmatrix,
        )

    # -------Read Triclinic Boxes---------
    else:
        # box boundaries of (x y z) with tilt factors
        boxbounds = np.zeros((ndim, 3))
        boxlength = np.zeros(ndim)  # box length along (x y z)
        for i in range(ndim):
            item = f.readline().split()
            boxbounds[i, :] = item[:3]  # with tilt factors
        if ndim < 3:
            for i in range(3 - ndim):
                item = f.readline().split()
                boxbounds = np.vstack(
                    (boxbounds, np.array(item[:3], dtype=np.float64)))

        xlo_bound, xhi_bound, xy = boxbounds[0, :]
        ylo_bound, yhi_bound, xz = boxbounds[1, :]
        zlo_bound, zhi_bound, yz = boxbounds[2, :]
        xlo = xlo_bound - min((0.0, xy, xz, xy + xz))
        xhi = xhi_bound - max((0.0, xy, xz, xy + xz))
        ylo = ylo_bound - min((0.0, yz))
        yhi = yhi_bound - max((0.0, yz))
        zlo = zlo_bound
        zhi = zhi_bound
        h0 = xhi - xlo
        h1 = yhi - ylo
        h2 = zhi - zlo
        h3 = yz
        h4 = xz
        h5 = xy

        realbounds = np.array(
            [xlo, xhi, ylo, yhi, zlo, zhi]).reshape((3, 2))
        reallength = (realbounds[:, 1] - realbounds[:, 0])[:ndim]
        boxbounds = boxbounds[:ndim, :2]
        hmatrix = np.zeros((3, 3))
        hmatrix[0] = [h0, 0, 0]
        hmatrix[1] = [h5, h1, 0]
        hmatrix[2] = [h4, h3, h2]
        hmatrix = hmatrix[:ndim, :ndim]

        item = f.readline().split()
        names = item[2:]
        positions = np.zeros((particle_number, ndim))
        particle_type = np.zeros(particle_number, dtype=int)
        if 'x' in names:
            for i in range(particle_number):
                item = f.readline().split()
                atom_index = int(item[0]) - 1
                particle_type[atom_index] = int(item[1])
                positions[atom_index] = [float(j) for j in item[2: ndim + 2]]

        elif 'xs' in names:
            for i in range(particle_number):
                item = f.readline().split()
                atom_index = int(item[0]) - 1
                particle_type[atom_index] = int(item[1])
                if ndim == 3:
                    positions[atom_index, 0] = xlo_bound + float(item[2]) * h0 + float(
                        item[3]) * h5 + float(item[4]) * h4
                    positions[atom_index, 1] = ylo_bound + \
                        float(item[3]) * h1 + float(item[4]) * h3
                    positions[atom_index, 2] = zlo_bound + float(item[4]) * h2
                elif ndim == 2:
                    positions[atom_index, 0] = xlo_bound + \
                        float(item[2]) * h0 + float(item[3]) * h5
                    positions[atom_index, 1] = ylo_bound + float(item[3]) * h1
                else:
                    logger.info(
                        f"cannot read for {ndim} dimensionality so far")
                    return None

        snapshot = SingleSnapshot(
            timestep=timestep,
            nparticle=particle_number,
            particle_type=particle_type,
            positions=positions,
            boxlength=reallength,
            boxbounds=boxbounds,
            realbounds=realbounds[:ndim],
            hmatrix=hmatrix,
        )
    return snapshot


def read_lammps_centertype(
    f: Any,
    ndim: int,
    moltypes: Dict[int, int]
) -> SingleSnapshot:
    """ Read a snapshot of molecules at one time from LAMMPS

    Inputs:
        1. f: open file type by python from reading input dump file

        2. ndim (int): dimensionality

        3. moltypes (dict, optional): only used for molecular system in LAMMPS, default is None.
            To specify, for example, if the system has 5 types of atoms in which 1-3 is
            one type of molecules and 4-5 is the other, and type 3 and 5 are the center of mass.
            Then moltypes should be {3:1, 5:2}. The keys ([3, 5]) of the dict (moltypes)
            are used to select specific atoms to present the corresponding molecules.
            The values ([1, 2]) is used to record the type of molecules.
    """

    item = f.readline()
    # End of file:
    if not item:
        logger.info("Reach end of file.")
        return None
    timestep = int(f.readline())
    item = f.readline()
    particle_number = int(f.readline())
    item = f.readline().split()
    # -------Read Orthogonal Boxes---------
    boxbounds = np.zeros((ndim, 2))  # box boundaries of (x y z)
    boxlength = np.zeros(ndim)  # box length along (x y z)
    for i in range(ndim):
        item = f.readline().split()
        boxbounds[i, :] = item[:2]

    boxlength = boxbounds[:, 1] - boxbounds[:, 0]
    if ndim < 3:
        for i in range(3 - ndim):
            f.readline()
    # shiftfactors = (boxbounds[:, 0] + boxlength / 2)
    # self.Boxbounds.append(boxbounds)
    hmatrix = np.diag(boxlength)

    item = f.readline().split()
    names = item[2:]
    particle_type = np.zeros(particle_number, dtype=int)
    positions = np.zeros((particle_number, ndim))
    # MoleculeType = np.zeros(particle_number, dtype=int)

    if 'xu' in names or 'x' in names:
        for i in range(particle_number):
            item = f.readline().split()
            atom_index = int(item[0]) - 1
            particle_type[atom_index] = int(item[1])
            positions[atom_index] = [float(j) for j in item[2: ndim + 2]]
            # MoleculeType[atom_index] = int(item[-1])

        conditions = [True if atomtype in moltypes.keys()
                      else False for atomtype in particle_type]
        positions = positions[conditions]
        particle_type = pd.Series(
            particle_type[conditions]).map(moltypes).values

        if 'x' in names:
            positions = np.where(
                positions < boxbounds[:, 0], positions + boxlength, positions)
            positions = np.where(
                positions > boxbounds[:, 1], positions - boxlength, positions)

    elif 'xs' in names:
        for i in range(particle_number):
            item = f.readline().split()
            atom_index = int(item[0]) - 1
            particle_type[atom_index] = int(item[1])
            positions[atom_index] = [float(j)
                                     for j in item[2: ndim + 2]] * boxlength
            # MoleculeType[atom_index] = int(item[-1])

        # choose only center-of-mass
        conditions = [True if atomtype in moltypes.keys()
                      else False for atomtype in particle_type]
        positions = positions[conditions]
        particle_type = pd.Series(
            particle_type[conditions]).map(
            moltypes).values
        positions += boxbounds[:, 0]

    snapshot = SingleSnapshot(
        timestep=timestep,
        nparticle=particle_type.shape[0],
        particle_type=particle_type,
        positions=positions,
        boxlength=boxlength,
        boxbounds=boxbounds,
        realbounds=None,
        hmatrix=hmatrix,
    )
    return snapshot


def read_lammps_vector(
    f: Any,
    ndim: int,
    columnsids: List[int]
) -> SingleSnapshot:
    """
    Read additional column(s) information from LAMMPS configurations
    For example, the dumped config is "id type x y vx vy", getting the information "vx vy"
    Note that `Positions` is used to take these additional column(s)

    Inputs:
        1. f: open file type by python from reading input dump file
        2. ndim (int): dimensionality
        3. columnsids (list of int): column id for additional information,
                for example, [5, 6] for "vx vy" above

    Return:
        single snapshot object
    """

    item = f.readline()
    # End of file:
    if not item:
        logger.info("Reach end of file.")
        return None
    timestep = int(f.readline())
    item = f.readline()
    particle_number = int(f.readline())
    item = f.readline().split()
    # -------Read Orthogonal Boxes---------
    boxbounds = np.zeros((ndim, 2))  # box boundaries of (x y z)
    boxlength = np.zeros(ndim)  # box length along (x y z)
    for i in range(ndim):
        item = f.readline().split()
        boxbounds[i, :] = item[:2]

    boxlength = boxbounds[:, 1] - boxbounds[:, 0]
    if ndim < 3:
        for i in range(3 - ndim):
            f.readline()
    # shiftfactors = (boxbounds[:, 0] + boxlength / 2)
    # self.Boxbounds.append(boxbounds)
    hmatrix = np.diag(boxlength)

    item = f.readline().split()
    particle_type = np.zeros(particle_number, dtype=int)
    particle_vector = np.zeros((particle_number, len(columnsids)))
    columns_index = [int(i - 1) for i in columnsids]
    for i in range(particle_number):
        item = f.readline().split()
        atom_index = int(item[0]) - 1
        particle_type[atom_index] = int(item[1])
        particle_vector[atom_index] = [float(item[j]) for j in columns_index]

    snapshot = SingleSnapshot(
        timestep=timestep,
        nparticle=particle_number,
        particle_type=particle_type,
        positions=particle_vector,
        boxlength=boxlength,
        boxbounds=boxbounds,
        realbounds=None,
        hmatrix=hmatrix,
    )
    return snapshot


def read_additions(dumpfile, ncol) -> npt.NDArray:
    """
    Read additional columns in the lammps dump file
    for example, read "order" from:
        id type x y z order

    Inputs:
        1. dumpfile (str): file name of input snapshots

        2. ncol (int): specifying the column number starting from 0 (zero-based)

    Return:
        in numpy array as [snapshot_number, particle_number] in float
    """

    with open(dumpfile, "r", encoding="utf-8") as f:
        content = f.readlines()

    nparticles = int(content[3])
    nsnapshots = int(len(content) / (nparticles + 9))
    results = np.zeros((nsnapshots, nparticles))

    for n in range(nsnapshots):
        items = content[n * nparticles +
                        (n + 1) * 9:(n + 1) * (nparticles + 9)]
        for item in items:
            item = item.split()
            atom_index = int(item[0]) - 1
            results[n, atom_index] = item[ncol]
    return results
