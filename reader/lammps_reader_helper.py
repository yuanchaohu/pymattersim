"""This module provide helper functions to read lammps files"""
from typing import Dict, Any
import numpy as np
import pandas as pd

from utils.logging_utils import get_logger_handle
from reader.reader_utils import SingleSnapshot, Snapshots

logger = get_logger_handle(__name__)


def read_lammps_wrapper(file_name: str, ndim: int) -> Snapshots:
    """Wrapper function around read lammps atomistic system"""
    logger.info('--------Start Reading LAMMPS Atomic Dump---------')
    f = open(file_name, 'r')
    nsnapshots = 0
    snapshots = []
    while True:
        snapshot = read_lammps(f, ndim)
        if not snapshot:
            break
        snapshots.append(snapshot)
        nsnapshots += 1
    f.close()
    logger.info('-------LAMMPS Atomic Dump Reading Completed--------')
    return Snapshots(nsnapshots=nsnapshots, snapshots=snapshots)


def read_lammps_centertype_wrapper(
        file_name: str, ndim: int, moltypes: Dict[int, int]) -> Snapshots:
    """Wrapper function around read lammps molecular system"""
    logger.info('-----Start Reading LAMMPS Molecule Center Dump-----')
    f = open(file_name, 'r')
    nsnapshots = 0
    snapshots = []
    while True:
        snapshot = read_lammps_centertype(f, ndim, moltypes)
        if not snapshot:
            break
        snapshots.append(snapshot)
        nsnapshots += 1
    f.close()
    logger.info('---LAMMPS Molecule Center Dump Reading Completed---')
    return Snapshots(nsnapshots=nsnapshots, snapshots=snapshots)


def read_lammps(f: Any, ndim: int) -> SingleSnapshot:
    """ Read a snapshot at one time from LAMMPS atomistic system"""
    try:
        item = f.readline()
        # End of file:
        if not item:
            logger.info("Reach end of file.")
            return
        timestep = int(f.readline().split()[0])
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
            shiftfactors = (boxbounds[:, 0] + boxlength / 2)
            # self.Boxbounds.append(boxbounds)
            hmatrix = np.diag(boxlength)

            item = f.readline().split()
            names = item[2:]
            positions = np.zeros((particle_number, ndim))
            particle_type = np.zeros(particle_number, dtype=int)

            if 'xu' in names:
                for i in range(particle_number):
                    item = f.readline().split()
                    # store particle type and sort particle by ID
                    particle_type[int(item[0]) - 1] = int(item[1])
                    # store particle positions and sort particle by ID
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]]

            elif 'x' in names:
                for i in range(particle_number):
                    item = f.readline().split()
                    particle_type[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]]

                # Moving particles outside the box back into the box while
                # applying periodic boundary conditions
                positions = np.where(
                    positions < boxbounds[:, 0], positions + boxlength, positions)
                # Moving particles outside the box back into the box while
                # applying periodic boundary conditions
                positions = np.where(
                    positions > boxbounds[:, 1], positions - boxlength, positions)
                # positions = positions - shiftfactors[np.newaxis, :]
                # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

            elif 'xs' in names:
                for i in range(particle_number):
                    item = f.readline().split()
                    particle_type[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]] * boxlength

                positions += boxbounds[:, 0]
                positions = np.where(
                    positions < boxbounds[:, 0], positions + boxlength, positions)
                positions = np.where(
                    positions > boxbounds[:, 1], positions - boxlength, positions)
                # positions = positions - shiftfactors[np.newaxis, :]
                # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

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
            return snapshot

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
                    particle_type[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]]

            elif 'xs' in names:
                for i in range(particle_number):
                    item = f.readline().split()
                    pid = int(item[0]) - 1
                    particle_type[pid] = int(item[1])
                    if ndim == 3:
                        positions[pid, 0] = xlo_bound + float(item[2]) * h0 + float(
                            item[3]) * h5 + float(item[4]) * h4
                        positions[pid, 1] = ylo_bound + \
                            float(item[3]) * h1 + float(item[4]) * h3
                        positions[pid, 2] = zlo_bound + float(item[4]) * h2
                    elif ndim == 2:
                        positions[pid, 0] = xlo_bound + \
                            float(item[2]) * h0 + float(item[3]) * h5
                        positions[pid, 1] = ylo_bound + float(item[3]) * h1

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

    except BaseException:
        logger.error("Exception when reading file.")
        return None


def read_lammps_centertype(f: Any,
                           ndim: int,
                           moltypes: Dict[int,
                                          int]) -> SingleSnapshot:
    """ Read a snapshot of molecules at one time from LAMMPS

        moltypes is a dict mapping atomic type to molecular type
        such as {3: 1, 5: 2}
        moltypes.keys() are atomic types of each molecule
        moltypes.values() are the modified molecule type

        ONLY dump the center-of-mass of each molecule
    """

    try:
        item = f.readline()
        # End of file:
        if not item:
            logger.info("Reach end of file.")
            return
        timestep = int(f.readline().split()[0])
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
        shiftfactors = (boxbounds[:, 0] + boxlength / 2)
        # self.Boxbounds.append(boxbounds)
        hmatrix = np.diag(boxlength)

        item = f.readline().split()
        names = item[2:]
        particle_type = np.zeros(particle_number, dtype=int)
        positions = np.zeros((particle_number, ndim))
        # MoleculeType = np.zeros(particle_number, dtype=int)

        if 'xu' in names:
            for i in range(particle_number):
                item = f.readline().split()
                particle_type[int(item[0]) - 1] = int(item[1])
                positions[int(item[0]) - 1] = [float(j)
                                               for j in item[2: ndim + 2]]
                # MoleculeType[int(item[0]) - 1] = int(item[-1])

            conditions = [True if atomtype in moltypes.keys()
                          else False for atomtype in particle_type]
            positions = positions[conditions]
            particle_type = pd.Series(
                particle_type[conditions]).map(
                moltypes).values

        elif 'x' in names:
            for i in range(particle_number):
                item = f.readline().split()
                particle_type[int(item[0]) - 1] = int(item[1])
                positions[int(item[0]) - 1] = [float(j)
                                               for j in item[2: ndim + 2]]
                # MoleculeType[int(item[0]) - 1] = int(item[-1])

            conditions = [True if atomtype in moltypes.keys()
                          else False for atomtype in particle_type]
            positions = positions[conditions]
            particle_type = pd.Series(
                particle_type[conditions]).map(
                moltypes).values
            positions = np.where(
                positions < boxbounds[:, 0], positions + boxlength, positions)
            positions = np.where(
                positions > boxbounds[:, 1], positions - boxlength, positions)
            # positions = positions - shiftfactors[np.newaxis, :]
            # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

        elif 'xs' in names:
            for i in range(particle_number):
                item = f.readline().split()
                particle_type[int(item[0]) - 1] = int(item[1])
                positions[int(item[0]) - 1] = [float(j)
                                               for j in item[2: ndim + 2]] * boxlength
                # MoleculeType[int(item[0]) - 1] = int(item[-1])

            conditions = [True if atomtype in moltypes.keys(
            ) else False for atomtype in particle_type]
            positions = positions[conditions]
            particle_type = pd.Series(
                particle_type[conditions]).map(
                moltypes).values
            positions += boxbounds[:, 0]
            positions = np.where(
                positions < boxbounds[:, 0], positions + boxlength, positions)
            positions = np.where(
                positions > boxbounds[:, 1], positions - boxlength, positions)
            # positions = positions - shiftfactors[np.newaxis, :]
            # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

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

    except BaseException:
        logger.error("Exception when reading file.")
        return None


def read_additions(dumpfile, ncol):
    """read additional columns in the lammps dump file

    ncol: int, specifying the column number starting from 0 (zero-based)
    return in numpy array as [particle_number, snapshotnumber] in float, sort particle by ID
    """

    with open(dumpfile) as f:
        content = f.readlines()

    nparticle = int(content[3])
    nsnapshots = int(len(content) / (nparticle + 9))

    results = np.zeros((nparticle, nsnapshots))

    for n in range(nsnapshots):
        items = content[n * nparticle + (n + 1) * 9:(n + 1) * (nparticle + 9)]
        for i in range(nparticle):
            item = items[i].split()
            item = np.array([float(_) for _ in item])
            results[int(item[0]) - 1, n] = item[ncol]   # sort particle by ID

    return results


def read_lammpslog(filename):
    """extract the thermodynamic quantities from lammp log file"""

    with open(filename, 'r') as f:
        data = f.readlines()

    # ----get how many sections are there----
    start = [i for i, val in enumerate(data) if val.startswith('Step ')]
    end = [i for i, val in enumerate(data) if val.startswith('Loop time of ')]

    if data[-1] != '\n':
        if data[-1].split()[0].isnumeric():  # incomplete log file
            end.append(len(data) - 2)

    start = np.array(start)
    end = np.array(end)
    linenum = end - start - 1
    logger.info(f'Section Number: {len(linenum)} Line Numbers: {str(linenum)}')
    del data

    final = []
    for i in range(len(linenum)):
        data = pd.read_csv(
            filename,
            sep='\\s+',
            skiprows=start[i],
            nrows=linenum[i])
        final.append(data)
        del data

    return final
