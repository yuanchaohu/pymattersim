import numpy as np
import pandas as pd

from typing import Dict, Any
from utils.logging_utils import get_logger_handle, log_table
from reader.reader_utils import DumpFileType, SingleSnapshot, Snapshots

logger = get_logger_handle(__name__)


def read_lammps_wrapper(file_name: str, ndim: int, **kwargs) -> Snapshots:
    logger.info('--------Start Reading LAMMPS Atomic Dump---------')
    f = open(file_name, 'r')
    snapshots = []
    while True:
        snapshot = read_lammps(f, ndim)
        if not snapshot:
            break
        snapshots.snapshots.append(snapshot)
        snapshots.snapshots_number += 1
    f.close()
    logger.info('-------LAMMPS Atomic Dump Reading Completed--------')
    return snapshots


def read_centertype_wrapper(
        file_name: str, ndim: int, moltypes: Dict[int, int]) -> Snapshots:
    logger.info('-----Start Reading LAMMPS Molecule Center Dump-----')
    f = open(file_name, 'r')
    snapshots = []
    while True:
        snapshot = read_centertype(f, ndim, moltypes)
        if not snapshot:
            break
        snapshots.snapshots.append(snapshot)
        snapshots.snapshots_number += 1
    f.close()
    logger.info('---LAMMPS Molecule Center Dump Reading Completed---')
    return snapshots


def read_lammps(f: Any, ndim: int) -> SingleSnapshot:
    """ Read a snapshot at one time from LAMMPS """

    try:
        item = f.readline()
        timestep = int(f.readline().split()[0])
        item = f.readline()
        ParticleNumber = int(f.readline())

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
            positions = np.zeros((ParticleNumber, ndim))
            ParticleType = np.zeros(ParticleNumber, dtype=np.int)

            if 'xu' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    ParticleType[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]]

            elif 'x' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    ParticleType[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]]

                positions = np.where(
                    positions < boxbounds[:, 0], positions + boxlength, positions)
                positions = np.where(
                    positions > boxbounds[:, 1], positions - boxlength, positions)
                # positions = positions - shiftfactors[np.newaxis, :]
                # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

            elif 'xs' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    ParticleType[int(item[0]) - 1] = int(item[1])
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
                particle_number=ParticleNumber,
                particle_type=ParticleType,
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
                        (boxbounds, np.array(item[:3], dtype=np.float)))

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
            positions = np.zeros((ParticleNumber, ndim))
            ParticleType = np.zeros(ParticleNumber, dtype=np.int)
            if 'x' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    ParticleType[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: ndim + 2]]

            elif 'xs' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    pid = int(item[0]) - 1
                    ParticleType[pid] = int(item[1])
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
                particle_number=ParticleNumber,
                particle_type=ParticleType,
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


def read_centertype(f: Any,
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
        timestep = int(f.readline().split()[0])
        item = f.readline()
        ParticleNumber = int(f.readline())
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
        ParticleType = np.zeros(ParticleNumber, dtype=np.int)
        positions = np.zeros((ParticleNumber, ndim))
        # MoleculeType = np.zeros(ParticleNumber, dtype=np.int)

        if 'xu' in names:
            for i in range(ParticleNumber):
                item = f.readline().split()
                ParticleType[int(item[0]) - 1] = int(item[1])
                positions[int(item[0]) - 1] = [float(j)
                                               for j in item[2: ndim + 2]]
                # MoleculeType[int(item[0]) - 1] = int(item[-1])

            conditions = [True if atomtype in moltypes.keys()
                          else False for atomtype in ParticleType]
            positions = positions[conditions]
            ParticleType = pd.Series(
                ParticleType[conditions]).map(
                moltypes).values

        elif 'x' in names:
            for i in range(ParticleNumber):
                item = f.readline().split()
                ParticleType[int(item[0]) - 1] = int(item[1])
                positions[int(item[0]) - 1] = [float(j)
                                               for j in item[2: ndim + 2]]
                # MoleculeType[int(item[0]) - 1] = int(item[-1])

            conditions = [True if atomtype in moltypes.keys()
                          else False for atomtype in ParticleType]
            positions = positions[conditions]
            ParticleType = pd.Series(
                ParticleType[conditions]).map(
                moltypes).values
            positions = np.where(
                positions < boxbounds[:, 0], positions + boxlength, positions)
            positions = np.where(
                positions > boxbounds[:, 1], positions - boxlength, positions)
            # positions = positions - shiftfactors[np.newaxis, :]
            # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

        elif 'xs' in names:
            for i in range(ParticleNumber):
                item = f.readline().split()
                ParticleType[int(item[0]) - 1] = int(item[1])
                positions[int(item[0]) - 1] = [float(j)
                                               for j in item[2: ndim + 2]] * boxlength
                # MoleculeType[int(item[0]) - 1] = int(item[-1])

            conditions = [True if atomtype in moltypes.keys(
            ) else False for atomtype in ParticleType]
            positions = positions[conditions]
            ParticleType = pd.Series(
                ParticleType[conditions]).map(
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
            particle_number=ParticleType.shape[0],
            particle_type=ParticleType,
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
