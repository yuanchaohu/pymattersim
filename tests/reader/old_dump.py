#!/usr/bin/python
# coding = utf-8
# This module is part of an analysis package

import numpy as np
import pandas as pd


class readdump:
    """Read snapshots from simulations"""

    def __init__(self, filename, ndim, filetype='lammps', moltypes='', *arg):
        self.filename = filename  # input snapshots
        self.ndim = ndim  # dimension
        self.TimeStep = []  # simulation timestep @ each snapshot
        self.ParticleNumber = []  # particle's Number @ each snapshot
        self.ParticleType = []  # particle's type @ each snapshot
        self.Positions = []  # a list containing all snapshots, each element is a snapshot
        self.SnapshotNumber = 0  # snapshot number
        self.Boxlength = []  # box length @ each snapshot
        self.Boxbounds = []  # box boundaries @ each snapshot
        self.Realbounds = []  # real box bounds of a triclinic box
        self.hmatrix = []  # h-matrix of the cells
        self.filetype = filetype  # trajectory type from different MD engines
        self.moltypes = moltypes  # molecule type for lammps molecular trajectories

    def read_onefile(self):
        """ Read all snapshots from one dump file

            The keyword filetype is used for different MD engines
            It has four choices:
            'lammps' (default)

            'lammpscenter' (lammps molecular dump with known atom type of each molecule center)
            moltypes is a dict mapping center atomic type to molecular type
            moltypes is also used to select the center atom
            such as moltypes = {3: 1, 5: 2}

            'gsd' (HOOMD-blue standard output for static properties)

            'gsd_dcd' (HOOMD-blue outputs for static and dynamic properties)

            The simulation box is centered at 0 by default for
            'x' and 'xs' coordinates of 'lammps' & 'lammpscenter'
        """

# ------------------lammps atomic trajectory---------------------------------
        if self.filetype == 'lammps':
            f = open(self.filename, 'r')
            positions = self.read_lammps(f)
            while positions.any():
                self.Positions.append(positions)
                positions = self.read_lammps(f)

            f.close()
            self.SnapshotNumber = len(self.TimeStep)
            print('--------LAMMPS Atomic Dump Reading Over---------')

# ------------------lammps molecular trajectory---------------------------
        if self.filetype == 'lammpscenter':
            f = open(self.filename, 'r')
            positions = self.read_centertype(f)
            while positions.any():
                self.Positions.append(positions)
                positions = self.read_centertype(f)

            f.close()
            self.SnapshotNumber = len(self.TimeStep)
            print('--------LAMMPS Molecule Center Dump Reading Over---------')

# ------------------hoomd-blue gsd trajectory---------------------------------
        if self.filetype == 'gsd':
            import gsd.hoomd
            import gsd

            f = gsd.hoomd.open(self.filename, mode='r')
            self.read_gsd(f)
            print('---------GSD file reading over-----------')

# ------------------hoomd-blue gsd with dcd trajectory--------------------
        if self.filetype == 'gsd_dcd':
            import gsd.hoomd
            import gsd
            import os
            from mdtraj.formats import DCDTrajectoryFile

            gsd_filename = self.filename
            gsd_filepath = os.path.dirname(gsd_filename)
            dcd_filename = gsd_filepath + '/' + \
                os.path.basename(gsd_filename)[:-3] + 'dcd'

            f_gsd = gsd.hoomd.open(gsd_filename, mode='r')
            f_dcd = DCDTrajectoryFile(dcd_filename, 'r')
            self.read_gsd_dcd(f_gsd, f_dcd)
            f_dcd.close()
            print('---------GSD & DCD file reading over-----------')

    def read_multiple():
        """ Read all snapshots from individual dump files """
        pass

    def read_lammps(self, f):
        """ Read a snapshot at one time from LAMMPS """

        try:
            item = f.readline()
            timestep = int(f.readline().split()[0])
            self.TimeStep.append(timestep)
            item = f.readline()
            ParticleNumber = int(f.readline())
            self.ParticleNumber.append(ParticleNumber)

            item = f.readline().split()
            # -------Read Orthogonal Boxes---------
            if 'xy' not in item:
                # box boundaries of (x y z)
                boxbounds = np.zeros((self.ndim, 2))
                boxlength = np.zeros(self.ndim)  # box length along (x y z)
                for i in range(self.ndim):
                    item = f.readline().split()
                    boxbounds[i, :] = item[:2]

                boxlength = boxbounds[:, 1] - boxbounds[:, 0]
                if self.ndim < 3:
                    for i in range(3 - self.ndim):
                        f.readline()
                shiftfactors = (boxbounds[:, 0] + boxlength / 2)
                # self.Boxbounds.append(boxbounds)
                self.Boxlength.append(boxlength)
                hmatrix = np.diag(boxlength)
                self.hmatrix.append(hmatrix)

                item = f.readline().split()
                names = item[2:]
                positions = np.zeros((ParticleNumber, self.ndim))
                ParticleType = np.zeros(ParticleNumber, dtype=int)

                if 'xu' in names:
                    for i in range(ParticleNumber):
                        item = f.readline().split()
                        ParticleType[int(item[0]) - 1] = int(item[1])
                        positions[int(item[0]) - 1] = [float(j)
                                                       for j in item[2: self.ndim + 2]]

                elif 'x' in names:
                    for i in range(ParticleNumber):
                        item = f.readline().split()
                        ParticleType[int(item[0]) - 1] = int(item[1])
                        positions[int(item[0]) - 1] = [float(j)
                                                       for j in item[2: self.ndim + 2]]

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
                                                       for j in item[2: self.ndim + 2]] * boxlength

                    positions += boxbounds[:, 0]
                    positions = np.where(
                        positions < boxbounds[:, 0], positions + boxlength, positions)
                    positions = np.where(
                        positions > boxbounds[:, 1], positions - boxlength, positions)
                    # positions = positions - shiftfactors[np.newaxis, :]
                    # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

                self.Boxbounds.append(boxbounds)
                self.ParticleType.append(ParticleType)
                return positions

            # -------Read Triclinic Boxes---------
            else:
                # box boundaries of (x y z) with tilt factors
                boxbounds = np.zeros((self.ndim, 3))
                boxlength = np.zeros(self.ndim)  # box length along (x y z)
                for i in range(self.ndim):
                    item = f.readline().split()
                    boxbounds[i, :] = item[:3]  # with tilt factors
                if self.ndim < 3:
                    for i in range(3 - self.ndim):
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
                self.Realbounds.append(realbounds[:self.ndim])
                reallength = (realbounds[:, 1] - realbounds[:, 0])[:self.ndim]
                self.Boxlength.append(reallength)
                boxbounds = boxbounds[:self.ndim, :2]
                self.Boxbounds.append(boxbounds)
                hmatrix = np.zeros((3, 3))
                hmatrix[0] = [h0, 0, 0]
                hmatrix[1] = [h5, h1, 0]
                hmatrix[2] = [h4, h3, h2]
                hmatrix = hmatrix[:self.ndim, :self.ndim]
                self.hmatrix.append(hmatrix)

                item = f.readline().split()
                names = item[2:]
                positions = np.zeros((ParticleNumber, self.ndim))
                ParticleType = np.zeros(ParticleNumber, dtype=int)
                if 'x' in names:
                    for i in range(ParticleNumber):
                        item = f.readline().split()
                        ParticleType[int(item[0]) - 1] = int(item[1])
                        positions[int(item[0]) - 1] = [float(j)
                                                       for j in item[2: self.ndim + 2]]

                elif 'xs' in names:
                    for i in range(ParticleNumber):
                        item = f.readline().split()
                        pid = int(item[0]) - 1
                        ParticleType[pid] = int(item[1])
                        if self.ndim == 3:
                            positions[pid, 0] = xlo_bound + float(item[2]) * h0 + float(
                                item[3]) * h5 + float(item[4]) * h4
                            positions[pid, 1] = ylo_bound + \
                                float(item[3]) * h1 + float(item[4]) * h3
                            positions[pid, 2] = zlo_bound + float(item[4]) * h2
                        elif self.ndim == 2:
                            positions[pid, 0] = xlo_bound + \
                                float(item[2]) * h0 + float(item[3]) * h5
                            positions[pid, 1] = ylo_bound + float(item[3]) * h1

                self.ParticleType.append(ParticleType)
                return positions

        except BaseException:
            positions = np.zeros((self.ParticleNumber[0], self.ndim))
            return positions

    def read_centertype(self, f):
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
            self.TimeStep.append(timestep)
            item = f.readline()
            ParticleNumber = int(f.readline())
            item = f.readline().split()
            # -------Read Orthogonal Boxes---------
            boxbounds = np.zeros((self.ndim, 2))  # box boundaries of (x y z)
            boxlength = np.zeros(self.ndim)  # box length along (x y z)
            for i in range(self.ndim):
                item = f.readline().split()
                boxbounds[i, :] = item[:2]

            boxlength = boxbounds[:, 1] - boxbounds[:, 0]
            if self.ndim < 3:
                for i in range(3 - self.ndim):
                    f.readline()
            shiftfactors = (boxbounds[:, 0] + boxlength / 2)
            # self.Boxbounds.append(boxbounds)
            self.Boxlength.append(boxlength)
            hmatrix = np.diag(boxlength)
            self.hmatrix.append(hmatrix)

            item = f.readline().split()
            names = item[2:]
            ParticleType = np.zeros(ParticleNumber, dtype=int)
            positions = np.zeros((ParticleNumber, self.ndim))
            # MoleculeType = np.zeros(ParticleNumber, dtype=int)

            if 'xu' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    ParticleType[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: self.ndim + 2]]
                    # MoleculeType[int(item[0]) - 1] = int(item[-1])

                conditions = [True if atomtype in self.moltypes.keys(
                ) else False for atomtype in ParticleType]
                positions = positions[conditions]
                ParticleType = pd.Series(
                    ParticleType[conditions]).map(
                    self.moltypes).values

            elif 'x' in names:
                for i in range(ParticleNumber):
                    item = f.readline().split()
                    ParticleType[int(item[0]) - 1] = int(item[1])
                    positions[int(item[0]) - 1] = [float(j)
                                                   for j in item[2: self.ndim + 2]]
                    # MoleculeType[int(item[0]) - 1] = int(item[-1])

                conditions = [True if atomtype in self.moltypes.keys(
                ) else False for atomtype in ParticleType]
                positions = positions[conditions]
                ParticleType = pd.Series(
                    ParticleType[conditions]).map(
                    self.moltypes).values
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
                                                   for j in item[2: self.ndim + 2]] * boxlength
                    # MoleculeType[int(item[0]) - 1] = int(item[-1])

                conditions = [True if atomtype in self.moltypes.keys(
                ) else False for atomtype in ParticleType]
                positions = positions[conditions]
                ParticleType = pd.Series(
                    ParticleType[conditions]).map(
                    self.moltypes).values
                positions += boxbounds[:, 0]
                positions = np.where(
                    positions < boxbounds[:, 0], positions + boxlength, positions)
                positions = np.where(
                    positions > boxbounds[:, 1], positions - boxlength, positions)
                # positions = positions - shiftfactors[np.newaxis, :]
                # boxbounds = boxbounds - shiftfactors[:, np.newaxis]

            self.Boxbounds.append(boxbounds)
            self.ParticleType.append(ParticleType)
            self.ParticleNumber.append(ParticleType.shape[0])
            return positions

        except BaseException:
            positions = np.zeros((3, self.ndim))
            return positions

    def read_gsd(self, f):
        """Read gsd file from HOOMD-blue
        gsd file provides all the configuration information
        ref: https://gsd.readthedocs.io/en/stable/hoomd-examples.html
        """

        self.SnapshotNumber = len(f)
        for onesnapshot in f:
            # ------------configuration information---------------
            self.TimeStep.append(onesnapshot.configuration.step)
            boxlength = onesnapshot.configuration.box[:self.ndim]
            self.Boxlength.append(boxlength)
            hmatrix = np.diag(boxlength)
            self.hmatrix.append(hmatrix)
            # ------------particles information-------------------
            self.ParticleNumber.append(onesnapshot.particles.N)
            self.ParticleType.append(onesnapshot.particles.typeid + 1)
            position = onesnapshot.particles.position[:, :self.ndim]
            self.Positions.append(position)
            boxbounds = np.column_stack(
                (position.min(
                    axis=0), position.max(
                    axis=0)))
            self.Boxbounds.append(boxbounds)

        if f[0].configuration.dimensions != self.ndim:
            print("---*Warning*: Wrong dimension information given---")

    def read_gsd_dcd(self, f_gsd, f_dcd):
        """Read gsd and dcd file from HOOMD-blue
        gsd file provides all the configuration information except positions with periodic boundary conditions
        dcd file provides the unwrap positions without periodic boundary conditions
        gsd is to get static information about the trajectory
        dcd is to get the absolute displacement to calculate dynamics
        ref: https://gsd.readthedocs.io/en/stable/hoomd-examples.html
        ref: http://mdtraj.org/1.6.2/api/generated/mdtraj.formats.DCDTrajectoryFile.html
        """

        # -----------------read gsd file-------------------------
        self.SnapshotNumber = len(f_gsd)
        for onesnapshot in f_gsd:
            # ------------configuration information---------------
            self.TimeStep.append(onesnapshot.configuration.step)
            boxlength = onesnapshot.configuration.box[:self.ndim]
            self.Boxlength.append(boxlength)
            hmatrix = np.diag(boxlength)
            self.hmatrix.append(hmatrix)
            # ------------particles information-------------------
            self.ParticleNumber.append(onesnapshot.particles.N)
            self.ParticleType.append(onesnapshot.particles.typeid + 1)
            position = onesnapshot.particles.position[:, :self.ndim]
            boxbounds = np.column_stack(
                (position.min(
                    axis=0), position.max(
                    axis=0)))
            self.Boxbounds.append(boxbounds)

        if f_gsd[0].configuration.dimensions != self.ndim:
            print("---*Warning*: Wrong dimension information given---")

        # -----------------read dcd file-------------------------
        position = f_dcd.read()[0]
        for i in range(position.shape[0]):
            self.Positions.append(position[i][:, :self.ndim])

        # ----------------gsd and dcd should be consistent--------
        if self.SnapshotNumber != len(self.Positions):
            print("---*Warning*: Inconsistent configuration in gsd and dcd files---")

        if self.ParticleNumber[0] != self.Positions[0].shape[0]:
            print("---*Warning*: Inconsistent particle number in gsd and dcd files---")


def read_additions(dumpfile, SnapshotNumber, ParticleNumber, ncol):
    """read additional columns in the dump file

    ncol is the column number starting from 0
    return in numpy array as [particlenumber, snapshotnumber] in float
    """

    results = np.zeros((ParticleNumber, SnapshotNumber))

    f = open(dumpfile)
    for n in range(SnapshotNumber):
        for i in range(9):
            f.readline()
        for i in range(ParticleNumber):
            item = f.readline().split()
            results[int(item[0]) - 1, n] = float(item[ncol])
    f.close()

    return results
