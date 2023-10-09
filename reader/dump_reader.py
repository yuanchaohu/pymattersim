#!/usr/bin/python
# coding = utf-8
# This module is part of an analysis package

import pandas as pd
import numpy as np
from utils.logging_utils import get_logger_handle, log_table
from reader.reader_utils import DumpFileType, SystemSnapshot, Snapshots
from lammps_reader_helper import read_lammps_wrapper, read_centertype_wrapper
from gsd_reader_helper import read_gsd_wrapper, read_gsd_dcd_wrapper


Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
             Reading partcles' positions from snapshots of molecular simulations

             -------------------for LAMMPS dump file----------------------------
             To run the code, dump file format (id type x y z ...) are needed
             It supports three types of coordinates now x, xs, xu
             To obtain the dump information, use
             ********************* from dump import readdump*********************
             ******************** d = readdump(filename, ndim) ******************
             ************************ d.read_onefile() **************************

             d.TimeStep:        a list of timestep from each snapshot
             d.ParticleNumber:  a list of particle number from each snapshot
             d.ParticleType:    a list of particle type in array in each snapshot
             d.Positions:       a list of particle coordinates in array in each snapshot
             d.SnapshotNumber:  snapshot number
             d.Boxlength:       a list of box length in array in each snapshot
             d.Boxbounds:       a list of box boundaries in array in each snapshot
             d.hmatrix:         a list of h-matrix of the cells in each snapshot

             The information is stored in list whose elements are mainly numpy arraies

             This module is powerful for different dimensions by giving 'ndim' for orthogonal boxes
             For a triclinic box, convert the bounding box back into the trilinic box parameters:
             xlo = xlo_bound - MIN(0.0,xy,xz,xy+xz)
             xhi = xhi_bound - MAX(0.0,xy,xz,xy+xz)
             ylo = ylo_bound - MIN(0.0,yz)
             yhi = yhi_bound - MAX(0.0,yz)
             zlo = zlo_bound
             zhi = zhi_bound
             See 'http://lammps.sandia.gov/doc/Section_howto.html#howto-12'

             -------------------------------------------------------------------
             Both atomic and molecular systems can be read (lammps versus lammpscenter)
             For molecular types, the new positions are the positions of the
             center atom of each molecule
             -------------------------------------------------------------------

             -------------------for HOOMD-blue dump file------------------------
             for static properties, gsd file is used
             for dynamic properties, both gsd and dcd files are used
             The gsd and dcd files are the same just with difference in position
             The information of the configuration is supplemented from gsd for dcd
             The output of the trajectory information is the same as lammps
             A keyword specifying different file type will be given
             ****Additional packages will be needed for these dump files****
         """


logger = get_logger_handle(__name__)

FILE_TYPE_MAP_READER = {
    DumpFileType.LAMMPS: read_lammps_wrapper,
    DumpFileType.LAMMPSCENTER: read_centertype_wrapper,
    DumpFileType.GSD: read_gsd_wrapper,
    DumpFileType.GSD_DCD: read_gsd_dcd_wrapper,
}


class DumpReader:
    """Read snapshots from simulations"""

    def __init__(
            self,
            filename: str,
            ndim: int,
            filetype: DumpFileType = DumpFileType.LAMMPS,
            moltypes: str = '') -> None:

        self.filename = filename  # input snapshots
        self.ndim = ndim  # dimension
        self.filetype = filetype  # trajectory type from different MD engines
        self.moltypes = moltypes  # molecule type for lammps molecular trajectories
        self.snapshots: Snapshots = None

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
        logger.info(
            f"Start Reading file {self.filename} of type {self.filetype}")
        self.snapshots = FILE_TYPE_MAP_READER[self.filetype](
            self.filename, self.ndim, self.moltypes)
        logger.info(f"Completed Reading file {self.filename}")


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
