#!/usr/bin/python
# coding = utf-8
# This module is part of an analysis package

try:
    import gsd, mdtraj
except ImportError:
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gsd"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mdtraj"])
        import gsd, mdtraj
    except Exception as e:
        print(f"An error occurred while installing gsd or mdtraj: {str(e)}")

from utils.logging_utils import get_logger_handle
from reader.reader_utils import DumpFileType, Snapshots
from reader.lammps_reader_helper import read_lammps_wrapper, read_lammps_centertype_wrapper
from reader.gsd_reader_helper import read_gsd_wrapper, read_gsd_dcd_wrapper


Docstr = """
            Reading partcles' positions from snapshots of molecular simulations

            -------------------for LAMMPS dump file----------------------------
            To run the code, dump file format (id type x y z ...) are needed
            It supports three types of coordinates now x, xs, xu
            To obtain the dump information, use
            ********************from reader.dump import DumpReader****************
            ***************from reader.reader_utils import DumpFileType***********
            *****reader=DumpReader(filename, ndim=3, filetype=DumpFileType(1))****
            *********************** reader.read_onefile() ************************

            reader.snapshots: stores a list of snapshot, which consisits:

                snapshot.timestep:         simulation timestep at each snapshot
                snapshot.nparticle:  particle number from each snapshot
                snapshot.particle_type:    particle type in array in each snapshot
                snapshot.positions:        particle coordinates in array in each snapshot
                snapshot.boxlength:        box length in array in each snapshot
                snapshot.boxbounds:        box boundaries in array in each snapshot
                snapshot.realbounds:       real box bounds of a triclinic box
                snapshot.hmatrix:          h-matrix of the cells in each snapshot

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

            -------------------for GSD dump file------------------------
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
    DumpFileType.LAMMPSCENTER: read_lammps_centertype_wrapper,
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
            moltypes: dict=None) -> None:

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

            'gsd' (gsd standard output for static properties)

            'gsd_dcd' (HOOMD-blue outputs for static and dynamic properties)

            The simulation box is centered at 0 by default for
            'x' and 'xs' coordinates of 'lammps' & 'lammpscenter'
        """
        logger.info(
            f"Start Reading file {self.filename} of type {self.filetype}")
        reader_inputs = {"file_name": self.filename, "ndim": self.ndim}
        if self.filetype == DumpFileType.LAMMPSCENTER:
            reader_inputs["moltypes"] = self.moltypes
        self.snapshots = FILE_TYPE_MAP_READER[self.filetype](**reader_inputs)
        logger.info(f"Completed Reading file {self.filename}")
