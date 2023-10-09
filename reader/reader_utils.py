import numpy as np
import numpy.typing as npt

from typing import List
from dataclasses import dataclass
from enum import Enum
from utils.logging_utils import get_logger_handle
logger = get_logger_handle(__name__)


class DumpFileType(Enum):
    """
    An Enum class holds dump file type:
    1. lammps: default, for more information see 'http://lammps.sandia.gov/doc/Section_howto.html#howto-12'
    2. lammpscenter: lammps molecular dump with known atom type of each molecule center
    3. gsd: HOOMD-blue standard output for static properties
    4. gsd_dcd: HOOMD-blue outputs for static and dynamic properties
    """
    LAMMPS = 1
    LAMMPSCENTER = 2
    GSD = 3
    GSD_DCD = 4


@dataclass
class SingleSnapshot:
    """
    A class to hold system snaptshot
    timestep:        simulation timestep at each snapshot
    ParticleNumber:  particle number from each snapshot
    ParticleType:    particle type in array in each snapshot
    Positions:       particle coordinates in array in each snapshot
    Boxlength:       box length in array in each snapshot
    Boxbounds:       box boundaries in array in each snapshot
    RealBounds:      real box bounds of a triclinic box
    hmatrix:         h-matrix of the cells in each snapshot
    """
    timestep: int
    particle_number: int
    particle_type: npt.NDArray
    positions: npt.NDArray
    boxlength: npt.NDArray
    boxbounds: npt.NDArray
    realbounds: npt.NDArray
    hmatrix: npt.NDArray


@dataclass
class Snapshots:
    """
    A data class to hold all system snapshots
    """
    snapshots_number: int
    snapshots: List[SingleSnapshot]
