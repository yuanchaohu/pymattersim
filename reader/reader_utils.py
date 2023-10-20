"""This module defines immutable data structure for saving snapshots."""

from typing import List
from dataclasses import dataclass
from enum import Enum
import numpy.typing as npt

from utils.logging_utils import get_logger_handle
logger = get_logger_handle(__name__)


class DumpFileType(Enum):
    """
    An Enum class holds dump file type:
        1. LAMMPS: default, for more information see
        'http://lammps.sandia.gov/doc/Section_howto.html#howto-12'
        
        2. LAMMPSCENTER: lammps molecular dump with known atom type of each molecule center
        
        3. GSD: HOOMD-blue standard output for static properties
        
        4. GSD_DCD: HOOMD-blue outputs for static and dynamic properties
    """
    LAMMPS = 1
    LAMMPSCENTER = 2
    GSD = 3
    GSD_DCD = 4


@dataclass(frozen=True)
class SingleSnapshot:
    """
    A frozen data class to hold single snaptshot information:
        timestep:         simulation timestep at each snapshot
        nparticle:        particle number from each snapshot
        particle_type:    particle type in array in each snapshot
        positions:        particle coordinates in array in each snapshot
        boxlength:        box length in array in each snapshot
        boxbounds:        box boundaries in array in each snapshot
        realbounds:       real box bounds of a triclinic box
        hmatrix:          h-matrix of the cells in each snapshot
    """
    # pylint: disable=too-many-instance-attributes
    timestep: int
    nparticle: int
    particle_type: npt.NDArray
    positions: npt.NDArray
    boxlength: npt.NDArray
    boxbounds: npt.NDArray
    realbounds: npt.NDArray
    hmatrix: npt.NDArray


@dataclass(frozen=True)
class Snapshots:
    """
    A frozen data class to hold all system snapshots
    snapshots: stores a list of snapshot, which consisits:
        snapshot.timestep:         simulation timestep at each snapshot
        snapshot.nparticle:        particle number from each snapshot
        snapshot.particle_type:    particle type in array in each snapshot
        snapshot.positions:        particle coordinates in array in each snapshot
        snapshot.boxlength:        box length in array in each snapshot
        snapshot.boxbounds:        box boundaries in array in each snapshot
        snapshot.realbounds:       real box bounds of a triclinic box
        snapshot.hmatrix:          h-matrix of the cells in each snapshot
    The information is stored in list whose elements are mainly numpy arraies.
    """
    nsnapshots: int
    snapshots: List[SingleSnapshot]
