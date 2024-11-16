"""This module defines immutable data structure for saving snapshots."""

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy.typing as npt

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


class DumpFileType(Enum):
    """
    An Enum class holds dump file type:
        1. LAMMPS: default, for more information see
        'http://lammps.sandia.gov/doc/Section_howto.html#howto-12'

        2. LAMMPSCENTER: lammps molecular dump with known atom type of each molecule center

        3. GSD: HOOMD-blue standard output for static properties

        4. GSD_DCD: HOOMD-blue outputs for static and dynamic properties

        5. LAMMPSVECTOR: additional column(s) from lammps configurations,
            for example, "vx vy" from "id type x y vx vy"
    """

    LAMMPS = 1
    LAMMPSCENTER = 2
    GSD = 3
    GSD_DCD = 4
    LAMMPSVECTOR = 5


@dataclass(frozen=True)
class SingleSnapshot:
    """
    A frozen data class to hold single snaptshot information:
        timestep:         simulation timestep at each snapshot
        nparticle:        particle number from each snapshot
        particle_type:    particle type in array in each snapshot
        positions:        particle coordinates in array in each snapshot
                          can be additional column information when
                          columnids is activated, see dump_reader.py
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
        snapshots[n].timestep:         simulation timestep at each snapshot
        snapshots[n].nparticle:        particle number from each snapshot
        snapshots[n].particle_type:    particle type in array in each snapshot
        snapshots[n].positions:        particle coordinates in array in each snapshot
        snapshots[n].boxlength:        box length in array in each snapshot
        snapshots[n].boxbounds:        box boundaries in array in each snapshot
        snapshots[n].realbounds:       real box bounds of a triclinic box
        snapshots[n].hmatrix:          h-matrix of the cells in each snapshot
    The information is stored in list whose elements are mainly numpy arraies.
    """

    nsnapshots: int
    snapshots: List[SingleSnapshot]
