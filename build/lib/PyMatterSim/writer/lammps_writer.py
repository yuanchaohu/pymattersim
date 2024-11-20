# coding = utf-8

"""see documentation @ ../../docs/writer.md"""

import numpy.typing as npt


def write_dump_header(timestep: int, nparticle: int, boxbounds: npt.NDArray, addson: str = None) -> str:
    """
    write the headers of lammps dump file

    Inputs:
        1. timestep (int): timestep for the current snapshot
        2. nparticle (int): number of particle
        3. boxbounds (npt.NDArray): the bounds of the simulation box
                                                     for two-dimensional box, [[xlo, xhi], [ylo, yhi]]
                                                     for three-dimensional box, [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
                                                     wherein xho and xhi represent minimum and maximum coordinate values
                                                     in the x-direction, respectively, same as [ylo, yhi] and [zlo, zhi]
        4. addson (str): the name of additional columns, such as "order Q6"

        Return:
                header of lammps dump file (str)
    """

    header = "ITEM: TIMESTEP\n"
    header += str(timestep) + "\n"

    header += "ITEM: NUMBER OF ATOMS\n"
    header += str(nparticle) + "\n"

    header += "ITEM: BOX BOUNDS pp pp pp\n"
    header += f"{boxbounds[0][0]:.6f} {boxbounds[0][1]:.6f}\n"
    header += f"{boxbounds[1][0]:.6f} {boxbounds[1][1]:.6f}\n"
    if len(boxbounds) == 3:
        header += f"{boxbounds[2][0]:.6f} {boxbounds[2][1]:.6f}\n"
        header += f"ITEM: ATOMS id type x y z {addson}\n"
    else:
        header += f"{-0.5:.6f} {0.5:.6f}\n"
        header += f"ITEM: ATOMS id type x y {addson}\n"

    return header


def write_data_header(nparticle: int, nparticle_type: int, boxbounds: npt.NDArray) -> str:
    """
    write the headers of lammps data file

    Inputs:
        1. nparticle (int): number of particle
        2. nparticle_type (int): number of particle type
        3. boxbounds (npt.NDArray): the bounds of the simulation box
                                 for two-dimensional box, [[xlo, xhi], [ylo, yhi]]
                                 for three-dimensional box, [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
                                 wherein xho and xhi represent minimum and maximum coordinate values
                                 in the x-direction, respectively, same as [ylo, yhi] and [zlo, zhi]

    Return:
        header of lammps data file (str)
    """

    header = "LAMMPS data file\n\n"
    header += f"{nparticle} atoms\n"
    header += f"{nparticle_type} atom types\n\n"
    header += f"{boxbounds[0][0]:.6f} {boxbounds[0][1]:.6f} xlo xhi\n"
    header += f"{boxbounds[1][0]:.6f} {boxbounds[1][1]:.6f} ylo yhi\n"
    if len(boxbounds) == 3:
        header += f"{boxbounds[2][0]:.6f} {boxbounds[2][1]:.6f} zlo zhi\n"
    else:
        header += "-0.5 0.5 zlo zhi\n"
    header += "\nAtoms #atomic\n\n"

    return header
