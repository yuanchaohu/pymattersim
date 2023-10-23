#coding = utf-8

import numpy as np

def write_data_header(
    nparticle: int, nparticletype: int, boxbounds: list) -> str:
    """
    write the headers of lammps data file

    Inputs:
        1. nparticle (int): number of particle numbers
        2. nparticletype (int): number of particle type
        3. boxbounds (list): the coordinates of the simulation box
                             for two-dimensional box, [[xlo, xhi], [ylo, yhi]]
                             for three-dimensional box, [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
                             wherein xho and xhi represent minimum and maximum coordinate values
                             in the x-direction, respectively, same as [ylo, yhi] and [zlo, zhi]

    Return:
        header of lammps data file (str)
    """

    header = 'LAMMPS data file\n\n'
    header += '%d atoms\n' %(nparticle)
    header += '%d atom types\n\n' %(nparticletype)
    header += '%.6f %.6f xlo xhi\n' %(boxbounds[0][0], boxbounds[0][1])
    header += '%.6f %.6f ylo yhi\n' %(boxbounds[1][0], boxbounds[1][1])
    if len(boxbounds) == 3:
        header += '%.6f %.6f zlo zhi\n' %(boxbounds[2][0], boxbounds[2][1])
    else:    
        header += '-0.5 0.5 zlo zhi\n'
    header += '\nAtoms #atomic\n\n'

    return header
