#coding = utf-8

import numpy as np

def write_dump_header(
	timestep: int, nparticle: int, boxbounds: list, additional_columns:str = '') -> str:
    """
    write the header of lammps dump file

    Inputs:
    	1. timestep (int): current timestep for the snapshot
    	2. nparticle (int): number of particle numbers
    	3. boxbounds (list): the coordinates of the simulation box
    						 for two-dimensional box, [[xlo, xhi], [ylo, yhi]]
    						 for three-dimensional box, [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    						 wherein xho and xhi represent minimum and maximum coordinate values
    						 in the x-direction, respectively, same as [ylo, yhi] and [zlo, zhi]
    	4. additional_columns (str): the name of additional_columns behind id type x y z for 3D box

	Return:
		header of lammps dump file (str)
    """

    header = 'ITEM: TIMESTEP\n'
    header += str(timestep) + '\n'

    header += 'ITEM: NUMBER OF ATOMS\n'
    header += str(nparticle) + '\n'

    header += 'ITEM: BOX BOUNDS pp pp pp\n'
    header += '%.6f %.6f\n' %(boxbounds[0][0], boxbounds[0][1])
    header += '%.6f %.6f\n' %(boxbounds[1][0], boxbounds[1][1])
    if len(boxbounds) == 3:
        header += '%.6f %.6f\n' %(boxbounds[2][0], boxbounds[2][1])
        header += 'ITEM: ATOMS id type x y z %s\n' %additional_columns
    else:
        header += '%.6f %.6f\n' %(-0.5, 0.5)
        header += 'ITEM: ATOMS id type x y %s\n' %additional_columns

    return header
