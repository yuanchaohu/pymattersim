[toc]

# Write LAMMPS dump/data format header

This module is used to write the header of lammps dump and data files for conveniently rewriting snapshots.

## Input Arguments

1. `writer.lammps_writer.write_dump_header` writes the header of lammps dump file:

   - `timestep` (`int`):  timestep for the current snapshot
   - `nparticle` (`int`): number of particle
   - `boxbounds` (`np.array`): the bounds of the simulation box. For two-dimensional box, `[[xlo, xhi], [ylo, yhi]]`. For three-dimensional box, `[[xlo, xhi], [ylo, yhi], [zlo, zhi]]`. `xho` and `xhi` represent minimum and maximum coordinate values in the x-direction, respectively, same as `[ylo, yhi]` and `[zlo, zhi]`
   - `addson` (`str`): the name of additional columns, such as "order Q6", default `addson = ''`

2. `writer.lammps_writer.write_data_header` writes the header of lammps data file:

   - `nparticle` (`int`): number of particle

   - `nparticle_type` (`int`): number of particle type

   - `boxbounds` (`np.array`): same as the definition in `write_dump_header`

## Return

A string for the header will be returned.

## Example

- Write the header of lammps dump file:

  ```python
  from writer.lammps_writer import write_dump_header
  
  write_dump_header(timestep, nparticle, boxbounds, addson = '')
  ```

- Write the header of lammps data file:

  ```python
  from writer.lammps_writer import write_data_header
  
  write_data_header(nparticle, nparticle_type, boxbounds)
  ```

