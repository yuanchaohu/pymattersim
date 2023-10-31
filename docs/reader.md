[TOC]

# Reading Computer Simulations Snapshots

This module read-in the static or time evolution of the particle positions (snapshots or trajectories) encapsulated in a simulation box from various simulators for materials science, chemistry, and physics etc. The popular ones includes but not limit yo LAMMPS and Hoomd-blue. This module is very flexible for future extension, just by adding suitable format reading of the simulation outputs. For example, the atomic/molecular configurations from *ab-initio* calculations (DFT) and Gromacs, and any others. This module is the base of all of the other physical calculations. As long as the simulation box information were read properly, they can be used for additional calculations. That means, we are trying to "forget about" the specific simulation box, but transform it to a specific data structures.

The main functions are captured by the module `reader`, in which there are several reading methods, listed currently as:
- `dump_reader` [main function to call other methods]
- `GSD_reader_helper` [specific for Hoomd-blue]
- `lammps_reader_helper` [specific for LAMMPS]
- `reader_utils` [define output data structures]
- `simulation_log` [simulation log reading, now only for LAMMPS]

The `dump_reader` is a main stream to include all other methods, which is useful for general purpose. The other helper functions are designed for specific purpose if any.

The typical features of `reader` are:

1. There are now four types of input snapshots format supported, which is specified by the `filetype` argument when initializing `reader.dump_reader.DumpReader`. Internally, this class calls the other specific methods:
   - `filetype=DumpFileType.LAMMPS`: Atomistic system in LAMMPS, by calling `reader.dump_reader.DumpReader` or `reader.lammps_reader_helper.read_lammps_wrapper `.
   - `filetype=DumpFileType.LAMMPSCENTER`: Molecular system in LAMMPS, by calling `reader.dump_reader.DumpReader` or `reader.lammps_reader_helper.read_lammps_centertype_wrapper`
   - `filetype=DumpFileType.GSD`: Static properties in Hoomd-blue, GSD file, by calling `reader.dump_reader.DumpReader` or `reader.GSD_reader_helper.read_GSD_wrapper`.  
   - `filetype=DumpFileType.GSD_DCD`: Dynamic properties in Hoomd-blue, both GSD and DCD files, by calling `reader.dump_reader.DumpReader` or `reader.GSD_reader_helper.read_GSD_DCD_wrapper`.

2. Supports any dimensionality. General cases include the two-dimensional (LAMMPS dump file format ***id type x y***) and three-dimensional snapshots (LAMMPS dump file format ***id type x y z***). As long as the input format includes ***id type***, only dimensional positions will be read-in and the other columns will be neglected. For example, for a two-dimensional system, the input format can be ***id type x y z xx yy zz nn...***, the positions ***(x y)*** can be read properly.

3. Supports both orthogonal and triclinic cells with the use of H-matrix to deal with the periodic boundary conditions. As long as the simulator supports these two types cell, the module can process it properly. For a triclinic box, it converts the bounding box back into the trilinic box parameters by refering to 'https://docs.lammps.org/Howto_triclinic.html':

   ```python
   xlo = xlo_bound - MIN(0.0,xy,xz,xy+xz)
   xhi = xhi_bound - MAX(0.0,xy,xz,xy+xz)
   ylo = ylo_bound - MIN(0.0,yz)
   yhi = yhi_bound - MAX(0.0,yz)
   zlo = zlo_bound
   zhi = zhi_bound
   ```

## Input Arguments

1. `filename` (`str`): the name of dump file
2. `ndim` (`int`): dimensionality
3. `filetype` (`DumpFileType`): input dump format, defined in `reader.reader_utils`, including the following format:
   - `DumpFileType.LAMMPS` (`default`)
   - `DumpFileType.LAMMPSCENTER`
   - `DumpFileType.GSD`
   - `DumpFileType.GSD_DCD`

4. `moltypes` (`dict`, optional): only used for molecular system in LAMMPS so far, default is `None`. To specify, for example, if the system has 5 types of atoms in which 1-3 is one type of molecules and 4-5 is the other, and type 3 and 5 are the center of mass. Then `moltypes` should be `{3:1, 5:2}`. The keys `[3, 5]` of  `moltypes` are used to select specific atoms to present the corresponding molecules. The values `[1, 2]` is used to record the type of molecules.

## Return

The input simulation box will be transformed to a list of 'digital' snapshot, by returning `snapshots` object. It is a list of `snapshot` that has all of the configuration information. `snapshots` and `snapshot` are two data class defined in `reader.reader_utils`.  `snapshot` consists:

- `snapshots[n].timestep` (int): simulation timestep at each snapshot
- `snapshots[n].nparticle` (int): particle number from each snapshot
- `snapshots[n].particle_type` (numpy array): particle type in array in each snapshot
- `snapshots[n].positions` (numpy array): particle coordinates in array in each snapshot
- `snapshots[n].boxlength` (numpy array): box length in array in each snapshot
- `snapshots[n].boxbounds` (numpy array): box boundaries in array in each snapshot
- `snapshots[n].realbounds` (numpy array): real box bounds of a triclinic box (optional)
- `snapshots[n].hmatrix` (numpy array): h-matrix of the cells in each snapshot

The information is stored in a list of which the elements are mainly numpy arrays. Particle-level information is referred by particle ID.

## Important Notes

1. In LAMMPS, ***x***, ***xs***, and ***xu*** format coordinates are acceptable. Such as with format The reduced ***xs*** will be rescaled to the absolute coordinates ***x***.
2. For the ***xs*** and ***x*** types in orthogonal cells with periodic boundary conditions, particle coordinates are **NOT** warpped to the inside of the box by default, which could be changed by hand when necessary. In non-periodic boundary conditions, there should be no particles at the outside of the cell.
3. Snapshots should be stored in one file at this stage.
4. In Hoomd-blue, GSD and DCD files are acceptable. GSD file has all the information with periodic boundary conditions, while DCD has the unwarpped coordinates. GSD file has all the information with periodic boundary conditions, while DCD only has the particle positions. Normally only GSD file is needed . But if one wants to analyze the dynamical properties, the DCD file should be dumped accompanying the GSD file to get the unwarp coordinates. More specifically, all the information about the trajectory will be obtained from the GSD file except the particle positions which will be obtained from the DCD file. Therefore, the DCD and GSD files shall be dumped with the same period or concurrently. Another important point in this case is the file name of GSD and DCD. They should share the same name with the only difference of the last three string, ie. ‘GSD ’or ‘DCD’. For example, if the file name of GSD is ***dumpfile.GSD*** then the DCD file name must be ***dumpfile.DCD***. To read the Hoomd-blue outputs, two new modules should be installed first: i) GSD; ii) mdtraj. These modules are available by conda. 

## Example

Some dump files are provided in `tests/sample_test_data`.

- Call `dump_reader` and specify `filetype`:

  ```python
  from reader.dump_reader import DumpReader
  from reader.reader_utils import DumpFileType
  
  readdump=DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS)
  readdump.read_onefile()
  
  # return the number of snapshots
  readdump.snapshots.nsnapshots
  
  # return the timestep of the first snapshot
  readdump.snapshots.snapshots[0].timestep
  
  # return the particle number of the first snapshot
  readdump.snapshots.snapshots[0].nparticle
  
  # return the particle type of the first snapshot
  readdump.snapshots.snapshots[0].particle_type
  
  # return the particle positions of the first snapshot
  readdump.snapshots.snapshots[0].positions
  
  # return the boxlength of the first snapshot
  readdump.snapshots.snapshots[0].boxlength
  
  # return the boxbounds of the first snapshot
  readdump.snapshots.snapshots[0].boxbounds
  
  # return the realbounds of the first snapshot, for triclinic box
  readdump.snapshots.snapshots[0].realbounds
  
  # return the hmatrix of the first snapshot
  readdump.snapshots.snapshots[0].hmatrix
  
  # get all the timesteps
  [snapshot.timestep for snapshot in readdump.snapshots.snapshots]
  ```

- The user can also directly call the wrapper functions, such as,

  ```python
  from reader.lammps_reader_helper import read_lammps_wrapper
  
  snapshots = read_lammps_wrapper(filename, ndim=3)
  
  # return number of snapshots
  snapshots.nsnapshots
  
  # return the particle positions of the first snapshot
  snapshots.snapshots[0].positions
  ```

- The `reader.lammps_reader_helper.read_additions` can read additional columns in the lammps dump file. `ncol` (`int`): specifying the column number starting from 0 (zero-based). `read_additions` returns a numpy array as shape [nsnapshots, nparticle] in float. For example, read ***order*** from ***id type x y z order***. 

  ```python
  from reader.lammps_reader_helper import read_additions
  
  # ncol starts in python style, i.e. from 0
  dumpfile ='./tests/sample_test_data/test_additional_columns.dump'
  read_additions(dumpfile, ncol=5)
  ```

- The `reader.simulation_log.read_lammpslog` can extract the thermodynamic quantities from lammps log file, input with `filename` (`str`) lammps log file name, and return list of pandas `DataFrame` for each logging section:

  ```python
  from reader.simulation_log import read_lammpslog
  
  read_lammpslog(filename)
  ```
