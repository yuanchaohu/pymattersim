# Read Snapshots

The module reads the time evolution of the particle positions (snapshots or trajectories) from various simulators for materials science, 
chemistry, and physics, and so on,
such as LAMMPS and Hoomd-blue engines. Our `reader` module is featured:

1. By specifying the `filetype` argument when initializing `reader.dump_reader.DumpReader` class, so far the module can read:
   - Atomistic system in LAMMPS, by calling `reader.lammps_reader_helper.read_lammps_wrapper `. In this case, `filetype=DumpFileType.LAMMPS`
   - Molecular system in LAMMPS, by calling `reader.lammps_reader_helper.read_lammps_centertype_wrapper`, In this case, `filetype=DumpFileType.LAMMPSCENTER`
   - Static properties in Hoomd-blue, gsd file, by calling `reader.gsd_reader_helper.read_gsd_wrapper`. In this case, `filetype=DumpFileType.GSD`
   - Dynamic properties in Hoomd-blue, both gsd and dcd files, by calling `reader.gsd_reader_helper.read_gsd_dcd_wrapper`. In this case, `filetype=DumpFileType.GSD_DCD`

2. Supports the two-dimensional (LAMMPS dump file format ***id type x y***) and three-dimensional snapshots (LAMMPS dump file format ***id type x y z***)

3. Supports the orthogonal and triclinic (not available in Hoomd-blue) cells with the use of h-matrix (diagonal). For a triclinic box, convert the bounding box back into the trilinic box parameters:

   ```python
   xlo = xlo_bound - MIN(0.0,xy,xz,xy+xz)
   xhi = xhi_bound - MAX(0.0,xy,xz,xy+xz)
   ylo = ylo_bound - MIN(0.0,yz)
   yhi = yhi_bound - MAX(0.0,yz)
   zlo = zlo_bound
   zhi = zhi_bound
   ```

   See 'https://docs.lammps.org/Howto_triclinic.html'

## Inputs

1. `filename` (`str`): the name of dump file
2. `ndim` (`int`): dimensionality
3. `filetype` (`DumpFileType`): input dump format, defined in `reader.reader_utils`, including the following format:
   - `DumpFileType.LAMMPS` (`default`)
   - `DumpFileType.LAMMPSCENTER`
   - `DumpFileType.GSD`
   - `DumpFileType.GSD_DCD`

4. `moltypes` (`dict`, optional): only used for molecular system in LAMMPS, default is `None`. To specify, for example, if the system has 5 types of atoms in which 1-3 is one type of molecules and 4-5 is the other, and type 3 and 5 are the center of mass. Then `moltypes` should be `{3:1, 5:2}`. The keys `[3, 5]` of  `moltypes` are used to select specific atoms to present the corresponding molecules. The values `[1, 2]` is used to record the type of molecules.

## Return

return `snapshots`, a list of `snapshot` that has all of the configuration information. `snapshots` and `snapshot` are two data class defined in `reader.reader_utils`.  `snapshot` consists:

- `snapshot.timestep`: simulation timestep at each snapshot
- `snapshot.nparticle`: particle number from each snapshot
- `snapshot.particle_type`: particle type in array in each snapshot
- `snapshot.positions`: particle coordinates in array in each snapshot
- `snapshot.boxlength`: box length in array in each snapshot
- `snapshot.boxbounds`: box boundaries in array in each snapshot
- `snapshot.realbounds`: real box bounds of a triclinic box
- `snapshot.hmatrix`: h-matrix of the cells in each snapshot

The information is stored in a list of which the elements are mainly numpy arrays. Particle-level information is referred by particle ID.

## Important Notes

1. In LAMMPS, ***x***, ***xs***, and ***xu*** format coordinates are acceptable. Such as with format The reduced ***xs*** will be rescaled to the absolute coordinates ***x***.
2. For the ***xs*** and ***x*** types in orthogonal cells with periodic boundary conditions, particle coordinates are warp to the inside of the box by default, which could be changed by hand when necessary. In non-periodic boundary conditions, there should be no particles at the outside of the cell.
3. Snapshots should be stored in one file at this stage.
5. In Hoomd-blue, gsd and dcd files are acceptable. gsd file has all the information with periodic boundary conditions, while dcd has the unwarpped coordinates. Gsd file has all the information with periodic boundary conditions, while dcd only has the particle positions. Normally only gsd file is needed . But if one wants to analyze the dynamical properties, the dcd file should be dumped accompanying the gsd file to get the unwarp coordinates. More specifically, all the information about the trajectory will be obtained from the gsd file except the particle positions which will be obtained from the dcd file. Therefore, the dcd and gsd files shall be dumped with the same period or concurrently. Another important point in this case is the file name of gsd and dcd. They should share the same name with the only difference of the last three string, ie. ‘gsd ’or ‘dcd’. For example, if the file name of gsd is ***dumpfile.gsd***then the dcd file name must be ***dumpfile.dcd***. To read the Hoomd-blue outputs, two new modules should be installed first: i) gsd; ii) mdtraj. These modules are available by conda. Currently, the dump files from Hoomd-blue only support orthogonal box.

## Example Usage

- Call `dump_reader` and specify `filetype`:

  ```python
  from reader.dump_reader import DumpReader
  from reader.reader_utils import DumpFileType
  readdump=DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS)
  readdump.read_onefile()
  
  # return the number of snapshots
  readdump.snapshots.nsnapshots
  
  # return the particle positions of the first snapshot
  readdump.snapshots.snapshots[0].positions
  ```

- The user can also directly call wrapper function, such as,

  ```python
  from reader.lammps_reader_helper import read_lammps_wrapper
  snapshots = read_lammps_wrapper(filename, ndim=3)
  
  # return the particle positions of the first snapshot
  snapshots.snapshots[0].positions
  ```

- The `reader.lammps_reader_helper.read_additions` can read additional columns in the lammps dump file. For example, read ***order*** from ***id type x y z order***:

  ```python
  from reader.lammps_reader_helper import read_additions
  read_additions(dumpfile, ncol=5)
  ```

​	`ncol` (`int`): specifying the column number starting from 0 (zero-based). `read_additions` returns a numpy arry as shape [particle_number, snapshotnumber] in float.

- The `reader.simulation_log.read_lammpslog` can extract the thermodynamic quantities from lammp log file, input with `filename` (`str`) lammps log file name, and return list of pandas `DataFrame` for each logging section:

  ```python
  from reader import simulation_log
  simulation_log.read_lammpslog(filename)
  ```
