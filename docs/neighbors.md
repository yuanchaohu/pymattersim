# Calculating Neighboring Particles

This module is used to calculate the neighboring particles of a particle, which is the base for many different analyzing methods.

There are many ways to recognize the neighbors, for example, by a certain number, by a certain cutoff, by particle-type specific cutoffs, by Voronoi tessellation etc. This module covers the above calculation methods and can be easily extended by individual purpose.

The format of the saved neighbor list file (named as ***neighborlist.dat*** in default) must be identification of the centered particle (***id***), coordination number of the centered particle (***cn***), and identification of neighboring particles (***neighborlist***). That is, ***id cn neighborlist***.

The neighbors in the output file is sorted by their distances to the centered particle in ascending order. Neighbor list of different snapshots is continuous without any gap and all start with the header ***id cn neighborlist***. This formatting is made to be consistent with reading neighbor lists for different analyzing methods.

## Nnearests

 `neighbors.calculate_neighbors.Nnearests` gets the `N` nearest neighbors around a particle. In this case, the coordination number is `N` for each particle.

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

- `N` (`int`): the specified number of nearest neighbors, default 12

- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no

  ​					   default `[1, 1, 1]`, that is, PBC is applied in all three dimensions for 3D box. Set [1,1] for two-dimensional system.

- `fnfile` (`str`): the name of output file that stores the neighbor list, default ***neighborlist.dat***

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from neighbors.calculate_neighbors import Nnearests

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

Nnearests(readdump.snapshots, N=12, ppp=[1,1,1], fnfile='neighborlist.dat')
```

## cutoffneighbors

`neighbors.calculate_neighbors.cutoffneighbors` gets the nearest neighbors around a particle by setting a global cutoff distance `r_cut`.  Usually, the cutoff distance can be determined as the position of the first deep valley in total pair correlation function.

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`):`Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

- `r_cut` (`float`): the global cutoff distance to screen the nearest neighbors

- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no

  ​					   default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box. Set [1,1] for two-dimensional system.

- `fnfile` (`str`): the name of output file that stores the neighbor list, default ***neighborlist.dat***

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from neighbors.calculate_neighbors import cutoffneighbors

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

cutoffneighbors(readdump.snapshots, r_cut=3.8, ppp=[1,1,1], fnfile='neighborlist.dat')
```

## cutoffneighbors_particletype

`neighbors.calculate_neighbors.cutoffneighbors_particletype` gets the nearest neighbors around a particle by setting a cutoff distance ` r_cut`. Taken Cu-Zr system as an example, `r_cut` should be a 2D numpy array:
$$
\begin{bmatrix}
  &r_{\rm cut}^{\rm Cu-Cu} & r_{\rm cut}^{\rm Cu-Zr}&\\
  &r_{\rm cut}^{\rm Zr-Cu} & r_{\rm cut}^{\rm Zr-Zr}&\\
\end{bmatrix}
$$
Usually, these cutoff distances can be determined as the position of the first deep valley in partial pair correlation function of each particle pair.

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

- `r_cut` (`np.array`): the cutoff distances of each particle pair

- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no

  ​					   default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box. Set [1,1] for a two-dimensional system

- `fnfile` (`str`): the name of output file that stores the neighbor list, default ***neighborlist.dat***

### Example

```python
import numpy as np
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from neighbors.calculate_neighbors import cutoffneighbors_particletype

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

# a binary system
r_cut = np.zeros((2, 2))
r_cut[0, 0] = 3.6
r_cut[0, 1] = 3.4
r_cut[1, 0] = 3.4
r_cut[1, 1] = 3.9
cutoffneighbors_particletype(readdump.snapshots, r_cut=r_cut, ppp=[1,1,1], fnfile='neighborlist.dat')
```

## read_neighbors

This module is used to read the property of neighboring particles from a saved file, as long as the format of file is compatible, as like ***neighborlist.dat***. Note that this module reads one Snapshot a time to save computer memory. If you have multiple snapshots, you can import this module in a loop (see example).

### Input Arguments

- `f` (`TextIO`): opened file which save the property of neighboring particles, `f = open(filename, 'r')`. Opening the file outside of the function ensures that when reading a file containing multiple snapshots, the file pointer can move to the next snapshot.
- `Nmax` (`int`): the maximum number of neighboring particles to consider. Setting `Nmax` to a sufficiently large value is ideal, with the default being 200. `Nmax` is defined to address the varying number of neighboring particles among different particles. In this way, we can create a regular two-dimensional NumPy array to save the property of neighboring particles.

### Return

two-dimensional numpy array, with shape (`nparticle`, `1+Nmax_fact`). `Nmax_fact` means the maximum coordination number for one particle in the system. The first column is the coordination number (***cn***), so number of columns plus 1. For particles with coordination number less than `Nmax_fact` (which is generally the case), the unoccupied positions in `neighborprop` (see source code) are padded with `0`.

### Important Notes

In the saved ***neighborlist.dat***, the particle ID is counted from 1. While in this read module, the returned neighborlist is counted from 0. This is to facilitate subsequent indexing operations.

### Example

- Read neighborlist:

  ```python
  from neighbors.read_neighbors import read_neighbors
  
  filename = 'neighborlist.dat'
  f = open(filename, 'r')
  neighborprop = read_neighbors(f, nparticle=8100, Nmax=200)
  f.close()
  ```

- Read facearealist:

  ```python
  from neighbors.read_neighbors import read_neighbors
  filename = 'dump.facearea.dat'	# generated by carrying out Voronoi analysis module
  f = open(filename, 'r')
  neighborprop = read_neighbors(f, nparticle=8100, Nmax=200)
  f.close()
  ```

- Read in multiple snapshots:

  ```python
  from neighbors.read_neighbors import read_neighbors
  
  neighbors_snapshots = []
  filename = 'neighborlist.dat'
  f = open(filename, 'r')
  for i in range(2):	# the example dump file contains 2 snapshots
      neighbors_snapshots.append(read_neighbors(f, nparticle=8100, Nmax=200))
  f.close()
  ```


## freud_neighbors

`freud` is a trajectory analysis package developed by [Glotzer's group](https://freud.readthedocs.io/en/stable/index.html). To use the package, some special attentions should be paid to the following points:

- The particle coordinates $$\in [-L/2, L/2]$$ 
- For 2D systems, the ***z*** component of coordinates should be input as 0

To use `freud` easily, a function `neighbors.freud_neighbors.convert_configuration` is used to convert the dump file from our `reader` module to `freud` style. Correspondingly the list of box information and the list of the particle coordinates are returned for further analysis.

### convert_configuration

#### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

#### Return

- `list_box` (`list`): the list of box information for `freud` analysis
- `list_points` (`list`): the list of particle positions for `freud` analysis

#### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType

from neighbors.freud_neighbors import convert_configuration

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

list_box, list_points = convert_configuration(readdump.snapshots)
```

### cal_neighbors

Calling `neighbors.freud_neighbors.Voro_neighbors` first and then performing Voronoi analysis with ***[voro++](https://math.lbl.gov/voro++/)*** package to calculate neighbors for both 2D and 3D systems. Three classes of information will be output:

- neighbor list: atom coordination and ids of each atom
- Voronoi face area (3D) or edge length (2D) list: between the center and each of its neighbors
- overall information: coordination number, Voronoi volume or area of each atom

#### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file
- `outputfile` (`str`): filename of neighbor list and bond info, such as edge length (2D) or facearea(3D)

#### Return

None (result is saved to file). For neighbor list, file with name outputfile+'.neighbor.dat'. For edgelength list (2D box), file with name outputfile+'.edgelength.dat'. For facearea list (3D box), file with name outputfile+'.facearea.dat'

#### Example

```python
from reader.lammps_reader_helper import read_lammps_wrapper
from neighbors.freud_neighbors import cal_neighbors

filename = 'dump_2D.atom'
snapshots = read_lammps_wrapper(filename, ndim=2)
cal_neighbors(snapshots, outputfile='dump')
```

