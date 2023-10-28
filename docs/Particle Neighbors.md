# Particle Neighbors

This module is used to calculate the neighboring particles of a particle, which is the base for many different analyzing methods. There are many ways to recognize the neighbors, for example, by a certain number, by a certain cutoff, by particle-type specific cutoffs, by Voronoi tessellation etc. This module covers the above calculation methods and can be easily extended by individual purpose.

The format of the saved neighbor list file (named as 'neighborlist.dat' in default) must be identification of the centered particle (***id***), coordination number of the centered particle (***cn***), and identification of neighboring particles (***neighborlist***). That is, "***id cn neighborlist***".

The neighbors in the output file is sorted by their distances to the centered particle in ascending order. Neighbor list of different snapshots is continuous without any gap and all start with the header "***id cn neighborlist***". This formatting is made to be consistent with reading neighbor lists for different analyzing methods.

## Nnearests

 `neighbors.particle_neighbors.Nnearests` gets the `N` nearest neighbors around a particle. In this case, the coordination number is `N` for each particle.

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
from neighbors.particle_neighbors import Nnearests

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

Nnearests(readdump.snapshots, N=12, ppp=[1,1,1], fnfile='neighborlist.dat')
```

## cutoffneighbors

`neighbors.particle_neighbors.cutoffneighbors` gets the nearest neighbors around a particle by setting a global cutoff distance `r_cut`.  Usually, the cutoff distance can be determined as the position of the first deep valley in total pair correlation function.

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
from neighbors.particle_neighbors import cutoffneighbors

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

cutoffneighbors(readdump.snapshots, r_cut=3.8, ppp=[1,1,1], fnfile='neighborlist.dat')
```

## cutoffneighbors_particletype

`neighbors.particle_neighbors.cutoffneighbors_particletype` gets the nearest neighbors around a particle by setting a cutoff distance ` r_cut`. Taken Cu-Zr system as an example, `r_cut` should be a 2D numpy array:
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
from neighbors.particle_neighbors import cutoffneighbors_particletype

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

