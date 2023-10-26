# Particle Neighbors

This module is used to calculate the neighboring particles around a particle. The format of the saved neighbor list file (named as 'neighborlist.dat' in default) must be identification of the centered particle (***id***), coordination number of the centered particle (***cn***), and identification of neighboring particles (***neighborlist***). The neighbors in the output file is sorted by their distances to the centered particle in ascending order. Neighbor list of different snapshots is continuous without any gap and all start with the header ***id cn neighborlist***.

## Nnearests

 `neighbors.particle_neighbors.Nnearests` gets the `N` nearest neighbors around a particle. In this case, the coordination number is `N` for each particle.

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader`

- `N` (`int`): the specified number of nearest neighbors, default 12

- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no

  ​					   default `[1, 1, 1]`, that is, PBC is applied in all three dimensions for 3D box

- `fnfile` (`str`): the name of output file that stores the neighbor list, default ***neighborlist.dat***

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType

from neighbors.particle_neighbors import Nnearests

filename = './tests/sample_test_data/test_additional_columns.dump'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()
snapshots = readdump.snapshots

Nnearests(snapshots, N = 12, ppp = [1, 1, 1], fnfile='neighborlist.dat')
```

## cutoffneighbors

`neighbors.particle_neighbors.cutoffneighbors` gets the nearest neighbors around a particle by setting a global cutoff distance `r_cut`.  Usually, the cutoff distance can be determined as the position of the first deep valley in total pair correlation function.

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`):`Snapshots` data class returned by `reader.dump_reader.DumpReader`

- `r_cut` (`float`): the global cutoff distance to screen the nearest neighbors

- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no

  ​					   default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box

- `fnfile` (`str`): the name of output file that stores the neighbor list, default ***neighborlist.dat***

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType

from neighbors.particle_neighbors import cutoffneighbors

filename = './tests/sample_test_data/test_additional_columns.dump'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()
snapshots = readdump.snapshots

cutoffneighbors(snapshots, r_cut = 3.8, ppp = [1, 1, 1], fnfile='neighborlist.dat')
```

## cutoffneighbors_particletype

`neighbors.particle_neighbors.cutoffneighbors_particletype` get the nearest neighbors around a particle by setting a cutoff distance ` r_cut`. Taken Cu-Zr system as an example, `r_cut` should be a 2D numpy array:
$$
\begin{bmatrix}
  &r_{cut}^{Cu-Cu} & r_{cut}^{Cu-Zr}&\\
  &r_{cut}^{Zr-Cu} & r_{cut}^{Zr-Zr}&\\
\end{bmatrix}
$$
Usually, these cutoff distances can be determined as the position of the first deep valley in partial pair correlation function of each particle pair.

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader`

- `r_cut` (`np.array`): the cutoff distances of each particle pair

- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no

  ​					   default [1, 1, 1], that is, PBC is applied in all three dimensions for 3D box

- `fnfile` (`str`): the name of output file that stores the neighbor list, default ***neighborlist.dat***

### Example

```python
import numpy as np

from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType

from neighbors.particle_neighbors import cutoffneighbors_particletype

filename = './tests/sample_test_data/test_additional_columns.dump'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()
snapshots = readdump.snapshots

cutoffneighbors_particletype(snapshots, r_cut = np.array(([3.6, 3.4], [3.4, 3.9])), 
                             ppp = [1, 1, 1], fnfile='neighborlist.dat')
```

