# Radical Voronoi Tessellation

This module is used to perform radical Voronoi tessellation using [voro++](https://math.lbl.gov/voro++/) for both PBC and non-PBC system. During calculations, the command line used will be printed.

The voro++ package considers non-periodic boundary conditions by default, so there may be some negative numbers in the neighbor list for non-periodic boundary conditions (please refer to the voro++ manual to know this well). A function `voronoi.voropp.voronowalls()` is designed to remove negative numbers in the neighbor list and other files correspondingly. Please choose `voronoi.voropp.cal_voro()` for all periodic boundary conditions and `voronowalls()` for the opposite. Note that the former is much faster than the latter.

## get_input

`voronoi.voropp.get_input` is used to design input file for Voro++ by considering particle radii.

### Inputs

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

- `radii` (`dict`): radii of particles, must be a dict like `{1 : 1.28, 2 : 1.60} `

  ​						   if you do not want to consider radii, set the radii the same, default `{1:1.0, 2:1.0} `

### Return

- `position` (`list`): input file for Voro++ with the format:

  ​								 ***particle_ID  x_coordinate  y_coordinate  z_coordinate radius***

- `bounds` (`list`): box bounds for snapshots

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from voronoi.voropp import get_input

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

get_input(readdump.snapshots, radii={1:1.0, 2:1.0})
```

## cal_voro

`voronoi.voropp.cal_voro` is used to perform radical Voronoi tessellation using voro++ for periodic boundary conditions.

### Inputs

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

- `ppp` (`str`): Make the container periodic in all three directions, default `ppp='-p'`

- `radii` (`dict`): radii of particles, must be a dict like `{1 : 1.28, 2 : 1.60} `

  ​						   if you do not want to consider radii, set the radii the same, default `{1:1.0, 2:1.0} `

- `outputfile` (`str`): filename of output, including ***neighborlist***, ***facearealist***, ***voronoi index***, ***overall*** (facearea and volume of particle).

### Return

- `None` [saved to file]
- for neighbor list, file with name `outputfile.neighbor.dat'`
- for facearea list, file with name `outputfile.facearea.dat'`
- for voronoi index, file with name `outputfile.voroindex.dat'`
- for overall, file with name `outputfile.overall.dat'`

Output files from ***voroindex*** and ***overall*** include a header for all snapshots. ***neighborlist*** and ***facearealist*** output files are in align with the format needed in the module `neighbors` and have headers for each individual snapshot.

### Example

```python
from reader.lammps_reader_helper import read_lammps_wrapper
from voronoi.voropp import cal_voro

test_file_3d = 'dump.atom'
snapshots = read_lammps_wrapper(test_file_3d, 3)

cal_voro(snapshots, outputfile='dump')
```

## voronowalls

`voronoi.voropp.voronowalls` is used to perform radical Voronoi tessellation using voro++ for periodic boundary conditions.

### Inputs

- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file

- `ppp` (`str`): Make the container periodic in a desired direction

  ​					 `'-px'`, `'-py'`, and `'-pz'` for x, y, and z directions, respectively

- `radii` (`dict`): radii of particles, must be a dict like `{1 : 1.28, 2 : 1.60} `

  ​						   if you do not want to consider radii, set the radii the same, default `{1:1.0, 2:1.0} `

- `outputfile` (`str`): filename of output, including ***neighborlist***, ***facearealist***, ***voronoi index***, ***overall*** (facearea and volume of particle).

### Return

- `None` [saved to file]
- for neighbor list, file with name `outputfile.neighbor.dat'`
- for facearea list, file with name `outputfile.facearea.dat'`
- for voronoi index, file with name `outputfile.voroindex.dat'`
- for overall, file with name `outputfile.overall.dat'`

Output files from ***voroindex*** and ***overall*** include a header for all snapshots. ***neighborlist*** and ***facearealist*** output files are in align with the format needed in the module `neighbors` and have headers for each individual snapshot.

## indicehis

Statistics the frequency of voronoi index from the output of voronoi analysis. Only the top 50 voronoi index will be output along with their fractions.

### Inputs

- `inputfile` (`str`): the filename of saved Voronoi index
- `outputfile` (`str`): the output filename of the frequency of voronoi index

### Return

- None [saved to file], frequency of Voronoi Indices

### Example

```python
from voronoi.voropp import indicehis

indicehis('dump.voroindex.dat', outputfile='voroindex_frction.dat')
```

