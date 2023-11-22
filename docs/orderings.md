# Local tetrahedral order $q_{\rm tetra}$

The class `static.tetrahedral` calculates the local tetrahedral order of the simulation system in three dimensions, such as for water-type and silicon/silica-type systems. Local tetrahedral order is defined as:
$$
q_{\rm tetra}=1-\frac{3}{8} \sum_{j=1}^3 \sum_{k=j+1}^4 \left(\cos \varphi_{jk}+\frac{1}{3} \right)^2
$$

In this calculation, only **4** nearest neighbors are taken into consideration. The algorithm of selecting nearest distances is from `numpy.argpartition` for fast computation. In this method, only the nearest neighbors are selected but not in a sorted order. $j$, $k$ run over these neighbors. 

## Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory
(returned by `reader.dump_reader.DumpReader`)
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.array([1,1,1])`
- `outputfile` (`str`): the name of file to save the calculated local tetrahedral order

## Return
- calculated local tetrahedral order in `np.ndarray` with shape `[nsnapshots, nparticle]`


## Example
```python
from reader.dump_reader import DumpReader
from static.tetrahedral import q_tetra

readdump = DumpReader(filename='dump_3d.atom', ndim=3)
readdump.read_onefile()

tetrahedral = q_tetra(readdump.snapshots)
```

# Packing capability $\Theta_o$
The module `static.packing_capability.theta_2D` calculates packing capability of a 2D system by comparing the bond angles in realistic configuration to a reference one (ref. [Tong & Tanaka PHYS. REV. X 8, 011041 (2018)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.011041)). $\Theta_o$ is defined as,
$$
\Theta_o = \frac{1}{N_o} \sum_{<i, j>} \left| \theta_{ij}^1 - \theta_{ij}^2 \right|
$$
where $N_o$ is the unique nearest pair number around the center, which equals to the coordination number.

## Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory
(returned by `reader.dump_reader.DumpReader`)
- `sigmas` (`np.ndarray`): particle sizes for each pair type (ascending order) in numpy array, can refer to  first peak position of partial g(r), shape as `[particle_type, particle_type]`
- `neighborfile` (`str`): file name of particle neighbors (see module `neighbors`)
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.array([1,1])`
- `outputfile` (`str`): the name of file to save the calculated theta_2D

## Return
- calculated packing capability of a 2D system in `np.ndarray` with shape `[nsnapshots, nparticle]`

## Example
```python
from reader.dump_reader import DumpReader
from static.packing_capability import theta_2D
from neighbors.freud_neighbors import cal_neighbors

filename = 'dump_2D.atom'
readdump = DumpReader(filename, 2)
readdump.read_onefile()

cal_neighbors(snapshots=readdump.snapshots, outputfile='test')
theta = theta_2D(readdump.snapshots, sigmas=np.array([[1.5,1.5],[1.5,1.5]]), neighborfile='test.neighbor.dat')
```
