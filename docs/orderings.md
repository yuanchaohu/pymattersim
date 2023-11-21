# Local tetrahedral order $q_{\rm tetra}$

The class `static.tetrahedral` calculates the local tetrahedral order of the simulation system in three dimensions, such as for water-type and silicon/silica-type systems. Local tetrahedral order is defined as:
$$
q_{\rm tetra}=\sum_{j=1}^3 \sum_{k=j+1}^4 \left(\cos \varphi_{jk}+\frac{1}{3} \right)^2
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
from tetrahedrality import local_tetrahedral_order

readdump = DumpReader(filename='dump_3d.atom', ndim=3)
readdump.read_onefile()

tetrahedral = q_tetra(readdump.snapshots)
```