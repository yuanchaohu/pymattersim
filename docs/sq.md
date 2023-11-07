# Structure Factors

`static.sq.sq` class is used to calculate the overall and partial static structure factors ***S(q)*** in orthogonal cells at various dimensional systems. This module is suitable for multi-component systems, from unary to senary. S(q) is defined as:

Fourier transformation of particle level physical quantity $A_i$ ($A_i=1$ for normal S(q) of unary system). Bool type $A_i$ makes the calculation for only selected particles with $A_i=True$.
$$
\rho_\alpha ({\vec q}) = \frac{1}{\sqrt{N_\alpha}} \sum_i^{N_\alpha} A_i \exp(-i \cdot {\vec q} \cdot {\vec r_i})
$$ 

$$
S_\alpha({\vec q}) = \lang \rho_\alpha({\vec q} \rho_\alpha({-\vec q})) \rang
$$

where ***N*** is particle number. The computation time is determined partly by the range of wavenumber, which now is regulated by the input argument `qrange`. The module now is only applicable for cubic systems.

## 1. Initializing Method
Initializing ***S(q)*** class

### Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file
- `qrange` (`float`): the wave number range to be calculated, default 10
- `onlypositive` (`bool`): whether only consider positive wave vectors, default `False`
- `qvector` (`np.ndarray`): input wave vectors, if `None` (default) use qrange & onlypositive
- `saveqvectors` (`bool`): whether to save ***S(q)*** for specific wavevectors, default `False`
- `outputfile` (`str`): the name of csv file to save the calulated ***S(q)***, default `None`

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from static.sq import sq

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

sq_cal = sq(readdump.snapshots, outputfile='sq.csv')
```

## 2. getresults
`getresults` method is used to to determine which function to call for calculating ***S(q)*** based on the number of different particle types in the system.

### Input Arguments
`None`

### Example
```python
sq_cal.getresults()
```

### Return
`Optional[Callable]`. For the sytem with one particle type, calling `self.unary()` function to calculate ***S(q)***, also including `self.binary()`, `self.ternary()`, `self.quarternary`, `self.quinary` for binary, ternary, quarternary, and quinary systems. For systems with more than five particle types, only overall ***S(q)*** will be calcuated by calling `self.unary()` function.