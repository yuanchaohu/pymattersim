# Pair Correlation Function

`static.gr.gr` class is used to calculate the overall and partial pair correlation functions ***g(r)*** in orthogonal and triclinic cells at various dimensional systems. This module is suitable for multi-component systems, from unary to senary. ***g(r)*** is defined as:
$$
g(r) = \lang \sum_{i \neq j} \delta (r - |\vec r_{ij}|) \rang
$$

$$
g_{\alpha \beta}(r) = \lang \sum_{i \neq j; i \in \alpha; j \in \beta} \delta (r - |\vec r_{ij}|) \rang
$$

$$
g_A(r) = \lang \sum_{i \neq j} \delta (r - |\vec r_{ij}|)A_i A_j \rang = \lang A(r) A(0) \rang
$$

spatial correlation function of particle-level quantity $A_i$ is $g_A(r) / g(r)$.

where ***N*** is particle number, ***ρ*** is number density. ***g(r)*** ranges to half of the box length minimum (***L_min/2***).

## 1. Initializing Method
Initializing ***g(r)*** class

### Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file
- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no
  
  default `[1,1,1]`, that is, PBC is applied in all three dimensions for 3D box. Set `[1,1]` for two-dimensional system.
- `rdelta` (`float`): bin size calculating g(r), default 0.01
- `outputfile` (`str`): the name of csv file to save the calculated g(r)

### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from static.gr import gr

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

gr_cal = gr(readdump.snapshots, ppp=[1,1,1], rdelta=0.01, outputfile='gr.csv')
```

## 2. getresults
`getresults` method is used to to determine which function to call for calculating ***g(r)*** based on the number of different particle types in the system.

### Input Arguments
`None`

### Example
```python
gr_cal.getresults()
```

### Return
`Optional[Callable]`. For the sytem with one particle type, calling `self.unary()` function to calculate ***g(r)***, also including `self.binary()`, `self.ternary()`, `self.quarternary`, `self.quinary` for binary, ternary, quarternary, and quinary systems. For systems with more than five particle types, only overall ***g(r)*** will be calcuated by calling `self.unary()` function.

The calculated ***g(r)*** is storted in the `outputfile`. Taken ternary sytem as an example, each column `outputfile` is ***r g(r) g11(r) g22(r)  g33(r) g12(r) g13(r) g23(r)***, respectively, with a header.

