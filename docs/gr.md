# Pair Correlation Function

`static.gr.gr` class is used to calculate the overall and partial pair correlation functions (PCF) $g(r)$ in orthogonal and triclinic cells at various dimensional systems. The calculated $g(r)$ ranges to half of the box length minimum (*L_min/2*). This module is suitable for multi-component systems, from unary to quinary.  The overall $g(r)$ is defined as:

$$
g(r) = \frac{1}{N \rho} \lang \sum_{i \neq j} \delta (r - |\vec r_{ij}|) \rang
$$

where $N$ is particle number, $\rho$ is number density ($\rho=N/V$, where $V$ is cell volume), and $|\vec r_{ij}|$ is the distance between centered particle $i$ and neighboring particles $j$. The partial PCF $g_{\alpha \beta}(r)$, in which $\alpha$ and $\beta$ representing two particle types, is defined as

$$
g_{\alpha \beta}(r) = \frac{V}{N_{\alpha} N_{\beta}} \lang \sum_{i \neq j; i \in \alpha; j \in \beta} \delta (r - |\vec r_{ij}|) \rang
$$

`static.gr.conditional_gr` function is used to calculate $g(r)$ of a single configuration for selected particles, and is also useful to calculate the spatial correlation of a particle-level physical quantity $A_i$. There are three conditions considered for $A_i$:
- condition is bool type, so calculate partial g(r) for selected particles
- condition is complex number, so calculate spatial correlation of complex number
- condition is float scalar, so calculate spatial correlation of scalar number
 
$A_i=1$ for normal overall $g(r)$, representing all particle are selected. Bool type $A_i$ makes the calculation for only selected particles with $A_i=True$. Pleause see more details in example.

$$
g_A(r) = \frac{1}{N \rho} \lang \sum_{i \neq j} \delta (r - |\vec r_{ij}|)A_i A_j \rang = \lang A(r) A(0) \rang
$$

The spatial correlation function of particle-level quantity $A_i$ is $g_A(r) / g(r)$.

## 1. `g(r)` class

### Initializing method
Initializing `g(r)` class

#### Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): `Snapshots` data class returned by `reader.dump_reader.DumpReader` from input configuration file
- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no
  
  default `[1,1,1]`, that is, PBC is applied in all three dimensions for 3D box. Set `[1,1]` for two-dimensional system.
- `rdelta` (`float`): bin size calculating g(r), default 0.01
- `outputfile` (`str`): the name of csv file to save the calculated g(r)

#### Example

```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from static.gr import gr

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

gr_cal = gr(readdump.snapshots, ppp=[1,1,1], rdelta=0.01, outputfile='gr.csv')
```

### `getresults`
`getresults` method is used to to determine which function to call for calculating $g(r)$ based on the number of different particle types in the system.

#### Input Arguments
`None`

#### Example
```python
gr_cal.getresults()
```

### Return
`Optional[Callable]`. For the sytem with one particle type, calling `self.unary()` function to calculate $g(r)$, also including `self.binary()`, `self.ternary()`, `self.quarternary`, `self.quinary` for binary, ternary, quarternary, and quinary systems. For systems with more than five particle types, only overall $g(r)$ will be calcuated by calling `self.unary()` function.

The calculated $g(r)$ is storted in the `outputfile`. Taken ternary sytem as an example, each column `outputfile` is ***r gr gr11 gr22  gr33 gr12 gr13 gr23***, respectively.

## 2. `conditional_gr`

### Input Arguments
- `snapshot` (`reader.reader_utils.SingleSnapshot`): single snapshot object of input trajectory
- `condition` (`np.ndarray`): particle-level condition for g(r)
- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no
  
  default `[1,1,1]`, that is, PBC is applied in all three dimensions for 3D box. Set `[1,1]` for two-dimensional system.
- `rdelta` (`float`): bin size calculating g(r), default 0.01

### Return

calculated conditional $g(r)$, `gA` column in the returned `pd.DataFrame`.

original overall `g(r)` is also calculated for reference or post-processing, `gr` column in the returned `pd.DataFrame`.

### Example
```python
from static.gr import conditional_gr

for snapshot in readdump.snapshots.snapshots:
    
    # Select particles with a particle type of 2 for g(r) calculation
    # in the returned DataFrame, the column gA is actually gr22 in this case
    grresults_selection = conditional_gr(snapshot, condition=snapshot.particle_type == 2)

    # Randomly generating values for particle quantity, ranging from 0 to 1
    particle_quantity = np.random.rand(snapshot.nparticle)
    
    # particle level quantity as conditoin to calculate g(r), also support complex-number quantity
    grresults_quantity = conditional_gr(snapshot, condition=particle_quantity)
    
    # calculate the spatial correlation function of particle-level quantity
    grresults_quantity["gA"] / grresults_quantity["gr"]
```
