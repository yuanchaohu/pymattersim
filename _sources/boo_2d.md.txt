### Bond Orientational Order Parameters at 2D

The class `static.boo.boo_2d()` calculates the bond orientational order (BOO) parameters in two dimensions. All calculations start from measuring the $l$-th order as a complex number $\varphi_l(j)$ for particle $j$:

$$
\varphi_l(j)=\frac{1}{N_j}\sum_{m=1}^{N_j}e^{i l \theta_m} \tag{1},
$$

where $N_j$ is the number of bonds of particles $j$ in consideration. This is the original one with all ($N_j$) neighboring particles treated as the same. However, in some cases, we need to treat these neighboring particles differently based on their properties with respect to the center particle $j$. Then we can obtain the weighted complex number $\varphi_l(j)$ as:

$$
\varphi_l(j)=\sum_{m=1}^{N_j}\frac{A_{jm}}{\sum |A_{jm}|}e^{i l \theta_m} \tag{2},
$$

where $A_{jm}$ is the property of the $(j-m)$ bond. Therefore, the normalized weights $\frac{A_{jm}}{\sum |A_{jm}|}$ are used in the calculation. Note that in the normalization we take the absolute values of $A_{jm}$
We can treat all $A_{jm}=1$ for the case shown in equation (1). The weights are provided in the same way as for the neighbors.

Based on the orientational order parameter $\varphi_l(j)$ in complex number, both the modulus and phase (or argument) are calculated by numpy functions.

In some cases, to remove thermal fluctuations, time average over a time period of $\tau$ can be performed in two ways:
1. time average of scalar modulus and phase

$$
\psi_l(j)=\frac{1}{\tau} \int_{t-\tau/2}^{t+\tau/2} |\varphi_l(j)| dt (\tag 3)
$$

2. time average of complex number $\varphi_l(j)$ and then get the scalar modulus and phase

$$
\psi_l(j)=\frac{1}{\tau} \int_{t-\tau/2}^{t+\tau/2} \varphi_l(j) dt (\tag 4)
$$ 


By using the complex number $\varphi_l(j)$, both spatial correlation and time correlation can be calculated by:

+ spatial correlation of the complex number
  
$$
  g_l(r)=\frac{L^2}{2\pi r \Delta r N(N-1)} \left< \sum_{j \neq k} \delta(\vec r - |\vec r_{jk}|) \varphi_l(j) \varphi_l^*(k) \right> \tag{5}
$$

+ time correlation of the complex number

$$
  C_l(t)=\frac{\left<le \sum_n \varphi_l^n(t) \varphi_l^{n*}(0) \right>le}{\left<le \sum_n |\varphi_l^n(0)|^2 \right>le} \tag{6}
$$

##### 1. `boo_2d` class

**Input Arguments**

- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `l` (`int`): degree of orientational order, like l=6 for hexatic order
- `neighborfile` (`str`): file name of particle neighbors (see module `neighbors`)
- `weightsfile` (`str`): file name of particle-neighbor weights (see module `neighbors`), one typical example is Voronoi cell edge length of the polygon; this file should be consistent with neighborfile, default `None`
- `ppp` (`npt.NDArray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([1,1])`,
- `Nmax` (`int`): maximum number for neighbors, default 10

**Return**:
- None

**Example**
```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from neighbors.freud_neighbors import cal_neighbors
from static.boo import boo_2d

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=2, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

cal_neighbors(readdump.snapshots, outputfile='test')
boo = boo_2d(readdump.snapshots,
             l=6,
             neighborfile='test.neighbor.dat'
             weightsfile='test.edgelength.dat')
```

##### 2. `lthorder()`
Calculate $l$-th orientational order in 2D, such as hexatic order

**Input Arguments**
- None

**Return**
- Calculated $l$-th order in complex number (`npt.NDArray`) with shape `[nsnapshots, nparticle]`

**Example**
```python
lthorder_results = boo2d.lthorder()
```

##### 3. `time_average()`
Calculate the time averaged $\varphi_l(j)$. There are two cases considered in this function:
- return the original complex order parameter (`time_period=None` or as default)
- return time averaged complex order parameter (`time_period` not `None`) by averaging complex number directly (`average_complex=True`, `default`) or by averaging mudulus and phase of the complex number first and then calculate complex order parameter (`average_complex=False`).

As equation (3) and (4) shows, one snapshot is intended to be averaged from $-\tau/2$ to $\tau/2$, so the middle snapshot number (`average_snapshot_id`) is also returned for reference. 

**Input Arguments**

- `time_period` (`float`): time average period, default `None`
- `dt` (`float`): simulation snapshots time step, default 0.002
- `average_complex` (`bool`): whether averaging the complex order parameter or not, default `True`
- `outputfile` (`float`): file name of the output modulus and phase, default `None`

**Return**
- `ParticlePhi` (`npt.NDArray`): the original complex order parameter if `time_period=None`
- `average_quantity` (`npt.NDArray`): time averaged $\varphi_l(j)$ results if `time_period` not `None`
- `average_snapshot_id` (`npt.NDArray`): middle snapshot number between time periods if `time_period` not `None`

**Example**
```python
modulus, phase = boo2d.modulus_phase(time_period=0.0, dt=0.002, average_complex=False)
```

##### 4. `spatial_corr()`
Calculate spatial correlation of the orientational order parameter

**Input Arguments**
- `rdelta` (`float`): bin size in calculating g(r) and Gl(r), default 0.01
- `outputfile` (`str`): csv file name for gl(r), default `None`

**Return**
- calculated gl(r) based on phi (`pd.DataFrame`)

**Example**
```python
spatial_corr_results = boo2d.spatial_corr()
```

##### 5. `time_corr()`
Calculate time correlation of the orientational order parameter

**Input Arguments**
- `dt` (`float`): timestep used in user simulations, default 0.002
- `outputfile` (`str`): csv file name for time correlation results, default `None`


**Return**
- time correlation quantity (`pd.DataFrame`)

**Example**
```python
time_corr_results = boo2d.time_corr()
```
