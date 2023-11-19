# Bond Orientational Order Parameters at 2D

The class `static.boo.boo_2d()` calculates the bond orientational order (BOO) parameters in two-dimensions. All calculations start from measuring the $l$-th order as a complex number $\varphi_l^i$:
$$
\varphi_l^i=\frac{1}{N_i}\sum_{j=1}^{N_i}e^{il\theta_j^i} \tag{1}
$$
where $N_i$ is the number of bonds of particles $j$. This is the original one with all ($N_i$) neighboring particles treated the same. However, in some cases, we need to treat these neighboring particles differently based on their properties with respect to the center particle $i$. Then we can obtain the weighted complex number $\varphi_l^i$ as:
$$
\varphi_l^i=\sum_{j=1}^{N_i}\frac{\ell_j}{\sum \ell_j}e^{il\theta_j^i} \tag{2}
$$
where $\ell_j$ is the property of the $j-$th neighbor. Therefore, the normalized weights $\frac{\ell_j}{\sum \ell_j}$ are used in the calculation. We can treat $\ell_j=1$ for the case shown in equation (1). The weights are provided in the same way as for the neighbors.

The time averaged $\varphi_l^i$ is expressed as:
$$
\psi_l^i=\frac{1}{t} \int_{0}^{t} \,dt |\varphi_l^i| \tag{3}
$$

By using the complex number $\varphi_l^i$ various BOO parameters can be calculated. For example,

+ spatial correlation of the complex number by `boo_2d.spatial_corr()`
  $$
  g_l(r)=\frac{L^2}{2\pi r \Delta r N(N-1)} \langle \sum_{j \neq k} \delta(\vec r - |\vec r_{jk}|) \varphi_l^j \varphi_l^{k*} \rangle \tag{4}
  $$

+ time correlation of the complex number by `boo_2d.time_corr()`
  $$
  C_l(t)=\frac{\langle \sum_n \varphi_l^n(t) \varphi_l^{n*}(0) \rangle}{\langle \sum_n |\varphi_l^n(0)|^2 \rangle} \tag{5}
  $$

## 1. `boo_2d()` class

### Input Arguments

- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `l` (`int`): degree of orientational order, like l=6 for hexatic order
- `neighborfile` (`str`): file name of particle neighbors (see module `neighbors`)
- `weightsfile` (`str`): file name of particle-neighbor weights (see module `neighbors`), one typical example is Voronoi cell edge length of the polygon; this file should be consistent with neighborfile, default `None`
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([1,1])`,
- `Nmax` (`int`): maximum number for neighbors, default 10

### Return:
- None

### Example
```python
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from neighbors.freud_neighbors import cal_neighbors
from static.boo import boo_2d

filename = 'dump.atom'
readdump = DumpReader(filename, ndim=3, filetype=DumpFileType.LAMMPS, moltypes=None)
readdump.read_onefile()

cal_neighbors(readdump.snapshots, outputfile='test')
boo = boo_2d(readdump.snapshots,
             l=6,
             neighborfile='test.neighbor.dat'
             weightsfile='test.edgelength.dat')
```

## 2. `lthorder`
Calculate $l$-th orientational order in 2D, such as hexatic order

### Input Arguments
- None

### Return
- Calculated $l$-th order in complex number (`np.ndarray`) with shape `[nsnapshots, nparticle]`

### Example
```python
lthorder_results = boo2d.lthorder()
```

## 3. `tavephi`
Compute PHI value and Time Averaged PHI

### Input Arguments

- `outputphi` (`str`): file name for absolute values of phi, default None
- `outputavephi` (`str`): file name for time averaged phi, default `None`
- `avet` (`float`): time used to average, default 0.0
- `dt` (`float`): timestep used in user simulations, default 0.002

### Return
- Calculated phi value or time averaged phi (`np.ndarray`)

### Example
```python
tavephi_results = boo2d.tavephi(avet=0.0, dt=0.002)
```

## 4. `spatial_corr`
Calculate spatial correlation of phi in 2D system

### Input Arguments
- `rdelta` (`float`): bin size in calculating g(r) and Gl(r), default 0.01
- `outputfile` (`str`): csv file name for gl(r)

### Return
- calculated gl(r) based on phi (`pd.DataFrame`)

### Example
```python
spatial_corr_results = boo2d.spatial_corr()
```

## 5. `time_corr`
Calculate time correlation of phi in 2D system

### Input Arguments
- `dt` (`float`): timestep used in user simulations, default 0.002
- `outputfile` (`str`): csv file name for time correlation results, default `None`


### Return
time correlation quantity (`pd.DataFrame`)

### Example
```python
time_corr_results = boo2d.time_corr()
```