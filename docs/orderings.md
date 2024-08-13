# 1. Local tetrahedral order $q_{\rm tetra}$

The class `static.tetrahedral` calculates the local tetrahedral order of the simulation system in three dimensions, such as for water-type and silicon/silica-type systems. Local tetrahedral order is defined as:
$$
q_{\rm tetra}=1-\frac{3}{8} \sum_{j=1}^3 \sum_{k=j+1}^4 \left(\cos \varphi_{jk}+\frac{1}{3} \right)^2
$$

In this calculation, only **4** nearest neighbors are taken into consideration. The algorithm of selecting nearest distances is from `numpy.argpartition` for fast computation. In this method, only the nearest neighbors are selected but not in a sorted order. $j$, $k$ run over these neighbors. 

## 1.1 Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory
(returned by `reader.dump_reader.DumpReader`)
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.array([1,1,1])`
- `outputfile` (`str`): file name to save the calculated local tetrahedral order, default `None`.
                        To reduce storage size and ensure loading speed, save npy file as default with extension ".npy".
                        If the file extension is ".dat" or ".txt", also saved a text file.

## 1.2 Return
- calculated local tetrahedral order in `np.ndarray` with shape `[nsnapshots, nparticle]`


## 1.3 Example
```python
from reader.dump_reader import DumpReader
from static.tetrahedral import q_tetra

readdump = DumpReader(filename='dump_3d.atom', ndim=3)
readdump.read_onefile()

tetrahedral = q_tetra(readdump.snapshots)
```

# 2. Packing capability $\Theta_o$
The module `static.packing_capability.theta_2D` calculates packing capability of a 2D system by comparing the bond angles in realistic configuration to a reference one (ref. [Tong & Tanaka PHYS. REV. X 8, 011041 (2018)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.011041)). $\Theta_o$ is defined as,
$$
\Theta_o = \frac{1}{N_o} \sum_{<i, j>} \left| \theta_{ij}^1 - \theta_{ij}^2 \right|
$$
where $N_o$ is the unique nearest pair number around the center, which equals to the coordination number.

## 2.1 Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory
(returned by `reader.dump_reader.DumpReader`)
- `sigmas` (`np.ndarray`): particle sizes for each pair type (ascending order) in numpy array, can refer to  first peak position of partial g(r), shape as `[particle_type, particle_type]`
- `neighborfile` (`str`): file name of particle neighbors (see module `neighbors`)
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.array([1,1])`
- `outputfile` (`str`): file name to save the calculated packing capability theta_2D, default `None`.
                        To reduce storage size and ensure loading speed, save npy file as default with extension ".npy".
                        If the file extension is ".dat" or ".txt", also saved as a text file.

## 2.2 Return
- calculated packing capability of a 2D system in `np.ndarray` with shape `[nsnapshots, nparticle]`

## 2.3 Example
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

# 3. Pair entropy $S_2$
The module `static.pair_entropy` calculates the particle-level pair entropy $S_2$, defined as,
$$
S_2^i = -2 \pi \rho k_B \int_0^{r_m} [g_m^i(r) \ln g_m^i(r) - g_m^i(r)+1] r^2 dr \quad (3D) \\
S_2^i = -\pi \rho k_B \int_0^{r_m} [g_m^i(r) \ln g_m^i(r) - g_m^i(r)+1] r dr \quad (2D)
\tag{1}
$$
where $r_m$ is is an upper integration limit that, in principle, should
be taken to infinity ($g(r_m \to \infty)=1$), and $g_m^i(r)$ is the pair correlation function centered at the $i$ th particle. We use a Gaussian smeared $g_m^i(r)$ to obtain a continuous and differentiable quantity,
$$
g_m^i(r) = \frac{1}{4 \pi \rho r^2} \sum_j \frac{1}{\sqrt {2 \pi \sigma^2}} e^{-(r-r_{ij})^2/(2 \sigma ^2)} \quad (3D) 
\\
g_m^i(r) = \frac{1}{2 \pi \rho r} \sum_j \frac{1}{\sqrt {2 \pi \sigma^2}} e^{-(r-r_{ij})^2/(2 \sigma ^2)} \quad (2D)
\tag{2}
$$
where $j$ are the neighbors of atom $i$, $r_{ij}$ is the pair distance, and $\sigma$ is a broadening parameter. 
The integration in Eq. (1) is calculated numerically using the trapezoid rule.

## `S2` class

### Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by `reader.dump_reader.DumpReader`)
- `sigmas` (`np.ndarray`): gaussian standard deviation for each pair particle type, can be set based on particle size. It must be a two-dimensional numpy array to cover all particle type pairs
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.ndarray=np.array([1,1,1])`, set `np.ndarray=np.array([1,1])` for two-dimensional systems
- `rdelta` (`float`): bin size calculating g(r), the default value is `0.02`
- `ndelta` (`int`): number of bins for g(r) calculation, `ndelta*rdelta` determines the range

### Return:
- None

### Example
```python
from static.pairentropy import S2
s2 = S2(readdump.snapshots,
        sigmas=np.array([[0.3, 0.2], [0.2, 0.4]]))
```

## `particle_S2()`
The function calculates the particle-level $g_m^i(r)$ by Gaussian smoothing and then calculate the particle-level $S_2^i$.

### Input Arguments
- `savegr` (`bool`): whether to save particle $g_m^i(r)$, default `False`
- `outputfile` (`str`): the name of csv file to save the calculated $S_2^i$

### Return:
- `s2_results`: particle level $S_2^i$ in shape `[nsnapshots, nparticle]`
- particle level $g_m^i$ if `savegr=True` in shape `[nsnapshots, nparticle, ndelta]`

### Example
```python
s2.particle_s2(savegr=True)
```

## `spatial_corr()`
`spatial_corr()` method calculates spatial correlation function of (normalized) $S_2^i$:
$$
g_s(r) = <S_2(0)S_2(r)>
$$

### Input Arguments
- `mean_norm` (`bool`): whether use mean normalized $S_2^i$
- `outputfile` (`str`): csv file name for $g_l$ of $S_2^i$, default `None`

### Return
- calculated spatial correlation results of $S_2^i$ (`pd.DataFrame`)

### Example
```python
glresults = s2.spatial_corr()

glresults_normalized = s2.spatial_corr(mean_norm=True)
```

## `time_corr()`
`time_corr()` method calculates time correlation of $S_2^i$
$$
g_s(t) = <S_2(0)S_2(t)>
$$

### Input Arguments
- `mean_norm` (`bool`): whether use mean normalized $S_2^i$
- `outputfile` (`str`): csv file name for $g_l$ of $S_2^i$, default `None`

### Return
- calculated time correlation results of $S_2^i$ (`pd.DataFrame`)

### Example
```python
s2.time_corr()
```


# 4. Gyration tensor of a group
This module calculates calculate gyration tensor of a cluster of atoms. This module calculates gyration tensor which is a tensor that describes the second moments of posiiton of a collection of particles gyration tensor is a symmetric matrix of shape (ndim, ndim). ref: https://en.wikipedia.org/wiki/Gyration_tensor. A group of atoms should be first defined. groupofatoms are the original coordinates of the selected group of a single configuration, the atom coordinates of the cluster should be removed from PBC which can be realized by ovito 'cluster analysis' method by choosing 'unwrap particle coordinates'.

### Input Arguments
- `pos_group` (`np.ndarray`): unwrapped particle positions of a group of atoms, shape as [num_of_particles, dimensionality]

### Return
`3D`: `radius_of_gyration`, `asphericity`, `acylindricity`, `shape_anisotropy`, `fractal_dimension` 

`2D`: `radius_of_gyration`, `acylindricity`, `fractal_dimension`

### Example
``` python
from static.shape import gyration_tensor

gyration_tensor(cluster_positions)
```