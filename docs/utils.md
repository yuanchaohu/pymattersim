# util functions
## [I. math functions](#i-funcs)
## [II. remove periodic boundary conditions](#ii-remove-pbc)
## [III. geometry](#iii-geometry-1)
## [IV. wavevector](#iv-wavevector-1)
## [V. Fast Fourier Transformation](#v-fast-fourier-transformation-1)
## [VI. spherical harmonics](#vi-spherical-harmonics-1)
## [VII. coarse graining](#vii-coarse-graining-1)

---

# I. funcs
mathmatical functions for feasible computations

## `1. nidealfac()`
`nidealfac()` is used to choose pre-factor of `Nideal` in g(r) calculation.

### Input Arguments
- `ndim` (`int`): system dimensionality, default 3

### Return
- Nideal (`float`)

### Example
```python
from utils import funcs

funcs.nidealfac(ndim=3)
```

## 2. `moment_of_inertia()`
`moment_of_inertia()` calculates moment of inertia for a rigid body made of n points/particles.

### Input Arguments
- `positions` (`np.ndarray`): positions of the point particles as [numofatoms, 3]
- `m` (`int`): assuming each point mass is 1.0/numofatoms, default 1
- `matrix` (`bool`): to return the results as a matrix of [ixx iyy izz ixy ixz iyz], default `False`

### Return
- moment of inertia (`np.ndarray`)

### Example
```python
from utils import funcs
funcs.moment_of_inertia(positions, m=1, matrix=False)
```


# II. Remove PBC

The module `utils.pbc()` removes periodic boundary conditions (PBC).

## Input Arguments
- `RIJ` (`np.ndarray`): position difference between particle pairs $i$ and $j$
- `hmatrix` (`np.ndarray`): h-matrix of the box
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no. Default `np.array([1, 1, 1])`, that is, PBC is applied in all three dimensions for 3D box

## Return
- A `np.ndarray` for the position difference between particle pairs $i$ and $j$ after removing PBC

## Example
```python
from utils.pbc import remove_pbc

remove_pbc(RIJ, hmatrix, ppp=np.array([1, 1, 1]))
```

# III. geometry
`utils.geometry()` includes math geometrical functions to assist other analysis

## 1. `triangle_area()`

`triangle_area()` function calculates the area of a triangle using Heron's equation

### Input Arguments
- `positions` (`np.ndarray`): numpy array of particle positions, `shape=(3, 2)` for 2D, `shape=(3, 3)` for 3D
- `hmatrix` (`np.ndarray`): h-matrix of the box
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting 1 for yes and 0 for no. default `np.array([1, 1])`, that is, PBC is applied in all two dimensions

### Return
- area of triangle (float)

### Example
```python
from utils import geometry

geometry.triangle_area(positions, hmatrix, ppp)
```

## 2. `triangle_angle()`

`triangle_angle()` function calculates the angle of a triangle based on side lengths

### Input Arguments
- `a`, `b`, `c` (float): side length

### Return
- corresponding angles `C` (`float`)

### Example
```python
from utils import geometry

geometry.triangle_angle(a=3, b=4, c=5)
```

## 3. `lines_intersection()`

`lines_intersection()` function extracts the line-line intersection for two lines `[P1, P2]` and `[P3, P4]` in two dimensions

### Input Arguments
- `P1` (`np.ndarray`): one point on line 1
- `P2` (`np.ndarray`): another point on line 1
- `P3` (`np.ndarray`): one point on line 2
- `P4` (`np.ndarray`): another point on line 2
- `P3` (`np.ndarray`): third point within a square
- `P4` (`np.ndarray`): fourth point within a square
- `R0` (`np.ndarray`): point within the sqaure
- `vector` (`np.ndarray`): pointing to R0 from R1 outside the square

### Return
- intersection of two lines (`np.ndarray`)

### Example
```python
from utils import geometry

geometry.lines_intersection(P1=np.array([0, 0]),
                            P2=np.array([1, 1]),
                            P3=np.array([1, 0]),
                            P4=np.array([0, 1]))
```

# IV. wavevector
`utils.wavevector` module generates wave-vector for calculations like static/dynamic structure factor.

## 1. `wavevector3d()`
`wavevector3d()` is used to define wave vectors for three dimensional systems.

### Input Arguments
- `numofq` (`int`): number of q

### Return
- wavevector (`np.ndarray`)

### Example
```python
from utils import wavevector
wavevector.wavevector3d(numofq=100)
```

## 2. `wavevector2d()`
`wavevector2d()` is used to define wave vectors for two dimensional systems.

### Input Arguments
- `numofq` (`int`): number of q

### Return
- wavevector (`np.ndarray`)

### Example
```python
from utils import wavevector
wavevector.wavevector2d(numofq=100)
```

## 3. `choosewavevector()`
`choosewavevector()` is used to define wave vector for 
$$
[n_x, n_y, n_z]
$$ 
as long as they are integers. Considering wave vector values from $[-N/2, N/2]$ or from $[0, N/2]$ (`onlypositive=True`). Only get the sqrt-able wave vector.

### Input Arguments
- `ndim` (`int`): dimensionality
- `numofq` (`int`): number of wave vectors
- `onlypositive` (`bool`): whether only consider positive wave vectors, , default `False`

### Example
```python
from utils import wavevector
wavevector.choosewavevector(ndim=3, numofq=100, onlypositive=False)
```

### Return
- qvectors (`np.ndarray`)

## 4. `continuousvector()`
`continuousvector()` is used to define wave vector for 
$$
[n_x, n_y, n_z]
$$ 
as long as they are integers. Considering wave vector values from $[-N/2, N/2]$ or from $[0, N/2]$ (`onlypositive=True`).

### Input Arguments
- `ndim` (`int`): dimensionality
- `numofq` (`int`): number of wave vectors, default 100
- `onlypositive` (`bool`): whether only consider positive wave vectors, default `False`

### Return
- qvectors (`np.ndarray`)

### Example
```python
from utils import wavevector
wavevector.continuousvector(ndim=3, numofq=100, onlypositive=False)
```

# V. Fast Fourier Transformation
`utils.fft` calculates the Fourier transformation of an autocorrelation function by Filon's integration method

## Input Arguments
- `C` (`np.ndarray`): the auto-correlation function
- `t` (`np.ndarray`): the time corresponding to C
- `a` (`float`): the frequency interval, default 0
- `outputfile` (`str`): filename to save the calculated results, default `None`
  
## Return
- FFT results (`pd.DataFrame`)

## Example
```python
from utils import fft
fft.Filon_COS(C, t, a, outputfile)
```

# VI. Spherical Harmonics
`utils.spherical_harmonics` calculates spherical harmonics of given ($\theta$, $\phi$) from ($l$ = 0) to ($l$ = 10). From `SphHarm0()` to `SphHarm10()` a list of [$-l$, $l$] values will be returned, if $l$>10 use scipy.special.sph_harm (this may be slower).

## Input Arguments
- For degree of harmonics $l$=0, the input is `None`.

- For $1<l\leq10$, the input is:
  - `theta` (`float`): Azimuthal (longitudinal) coordinate; must be in $[0, 2\pi]$
  - `phi` (`float`): Polar (colatitudinal) coordinate; must be in $[0, \pi]$

- For $l>10$, beside `theta` and `phi`, `l` should also be taken as input.

## Return
- spherical harmonics (`np.ndarray`)

## Example
```python
from utils import spherical_harmonics

spherical_harmonics.SphHarm4(theta=60*np.pi/180, phi=30*np.pi/180)

spherical_harmonics.SphHarm6(theta=60*np.pi/180, phi=30*np.pi/180)

spherical_harmonics.SphHarm_above(l=12, theta=60*np.pi/180, phi=30*np.pi/180)
```

# VII. Coarse Graining
`utils.coarse_graining` calculates the time average, spatial average, gaussian blurring, and atomic position average of input property.

## 1. `time_average()`
Calculate time average of the input property over a certain time window

### Input Arguments
- `snapshots` (`reader.reader_utils.Snapshots`): snapshot object of input trajectory
(returned by `reader.dump_reader.DumpReader`)
- `input_property` (`np.ndarray`): the input particle-level property, in `np.ndarray` with shape `[nsnapshots, nparticle]`
- `time_period` (`float`): time used to average, default 0.0
- `dt` (`float`): timestep used in user simulations, default 0.002

### Return
- `results` (`np.ndarray`): Calculated time averaged input results with shape `[nsnapshots_updated, nparticles]`
- `results_middle_snapshots` (`np.ndarray`): Corresponding snapshot id of the middle snapshot of each time period with shape `[nsnapshots_updated]`

### Example
```python
from utils.coarse_graining import time_average

time_average(snapshots, input_property, time_period, dt)
```

## 2. `spatial_average()`
Calculate spatial average of the input property over a certain distance, which is demonstrated by the neighbor definition.

### Input Arguments
- `input_property` (`np.ndarray`): input property to be coarse-grained, should be in the shape [num_of_snapshots, num_of_particles, xxx]. The input property can be scalar or vector or tensor
- `neighborfile` (`str`): file name of pre-defined neighbor list
- `Namx` (`int`): maximum number of particle neighbors
- `outputfile` (`str`): file name of coarse-grained variable


### Return
- `cg_input_property` (`np.ndarray`): coarse-grained input property in numpy ndarray


### Example
```python
import numpy as np
from reader.dump_reader import DumpReader
from utils.coarse_graining import spatial_average 

test_file = "test.atom"
readdump = DumpReader(test_file, ndim=2)
readdump.read_onefile()
input_property = np.random.rand([readdump.snapshots.nsnapshots, readdump.snapshots.snapshots.nparticle])
Nnearests(readdump.snapshots, N=6, ppp=np.array([1,1]), fnfile='neighborlist.dat')
neighborfile='neighborlist.dat'
sa = spatial_average(input_property,neighborfile)
```


## 3. `gaussian_blurring()`
Project input properties into a grid made from the simulation box. Basically, the calculation is to define a grid based on the simulation box, and then project the particle-level property to the grids based on the gaussian distribution function:
$$
p(\vec{r_j}) = \sum_i \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left(-\frac{(\vec{r_j}-\vec{r}_i)^2}{2\sigma^2}\right) p_i,
$$
in which $\vec{r_j}$ is the position of the $j_{th}$ grid, and $\vec{r_i}$ is the position of the $i_{th}$ particle. $p_i$ is the particle-level property of the $i_{th}$ particle, which can be a scalar, vector or tensor.

### Input Arguments
- `snapshots` (`read.reader_utils.snapshots)`: multiple trajectories dumped linearly or in logscale
- `condition` (`np.ndarray`): particle-level condition / property, type should be float, shape: [num_of_snapshots, num_of_particles, xxx], The input property can be scalar or vector or tensor, based on the shape of condition, mapping as {"scalar": 3, "vector": 4, "tensor": 5}
- `ngrids` (`np.ndarray` of `int`): predefined grid number in the simulation box, shape as the dimension, for example, [25, 25] for 2D systems
- `sigma` (`float`): standard deviation of the gaussian distribution function, default 2.0
- `ppp` (`np.ndarray`): the periodic boundary conditions (PBCs), setting 1 for yes and 0 for no, default `np.array([1,1,1]`)
- `gaussian_cut` (`float`): the longest distance to consider the gaussian probability or the contribution from the simulation particles. default 6.0.
- `outputfile` (`str`): file name to save the grid positions and the corresponding properties

### Return
- `grid_positions` (`np.ndarray`): Positions of the grids of each snapshot
- `grid_property` (`np.ndarray`): properties of each grid of each snapshot

### Example
```python
import numpy as np
from reader.dump_reader import DumpReader
from utils.coarse_graining import gaussian_blurring

test_file = "test.atom"
readdump = DumpReader(test_file, ndim=2)
readdump.read_onefile()
ppp = np.array([1,1])
ngrids = np.array([20,20])
input_property = np.random.rand([
  readdump.snapshots.nsnapshots,
  readdump.snapshots.snapshots.nparticle
])

gb_position, gb_property = gaussian_blurring(readdump.snapshots,input_property,ngrids,2.0,ppp)
```