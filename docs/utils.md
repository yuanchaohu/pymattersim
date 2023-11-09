# util functions
## [I. math funcs](#i-funcs)
## [II. remove periodic boundary conditions](#ii-remove-pbc)
## [III. geometry](#iii-geometry-1)
## [IV. wavevector](#iv-wavevector-1)
## [V. Fast Fourier Transformation](#v-fast-fourier-transformation-1)
## [VI. spherical harmonics](#vi-spherical-harmonics-1)

---

# I. funcs
mathmatical functions for feasible computations

## 1. nidealfac
`nidealfac` is used to choose pre-factor of `Nideal` in g(r) calculation.

### Input Arguments
- `ndim` (`int`): system dimensionality, default 3

### Return
- Nideal (`float`)

### Example
```python
from utils import funcs
funcs.nidealfac(ndim=3)
```

## 2. moment_of_inertia
`moment_of_inertia` is used to calculate moment of inertia for a rigid body made of n points/particles.

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

The module `utils.pbc` is used to remove periodic boundary conditions (PBC) and is usually embedded in other analysis modules.

## Input Arguments
- `RIJ` (`np.ndarray`): position difference between particle pairs $i$ and $j$
- `hmatrix` (`np.ndarray`): h-matrix of the box
- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no. Default `[1, 1, 1]`, that is, PBC is applied in all three dimensions for 3D box

## Return
- A `np.ndarray` for the position difference between particle pairs $i$ and $j$ after removing PBC

## Example

```python
from utils.pbc import remove_pbc

remove_pbc(RIJ, hmatrix, ppp=[1,1,1])
```

# III. geometry
`utils.geometry` includes math geometrical functions to assist other analysis

## 1. triangle_area

`triangle_area` function claculates the area of a triangle using Heron's equation

### Input Arguments
- `positions` (`np.ndarray`): numpy array of particle positions, `shape=(3, 2)` for 2D, `shape=(3, 3)` for 3D
- `hmatrix` (`np.ndarray`): h-matrix of the box
- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no. default [1, 1], that is, PBC is applied in all two dimensions

### Return
- area of triangle (float)

### Example

```python
from utils import geometry
geometry.triangle_area(positions, hmatrix, ppp)
```

## 2. triangle_angle

`triangle_angle` function calculates the angle of a triangle based on side lengths

### Input Arguments
- `a`, `b`, `c` (float): side length

### Return
- corresponding angles `A`, `B`, `C` (`np.ndarray`)

### Example

```python
from utils import geometry
geometry.triangle_angle(a=3, b=4, c=5)
```

## 3. lines_intersection

`lines_intersection` function extracts the line-line intersection for two lines `[P1, P2]` and `[P3, P4]` in two dimensions

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
- line segment (`np.ndarray`)

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

## 1. wavevector3d
`wavevector3d` is used to define wave vectors for three dimensional systems.

### Input Arguments
- `numofq` (`int`): number of q

### Return
- wavevector (`np.ndarray`)

### Example
```python
from utils import wavevector
wavevector.wavevector3d(numofq=100)
```

## 2. wavevector2d
`wavevector2d` is used to define wave vectors for two dimensional systems.

### Input Arguments
- `numofq` (`int`): number of q

### Return
- wavevector (`np.ndarray`)

### Example
```python
from utils import wavevector
wavevector.wavevector2d(numofq=100)
```

## 3. choosewavevector
`choosewavevector` is used to define wave vector for 
$$
[n_x, n_y, n_z]
$$ 
as long as they are integers. Considering wave vector values from $[-N/2, N/2]$ or from $[0, N/2]$ (`onlypositive=True`). Only get the sqrt-able wave vector.

### Input Arguments
- `ndim` (`int`): dimensionality
- `numofq` (`int`): number of wave vectors
- `onlypositive` (`bool`): whether only consider positive wave vectors

### Example
```python
from utils import wavevector
wavevector.choosewavevector(ndim=3, numofq=100, onlypositive=False)
```

### Return
- qvectors (`np.ndarray`)

## 4. continuousvector
`continuousvector` is used to define wave vector for 
$$
[n_x, n_y, n_z]
$$ 
as long as they are integers. Considering wave vector values from $[-N/2, N/2]$ or from $[0, N/2]$ (`onlypositive=True`).

### Input Arguments
- `ndim` (`int`): dimensionality
- `numofq` (`int`): number of wave vectors
- `onlypositive` (`bool`): whether only consider positive wave vectors

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