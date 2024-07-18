# Dynamical properties at two & three dimensions

This module calculates the dynamical properties at different dimensions. It contains two different calculation classes: 
+ linear dynamics (`Dynamics()`) [snapshots are dumped in a linear manner]
+ log dynamics (`LogDynamics()`) [snapshots are dumped in a log scale]

To calculate the atomic-scale dynamics, the absolute coordinates, for example, the 'xu,yu,zu' type in LAMMPS, are preferred, without dealing with the periodic boundary conditions. This is done by providing the trajectories through the variable `xudump`. Nevertheless, the other coordinates are acceptable. There cases are included:

1. Providing 'xu' type trajectory to `xudump` and 'x' or 'xs' type trajectory to `xdump`; the absolute coordinates from `xudump` will be used to calculate the dynamical quantities like displacement vector while the wrapped positions from `xdump` will be used only for calculating the dynamical structure factor `Sq4()`.
2. Providing only 'xu' type trajectory to `xudump` and set `None` to `xdump`; the absolute positions from `xudump` will be used to calculate all the quantities.
3. Providing only 'x' or 'xs' type trajectory to `xdump` and set `None` to `xudump`; the wrapped positions from `xdump` will be used to calculate all the quantities, by considering periodic boundary conditions (setting the correct `ppp` argument). This is not recommended and should be avoided for calculating mean-square displacements related quantities at long time scales.

## `cage_relative()`
Getting the cage-relative or coarse-grained displacement vectors of the center particle with respect to the neighbors in a certain range for single configuration.
The coarse-graining is done based on the given the neighboring particles, which are accessible by the `neighbors` module. Specifically,
$$
\Delta {\vec r_j} = \left[{\vec r_j} (t) - {\vec r_j}(0) \right] - \frac{1}{N_j}\sum_{i}^{N_j} \left[ {\vec r_i}(t) - {\vec r_i}(0) \right],
$$
in which $j$ is the center particle and $i$ is its neighbor.

#### Input Arguments:
- `RII`(`np.ndarray`): original (absolute) displacement matrix with shape `[num_of_particles, ndim]`                  
- `cnlist`(`np.ndarray`): neighbor list of the initial or reference configuration with shape `[num_of_particles, num_of_neighbors]`. This gives the particle identifiers ($i$) for each center ($j$). It available from the 'neighbors' module.

#### Return:
- `RII_relative`(`np.ndarray`): cage-relative displacement matrix with shape `[num_of_particles, ndim]` (the same as the input `RII`)

#### Example
```python

```

## 1. `Dynamics()` class

This module calculates particle-level dynamical quantities with absolute or wrapped coordinates. It considers both two-dimensional systems and three-dimensional system, and for both absolute and cage-relative dynamics.

This module recommends to use absolute coordinates (like 'xu' in LAMMPS) to calculate particle-level displacement vectors, such as mean-squared displacements, while PBC is taken care of others as well.
The functions are listed below.


#### Input Arguments:

- `xu_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `x_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `dt`(`float`): timestep used in user simulations, default 0.002
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([0,0,0])` for 3D. It requires to be precisely given to determine the model dimensionality.
- `diameters` (`dict[int, float]`): map particle types to particle diameters, for example, `{1: 1.0, 2:1.5}` for binary systems.
- `a`(`float`): mobility cutoff scaling factor, used together with `diameters` to determine the cutoff for each particle type, default 0.3.
- `cal_type`(`str`): calculation type for the dynamical structure factor, can be either `slow`(default) or `fast`, accounting for the slow and fast dynamics.
- `neighborfile`(`str`): neighbor list filename for coarse-graining, only provided when calculating (cage-)relative displacements.
- `max_neighbors`(`int`): maximum of particle neighbors considered, default 30.

#### Return:
- None


#### Example:

```python                   
```


### 1.1 `relaxation()`
Compute self-intermediate scattering functions ISF,

$$
F_s(q,t) = \frac{1}{N} \bigg\langle \sum_{j=1}^{N}\exp\lbrack iq \cdot ({\bf r}_j(t)-{\bf r}_j(0))\rbrack\bigg\rangle \tag{1}.
$$


Overlap function $Q(t)$:

$$
Q(t) = \frac{1}{N} \bigg\langle \sum_{j = 1}^{N}\omega\left( \left| \mathbf{r}_j\left( t \right) - \mathbf{r}_j\left( 0 \right) \right| \right) \bigg\rangle \tag{2},
$$
in which $\omega(x)=1$ if $x<cuoff$ for slow dynamics or $x>cutoff$ for fast dynamics, else $\omega(x)=0$. This is related to the input argument `cal_type`. Its corresponding dynamic susceptibility $X_4(t)$ is defined as:

$$
\chi_{4}\left( t \right) = N^{- 1}\left\lbrack \left\langle {Q\left( t \right)}^{2} \right\rangle - \left\langle Q\left( t \right) \right\rangle^{2} \right\rbrack
\tag{3},
$$
in which the quantity $Q(t)$ should be non-averaged value.

The mean-squared displacement is defined by

$$
\langle \Delta {r^2}(t)\rangle = \frac{1}{N} \bigg \langle \sum_{j=1}^{N} \lbrack {\bf r}_j(t)-{\bf r}_j(0) \rbrack ^2 \bigg \rangle
\tag{4},
$$
in which the prefered coordiantes are the absolute one, given by 'xudump'.

The non-Gaussion parameter $\alpha_2(t)$ is defined as

$$
\alpha_{2}\left( t \right) = \frac{3\left\langle \Delta r^{4}\left( t \right) \right\rangle}{5\left\langle \Delta r^{2}\left( t \right) \right\rangle^{2}} - 1\left( 3D \right); 
\qquad
\alpha_{2}\left( t \right) = \frac{\left\langle \Delta r^{4}\left( t \right) \right\rangle}{2\left\langle \Delta r^{2}\left( t \right) \right\rangle^{2}} - 1\left( 2D \right)
\tag{5},
$$
which is usually used to measure the degree of dynamical heterogeneity in the disordered states revealed by the particle-level mobility.

#### Input Arguments:
- `qconst`(`float`): the wavenumber factor for calculating self-intermediate scattering function. default $2\pi$, used internally as `qconst/diameters` to define the wavenumber for each particle. For example, if input `qconst=2PI` and `diamter=1.0` then the used wavenumber is `2PI/1.0`. This setting provides flexibility for polydisperse systems or considering dynamics differently for different particle types.
- `condition`(`np.ndarray`): particle-level selection with the shape `[num_of_snapshots, num_of_particles]` , preferring the `bool` type. Default `None`.
- `outputfile`(`str`): file name to save the calculated dynamics results

#### Return:
- `results`(`pd.DataFrame`): Calculated dynamics results

#### Example
```python

```

### 1.2 `sq4()`
Compute four-point dynamical structure factor of specific atoms at a characteristic timescale. The atoms can exhibit fast or slow dynamics, given by the input argument `cal_type`. It is defined in the same spirit as the static structure factor as below:
$$
S_4\left( q,t \right) = N^{- 1}\left\langle W\left( \mathbf{q},t \right)W\left( - \mathbf{q},t \right) \right\rangle, 
$$
in which 
$$
  W\left( \mathbf{q},t \right) = \sum_{j = 1}^{N}{\exp\left\lbrack i\mathbf{q} \cdot \mathbf{r}_{j}\left( 0 \right) \right\rbrack\omega\left( \left| \mathbf{r}_j\left( t \right) - \mathbf{r}_j\left( 0 \right) \right| \right)}, 
$$
which is essentially the same as the quantity $Q(t)$ as above, accounting for the slow or fast dynamics at the particle level. The calculation is done in the same way as above.

#### Input Arguments:
-   `t` (`float`): characteristic time, typically peak time of $\chi_4(t)$, see self.relaxation()
-   `qrange` (`float`): the wave number range to be calculated, default 10.0. It determines how many wavevector is involved in the computation.
-   `condition` (`np.ndarray`): particle-level condition / property, shape `[num_of_snapshots, num_of_particles]`. This supplements the fast or slow dynamics, for example, to calculate the slow dynamics of particles with type=1 will require this `condition` to be a `bool` type with `True` only for particle type of 1.
-   `outputfile` (`str`): output filename for the calculated dynamical structure factor

#### Return:
- `results`(`pd.DataFrame`): calculated dynamical structure factor

#### Example:
```python

```

## 2. `LogDynamics()` class
This module is designed to calculate the dynamical properties of a trajectory as above but dumped in a log scale. It uses the 'log' style to output the atomic trajectory. Therefore, no ensemble-average is performed. The first configuration is the only one used to consider the particle neighbors and as the reference for calculating the dynamics.

Ensemble average is absent compared to the above `Dynamics()` class!

#### Input Arguments:

- `xu_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `x_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `dt`(`float`): timestep used in user simulations, default 0.002
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([0,0,0])` for 3D. It requires to be precisely given to determine the model dimensionality.
- `diameters` (`dict[int, float]`): map particle types to particle diameters, for example, `{1: 1.0, 2:1.5}` for binary systems.
- `a`(`float`): mobility cutoff scaling factor, used together with `diameters` to determine the cutoff for each particle type, default 0.3.
- `cal_type`(`str`): calculation type for the dynamical structure factor, can be either `slow`(default) or `fast`, accounting for the slow and fast dynamics.
- `neighborfile`(`str`): neighbor list filename for coarse-graining, only provided when calculating (cage-)relative displacements.
- `max_neighbors`(`int`): maximum of particle neighbors considered, default 30.

#### Return:
- None

#### Example:

```python     

```


### 2.1 `relaxation()`
Compute the self-intermediate scattering functions ISF, overlap function Qt and set its corresponding dynamical susceptibility X4_Qt ($\chi_4(t)$) as 0, mean-square displacements msd, and non-Gaussion parameter $\alpha_2(t)$

#### Input Arguments:
- `qconst`(`float`): the wavenumber factor for calculating self-intermediate scattering function. default $2\pi$, used internally as `qconst/diameters` to define the wavenumber for each particle. For example, if input `qconst=2PI` and `diamter=1.0` then the used wavenumber is `2PI/1.0`. This setting provides flexibility for polydisperse systems or considering dynamics differently for different particle types.
- `condition`(`np.ndarray`): particle-level selection with the shape `[num_of_snapshots, num_of_particles]` , preferring the `bool` type. Default `None`.
- `outputfile`(`str`): file name to save the calculated dynamics results

#### Return:
- `results`(`pd.DataFrame`): Calculated dynamics results

#### Example:
```python

```
