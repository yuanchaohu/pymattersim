# Dynamical properties at two & three dimensions

This module calculates the dynamical properties at different dimensions. It contains two different class for calculation: linear dynamics(`Dynamics`) and log dynamics(`LogDynamics`).

True trajectories by lammps xu format are prefered for calculating dynamics, without considering the periodic boundary conditions. This is done by provide the viariable 'xudump'. Three types of inputs are available:

1. provide 'xu' type format to xudump and 'x/xs' type format to xdump; the true positions from xudump will be used to calculate the dynamics while the wrapped positions from xdump will be used only for calculating Sq4.
2. only provide 'xu' type format to xudump and set None to xdump; the true positions from xudump will be used to calculate all the quantities
3. only provide 'x' type format to xdump and set None to xudump; the wrapped positions from xdump will be used to calculate all the quantities, by considering periodic boundary conditions (setting ppp argument). This is not recommended and should be avoided for calculating mean-square displacement related quantities

## `cage_relative()`
get the cage-relative or coarse-grained motion for single configuration
The coarse-graining is done based on given neighboring particles

$$ \Delta {\vec r_j} = \left[{\vec r_j} (t) - {\vec r_i}(0) \right] - \frac{1}{N_j}\sum_{i}^{N_j} \left[ {\vec r_i}(t) - {\vec r_i}(0) \right] $$
#### Input Arguments:
- `RII`(`np.ndarray`): original (absolute) displacement matrix with shape `[num_of_particles, ndim]`                  
- `cnlist`(`np.ndarray`): neighbor list of the initial or reference configuration with shape `[num_of_particles, num_of_neighbors]`. It available from the 'neighbors' module

#### Return:
- `RII_relative`(`np.ndarray`): cage-relative displacement matrix with shape `[num_of_particles, ndim]`

#### Example
```python

```

## 1. `Dynamics` class




#### Input Arguments:

- `xu_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `x_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `dt`(`float`): timestep used in user simulations, default 0.002
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([0,0,0])` for 3D
- `diameters` (`dict[int, float]`): map particle types to particle diameters
- `a`(`float`): slow mobility cutoff scaling factor
- `cal_type`(`str`): calculation type, can be either `slow`(default) or `fast`
- `neighborfile`(`str`): neighbor list filename for coarse-graining, only provided when calculating (cage-)relative displacements
- `max_neighbors`(`int`): maximum of particle neighbors considered, default 30

#### Return:
- None


#### Example:

```python                   
```


### 1.1 `Dynamics.relaxation()`
Compute self-intermediate scattering functions ISF,

$$
F_s(q,t) = \frac{1}{N} \bigg\langle \sum_{j=1}^{N}\exp\lbrack iq \cdot (r_j(t)-j(0))\rbrack\bigg\rangle \tag{1}
$$


Overlap function $Q(t)$:

$$
Q(t) = \frac{1}{N}\sum_{j = 1}^{N}\omega\left( \left| \mathbf{r}_j\left( t \right) - \mathbf{r}_j\left( 0 \right) \right| \right)\tag{2}
$$

and its corresponding dynamic susceptibility $Q(t)X_4$:

$$
\chi_{4}\left( t \right) = N^{- 1}\left\lbrack \left\langle {Q\left( t \right)}^{2} \right\rangle - \left\langle Q\left( t \right) \right\rangle^{2} \right\rbrack
\tag{3}
$$

Mean-square displacements msd,

$$
\langle \Delta {r^2}(t)\rangle = \frac{1}{N} \bigg \langle \sum_{j=1}^{N} \lbrack r_j(t)-r_j(0) \rbrack ^2 \bigg \rangle
\tag{4}
$$

Non-Gaussion parameter alpha2,

$$
\alpha_{2}\left( t \right) = \frac{3\left\langle \Delta r^{4}\left( t \right) \right\rangle}{5\left\langle \Delta r^{2}\left( t \right) \right\rangle^{2}} - 1\left( 3D \right);\alpha_{2}\left( t \right) = \frac{\left\langle \Delta r^{4}\left( t \right) \right\rangle}{2\left\langle \Delta r^{2}\left( t \right) \right\rangle^{2}} - 1\left( 2D \right)
\tag{5}
$$

#### Input Arguments:
- `qconst`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `condition`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `outputfile`(`str`): file name to save the calculated dynamics results

#### Return:
- `results`(`pd.DataFrame`): Calculated dynamics results


#### Example
```python

```
### 1.2 `Dynamics.sq4()`
Compute four-point dynamic structure factor of specific atoms at characteristic timescale


$
S_4\left( q,t \right) = N^{- 1}\left\langle W\left( \mathbf{q},t \right)W\left( - \mathbf{q},t \right) \right\rangle $
in which 
$
  W\left( \mathbf{q},t \right) = \sum_{j = 1}^{N}{\exp\left\lbrack i\mathbf{q} \cdot \mathbf{r}_{j}\left( 0 \right) \right\rbrack\omega\left( \left| \mathbf{r}_j\left( t \right) - \mathbf{r}_j\left( 0 \right) \right| \right)} 
$

#### Input Arguments:
-   `t` (`float`): characteristic time, typically peak time of X4, see self.relaxation()
-   `qrange` (`float`): the wave number range to be calculated, default 10.0
-   `condition` (`np.ndarray`): particle-level condition / property, shape [nsnapshots, nparticles]
-   `outputfile` (`str`): output filename for the calculated dynamical structure factor

#### Return:
- `results`(`pd.DataFrame`): calculated dynamical structure factor

#### Example:
```python

```



## 2. `LogDynamics` class
This class calculates particle-level dynamics with orignal coordinates.

#### Input Arguments:

- `xu_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `x_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `dt`(`float`): timestep used in user simulations, default 0.002
- `ppp` (`np.ndarray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([0,0,0])` for 3D
- `diameters` (`dict[int, float]`): map particle types to particle diameters
- `a`(`float`): slow mobility cutoff scaling factor
- `cal_type`(`str`): calculation type, can be either `slow`(default) or `fast`
- `neighborfile`(`str`): neighbor list filename for coarse-graining, only provided when calculating (cage-)relative displacements
- `max_neighbors`(`int`): maximum of particle neighbors considered, default 30

#### Return:
- None


#### Example:

```python                   
```



### 2.1 `relaxation()`

#### Input Arguments:
- `qconst`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `condition`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `outputfile`(`str`): file name to save the calculated dynamics results

#### Return:
- `results`(`pd.DataFrame`): Calculated dynamics results

#### Example:
```python

```
