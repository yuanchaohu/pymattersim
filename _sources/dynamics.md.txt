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
- `RII`(`npt.NDArray`): original (absolute) displacement matrix with shape `[num_of_particles, ndim]`                  
- `cnlist`(`npt.NDArray`): neighbor list of the initial or reference configuration with shape `[num_of_particles, num_of_neighbors]`. This gives the particle identifiers ($i$) for each center ($j$). It available from the 'neighbors' module.

#### Return:
- `RII_relative`(`npt.NDArray`): cage-relative displacement matrix with shape `[num_of_particles, ndim]` (the same as the input `RII`)

#### Example
```python
from dynamic.dynamics import cage_relative
from neighbors.read_neighbors import read_neighbors
from neighbors.calculate_neighbors import Nnearests
from reader.dump_reader import DumpReader
import numpy as np

#here is a example of how it works, it used within Dynamics and LogDynamics class
filename = 'dump.atom'
readdump = DumpReader(filename, ndim=2)
readdump.read_onefile()
neighborfile='neighborlist.dat'
ppp=np.array([0,0])
snapshots = readdump.snapshots
Nnearests(readdump.snapshots, N=6, ppp=ppp,  fnfile=neighborfile)

neighborlists = []
if neighborfile:
    fneighbor = open(neighborfile, "r", encoding="utf-8")
    for n in range(snapshots.nsnapshots):
        medium = read_neighbors(
            f=fneighbor,
            nparticle=snapshots.snapshots[n].nparticle,
            Nmax=30
        )
        neighborlists.append(medium)
    fneighbor.close()
    
pos_init = snapshots.snapshots[10-1].positions
pos_end = snapshots.snapshots[10].positions
RII = pos_end - pos_init
cage_relative(RII,neighborlists[10-1])
```

## 1. `Dynamics()` class

This module calculates particle-level dynamical quantities with absolute or wrapped coordinates. It considers both two-dimensional systems and three-dimensional system, and for both absolute and cage-relative dynamics.

This module recommends to use absolute coordinates (like 'xu' in LAMMPS) to calculate particle-level displacement vectors, such as mean-squared displacements, while PBC is taken care of others as well.
The functions are listed below.


#### Input Arguments:

- `xu_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `x_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `dt`(`float`): timestep used in user simulations, default 0.002
- `ppp` (`npt.NDArray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([0,0,0])` for 3D. It requires to be precisely given to determine the model dimensionality.
- `diameters` (`dict[int, float]`): map particle types to particle diameters, for example, `{1: 1.0, 2:1.5}` for binary systems.
- `a`(`float`): mobility cutoff scaling factor, used together with `diameters` to determine the cutoff for each particle type, default 0.3.
- `cal_type`(`str`): calculation type for the dynamical structure factor, can be either `slow`(default) or `fast`, accounting for the slow and fast dynamics.
- `neighborfile`(`str`): neighbor list filename for coarse-graining, only provided when calculating (cage-)relative displacements.
- `max_neighbors`(`int`): maximum of particle neighbors considered, default 30.

#### Return:
- None


#### Example:

```python
from dynamic.dynamics import Dynamics
from neighbors.calculate_neighbors import Nnearests
from reader.dump_reader import DumpReader

# example for 2d dump
# read the dump file
file_2d_x = "2ddump.s.atom"
input_x = DumpReader(file_2d_x, ndim=2)
input_x.read_onefile()
file_2d_xu = "2ddump.u.atom"
input_xu = DumpReader(file_2d_xu, ndim=2)
input_xu.read_onefile()

#calculate the neighbor, neighborlist can also be None
ppp=np.array([0,0])
Nnearests(input_x.snapshots, N = 6, ppp = ppp,  fnfile = 'neighborlist.dat')
neighborfile='neighborlist.dat' #or None

dynamic = Dynamics(xu_snapshots=input_xu.snapshots,
                   x_snapshots=input_x.snapshots,
                   dt=0.002, 
                   ppp=ppp,
                   diameters={1:1.0, 2:1.0},
                   a=0.3,
                   cal_type = "slow",
                   neighborfile=neighborfile,
                   max_neighbors=30)
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
- `condition`(`npt.NDArray`): particle-level selection with the shape `[num_of_snapshots, num_of_particles]` , preferring the `bool` type. Default `None`.
- `outputfile`(`str`): file name to save the calculated dynamics results

#### Return:
- `results`(`pd.DataFrame`): Calculated dynamics results

#### Example
```python
#calculate the particle-level condition, can alos be None
condition=[]
for snapshot in input_x.snapshots.snapshots:
    condition.append(snapshot.particle_type==1)
condition = np.array(condition)

result = dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="")
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
-   `condition` (`npt.NDArray`): particle-level condition / property, shape `[num_of_snapshots, num_of_particles]`. This supplements the fast or slow dynamics, for example, to calculate the slow dynamics of particles with type=1 will require this `condition` to be a `bool` type with `True` only for particle type of 1.
-   `outputfile` (`str`): output filename for the calculated dynamical structure factor

#### Return:
- `results`(`pd.DataFrame`): calculated dynamical structure factor

#### Example:
```python
dynamic.sq4(t=10,qrange=10,condition=condition, outputfile="")
```

## 2. `LogDynamics()` class
This module is designed to calculate the dynamical properties of a trajectory as above but dumped in a log scale. It uses the 'log' style to output the atomic trajectory. Therefore, no ensemble-average is performed. The first configuration is the only one used to consider the particle neighbors and as the reference for calculating the dynamics.

Ensemble average is absent compared to the above `Dynamics()` class!

#### Input Arguments:

- `xu_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `x_snapshots`(`reader.reader_utils.Snapshots`): snapshot object of input trajectory (returned by reader.dump_reader.DumpReader)
- `dt`(`float`): timestep used in user simulations, default 0.002
- `ppp` (`npt.NDArray`): the periodic boundary conditions, setting `1` for yes and `0` for no, default `np.array([0,0,0])` for 3D. It requires to be precisely given to determine the model dimensionality.
- `diameters` (`dict[int, float]`): map particle types to particle diameters, for example, `{1: 1.0, 2:1.5}` for binary systems.
- `a`(`float`): mobility cutoff scaling factor, used together with `diameters` to determine the cutoff for each particle type, default 0.3.
- `cal_type`(`str`): calculation type for the dynamical structure factor, can be either `slow`(default) or `fast`, accounting for the slow and fast dynamics.
- `neighborfile`(`str`): neighbor list filename for coarse-graining, only provided when calculating (cage-)relative displacements.
- `max_neighbors`(`int`): maximum of particle neighbors considered, default 30.

#### Return:
- None

#### Example:

```python     
from dynamic.dynamics import LogDynamics
from neighbors.calculate_neighbors import Nnearests
from reader.dump_reader import DumpReader

# example for 2d dump
# read the dump file
file_2d_x = "2ddump.log.s.atom"
input_x = DumpReader(file_2d_x, ndim=2)
input_x.read_onefile()
file_2d_xu = "2ddump.log.u.atom"
input_xu = DumpReader(file_2d_xu, ndim=2)
input_xu.read_onefile()

#calculate the neighbor, neighborlist can also be None
ppp=np.array([0,0])
Nnearests(input_x.snapshots, N = 6, ppp = ppp,  fnfile = 'neighborlist.dat')
neighborfile='neighborlist.dat'

log_dynamic = LogDynamics(xu_snapshots=input_xu.snapshots,
                   x_snapshots=input_x.snapshots,
                   dt=0.002, 
                   ppp=ppp,
                   diameters={1:1.0, 2:1.0},
                   a=0.3,
                   cal_type = "slow",
                   neighborfile=neighborfile,
                   max_neighbors=30)   
```


### 2.1 `relaxation()`
Compute the self-intermediate scattering functions ISF, overlap function Qt and set its corresponding dynamical susceptibility X4_Qt ($\chi_4(t)$) as 0, mean-square displacements msd, and non-Gaussion parameter $\alpha_2(t)$

#### Input Arguments:
- `qconst`(`float`): the wavenumber factor for calculating self-intermediate scattering function. default $2\pi$, used internally as `qconst/diameters` to define the wavenumber for each particle. For example, if input `qconst=2PI` and `diamter=1.0` then the used wavenumber is `2PI/1.0`. This setting provides flexibility for polydisperse systems or considering dynamics differently for different particle types.
- `condition`(`npt.NDArray`): particle-level selection with the shape `[num_of_snapshots, num_of_particles]` , preferring the `bool` type. Default `None`.
- `outputfile`(`str`): file name to save the calculated dynamics results

#### Return:
- `results`(`pd.DataFrame`): Calculated dynamics results

#### Example:
```python
#calculate the particle-level condition, can alos be None
condition=[]
condition=(input_xu.snapshots.snapshots[0].particle_type==1)

result = dynamic.relaxation(qconst=2*np.pi, condition=condition, outputfile="")
```


## 3. `time_correlation()`

Calculate the time correlation of the input property given by condition

There are three cases considered, given by the shape of condition:
1. condition is float scalar type, for example, density
2. condition is float vector type, for example, velocity
3. condition is float tensor type, for example, nematic order

#### Input:
- `snapshots` (`read.reader_utils.snapshots`): multiple trajectories dumped linearly or in logscale
- `condition` (`npt.NDArray`): particle-level condition / property, type should be float
                            shape: `[num_of_snapshots, num_of_particles, xxx]`
- `dt` (`float`): time step of input snapshots, default 0.002
- `outputfile` (`str`): output file name, default "" (None)
    
#### Return:
- calculated time-correlation information in pandas dataframe

#### Example:
```python
#example for random scalar condition
from dynamic.time_corr import time_correlation
readdump = DumpReader(test_file, ndim=2)
readdump.read_onefile()
condition = np.random.rand(readdump.snapshots.nsnapshots, readdump.snapshots.snapshots[0].positions)
tc = time_correlation(readdump.snapshots,condition)
```