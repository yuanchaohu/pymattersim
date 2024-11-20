### Hessian matrix calculation and diagonalization


---

##### I. Pair interactions

This module calculates the hessian matrix of a simulation configuration with a pair potential. Currently, there are three types of pair potentials at 2D and 3D are supported, i.e. lennard-jones, inverse-power law, and harmonic or hertz potentials. They are defined as

1. Lennard-Jones interaction:
$$
s(r) = 4 \epsilon \left[(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^{6} \right]
$$

$$
s'=\frac{ds}{dr}=\frac{-24\epsilon}{r} \left[2(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^{6} \right]
\qquad\qquad
s'(r_c) = \frac{-24\epsilon}{r_c} \left[2(\frac{\sigma}{r_c})^{12} - (\frac{\sigma}{r_c})^{6} \right]
$$

$$
s''=\frac{d^2 s}{d r^2} = \frac{ds'}{dr} = \frac{24\epsilon}{r^2} \left[26(\frac{\sigma}{r})^{12} - 7(\frac{\sigma}{r})^6 \right] \qquad\qquad\qquad\qquad\qquad\qquad\qquad
$$


2. Inverse power law interaction:
$$
s(r) = A \epsilon (\frac{\sigma}{r})^n
$$

$$
s'=\frac{ds}{dr} = -\frac{A \epsilon n}{r} (\frac{\sigma}{r})^n
\qquad\qquad s'(r_c) = -\frac{A \epsilon n}{r_c} (\frac{\sigma}{r_c})^n
$$

$$
s''=\frac{d^2s}{dr^2}=\frac{ds'}{dr} = \frac{A \epsilon n (n+1)}{r^2} (\frac{\sigma}{r})^n \qquad\qquad\qquad\qquad
$$


3. Harmonic & Hertz interactions:
$$
s(r) = \frac{\epsilon}{\alpha} \left(1 - \frac{r}{\sigma} \right)^\alpha
$$

$$
s' = \frac{ds}{dr} = -\frac{\epsilon}{\sigma} \left(1 - \frac{r}{\sigma} \right)^{\alpha -1} 
\qquad\qquad
s'(r_c) = 0
$$

$$
s'' = \frac{d^2s}{dr^2} = \frac{ds'}{dr} = \frac{\epsilon}{\sigma^2} (\alpha-1) \left(1 - \frac{r}{\sigma} \right)^{\alpha-2} \qquad
$$

After measuring the Hessian matrix, it can be diagonalized by numpy method and then provides eigenvalues and eigenvectors. For each eigenvector, the participation ratio (see the `vector` module) can be evaluated accompanied by its corresponding frequency.

#### 1. `PairInteractions` class
This class aims to calculate the pair potential and force between a pair of particles. The return will be used for hessian matrix calculation. For each of the above pair potential, a class method is defined that can be called independently and a `caller` function is used to easily access each potential.

**Input Arguments**
- `r` (`float`): pair distance
- `epsilon` (`float`): cohesive enerrgy between the pair
- `sigma` (`float`): diameter or reference length between the pair
- `r_c` (`float`): cutoff distance where the potential energy is cut to 0
- `shift` (`bool`): whether shift the potential energy at `r_c` to 0, default True

**Return**
- None

**Example** 
```python
from hessians import PairInteractions

pair_interaction = PairInteractions(r, epsilon, sigma, r_c, shift)
```

##### 1.1 `caller()`

Calculate pair potential and force based on model inputs

##### Input
- `interaction_params` (`InteractionParams`): define the name and parameters for the pair potential

**Return**
- [`s1`, `s1rc`, `s2`] (list of `float`)

**Example** 
```python
from hessians import InteractionParams, ModelName

interaction_params = InteractionParams(
    model_name=ModelName.inverse_power_law,
    ipl_n=10,
    ipl_A=1.0
)

pair_interaction.caller(interaction_params)
```

##### 1.2 `lennard_jones()`
Lennard-Jones interaction

##### Input
- `None`

**Return** 
- [`s1`, `s1rc`, `s2`] (list of `float`)

**Example**
```python
pair_interaction.lennard_jons()
```

##### 1.3 `inverse_power_law()`
Inverse power-law potential

##### Inputs
- `n` (`float`): power law exponent for a pair
- `A` (`float`): power law prefactor for a pair, default 1.0

**Return** 
- [`s1`, `s1rc`, `s2`] (list of `float`)

**Example**
```python
pair_interaction.inverse_power_law(n=10, A=1.0)
```

##### 1.4 `harmonic_hertz()`
harmonic or hertz potential

##### Inputs
- `alpha` (`float`): spring exponent

**Return** 
- [`s1`, `s1rc`, `s2`] (list of `float`)

**Example**
```python
pair_interaction.harmonic_hertz(alpha=2.0)
```

#### II. Hessian matrix
##### 2. `HessianMatrix` class
This class calculates the hessian matrix of the simulation configuration and diagonalize it with numpy. The eigenvalues and eigenvectors are given for further analysis, such as the frequency and participation ratio of each eigenvector.

**Input Arguments**
- `snapshot` (`reader.reader_utils.SingleSnapshot`): single snapshot object of input trajectory
- `masses` (`dict` of `int` to `float`): mass for each particle type, example `{1:1.0, 2:2.0}`
- `epsilons` (`npt.NDArray`): cohesive energies for all pairs of particle type, shape as [num_of_particletype, num_of_particletype]
- `sigmas` (`npt.NDArray`): diameters for all pairs of particle type, shape as [num_of_particletype, num_of_particletype]
- `r_cuts` (`npt.NDArray`): cutoff distances of pair potentials for all pairs of particle type, shape as [num_of_particletype, num_of_particletype]
- `ppp` (`npt.NDArray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.array([1,1,1])`, set `np.array([1,1])` for two-dimensional systems
- `shiftpotential` (`bool`): whether shift the potential energy at `r_c` to 0, default `True`

**Return**
- `None`

**Example**
```python
import numpy as np 
from PyMatterSim.static.hessians import  HessianMatrix, InteractionParams, ModelName
from PyMatterSim.reader.dump_reader import DumpReader

### inverse-power law potential model in 2D
dumpfile = "test.atom"
masses = {1:1.0, 2:1.0}
epsilons = np.ones((2, 2))

sigmas = np.zeros((2, 2))
sigmas[0, 0] = 1.00
sigmas[0, 1] = 1.18
sigmas[1, 0] = 1.18
sigmas[1, 1] = 1.40

r_cuts = np.zeros((2, 2))
r_cuts[0, 0] = 1.48
r_cuts[0, 1] = 1.7464
r_cuts[1, 0] = 1.7464
r_cuts[1, 1] = 2.072

interaction_params = InteractionParams(
    model_name=ModelName.inverse_power_law,
    ipl_n=10,
    ipl_A=1.0
)

readdump = DumpReader(dumpfile, ndim=2)
readdump.read_onefile()

h = HessianMatrix(
    snapshot=readdump.snapshots.snapshots[0],
    masses=masses,
    epsilons=epsilons,
    sigmas=sigmas,
    r_cuts=r_cuts,
    ppp=np.array([1,1]),
    shiftpotential=True
)
```

##### 2.1 `pair_matrix()`
Hessian matrix block for a pair

##### Inputs
- `Rji` (`npt.NDArray`): $\vec R_i - \vec R_j$, positional vector for i-j pair, shape as [`ndim`]
- `dudrs` (list of `float`): a list of [`s'(r)`, `s'(r_c)`, `s''(r)`], named as [`s1`, `s1rc`, `s2`], returned by the PairInteractions class

**Return**
- `dudr2i` (`npt.NDArray`): hessian matrix block of pair i-j centered on i
- `dudr2j` (`npt.NDArray`): hessian matrix block of pair i-j centered on j
Both have the shape [`ndim`, `ndim`]

**Example**
```python
h.pair_matrix([1.0, 3.0, 2.0], [0.1, 0, 10])
```


##### 2.2 `diagonalize_hessian()`
This function is used to calculate the Hessian matrix of a simulation configuration and use numpy package to diagonalize the hessian matrix to get the eigenvalues and eigenvectors. The participation ratio of each eigenvector will be calculated as well. The limitation here is the particle number or system size to be considered as numpy cannot diagonalize very large matrix.

**Input Arguments**       
- `interaction_params` (`InteractionParams`): pair interaction parameters
- `saveevecs` (`bool`): save the eigenvectors or not, default True
- `savehessian` (`bool`): save the hessian matrix in dN*dN or not, default False
- `outputfile` (`str`): filename to save the computational results, defult None to use model name

**Return**
- None

**Example**
```python
h.diagonalize_hessian(
    interaction_params=interaction_params,
    saveevecs=True,
    savehessian=True,
    outputfile="calfile"
)
```