
*general derivations*

**Potential energy (shifted):**
$$
U(r) = s(r) - s(r_c) - (r-r_c) \cdot \frac{ds}{dr}|_{r=r_c}
= s(r) - s(r_c) - (r-r_c) \cdot s'(r_c)
$$

$$
U'(r) = \frac{dU}{dr} = s'(r) - s'(r_c)
$$

**Three parameters are important as input for hessian:** $s'(r)$, $s'(r_c)$, $s''(r)$

*first derivatives (force part)*
$$
\frac{\partial s}{\partial x_i} = \frac{\partial s}{\partial r}\frac{\partial r}{\partial x_i} =s'(\frac{x}{r}) \qquad\qquad{\rm shifted:}\quad \frac{\partial U}{\partial x_i}= [s'-s'(r_c)] (\frac{x}{r})
$$

$$
\frac{\partial s}{\partial y_i} = \frac{\partial s}{\partial r} \frac{\partial r}{\partial y_i} =s'(\frac{y}{r}) \qquad\qquad{\rm shifted:}\quad \frac{\partial U}{\partial y_i}=[s'-s'(r_c)](\frac{y}{r})
$$

$$
\frac{\partial s}{\partial z_i} = \frac{\partial s}{\partial r} \frac{\partial r}{\partial z_i} =s'(\frac{z}{r}) \qquad\qquad{\rm shifted:}\quad \frac{\partial U}{\partial z_i}=[s'-s'(r_c)](\frac{z}{r})
$$

*second derivatives (hessian part) -- only shifted version*
$$
\frac{\partial ^2 U}{\partial x_i^2} = s'' (\frac{x}{r})^2 + [s'-s'(r_c)] \frac{r^2-x^2}{r^3}
$$

$$
\frac{\partial ^2 U}{\partial y_i^2} = s'' (\frac{y}{r})^2 + [s'-s'(r_c)] \frac{r^2-y^2}{r^3}
$$

$$
\frac{\partial ^2 U}{\partial z_i^2} = s'' (\frac{z}{r})^2 + [s'-s'(r_c)] \frac{r^2-z^2}{r^3}
$$

$$
\frac{\partial^2 U}{\partial x_i \partial y_i}= s''(\frac{xy}{r^2}) + [s'-s'(r_c)](\frac{-xy}{r^3})
$$

$$
\frac{\partial^2 U}{\partial x_i \partial z_i}= s''(\frac{xz}{r^2}) + [s'-s'(r_c)](\frac{-xz}{r^3})
$$

$$
\frac{\partial^2 U}{\partial y_i \partial z_i}= s''(\frac{yz}{r^2}) + [s'-s'(r_c)](\frac{-yz}{r^3})
$$

$$
\frac{\partial^2 U}{\partial x_i \partial x_j} = -\frac{\partial^2 U}{\partial x_i^2}
\qquad\quad
\frac{\partial^2 U}{\partial x_i \partial y_j} = -\frac{\partial^2 U}{\partial x_i \partial y_i}
\qquad\quad
\frac{\partial^2 U}{\partial x_i \partial z_j} = -\frac{\partial^2 U}{\partial x_i \partial z_i}
$$

$$
\frac{\partial^2 U}{\partial y_i \partial x_j} = -\frac{\partial^2 U}{\partial x_i \partial y_i}
\qquad\quad
\frac{\partial^2 U}{\partial y_i \partial y_j} = -\frac{\partial^2 U}{\partial y_i^2}
\qquad\quad
\frac{\partial^2 U}{\partial y_i \partial z_j} = -\frac{\partial^2 U}{\partial y_i \partial z_i}
$$

$$
\frac{\partial^2 U}{\partial z_i \partial x_j} = -\frac{\partial^2 U}{\partial x_i \partial z_i}
\qquad\quad
\frac{\partial^2 U}{\partial z_i \partial y_j} = -\frac{\partial^2 U}{\partial y_i \partial z_i}
\qquad\quad
\frac{\partial^2 U}{\partial z_i \partial z_j} = -\frac{\partial^2 U}{\partial z_i^2}
$$



*an example of hessian matrix*
$$
pair\ i-i \qquad\quad\quad\quad\quad pair\ i-j
\\
\left[
\matrix{
x_i^2     & x_iy_i    & x_iz_i &|& x_ix_j    & x_iy_j    & x_iz_j   \\
y_ix_i    & y_i^2     & y_iz_i &|& y_ix_j    & y_iy_j    & y_iz_j   \\
z_ix_i    & z_iy_i    & z_i^2  &|& z_ix_j    & z_iy_j    & z_iz_j \\
}
\right]
$$


---

**Lennard-Jones interaction**
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



---

**Inverse power law (IPL) interaction**
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

---

**Harmonic & Hertz interactions**
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


# 2. HessianMatrix class

### Input Arguments
- `snapshot` (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
- masses (dict of int to float): mass for each particle type, example {1:1.0, 2:2.0}
- epsilons (npt.NDArray): cohesive energies for all pairs of particle type, shape as [num_of_particletype, num_of_particletype]
- `sigmas` (npt.NDArray): diameters for all pairs of particle type, shape as [num_of_particletype, num_of_particletype]
- `r_cuts` (`npt.NDArray`): cutoff distances of pair potentials for all pairs of particle type, shape as [num_of_particletype, num_of_particletype]
- ppp (npt.NDArray): the periodic boundary conditions, setting 1 for yes and 0 for no, default np.array([1,1,1]), set np.array([1,1]) for two-dimensional systems
- shiftpotential (bool): whether shift the potential energy at r_c to 0, default True

### Return
- `None`

### Example
```python
import numpy as np 
from static.hessians import  HessianMatrix, InteractionParams, ModelName
from reader.dump_reader import DumpReader

# inverse-power law potential model in 2D
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

## 2.1 `pair_matrix()`

## 2.2 diagonalize_hessian()
This function is used to calculate the Hessian matrix of a simulation configuration and use numpy package to diagonalize the hessian matrix to get the eigenvalues and eigenvectors. The participation ratio of each eigenvector will be calculated as well.

### Input Arguments

### Return

### Example
```python
h.diagonalize_hessian(
    interaction_params=interaction_params,
    saveevecs=True,
    savehessian=True,
    outputfile="calfile"
)
```