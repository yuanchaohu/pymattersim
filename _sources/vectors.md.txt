### vector analysis


---

#### 1. participation ratio
This module calculates the participation ratio of one vibration mode. It is measured as
$$
PR_n = \frac{\left[ \sum_i^N |\vec e_{i,n}|^2 \right]^2}
{N\sum_i ({\vec e_{i, n}} \cdot {\vec e_{i,n}})^2},
$$
where $e_{i,n}$ is the eigenvector of particle $i$ in mode $n$. $N$ is the particle number.

##### Input Argument
- `vector` (`npt.NDArray`): input vector field, shape as [num_of_particles, ndim]

**Return**
- participation ratio of the vector field (float)

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from static.vector import participation_ratio

test_file = "test.atom"
Readdump =DumpReader(test_file, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
Readdump.read_onefile()
v = test_file.snapshots.snapshots[0].positions
pr = participation_ratio(v)
```

#### 2. local vector alignment
This module calculates the orientational ordering of each particle based on the vector alignments. It is measured as
$$
\Psi_i = \frac{1}{CN_i} \sum_j^{CN_i} {\vec e_i} \cdot {\vec e_j},
$$
where ${\vec e_i}$ is the eigenvector of particle $i$ and ${CN_i}$ is the number of neighbors of particle $i$.

##### Input Argument
- `vector` (`npt.NDArray`): input vector field, shape as [num_of_particles, ndim]
- `neighborfile` (`str`): file name of particle neighbors (see module `neighbors`)

**Return**
- phase quotient measured as a `float`

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from reader.reader_utils import DumpFileType
from static.vector import local_vector_alignment
from neighbors.calculate_neighbors import Nnearests


test_file = "test.atom"
input_vp =DumpReader(test_file, ndim=2)
input_vp.read_onefile()

input_v =DumpReader(test_file, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
input_v.read_onefile()
v = input_v.snapshots.snapshots[0].positions

Nnearests(input_vp.snapshots, N = 6, ppp = np.array([1,1]),  fnfile = 'neighborlist.dat')
neighborfile='neighborlist.dat'

pr = local_vector_alignment(v,neighborfile)
```

#### 3. phase quotient
This module calculates the phase quotient of a vector field. Maximum 200 neighbors are considered. It is defined as
$$
PQ_n = \frac{\sum_i^{N} \sum_j^{CN_i} {\vec e_{i, n}} \cdot {\vec e_{j, n}}}{\sum_i^{N} \sum_j^{CN_i} |{\vec e_{i, n}} \cdot {\vec e_{j, n}}|}.
$$

where $i$ and $j$ are atom indices, $N$ is the total atom number, ${CN_i}$ is the coordination number of atom $i$, n is the mode number, $\vec e_{i, n}$ is the eigenvector of atom $i$ in mode $n$. The upper bound $PQ_n=1$ means acoustic nature of the mode (an atom has the same direction as the neighbors), while the lower bound of -1 means optical nature of the mode (an atom has the opposite direction as the neighbors). If $PQ_n \gt 0$, the mode is acoustic-like otherwise is optical-like mode.

##### Input Argument
- `vector` (`npt.NDArray`): input vector field, shape as [num_of_particles, ndim]
- `neighborfile` (`str`): file name of particle neighbors (see module neighbors)

**Return**
- phase quotient measured as a float

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from static.vector import phase_quotient
from neighbors.calculate_neighbors import Nnearests

test_file = "test.atom"
input_vp =DumpReader(test_file, ndim=2)
input_vp.read_onefile()

input_v =DumpReader(test_file, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
input_v.read_onefile()
v = input_v.snapshots.snapshots[0].positions

Nnearests(input_vp.snapshots, N = 6, ppp = np.array([1,1]),  fnfile = 'neighborlist.dat')
neighborfile='neighborlist.dat'

pq = phase_quotient(v,neighborfile)
```

#### 4. divergence and curl
This module calculates the divergence and curl of a vector field at 2D and 3D. Divergence is scalar over all dimensions. Curl only exists in 3D as a vector. Maximum 200 neighbors are considered. Differently, **curl** can only be defined 3D as a vector, its direction shows the local rotational direction while its magnitude means the amplitude of such motion.

Both divergence and curl can be defined anywhere in space within the vector field, so i and j in above equations can be any lattice position in space.
- ${\rm divergence\ (scalar):\quad} div\ {\vec u_i} = \nabla \cdot {\vec u} = \frac{1}{N_i} \sum_j^{N_i} (\vec R_j - \vec R_i) \cdot (\vec u_j - \vec u_i)$
- ${\rm curl\ (vector):\quad} curl\ {\vec u_i} = \nabla \times {\vec u} = \frac{1}{N_i} \sum_j^{N_i} (\vec R_j - \vec R_i) \times (\vec u_j - \vec u_i)$
  
in which $\vec R_i$ is the position vector of atom $i$, $\vec u_i$ is the vector field (e.g. eigenvector) at atom $i$, $N_i$ is the number of nearest neighbors of atom $i$.

**Input Arguments**
- `snapshots` (`reader.reader_utils.SingleSnapshot`): snapshot object of input trajectory (returned by `reader.dump_reader.DumpReader`)
- `vector` (`npt.NDArray`): vector field shape as [num_of_partices, ndim], it determines the dimensionality of the calculation.
- `ppp` (`npt.NDArray`): the periodic boundary conditions, setting 1 for yes and 0 for no, default `np.array([1,1,1])`, set `np.array([1,1])` for two-dimensional systems
- `neighborfile` (`str`): file name of particle neighbors (see module `neighbors`)

**Return**
- divergence and curl (only 3D) in numpy array of the input vector

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from static.vector import divergence_curl
from neighbors.calculate_neighbors import Nnearests

test_file = "test.atom"
input_vp =DumpReader(test_file, ndim=2)
input_vp.read_onefile()
input_v =DumpReader(test_file, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
input_v.read_onefile()

ppp = np.array([1,1])
Nnearests(input_vp.snapshots, N = 6, ppp = ppp,  fnfile = 'neighborlist.dat')
neighborfile='neighborlist.dat'
        
v = input_v.snapshots.snapshots[0].positions
pq = divergence_curl(input_vp.snapshots.snapshots[0],v,ppp,neighborfile)[:10]
```

#### 5. vibrability
This calculates the susceptibility of particle motion to infinitesimal thermal excitation in the zero temperature limit, defined as:
$$
\Psi_i = \sum_{l=1}^{dN-d} \frac{1}{\omega_l^2} |{\vec e}_{l, i}|^2,
$$

where $\omega_l^2$ is the eigenvalue of the lth mode, while ${\vec e}_{l, i}$ is the corresponding eigenvector for particle $i$. Larger $\Psi$ indicates more susceptible to excitation and hence more disorder.

**Input Arguments**
- `eigenfrequencies` (`npt.NDArray`): eigen frequencies generally from Hessian diagonalization, shape as [num_of_modes,]
- `eigenvectors` (`npt.NDArray`): eigen vectors associated with eigenfrequencies, each column represents an eigen mode as from `np.linalg.eig` method

**Return**
- particle-level vibrability in a numpy array

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from static.vector import vibrability

###example for 10x10 matrix
a = np.array([[2, 3, 4, 1, 3, 3, 2, 2, 2, 2],
            [3, 2, 2, 2, 3, 3, 3, 2, 3, 3],
            [2, 2, 3, 3, 1, 2, 4, 1, 4, 3],
            [2, 1, 2, 2, 3, 1, 1, 2, 3, 4],
            [3, 2, 3, 4, 3, 2, 2, 2, 1, 4],
            [4, 3, 4, 1, 2, 2, 4, 2, 2, 3],
            [3, 4, 3, 1, 1, 1, 2, 2, 3, 1],
            [3, 1, 2, 3, 2, 4, 2, 4, 4, 1],
            [2, 2, 2, 4, 3, 4, 3, 3, 3, 2],
            [1, 1, 1, 1, 1, 1, 2, 1, 2, 3]])
eigenfrequencies, eigenvectors = np.linalg.eig(a)
eigenfrequencies = np.abs(eigenfrequencies.real)
eigenfrequencies = np.sqrt(eigenfrequencies)
eigenvectors = eigenvectors.real

vb = vibrability(eigenfrequencies,eigenvectors,10)
```

#### 6. vector decompostion structure factor
Decomposing the vector into transverse and longitudinal components and calculate their corresponding stucture factor.

$$
{\rm Longitudinal:} \qquad
S_L (q, \omega) = \frac{k_B T}{M} \left(\frac{q}{\omega} \right)^2 \sum_{\lambda=1}^{dN-d} E_{\lambda, L} (\mathbf{q}) \delta(\omega - \omega_{\lambda})
$$

$$
{\rm Transverse:} \qquad
S_T (q, \omega) = \frac{k_B T}{M} \left(\frac{q}{\omega} \right)^2 \sum_{\lambda=1}^{dN-d} E_{\lambda, T} (\mathbf{q}) \delta(\omega - \omega_{\lambda})
$$

where

$$
E_{\lambda, L} (\bf{q}) = \frac{1}{N} \left| \sum_{i=1}^N \left(\bf{\hat q} \cdot \frac{\bf{e}_i^{\lambda}}{\sqrt{m_i}} \right) \cdot \exp(i \bf{q} \cdot \bf{r}_i) \right|^2
$$

$$
E_{\lambda, T} (\bf{q}) = \frac{1}{N} \left| \sum_{i=1}^N \left(\bf{\hat q} \times \frac{\bf{e}_i^{\lambda}}{\sqrt{m_i}} \right) \cdot \exp(i \bf{q} \cdot \bf{r}_i) \right|^2
$$

in which $\bf \hat q = \bf q/|\bf q|$, $m_i$ and $\bf{r}_i$ are the mass and positional vector in the inherent structure of particle $i$. The average mass is approximated as $M^{-1} = \sum_l m_l^{-1} /N_l
$ in which $l$ runs over all species. The code calculates $E(q)$.

Another faster method is first taking the FFT of the eigenvector and then take dot product for the longitudinal component and cross product for the transverse component, as
$$
\tilde e = \sum_{i=1}^N \frac{\bf{e}_i^{\lambda}}{\sqrt{m_i}} \cdot \exp(i \bf{q} \cdot \bf{r}_i)
$$ 

$$
E_{\lambda, L} (\bf{q}) = \frac{1}{N} \left| \hat{\bf{q}} \cdot \tilde e \right|^2
$$

$$
E_{\lambda, T} (\bf{q}) = \frac{1}{N} \left| \hat{\bf{q}} \times \tilde e \right|^2
$$

This new method is utilized in the calculation. Basically, the vector field is first Fourier transformed and then decomposed into transverse and longitudinal components. The corresponding structure factor is then calculated.

**Input Arguments**
- `snapshot` (`reader.reader_utils.SingleSnapshot`): single snapshot object of input trajectory
- `qvector` (`npt.NDArray` of `int`): designed wavevectors in two-dimensional `np.array` (see `utils.wavevector`)
- `vector` (`npt.NDArray`): particle-level vector, shape as [num_of_particles, ndim], for example, eigenvector field and velocity field
- `outputfile` (`str`): filename.csv to save the calculated `S(q)`, default `None`

**Return**
- `vector_fft`: calculated transverse and longitudinal `S(q)` for each input wavevector (`pd.DataFrame`), FFT in complex number is also returned for reference
- `ave_sqresults`: the ensemble averaged `S(q)` over the same wavenumber (`pd.DataFrame`)

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from static.vector import vector_decomposition_sq
from utils.wavevector import choosewavevector

test_file = "test.atom"

input_vp =DumpReader(test_file, ndim=2)
input_vp.read_onefile()

input_v =DumpReader(test_file, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
input_v.read_onefile()
v = input_v.snapshots.snapshots[0].positions

qvector = choosewavevector(2,10)
vector_fft, ave_sqresults=vector_decomposition_sq(input_vp.snapshots.snapshots[0],qvector,v)
```

#### 7. vectors Fourier transformaiton and their time correlations
This module calculates spectra and time correlation of the longitudinal and tranverse components of a vector field by fast fourier transformation (FFT). This is usually used to calculate the current-current correlation function and its power spectra. This is also useful to calculate the dynamic structure factor based on the velocity field.

$$
\vec j (\vec q, t) = \sum_{m=1}^N \vec v_m(t) \exp[i \vec q \cdot \vec r_m(t)],
$$

where $\vec v_m(t)$ is the vector of particle $m$ at time $t$, and $\vec r_m(t)$ is the particle position.. It is decomposed into transverse and longitudinal components as 
$$
\vec j_L (\vec q, t) = (\vec j (\vec q, t) \cdot \mathbf{\hat q}) \cdot \mathbf{\hat q}
\qquad\qquad
\vec j_T (\vec q, t) = \vec j (\vec q, t) - \vec j_L (\vec q, t)
$$

where $\mathbf{\hat q}$ is the unit vector ${\bf \hat q}={\vec q}/|{\bf q}|$.

The ensemble averaged power spectra is calculated as
$$
E_L (q) = \left < \frac{1}{N} \left| \vec j_L(\vec q, t) \right|^2 \right>
\qquad\qquad\qquad
E_T (q) = \left < \frac{1}{N} \left| \vec j_T(\vec q, t) \right|^2 \right>
$$

and the time correlation functions are calculated as
$$
C_\alpha(q, t)= \frac{1}{N} \left< \vec j_\alpha({\vec q}, t) \cdot \vec j_\alpha(-{\vec q}, 0) \right>
$$

where $\alpha$ represents longitudinal ($L$) or transverse ($T$) current.

**Input Arguments**
- `snapshots` (`read.reader_utils.snapshots`): multiple trajectories dumped linearly or in logscale
- `qvector` (`npt.NDArray` of `int`): designed wavevectors in two-dimensional `np.array` (see `utils.wavevector`)
- `vectors` (`npt.NDArray`): particle-level vector, shape as [num_of_snapshots, num_of_particles, ndim], for example, eigenvector field and velocity field
- `dt` (`float`): time step of input snapshots, default 0.002
- `outputfile` (`str`): filename.csv to save the calculated `S(q)`, default `None`

**Return**
- the averaged spectra of full, transverse, and longitudinal mode, saved into a csv dataset
- time correlation of FFT of vectors in full, transverse, and longitudinal mode. Dict as `{"FFT": pd.DataFrame, "T_FFT": pd.DataFrame, "L_FFT": pd.DataFrame}`

**Example**
```python
import numpy as np
from reader.dump_reader import DumpReader
from static.vector import vector_fft_corr
from utils.wavevector import choosewavevector

test_file = "test.atom"
input_vp =DumpReader(test_file, ndim=2)
input_vp.read_onefile()

input_v =DumpReader(test_file, ndim=2,filetype=DumpFileType.LAMMPSVECTOR,columnsids=[5,6])
input_v.read_onefile()

qvector = choosewavevector(2, 27, False)
v =[]
for i in range(input_v.snapshots.nsnapshots):
    temp = input_v.snapshots.snapshots[i].positions
    v.append(temp)
v=np.array(v)

alldata = vector_fft_corr(input_vp.snapshots,qvector,v,outputfile="test")
```