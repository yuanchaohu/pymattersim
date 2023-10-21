# Particle Neighbors

The format of the saved neighbor list file (named as 'neighborlist.dat' in default) must be identification of the centered particle, coordination number of the centered particle, and identification of neighboring particles. In default, the neighbors in the output file is sorted by their distances to the centered particle in the ascending order.

## N Neighbors

Getting the N nearest neighbors around a particle. In this case, the coordination number is N for each centered particle.  The parameter *N* receives the desired N.

```python
from ParticleNeighbors import Nnearests

Nnearests(dumpfile, ndim = 3, filetype = 'lammps', moltypes = '',
          N = 12, ppp = [1,1,1],  fnfile = 'neighborlist.dat')
```



## Cutoff Neighbors

Get the nearest neighbors around a particle by setting a cutoff distance. The parameter *r_cut* receives the cutoff distances. Usually, the cutoff distance can be determined as the position of the first deep valley in total pair correlation function.

Usage example:

```python
from ParticleNeighbors import cutoffneighbors

cutoffneighbors(dumpfile, r_cut, ndim = 3, filetype = 'lammps', moltypes = '',
                ppp = [1,1,1], fnfile = 'neighborlist.dat')
```



## Cutoff Neighbors with atom types
Get the nearest neighbors around a particle by setting a cutoff distance for each atom pair. The parameter *r_cut* receives the user-specified cutoff distances of each atom pair. Taken Cu-Zr system as an example, *r_cut* should be a 2D numpy array: 
$$
\begin{bmatrix}
  &r_{cut}^{Cu-Cu} & r_{cut}^{Cu-Zr}&\\
  &r_{cut}^{Zr-Cu} & r_{cut}^{Zr-Zr}&\\
\end{bmatrix}
$$
Usually, these cutoff distances can be determined as the position of the first deep valley in partial pair correlation function for each atom pair.

Usage example:

```python
from ParticleNeighbors import cutoffneighbors_atomtypes

cutoffneighbors_atomtypes(dumpfiles, r_cut, n_dim=3, filetype = 'lammps', moltypes = '', 
                          ppp = [1,1,1], fnfile = 'neighborlist.dat')
```



