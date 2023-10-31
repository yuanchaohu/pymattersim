[TOC]

# Remove PBC

The module `utils.pbc` is used to remove periodic boundary conditions (PBC) and is usually embedded in other analysis modules.

## Input Arguments

- `RIJ` (`np.array`): position difference between particle ***i*** (center) and ***j*** (neighbors) with PBC
- `hmatrix` (`np.array`): h-matrix of the box
- `ppp` (`list`): the periodic boundary conditions, setting 1 for yes and 0 for no. Default `[1, 1, 1]`, that is, PBC is applied in all three dimensions for 3D box

## Return

A `np.array` for the position difference between particle ***i*** (center) and ***j*** (neighbors) after removing PBC

## Example

```python
from utils.pbc import remove_pbc

remove_pbc(RIJ, hmatrix, ppp)
```

