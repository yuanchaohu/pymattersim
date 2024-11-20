# pymattersim

## Summary
Physics-driven data analyis of computer simulations for materials science, chemistry, physics, and beyond.

## Installation
`pip install PyMatterSim`

`pip install git@github.com:yuanchaohu/pymattersim.git`

## Requirements
- python 3.6 or higher
- numpy
- pandas
- freud-analysis
- scipy
- sympy
- gsd (optional)
- mdtraj (optional)
- voro++ (optional, standalone binary)

## Usage
Please refer to the `/docs/` for documentation and examples.
Some examples are provided from the unittest modules (`tests/`)

## Types of computer simulations
1. LAMMPS
   1. atom type & molecular type such as patchy particle, rigid body, molecules et al.
   2. x, xs, xu type particle positions
   3. orthagonal / triclinic box
2. Hoomd-blue
   1. GSD for structure analysis (need `gsd==3.2.0`)
   2. GSD + DCD for dynamics analysis (need `gsd==3.2.0` and `mdtraj==1.9.9`)
3. VASP (to be added)
4. Any type of simulators as long as the input were formatted well, modifying the `reader` module to use the computational modules.


## Notes
[Voro++](https://math.lbl.gov/voro++/) is recommend to install separately for specific Voronoi analysis. Some of the analysis from the original voro++ is maintained from the [freud-analysis package](https://freud.readthedocs.io/en/stable/gettingstarted/installation.html) developed by the Glozter group.

## Citation


## UnitTest
Please run the bash scripts available from `shell/` for unittests. As follows are test statistics:
| Test              | # Tests and Runtime | Status |
| :---------------- | :-------------------------  | :----- |
| test_dynamics     |  Ran 15 tests in 10.303s    | OK     |
| test_neighbors    |  Ran 11 tests in 91.711s    | OK     |
| test_reader       |  Ran 11 tests in 0.270s     | OK     |
| test_static       |  Ran 28 tests in 298.248s   | OK     |
| test_utils        |  Ran 30 tests in 4.997s     | OK     |
| test_writer       |  Ran 3 tests in 0.005s      | OK     |