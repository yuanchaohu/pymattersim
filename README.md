# pymattersim

## Summary
Physics-driven data analyis of computer simulations for materials science, chemistry, physics, and beyond.

## Installation
`pip install PyMatterSim`

`pip install git@github.com:yuanchaohu/pymattersim.git`

## Documentation
The [documentation](https://yuanchaohu.github.io/pymattersim/) is now available online.

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
```
@article{hu2024pymattersimpythondataanalysis,
      title={PyMatterSim: a Python Data Analysis Library for Computer Simulations of Materials Science, Physics, Chemistry, and Beyond}, 
      author={Y. -C. Hu and J. Tian},
      year={2024},
      eprint={2411.17970},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2411.17970}, 
}
```

## References
- Y.-C. Hu et al. [Origin of the boson peak in amorphous solids](https://doi.org/10.1038/s41567-022-01628-6). **Nature Physics**, 18(6), 669-677 (2022) 
- Y.-C. Hu et al. [Revealing the role of liquid preordering in crystallisation of supercooled liquids](https://doi.org/10.1038/s41467-022-32241-z). **Nature Communications**, 13(1), 4519 (2022)
- Y.-C. Hu et al. [Physical origin of glass formation from multicomponent systems](https://www.science.org/doi/10.1126/sciadv.abd2928). **Science Advances** 6 (50), eabd2928 (2020)
- Y.-C. Hu et al. [Configuration correlation governs slow dynamics of supercooled metallic liquids](https://doi.org/10.1073/pnas.1802300115). **Proceedings of the National Academy of Sciences U.S.A.**, 115(25), 6375-6380 (2018)
- Y.-C. Hu et al. [Five-fold symmetry as indicator of dynamic arrest in metallic glass-forming liquids](https://doi.org/10.1038/ncomms9310). **Nature Communications**, 6(1), 8310 (2015) 


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
