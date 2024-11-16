# pymattersim

molecular dynamics simulations analysis

Simulators:
1. LAMMPS
   1. atom type & molecular type such as patchy particle, rigid body, molecules et al.
   2. x, xs, xu type particle positions
   3. orthagonal / triclinic box
2. Hoomd-blue
   1. GSD for structure analysis (need `gsd==3.2.0`)
   2. GSD + DCD for dynamics analysis (need `gsd==3.2.0` and `mdtraj==1.9.9`)
3. VASP (to be added)
4. Any type of simulators as long as the input were formatted well, modifying the `reader` module to use the computational modules.
5. Voro++ is recommend to install separately for some specific analysis (optional)
