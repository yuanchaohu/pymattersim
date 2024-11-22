.. py_Matters_Simulations documentation master file, created by
   sphinx-quickstart on Tue Nov 19 14:08:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Python for Matters Simulations documentation!
==================================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Getting started <self>
   reader
   writer
   neighbors
   gr
   sq
   boo_2d
   boo_3d
   orderings
   dynamics
   hessian
   vectors
   utils

Getting started
----------------

This toolkit is a Physics-driven data analyis of computer simulations 
for materials science, chemistry, physics, and beyond. Currently, the
library is mainly designed for computer simulations of amorphous materials
and supercooled liquids from the open source simulator LAMMPS. But the
analyis is in principle useful for any simulations, and it is straightforward
to make extensions.

Installation
----------------

.. code:: bash

   pip install PyMatterSim


The source code is available from this `github project <https://github.com/yuanchaohu/pymattersim>`_,
and the package is deployed at `this PYPI page <https://pypi.org/project/PyMatterSim/>`_.

Documentation
----------------

The documentation for **PyMatterSim** is `hosted online <https://yuanchaohu.github.io/pymattersim/>`_

Citation
----------------

Working in progress.

References
----------------

- Y.-C. Hu et al. `Origin of the boson peak in amorphous solids <https://doi.org/10.1038/s41567-022-01628-6>`_. **Nature Physics**, 18(6), 669-677 (2022) 
- Y.-C. Hu et al. `Revealing the role of liquid preordering in crystallisation of supercooled liquids <https://doi.org/10.1038/s41467-022-32241-z>`_. **Nature Communications**, 13(1), 4519 (2022)
- Y.-C. Hu et al. `Physical origin of glass formation from multicomponent system <https://www.science.org/doi/10.1126/sciadv.abd2928>`_. **Science Advances** 6 (50), eabd2928 (2020)
- Y.-C. Hu et al. `Configuration correlation governs slow dynamics of supercooled metallic liquids <https://doi.org/10.1073/pnas.1802300115>`_. **Proceedings of the National Academy of Sciences U.S.A.**, 115(25), 6375-6380 (2018)
- Y.-C. Hu et al. `Five-fold symmetry as indicator of dynamic arrest in metallic glass-forming liquids <https://doi.org/10.1038/ncomms9310>`_. **Nature Communications**, 6(1), 8310 (2015) 


Unit Tests
----------------
The unit tests for **PyMatterSim** are included in the github repository
and are configured to be run using the python :mod:`UnitTest` library:

.. code-block:: bash

   # auto-run tests with shell scripts
   cd shell
   bash *sh

   # run individual tests
   cd tests/yourdir/
   python *py


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
