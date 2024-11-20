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

Citation
----------------

This section is still a work in progress.


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


Documentation
----------------

The documentation for **PyMatterSim** is `hosted online <https://yuanchaohu.github.io/pymattersim/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
