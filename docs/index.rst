
FOLPSpipe: Documentation
========================

FOLPS (Fast and Efficient Computation of the Redshift-Space Power Spectrum and Bispectrum) is a high-performance tool for cosmological modeling, designed for large-scale structure (LSS) analyses with support for massive neutrinos and modified gravity theories.

Main features:
- Robust and accurate predictions for the power spectrum and bispectrum in redshift space.
- Support for different backends (NumPy/JAX).
- Auxiliary tools in `tools.py`.

Installation
============

Clone the repository and navigate to the main folder:

.. code-block:: bash

   git clone <your-repo>
   cd FOLPSpipe
   pip install -r requirements.txt

Getting Started
===============

After installation, you can start using FOLPSpipe in your Python scripts or Jupyter notebooks. For detailed API documentation, see the `api` section. Example notebooks and scripts are provided in the repository.

Basic Usage
===========

Example usage in Python:

.. code-block:: python

   from folps import folps
   # Initialize and use the main functions here

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`