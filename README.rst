Benchmark Repository for SLOPE
==============================

|Build Status| |Python 3.6+|

This repository is dedicated to regression with the Sorted L-One Penalized Estimation (SLOPE) estimator which consists in solving the following program:

$$ \\min_{\\beta} \\, \\tfrac{1}{2n} \\Vert y - X\\beta \\Vert^2_2 + J(\\beta, \\lambda) $$

where

$$ J(\\beta, \\lambda) = \\sum_{j=1}^p \\lambda_j \| \\beta_{(j)}\| $$

with $\\lambda_1 \\geq \\lambda_2 \\geq ... \\geq \\lambda_p$ and $\|\\beta_{(1)}\| \\geq \|\\beta_{(2)}\| \\geq ... \\geq \|\\beta_{(p)}\|$.

We note $n$ (or n_samples) the number of samples and $p$ (or n_features) the number of features.
We also have that $X\\in \\mathbb{R}^{n\\times p}$ and $y\\in \\mathbb{R}^n$.

Installling Benchopt
--------------------

This benchmark relies on benchopt, a generic framework for running numerical benchmarks.
The recommended way to use benchopt is within a conda environment. So, begin by creating and activating
a new conda environment and install benchopt in it:

.. code-block::

   $ conda create -n benchopt python
   $ conda activate benchopt
   $ pip install -U benchopt

Installing the Benchmark
------------------------

To install the benchmark, clone this repository and move to its folder:

.. code-block::

   $ git clone https://github.com/benchopt/benchmark_slope
   $ cd benchmark_slope/

To install the dependencies for the solvers and datasets for the benchmark,
first make sure that you have activated the conda environment where benchopt is
installed. Then, you can either install all the dependencies with:

.. code-block::

   $ benchopt install .

Or you can install only a subset of solvers by specifying them with the ``-s`` option.

.. code-block::
   $ benchopt install . -s sortedl1 -d libsvm

Running the Benchmark
---------------------

To run the benchmark, simply use the `benchopt run` command:

.. code-block::

	$ benchopt run .

By default, all solvers and datasets are run. You can restrict the benchmark to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run -s PGD[prox=prox_fast_stack] -d libsvm[dataset=real-sim,standardize=True]

You can also specify a YAML configuration file to set the parameters of the benchmark.
An example config is provided in <example_config.yml>.

.. code-block::

	$ benchopt run --config example_config.yml .

Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_slope/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_slope/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/

Acknowledgments
---------------

This repository is based on the work of Johan Larsson, Quentin Klopfenstein, Mathurin Massias and Jonas Wallin at https://github.com/Klopfe/benchmark_slope.
