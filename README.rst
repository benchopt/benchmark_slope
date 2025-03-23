Benchmark repository for SLOPE
==============================

|Build Status| |Python 3.6+|

This repository is based on the work of Johan Larsson, Quentin Klopfenstein, Mathurin Massias and Jonas Wallin at https://github.com/Klopfe/benchmark_slope.

This repository is dedicated to regression with the Sorted L-One Penalized Estimation (SLOPE) estimator which consists in solving the following program:

$$ \\min_{\\beta} \\, \\tfrac{1}{2n} \\Vert y - X\\beta \\Vert^2_2 + J(\\beta, \\lambda) $$

where

$$ J(\\beta, \\lambda) = \\sum_{j=1}^p \\lambda_j \| \\beta_{(j)}\| $$

with $\\lambda_1 \\geq \\lambda_2 \\geq ... \\geq \\lambda_p$ and $\|\\beta_{(1)}\| \\geq \|\\beta_{(2)}\| \\geq ... \\geq \|\\beta_{(p)}\|$.

We note $n$ (or n_samples) the number of samples and $p$ (or n_features) the number of features.
We also have that $X\\in \\mathbb{R}^{n\\times p}$ and $y\\in \\mathbb{R}^n$.



Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_slope
   $ benchopt install ./benchmark_slope
   $ benchopt run ./benchmark_slope  -config example_config

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run ./benchmark_slope -s PGD -d simulated --max-runs 10 --n-repetitions 5


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/cli.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_slope/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_slope/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
