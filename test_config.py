import sys

import pytest


def check_test_solver_install(solver_class):
    if solver_class.name.lower() == "rslope" and sys.platform == "darwin":
        pytest.xfail("Dependencies for R packages fail to install on MacOS.")
