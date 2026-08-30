from types import SimpleNamespace

import pytest

from solvers.newt_alm import Solver


@pytest.mark.parametrize(
    ("inner_solver", "n_samples", "expected_skip"),
    [
        ("standard", 10_001, True),
        ("standard", 10_000, False),
        ("auto", 20_000, False),
    ],
)
def test_skip_standard_solver_for_large_dense_systems(
    inner_solver,
    n_samples,
    expected_skip,
):
    solver = Solver.get_instance(inner_solver=inner_solver)
    X = SimpleNamespace(shape=(n_samples, 100))

    skip, reason = solver.skip(
        X=X,
        y=None,
        alphas=None,
        fit_intercept=False,
    )

    assert skip is expected_skip
    assert (reason is not None) is expected_skip
