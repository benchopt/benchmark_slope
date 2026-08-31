import pytest

from solvers.sortedl1 import MIN_TOLERANCE, Solver


@pytest.mark.parametrize("constant_objective", [False, True])
def test_tolerance_sampling_stops_at_minimum_tolerance(constant_objective):
    solver = Solver.get_instance()
    criterion = solver.stopping_criterion.get_runner_instance(
        solver=solver,
        max_runs=100,
    )
    stop_val = criterion.init_stop_val()
    evaluated_tolerances = []
    objective_values = []

    while True:
        evaluated_tolerances.append(stop_val)
        objective_value = (
            0.0 if constant_objective else -float(len(evaluated_tolerances))
        )
        objective_values.append({"objective_value": objective_value})
        stop, status, stop_val = criterion.should_stop(
            stop_val,
            objective_values,
        )
        if stop:
            break

    assert status == "done"
    assert MIN_TOLERANCE == pytest.approx(1e-7)
    assert evaluated_tolerances[-1] == pytest.approx(1e-7)
    assert min(evaluated_tolerances) >= MIN_TOLERANCE
