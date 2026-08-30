import pytest

from solvers.sortedl1 import MIN_TOLERANCE, Solver


def test_tolerance_sampling_stops_at_minimum_tolerance():
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
        objective_values.append(
            {"objective_value": -float(len(evaluated_tolerances))}
        )
        stop, status, stop_val = criterion.should_stop(
            stop_val,
            objective_values,
        )
        if stop:
            break

    assert status == "done"
    assert evaluated_tolerances[-1] == pytest.approx(MIN_TOLERANCE)
    assert min(evaluated_tolerances) >= MIN_TOLERANCE
