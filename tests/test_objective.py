import numpy as np
import pytest

from objective import Objective, TargetObjectiveCriterion


def test_intercept_dual_point_is_centered():
    objective = Objective(fit_intercept=True)
    objective.X = np.array([[-1.0], [1.0]])
    objective.y = np.ones(2)
    objective.alphas = np.ones(1)

    result = objective.evaluate_result(beta=np.array([0.75, 0.0]))

    np.testing.assert_allclose(result["duality_gap"], 0.03125)
    assert result["duality_gap"] >= 0.0
    assert result["target_rel_duality_gap"] == 1e-7


@pytest.mark.parametrize(
    ("gap", "should_stop"),
    [(2e-7, False), (1e-7, True), (5e-8, True)],
)
def test_stops_at_target_relative_duality_gap(gap, should_stop):
    criterion = TargetObjectiveCriterion(key_to_monitor="rel_duality_gap")
    criterion.terminal = None
    objective_list = [
        {
            "objective_rel_duality_gap": gap,
            "objective_target_rel_duality_gap": 1e-7,
        }
    ]

    stop, _ = criterion.check_convergence(objective_list)

    assert stop is should_stop


def test_target_relative_duality_gap_must_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        Objective(target_rel_duality_gap=0.0)


def test_objective_criterion_supports_mixed_sampling_strategies():
    class DummySolver:
        def __init__(self, sampling_strategy):
            self.sampling_strategy = sampling_strategy

    criterion = TargetObjectiveCriterion()
    iteration_runner = criterion.get_runner_instance(
        solver=DummySolver("iteration")
    )
    tolerance_runner = criterion.get_runner_instance(
        solver=DummySolver("tolerance")
    )

    assert iteration_runner.strategy == "iteration"
    assert tolerance_runner.strategy == "tolerance"
    assert criterion.strategy is None
