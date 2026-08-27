import numpy as np

from objective import Objective


def test_intercept_dual_point_is_centered():
    objective = Objective(fit_intercept=True)
    objective.X = np.array([[-1.0], [1.0]])
    objective.y = np.ones(2)
    objective.alphas = np.ones(1)

    result = objective.evaluate_result(beta=np.array([0.75, 0.0]))

    np.testing.assert_allclose(result["duality_gap"], 0.03125)
    assert result["duality_gap"] >= 0.0
