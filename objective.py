# Author: Quentin Klopfenstein
#         Jonas Wallin
#         Johan Larsson

import math

from benchopt import BaseObjective, safe_import_context
from benchopt.stopping_criterion import StoppingCriterion


class TargetObjectiveCriterion(StoppingCriterion):
    """Stop a convergence curve after it reaches an objective-level target."""

    def __init__(
        self,
        target_key="target_rel_duality_gap",
        strategy=None,
        key_to_monitor="rel_duality_gap",
        minimize=True,
    ):
        self.target_key = target_key
        self.target_key_ = (
            target_key
            if target_key.startswith("objective_")
            else f"objective_{target_key}"
        )
        super().__init__(
            target_key=target_key,
            strategy=strategy,
            key_to_monitor=key_to_monitor,
            minimize=minimize,
        )

    def check_convergence(self, objective_list):
        result = objective_list[-1]
        value = result[self.key_to_monitor_]
        target = result[self.target_key_]

        if value <= target:
            self.debug(f"Exit with {self.key_to_monitor_} = {value:.2e}.")
            return True, 1.0

        if not math.isfinite(value):
            return False, 0.0

        return False, min(1.0, target / value)

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import stats


class Objective(BaseObjective):
    name = "SLOPE"
    min_benchopt_version = "1.5"
    requirements = ["numba", "numpy", "scipy"]
    stopping_criterion = TargetObjectiveCriterion(
        key_to_monitor="rel_duality_gap"
    )
    parameters = {
        "reg": [0.5, 0.1, 0.02],
        "q": [0.2, 0.1, 0.05],
        "fit_intercept": [False],
        "target_rel_duality_gap": [1e-7],
    }

    def __init__(
        self,
        reg=0.1,
        q=0.1,
        fit_intercept=False,
        target_rel_duality_gap=1e-7,
    ):
        if target_rel_duality_gap <= 0:
            raise ValueError("target_rel_duality_gap must be strictly positive")

        self.q = q
        self.reg = reg
        self.fit_intercept = fit_intercept
        self.target_rel_duality_gap = target_rel_duality_gap

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape
        self.alphas = self._get_lambda_seq()

    def get_one_result(self):
        return dict(beta=np.zeros(self.n_features + 1))

    def evaluate_result(self, beta):
        intercept, coefs = beta[0], beta[1:]

        X, y = self.X, self.y
        n_samples = X.shape[0]
        # compute residuals
        diff = y - X @ coefs - intercept

        # compute primal
        p_obj = 1.0 / (2 * n_samples) * diff @ diff + np.sum(
            self.alphas * np.sort(np.abs(coefs))[::-1]
        )

        # compute dual
        theta = diff.copy()
        if self.fit_intercept:
            # An unpenalized intercept imposes sum(theta) = 0 in the dual.
            theta -= np.mean(theta)
        theta /= max(1, self._dual_norm_slope(theta, self.alphas))
        d_obj = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / (2 * n_samples)

        return dict(
            value=p_obj,
            duality_gap=p_obj - d_obj,
            rel_duality_gap=(p_obj - d_obj) / (1e-10 + np.abs(p_obj)),
            target_rel_duality_gap=self.target_rel_duality_gap,
        )

    def get_objective(self):
        return dict(
            X=self.X, y=self.y, alphas=self.alphas, fit_intercept=self.fit_intercept
        )

    def _dual_norm_slope(self, theta, alphas):
        Xtheta = np.sort(np.abs(self.X.T @ theta))[::-1]
        taus = 1 / np.cumsum(alphas)
        return np.max(np.cumsum(Xtheta) * taus)

    def _get_lambda_seq(self):
        randnorm = stats.norm(loc=0, scale=1)
        q = self.q
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, self.X.shape[1] + 1) * q / (2 * self.X.shape[1])
        )

        alpha_max = self._dual_norm_slope(
            (self.y - self.fit_intercept * np.mean(self.y)) / len(self.y), alphas_seq
        )
        return alpha_max * alphas_seq * self.reg
