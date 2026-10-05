from typing import ClassVar

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from scipy.optimize import brentq
    from scipy.stats import norm
    from tick.linear_model import ModelLinReg
    from tick.prox import ProxSlope
    from tick.solver import AGD


def _slope_parameters(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0 or not np.all(np.isfinite(weights)):
        return None
    if np.all(weights == 0):
        return 0.1, 0.0
    if weights[-1] <= 0 or weights[0] < weights[-1]:
        return None

    p = len(weights)
    if p == 1:
        q = 0.1
    else:
        ratio = weights[0] / weights[-1]

        def ratio_error(candidate):
            return norm.isf(candidate / (2 * p)) / norm.isf(candidate / 2) - ratio

        lower, upper = 1e-12, 1 - 1e-12
        if ratio_error(lower) > 0 or ratio_error(upper) < 0:
            return None
        q = brentq(ratio_error, lower, upper, xtol=1e-14)

    bh_weights = norm.isf(q * np.arange(1, p + 1) / (2 * p))
    strength = weights[0] / bh_weights[0]
    if not np.allclose(weights, strength * bh_weights, rtol=1e-6, atol=1e-12):
        return None
    return q, strength


class Solver(BaseSolver):
    name = "tick"
    sampling_strategy = "iteration"
    install_cmd = "conda"
    requirements: ClassVar[list[str]] = [
        "numpy",
        "scipy",
        "numpydoc",
        "pip::tick>=0.8.0.2",
    ]
    references: ClassVar[list[str]] = [
        (
            "Bacry, E., Bompaire, M., Gaiffas, S., & Poulsen, S. V. (2017). "
            "tick: a Python library for statistical learning, with a particular "
            "emphasis on time-dependent modeling. arXiv:1707.03003."
        )
    ]

    def skip(self, X, y, alphas, fit_intercept):
        if _slope_parameters(alphas) is None:
            return True, "tick requires Benjamini-Hochberg SLOPE weights"
        return False, None

    def set_objective(self, X, y, alphas, fit_intercept):
        self.n_features = X.shape[1]
        self.fit_intercept = fit_intercept
        parameters = _slope_parameters(alphas)
        if parameters is None:
            raise ValueError("tick requires Benjamini-Hochberg SLOPE weights")
        q, strength = parameters

        if sparse.issparse(X):
            X = sparse.csr_matrix(X, dtype=np.float64)
        else:
            X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        model = ModelLinReg(fit_intercept=fit_intercept).fit(X, y)
        penalty_range = (0, self.n_features) if fit_intercept else None
        prox = ProxSlope(strength=strength, fdr=q, range=penalty_range)
        self.solver = AGD(
            # The average sample Lipschitz constant bounds the full loss and
            # works for sparse designs, unlike tick's dense-only best bound.
            step=1 / model.get_lip_mean(),
            max_iter=1,
            tol=0.0,
            linesearch=False,
            verbose=False,
        )
        self.solver.set_model(model).set_prox(prox)
        self.n_coeffs = model.n_coeffs

    @staticmethod
    def get_next(previous):
        return max(1, 2 * previous)

    def run(self, n_iter):
        self.coef = np.zeros(self.n_coeffs)
        if n_iter:
            self.solver.max_iter = n_iter
            self.coef = self.solver.solve(x0=self.coef)

    def get_result(self):
        if self.fit_intercept:
            beta = np.concatenate(([self.coef[-1]], self.coef[:-1]))
        else:
            beta = np.concatenate(([0.0], self.coef))
        return {"beta": beta}
