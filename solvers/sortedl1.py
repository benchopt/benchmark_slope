from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY

with safe_import_context() as import_ctx:
    import numpy as np
    from sortedl1 import Slope


class Solver(BaseSolver):
    name = "sortedl1"
    sampling_strategy = "tolerance"
    install_cmd = "conda"
    requirements = ["pip:sortedl1"]

    references = [
        "J. Larsson, Q. Klopfenstein, M. Massias, and J. Wallin, "
        "“Coordinate descent for SLOPE,” in Proceedings of the 26th "
        "international conference on artificial intelligence and statistics, "
        "F. Ruiz, J. Dy, and J.-W. van de Meent, Eds., in Proceedings of "
        "machine learning research, vol. 206. Valencia, Spain: PMLR, Apr. 2023, "
        "pp. 4802–4821. [Online]. Available: "
        "https://proceedings.mlr.press/v206/larsson23a.html"
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = fit_intercept

        self.model = Slope(
            lam=self.lambdas,
            alpha=1.0,
            fit_intercept=self.fit_intercept,
            max_iter=1_000_000,
        )

    def run(self, tol):
        if tol == INFINITY:
            self.coef = np.zeros(self.X.shape[1] + 1)
        else:
            self.model.tol = tol
            self.model.fit(self.X, self.y)

            coef = self.model.coef_.flatten()

            if self.fit_intercept:
                self.coef = np.hstack((np.atleast_1d(self.model.intercept_), coef))
            else:
                self.coef = np.hstack(([0.0], coef))

    @staticmethod
    def get_next(prev_tol):
        if prev_tol == INFINITY:
            return 0.1
        else:
            return prev_tol / 5

    def get_result(self):
        return dict(beta=self.coef)
