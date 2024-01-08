from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sortedl1 import Slope


class Solver(BaseSolver):
    name = "sortedl1"
    sampling_strategy = "iteration"
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

        # n_samples = self.X.shape[0]
        self.model = Slope(
            lam=self.lambdas,
            alpha=1.0,
            fit_intercept=self.fit_intercept,
            standardize=False,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1] + 1)
        else:
            self.model.max_iter = n_iter
            self.model.fit(self.X, self.y)

            coef = self.model.coef_.flatten()

            if self.fit_intercept:
                self.coef = np.hstack((np.atleast_1d(self.model.intercept_), coef))
            else:
                self.coef = np.hstack(([0.0], coef))

    def get_result(self):
        return dict(beta=self.coef)
