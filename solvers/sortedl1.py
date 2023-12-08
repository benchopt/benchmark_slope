from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sortedl1 import Slope


class Solver(BaseSolver):
    name = "sortedl1"
    sampling_strategy = "iteration"
    install_cmd = "conda"
    # TODO: use PyPi once the package is available there
    requirements = ["pip:git+https://github.com/jolars/sortedl1"]

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
