from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import warnings

    import numpy as np
    from skglm import GeneralizedLinearEstimator
    from skglm.datafits import Quadratic
    from skglm.penalties import SLOPE
    from skglm.solvers import FISTA
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    sampling_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["pip:skglm"]

    references = [
        "Q. Bertrand, Q. Klopfenstein, P.-A. Bannier, G. Gidel, and M. Massias,"
        "“Beyond L1: faster and better sparse models with skglm,” in Advances in"
        "Neural Information Processing Systems 35, New Orleans, USA: Curran Associates,"
        "Inc., Dec. 2022, pp. 38950–38965. Accessed: Jan. 08, 2024."
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = fit_intercept

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.model = GeneralizedLinearEstimator(
            Quadratic(), SLOPE(alphas), FISTA(opt_strategy="fixpoint", tol=1e-20)
        )

        # Cache Numba compilation
        self.run(1)

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

    def get_next(self, stop_val):
        return stop_val + 1

    def skip(self, X, y, alphas, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle intercept fitting"

        return False, None

    def get_result(self):
        return dict(beta=self.coef)
