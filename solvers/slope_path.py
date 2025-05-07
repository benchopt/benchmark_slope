from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from modules import path_solver
    from scipy import sparse


class Solver(BaseSolver):
    name = "SlopePath"
    sampling_strategy = "iteration"
    install_cmd = "conda"
    requirements = ["pip:git+https://github.com/jolars/slope-path"]
    # TODO when benchopt 1.7 is released, update to
    # "pip::git..."

    references = [
        "Dupuis, X., & Tardivel, P. (2024). The solution path of SLOPE. "
        "Proceedings of The 27th International Conference on Artificial "
        "Intelligence and Statistics, 238, 775–783. "
        "https://proceedings.mlr.press/v238/dupuis24a.html"
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas * X.shape[0]
        self.fit_intercept = fit_intercept

    def warm_up(self):
        # Needs numba JIT compilation
        self.run_once()

    def skip(self, X, y, alphas, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle intercept fitting"

        if sparse.issparse(X):
            return True, f"{self.name} does not handle sparse design matrices"

        return False, None

    def run(self, it):
        sol, _, _ = path_solver(
            self.X,
            self.y,
            self.alphas,
            k_max=it,
            rtol_pattern=1e-10,
            atol_pattern=1e-10,
            rtol_gamma=1e-10,
            split_max=1e1,
            log=0,
        )

        self.w = np.hstack((np.array([0.0]), sol))

    def get_result(self):
        return dict(beta=self.w)
