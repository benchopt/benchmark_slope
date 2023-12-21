from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse

    from benchmark_utils import prox_fast_stack, prox_isotonic


class Solver(BaseSolver):
    name = "PGD"  # proximal gradient
    sampling_strategy = "callback"

    install_cmd = "conda"
    requirements = ["numpy", "scipy", "numba", "scikit-learn"]

    # any parameter defined here is accessible as a class attribute
    parameters = {"prox": ["prox_isotonic", "prox_fast_stack"]}
    references = [
        "I. Daubechies, M. Defrise and C. De Mol, "
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        "vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)",
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        "vol. 2, no. 1, pp. 183-202 (2009)",
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = fit_intercept

    def skip(self, X, y, alphas, fit_intercept):
        if fit_intercept:
            return True, "Intercept not supported"
        return False, None

    def get_result(self):
        # this doesn't seem to be called
        return dict(beta=self.w)

    def run(self, callback):
        n_samples, n_features = self.X.shape
        self.w = w = np.zeros(n_features + 1)

        if sparse.issparse(self.X):
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / n_samples
        else:
            L = np.linalg.norm(self.X, ord=2) ** 2 / n_samples

        if self.prox == "prox_fast_stack":
            prox_func = prox_fast_stack
        elif self.prox == "prox_isotonic":
            prox_func = prox_isotonic
        else:
            raise ValueError(f"fUnsupported prox {self.prox}")

        while callback():
            self.w[1:] = prox_func(
                w[1:] + self.X.T @ (self.y - self.X @ w[1:]) / (L * n_samples),
                self.lambdas / L,
            )
            # TODO intercept in gradient + update intercept
