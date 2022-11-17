from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    prox_isotonic = import_ctx.import_from('utils', 'prox_isotonic')
    prox_fast_stack = import_ctx.import_from('utils', 'prox_fast_stack')


class Solver(BaseSolver):
    name = 'PGD'  # proximal gradient
    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {'prox': ['prox_isotonic', 'prox_fast_stack']}
    references = [
        'I. Daubechies, M. Defrise and C. De Mol, '
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        'vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)',
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = False

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def get_result(self):
        return self.w

    def run(self, callback):
        n_samples, n_features = self.X.shape
        w = np.zeros(n_features)

        if sparse.issparse(self.X):
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / n_samples
        else:
            L = np.linalg.norm(self.X, ord=2) ** 2 / n_samples

        if self.prox == "prox_fast_stack":
            prox_func = prox_fast_stack
        elif self.prox == "prox_isotonic":
            prox_func = prox_isotonic
        else:
            raise ValueError(f"Unsupported prox {self.prox}")

        while callback(w):
            w = prox_func(
                w + self.X.T @ (self.y - self.X @ w) / (L * n_samples),
                self.lambdas / L)
        self.w = w
