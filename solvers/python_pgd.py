from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from numpy.linalg import norm
    prox_l1sorted = import_ctx.import_from('utils', 'prox_l1sorted')
    prox_slope = import_ctx.import_from('utils', 'prox_slope')


class Solver(BaseSolver):
    name = 'Slope-PGD'  # proximal gradient
    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {'prox': ['prox_l1sorted', 'prox_slope']}
    references = [
        'I. Daubechies, M. Defrise and C. De Mol, '
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        'vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)',
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def skip(self, X, y, alphas):
        fit_intercept = False
        # XXX - not implemented but not too complicated to implement
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = False

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def get_result(self):
        return self.w

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        return L

    def run(self, callback):
        fit_intercept = False
        n_samples, n_features = self.X.shape
        R = self.y.copy()
        w = np.zeros(n_features)
        intercept = 0.0

        z = w.copy()
        t = 1

        if sparse.issparse(self.X):
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / n_samples
        else:
            L = norm(self.X, ord=2) ** 2 / n_samples

        it = 0
        if fit_intercept:
            w = np.hstack((intercept, w))
        while callback(w):
            if self.prox == 'prox_l1sorted':
                w_new = prox_l1sorted(z + (self.X.T @ R) / (L * n_samples),
                                      self.lambdas / L)
            elif self.prox == 'prox_slope':
                w_new = prox_slope(z + (self.X.T @ R) / (L * n_samples),
                                        self.lambdas / L)

            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z = w_new + (t - 1) / t_new * (w_new - w)
            w = w_new
            t = t_new
            R[:] = self.y - self.X @ z
            R -= intercept
            it += 1
        self.w = w
