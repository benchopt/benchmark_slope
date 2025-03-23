from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from scipy import sparse
    from sklearn.isotonic import isotonic_regression
    from numba import njit

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


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

    def warm_up(self):
        self.run_once()

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
            prox_func = _prox_fast_stack
        elif self.prox == "prox_isotonic":
            prox_func = self._prox_isotonic
        else:
            raise ValueError(f"fUnsupported prox {self.prox}")

        while callback():
            self.w[1:] = prox_func(
                w[1:] + self.X.T @ (self.y - self.X @ w[1:]) / (L * n_samples),
                self.lambdas / L,
            )
            # TODO intercept in gradient + update intercept
            #

    def _prox_isotonic(self, beta, lambdas):
        """Proximal operator of the OWL norm
        dot(lambdas, reversed(sort(abs(beta))))
        Follows description and notation from:
        X. Zeng, M. Figueiredo,
        The ordered weighted L1 norm: Atomic formulation, dual norm,
        and projections.
        eprint http://arxiv.org/abs/1409.4271
        (From pyowl)
        XXX

        Parameters
        ----------
        beta: array
            vector of coefficients
        lambdas: array
            vector of regularization weights

        Returns
        -------
        array
            the result of the proximal operator
        """
        # from https://github.com/svaiter/gslope_oracle_inequality/
        # blob/master/graphslope/core.py
        beta_abs = np.abs(beta)
        ix = np.argsort(beta_abs)[::-1]
        beta_abs = beta_abs[ix]
        # project to K+ (monotone non-negative decreasing cone)
        beta_abs = isotonic_regression(beta_abs - lambdas, y_min=0, increasing=False)

        # undo the sorting
        inv_ix = np.zeros_like(ix)
        inv_ix[ix] = np.arange(len(beta))
        beta_abs = beta_abs[inv_ix]

        return np.sign(beta) * beta_abs


@njit
def _prox_fast_stack(beta, lambdas):
    """Compute the sorted L1 proximal operator.
    Parameters
    ----------
    beta : array
        vector of coefficients
    lambdas : array
        vector of regularization weights
    Returns
    -------
    array
        the result of the proximal operator
    """
    # from https://github.com/jolars/slopecd/blob/main/code/slope/utils.py
    beta_sign = np.sign(beta)
    beta = np.abs(beta)
    ord = np.flip(np.argsort(beta))
    beta = beta[ord]

    p = len(beta)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = beta[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    for j in range(k):
        d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            beta[i] = d

    beta[ord] = beta.copy()
    beta *= beta_sign

    return beta
