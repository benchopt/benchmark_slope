from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from scipy import sparse
    from sklearn.isotonic import isotonic_regression

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


class Solver(BaseSolver):
    name = "PGD"  # proximal gradient
    sampling_strategy = "callback"

    install_cmd = "conda"
    requirements = ["numpy", "scipy", "numba", "scikit-learn"]

    # any parameter defined here is accessible as a class attribute
    parameters = {
        "prox": ["prox_isotonic", "prox_fast_stack"],
        "acceleration": ["none", "fista", "bb", "anderson"],
    }
    references = [
        "I. Daubechies, M. Defrise and C. De Mol, "
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        "vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)",
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        "vol. 2, no. 1, pp. 183-202 (2009)",
        "Barzilai, J., & Borwein, J. M. (1988). "
        "Two-point step size gradient methods. IMA Journal of Numerical "
        "Analysis, 8(1), 141–148. https://doi.org/10.1093/imanum/8.1.141",
        "Anderson, D. G. (1965). Iterative procedures for nonlinear "
        "integral equations. J. ACM, 12(4), 547–560. "
        "https://doi.org/10.1145/321296.321305",
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = fit_intercept

    def warm_up(self):
        self.run_once()

    def get_result(self):
        # this doesn't seem to be called
        return dict(beta=self.w)

    def run(self, callback):
        n_samples, n_features = self.X.shape
        self.w = np.zeros(n_features + 1)

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

        if self.acceleration == "fista":
            # FISTA acceleration
            z = np.zeros_like(self.w[1:])
            t = 1.0
            while callback():
                w_prev = self.w[1:].copy()

                # Calculate residuals based on extrapolated point z
                residuals = self.y - self.X @ z - self.w[0]

                # Update weights using proximal operator
                self.w[1:] = prox_func(
                    z + self.X.T @ residuals / (L * n_samples),
                    self.lambdas / L,
                )

                # Update FISTA parameters
                t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
                z = self.w[1:] + ((t - 1) / t_next) * (self.w[1:] - w_prev)
                t = t_next

        elif self.acceleration == "bb":
            # Barzilai-Borwein stepsize
            w_old = self.w[1:].copy()
            grad_old = np.zeros_like(w_old)

            while callback():
                residuals = self.y - self.X @ self.w[1:] - self.w[0]
                grad = -self.X.T @ residuals / n_samples

                # BB stepsize calculation
                s = self.w[1:] - w_old
                y = grad - grad_old
                ss = np.dot(s, s)
                sy = np.dot(s, y)

                if sy > 1e-10:
                    step = ss / sy  # BB1 variant
                else:
                    # Fallback to fixed step size if sy is too small
                    step = 1.0 / L

                # Store current values for next iteration
                w_old[:] = self.w[1:]
                grad_old[:] = grad

                # Apply proximal step
                self.w[1:] = prox_func(self.w[1:] - step * grad, self.lambdas * step)

        elif self.acceleration == "anderson":
            # Anderson acceleration
            K = 5
            last_K_w = np.zeros([K + 1, n_features])
            U = np.zeros([K, n_features])
            it = 0

            while callback():
                residuals = self.y - self.X @ self.w[1:] - self.w[0]
                w_new = prox_func(
                    self.w[1:] + self.X.T @ residuals / (L * n_samples),
                    self.lambdas / L,
                )

                if it < K + 1:
                    last_K_w[it] = w_new
                else:
                    for k in range(K):
                        last_K_w[k] = last_K_w[k + 1]

                    last_K_w[K] = w_new

                    for k in range(K):
                        U[k] = last_K_w[k + 1] - last_K_w[k]

                    C = np.dot(U, U.T)

                    try:
                        coefs = np.linalg.solve(C, np.ones(K))
                        c = coefs / coefs.sum()
                        w_acc = np.sum(last_K_w[:-1] * c[:, None], axis=0)

                        p_obj = np.linalg.norm(
                            self.y - self.X @ w_new - self.w[0]
                        ) ** 2 / (2 * n_samples) + np.sum(
                            self.lambdas * np.sort(np.abs(w_new))[::-1]
                        )
                        p_obj_acc = np.linalg.norm(
                            self.y - self.X @ w_acc - self.w[0]
                        ) ** 2 / (2 * n_samples) + np.sum(
                            self.lambdas * np.sort(np.abs(w_acc))[::-1]
                        )

                        if p_obj_acc < p_obj:
                            w_new = w_acc

                    except np.linalg.LinAlgError:
                        pass

                self.w[1:] = w_new
                it += 1

        else:
            # Standard PGD
            while callback():
                residuals = self.y - self.X @ self.w[1:] - self.w[0]

                self.w[1:] = prox_func(
                    self.w[1:] + self.X.T @ residuals / (L * n_samples),
                    self.lambdas / L,
                )

        if self.fit_intercept:
            self.w[0] += np.mean(residuals)

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
