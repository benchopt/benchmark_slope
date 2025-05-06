from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import sparse
    from scipy.linalg import cholesky, solve_triangular
    from scipy.sparse.linalg import lsqr
    from sklearn.isotonic import isotonic_regression


class Solver(BaseSolver):
    name = "ADMM"
    sampling_strategy = "callback"
    parameters = {"adaptive_rho": [False, True], "rho": [10, 100, 1000]}

    install_cmd = "conda"
    requirements = ["numpy", "scipy", "scikit-learn"]
    references = [
        "Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2010). "
        "Distributed optimization and statistical learning via the "
        "alternating direction method of multipliers. Foundations and "
        "Trends in Machine Learning, 3(1), 1-122. "
        "https://doi.org/10.1561/2200000016"
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept

    def get_result(self):
        beta = self.w if self.fit_intercept else np.hstack((np.array([0.0]), self.w))
        return dict(beta=beta)

    def run(self, callback):
        # implementation from
        # https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
        # splitting: min f(w) + g(z) subject to z = w + use
        # *scaled* Lagrangian variable u

        lambdas = self.alphas
        alpha = 1.0
        rho = self.rho
        adaptive_rho = self.adaptive_rho
        lsqr_atol = 1e-6
        lsqr_btol = 1e-6

        # parameters
        mu = 10
        tau_incr = 2
        tau_decr = 2

        X = self.X

        n, p = X.shape

        if self.fit_intercept:
            X = self._add_intercept_column(X)
            p += 1

        self.w = np.zeros(p)
        z = np.zeros(p)
        u = np.zeros(p)

        do_lsqr = sparse.issparse(X) and min(n, p) > 1000

        # cache factorizations if dense
        if not do_lsqr:
            if n >= p:
                XtX = X.T @ X
                if sparse.issparse(X):
                    XtX = XtX.toarray()
                np.fill_diagonal(XtX, XtX.diagonal() + rho)
                L = cholesky(XtX, lower=True)
            else:
                XXt = X @ X.T
                if sparse.issparse(X):
                    XXt = XXt.toarray()
                XXt *= 1 / rho
                np.fill_diagonal(XXt, XXt.diagonal() + 1)
                L = cholesky(XXt, lower=True)

            U = L.T

        Xty = X.T @ self.y

        while callback():
            if do_lsqr:
                res = lsqr(
                    sparse.vstack((X, np.sqrt(rho) * sparse.eye(p))),
                    np.hstack((self.y, np.sqrt(rho) * (z - u))),
                    x0=self.w,
                    atol=lsqr_atol,
                    btol=lsqr_btol,
                )
                self.w = res[0]
            else:
                q = Xty + rho * (z - u)

                U = L.T

                if n >= p:
                    self.w = solve_triangular(U, solve_triangular(L, q, lower=True))
                else:
                    tmp = solve_triangular(U, solve_triangular(L, X @ q, lower=True))
                    self.w = q / rho - (X.T @ tmp) / (rho**2)

            z_old = z.copy()
            w_hat = alpha * self.w + (1 - alpha) * z_old

            z = w_hat + u
            z[self.fit_intercept :] = self._prox_isotonic(
                z[self.fit_intercept :], lambdas * (n / rho)
            )

            u += w_hat - z

            if adaptive_rho:
                # update rho
                r_norm = norm(self.w - z)
                s_norm = norm(-rho * (z - z_old))

                rho_old = rho

                if r_norm > mu * s_norm:
                    rho *= tau_incr
                    u /= tau_incr
                elif s_norm > mu * r_norm:
                    rho /= tau_decr
                    u *= tau_decr

                if rho_old != rho and not do_lsqr:
                    # need to refactorize since rho has changed
                    if n >= p:
                        np.fill_diagonal(XtX, XtX.diagonal() + (rho - rho_old))
                        L = cholesky(XtX, lower=True)
                    else:
                        np.fill_diagonal(XXt, XXt.diagonal() - 1)
                        XXt *= rho_old / rho
                        np.fill_diagonal(XXt, XXt.diagonal() + 1)
                        L = cholesky(XXt, lower=True)

                    U = L.T

    def _prox_isotonic(self, beta, lambdas):
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

    def _add_intercept_column(self, X):
        n = X.shape[0]

        if sparse.issparse(X):
            return sparse.hstack((sparse.csc_array(np.ones((n, 1))), X), format="csc")
        else:
            return np.hstack((np.ones((n, 1)), X))
