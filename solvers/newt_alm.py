from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import copy
    import warnings

    import numpy as np
    from numba import njit
    from scipy import sparse
    from scipy.linalg import cho_factor, cho_solve, solve
    from scipy.sparse.linalg import cg, spsolve

if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


class Solver(BaseSolver):
    name = "Newt-ALM"
    sampling_strategy = "callback"

    install_cmd = "conda"
    requirements = ["numpy", "scipy", "numba"]

    parameters = {"inner_solver": ["auto", "standard", "woodbury", "cg"]}
    references = [
        "Luo, Z., Sun, D., Toh, K.-C., & Xiu, N. (2019). Solving the OSCAR and "
        "SLOPE models using a semismooth Newton-based augmented Lagrangian method. "
        "Journal of Machine Learning Research, 20(106), 1–25."
    ]

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.lambdas = X, y, alphas
        self.fit_intercept = fit_intercept

    def warm_up(self):
        self.run_once()

    def get_result(self):
        return dict(beta=self.w)

    @staticmethod
    def get_next(previous):
        # Linear growth for number of iterations
        return previous + 1

    def run(
        self,
        callback=None,
    ):
        if self.inner_solver not in ["auto", "standard", "woodbury", "cg"]:
            raise ValueError("`solver` must be one of auto, standard, woodbury, and cg")

        line_search_param = {"mu": 0.2, "delta": 0.5, "beta": 2}
        local_param = {"epsilon": 1.0, "delta": 1.0, "delta_prime": 1.0, "sigma": 0.5}
        max_inner_it = 50

        A = self.X
        b = self.y

        m, n = A.shape

        self.w = np.zeros(n + 1)

        if sparse.issparse(A):
            L = sparse.linalg.svds(A, k=1)[1][0] ** 2
        else:
            L = np.linalg.norm(A, ord=2) ** 2

        local_param["sigma"] = min(1.0, 1 / np.sqrt(L))

        if self.fit_intercept:
            A = _add_intercept_column(A)
            n += 1

        lambdas = self.lambdas.copy() * m

        x = np.zeros(n)
        y = np.zeros(m)

        ATy = A.T @ y

        while callback():
            # step 1
            local_param["delta_prime"] *= 0.999
            local_param["epsilon"] *= 0.9
            local_param["delta"] *= 0.999

            x_old = x.copy()

            for j in range(max_inner_it):
                converged, x, y, ATy = self._inner_step(
                    A,
                    b,
                    x_old,
                    y,
                    ATy,
                    lambdas,
                    x_old,
                    local_param,
                    line_search_param,
                )
                if converged:
                    break

                if j == max_inner_it - 1:
                    warnings.warn("The inner solver did not converge.")

            # step 3, update sigma
            # TODO: The paper says nothing about how sigma is updated except
            # that it is always increased.
            local_param["sigma"] *= 1.1

            if self.fit_intercept:
                self.w = x.copy()
            else:
                self.w[1:] = x.copy()

    def _inner_step(
        self,
        A,
        b,
        x,
        y,
        ATy,
        lambdas,
        x_old,
        local_param,
        line_search_param,
    ):
        sigma = local_param["sigma"]

        d, nabla_psi = self._compute_direction(x_old, sigma, A, b, y, ATy, lambdas)
        ATd = A.T @ d
        alpha = _line_search(
            y,
            d,
            x_old,
            ATy,
            ATd,
            b,
            lambdas,
            sigma,
            nabla_psi,
            line_search_param["delta"],
            line_search_param["mu"],
            line_search_param["beta"],
            self.fit_intercept,
        )

        # step 1c, update y
        y += alpha * d
        ATy += alpha * ATd

        # step 2, update x
        x = x_old - sigma * ATy
        x[self.fit_intercept :] = _prox_slope(x[self.fit_intercept :], sigma * lambdas)

        # check for convergence
        x_diff_norm = np.linalg.norm(x - x_old)
        converged = self._check_convegence(
            x_diff_norm,
            nabla_psi,
            local_param["epsilon"],
            sigma,
            local_param["delta"],
            local_param["delta_prime"],
        )
        return converged, x, y, ATy

    def _build_W(self, x_tilde, sigma, lambdas, A):
        m = A.shape[0]

        ord = np.argsort(np.abs(x_tilde[self.fit_intercept :]))[::-1]
        x_lambda = np.abs(_prox_slope(x_tilde[self.fit_intercept :], lambdas)[ord])

        z = _BBT_inv_B(np.abs(x_tilde[self.fit_intercept :][ord]) - lambdas - x_lambda)

        Gamma = np.where(np.logical_and(z != 0, _B(x_lambda) == 0))[0]
        GammaC = np.setdiff1d(np.arange(len(x_lambda)), Gamma)

        nC = len(GammaC)

        pi_list, _ = _build_pi(x_tilde[self.fit_intercept :])

        if sparse.issparse(A):
            W_row, W_col, W_data = _assemble_sparse_W(
                nC, GammaC, pi_list, A.data, A.indices, A.indptr, m, self.fit_intercept
            )
            # we use CSR here because transpose later on makes it CSC, which
            # is what we really want.
            W = sparse.coo_matrix(
                (W_data, (W_row, W_col)), shape=(m, nC + self.fit_intercept)
            ).tocsr()
        else:
            W = _assemble_dense_W(nC, GammaC, pi_list, A, self.fit_intercept)

        return np.sqrt(sigma) * W

    def _compute_direction(self, x, sigma, A, b, y, ATy, lambdas):
        x_tilde = x / sigma - ATy

        x_tilde_prox = x - sigma * ATy
        x_tilde_prox[self.fit_intercept :] = _prox_slope(
            x_tilde_prox[self.fit_intercept :], sigma * lambdas
        )

        nabla_psi = y + b - A @ x_tilde_prox

        W = self._build_W(x_tilde, sigma, lambdas, A)

        m, r1_plus_r2 = W.shape

        inner_solver = copy.deepcopy(self.inner_solver)

        if inner_solver == "auto":
            if r1_plus_r2 <= 100 * m:
                inner_solver = "woodbury"
            elif m > 10_000 and m / r1_plus_r2 > 0.1:
                inner_solver = "cg"
            else:
                inner_solver = "standard"

        if inner_solver == "woodbury":
            # Woodbury factorization solver
            WTW = W.T @ W
            if sparse.issparse(A):
                V_inv = sparse.eye(m, format="csc") - W @ spsolve(
                    sparse.eye(r1_plus_r2, format="csc") + WTW, W.T
                )
            else:
                np.fill_diagonal(WTW, WTW.diagonal() + 1)
                V_inv = np.eye(m) - W @ solve(WTW, W.T)

            d = V_inv @ (-nabla_psi)
        elif inner_solver == "cg":
            # Conjugate gradient
            V = W @ W.T
            if sparse.issparse(A):
                V += sparse.eye(m, format="csc")
            else:
                np.fill_diagonal(V, V.diagonal() + 1)

            # preconditioner
            M = sparse.diags(V.diagonal())

            d, _ = cg(V, -nabla_psi, M=M)
        else:
            V = W @ W.T
            if sparse.issparse(A):
                V = sparse.eye(V.shape[0]) + W @ W.T
                d = spsolve(V, -nabla_psi)
            else:
                np.fill_diagonal(V, V.diagonal() + 1)
                d = cho_solve(cho_factor(V), -nabla_psi)

        return d, nabla_psi

    def _check_convegence(
        self, x_diff_norm, nabla_psi, epsilon_k, sigma, delta_k, delta_prime_k
    ):
        # check for convergence
        norm_nabla_psi = np.linalg.norm(nabla_psi)

        eps = np.sqrt(np.finfo(float).eps)

        a = epsilon_k / np.sqrt(sigma) + eps
        b1 = (delta_k / np.sqrt(sigma)) * x_diff_norm + eps
        b2 = (delta_prime_k / sigma) * x_diff_norm + eps

        crit_A = norm_nabla_psi <= a
        crit_B1 = norm_nabla_psi <= b1
        crit_B2 = norm_nabla_psi <= b2

        return crit_A and crit_B1 and crit_B2


@njit
def _build_psi(y, x, ATy, b, lambdas, sigma, fit_intercept):
    w = x - sigma * ATy
    u = np.zeros(len(w))
    u[fit_intercept:] = (1 / sigma) * (
        w[fit_intercept:] - _prox_slope(w[fit_intercept:], lambdas * sigma)
    )
    if fit_intercept:
        u[0] = 0.0

    phi = 0.5 * np.linalg.norm(u - w / sigma) ** 2

    return (
        0.5 * np.linalg.norm(y) ** 2
        + b @ y
        - (0.5 / sigma) * np.linalg.norm(x) ** 2
        + sigma * phi
    )


@njit
def _line_search(
    y, d, x, ATy, ATd, b, lambdas, sigma, nabla_psi, delta, mu, beta, fit_intercept
):
    # step 1b, line search
    mj = 0

    psi0 = _build_psi(y, x, ATy, b, lambdas, sigma, fit_intercept)
    nabla_psi_d = nabla_psi @ d
    while True:
        alpha = delta**mj
        lhs = _build_psi(
            y + alpha * d, x, ATy + alpha * ATd, b, lambdas, sigma, fit_intercept
        )
        rhs = psi0 + mu * alpha * nabla_psi_d

        if lhs <= rhs:
            break

        mj = 1 if mj == 0 else mj * beta

    return alpha


@njit
def _nonzero_sign(x):
    n = len(x)
    out = np.empty(n)

    for i in range(n):
        s = np.sign(x[i])
        out[i] = s if s != 0 else 0

    return out


def _permutation_matrix(x):
    n = len(x)

    signs = _nonzero_sign(x)
    order = np.argsort(np.abs(x))[::-1]

    pi = sparse.lil_matrix((n, n), dtype=int)

    for j, ord_j in enumerate(order):
        pi[j, ord_j] = signs[ord_j]

    return sparse.csc_array(pi)


# build the signedpermutation object
@njit
def _build_pi(x):
    n = len(x)
    pi_list = np.empty((n, 2), dtype=np.int64)
    piT_list = np.empty((n, 2), dtype=np.int64)
    signs = _nonzero_sign(x)
    order = np.argsort(np.abs(x))[::-1]

    for j, ord_j in enumerate(order):
        pi_list[j, 0] = ord_j
        pi_list[j, 1] = signs[ord_j]
        piT_list[ord_j, 0] = j
        piT_list[ord_j, 1] = signs[ord_j]

    return pi_list, piT_list


# multiplaction of signed permuation
@njit
def _pix(x, pi_list):
    return x[pi_list[:, 0]] * pi_list[:, 1]


# inverse of the matrix B.T
def _BTinv(x):
    return np.cumsum(x)


# inverse of the matrix B
@njit
def _Binv(x):
    return np.cumsum(x[::-1])[::-1]


@njit
def _B(x):
    y = x.copy()
    y[:-1] -= x[1:]
    return y


# Bt^-1B^-1
# returns (BBt) ^-1 B x
def _BBT_inv_B(x):
    return _BTinv(x)


@njit
def _assemble_sparse_W(
    nC, GammaC, pi_list, A_data, A_indices, A_indptr, m, fit_intercept
):
    W_row = []
    W_col = []
    W_data = []

    start = 0

    if fit_intercept:
        for i in range(m):
            W_col.append(0)
            W_row.append(i)
            W_data.append(1.0)

    for i in range(nC):
        nCi = GammaC[i] + 1 - start
        for j in range(start, GammaC[i] + 1):
            k = pi_list[j, 0]
            pi_list_j1 = pi_list[j, 1]
            for ind in range(
                A_indptr[k + fit_intercept], A_indptr[k + 1 + fit_intercept]
            ):
                W_row.append(A_indices[ind])
                W_col.append(i + fit_intercept)
                val = pi_list_j1 * A_data[ind]
                if nCi > 1:
                    val /= np.sqrt(nCi)
                W_data.append(val)
        start = GammaC[i] + 1

    return np.array(W_row), np.array(W_col), np.array(W_data)


@njit
def _assemble_dense_W(nC, GammaC, pi_list, A, fit_intercept):
    m = A.shape[0]

    W = np.zeros((m, nC + fit_intercept))

    if fit_intercept:
        W[:, 0] = np.ones(m, dtype=np.float64)

    start = 0
    for i in range(nC):
        nCi = GammaC[i] + 1 - start
        for j in range(start, GammaC[i] + 1):
            W[:, i + fit_intercept] += (
                pi_list[j, 1] * A[:, pi_list[j, 0] + fit_intercept]
            )
        if nCi > 1:
            W[:, i + fit_intercept] /= np.sqrt(nCi)
        start = GammaC[i] + 1

    return W


@njit
def _prox_slope(beta, lambdas):
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


def _add_intercept_column(X):
    n = X.shape[0]

    if sparse.issparse(X):
        return sparse.hstack((sparse.csc_array(np.ones((n, 1))), X), format="csc")
    else:
        return np.hstack((np.ones((n, 1)), X))
