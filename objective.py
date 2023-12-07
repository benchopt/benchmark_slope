# Author: Quentin Klopfenstein
#         Jonas Wallin
#         Johan Larsson

from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import stats


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "SLOPE"
    parameters = {
        "reg": [0.5, 0.1, 0.02],
        "q": [0.2, 0.1, 0.05],
        "fit_intercept": [False],
    }

    def __init__(self, reg, q, fit_intercept):
        self.q = q
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape
        self.alphas = self._get_lambda_seq()

    def compute(self, res):
        intercept, beta = res[0], res[1:]

        X, y = self.X, self.y
        n_samples = X.shape[0]
        # compute residuals
        diff = y - X @ beta - intercept

        # compute primal
        p_obj = 1.0 / (2 * n_samples) * diff @ diff + np.sum(
            self.alphas * np.sort(np.abs(beta))[::-1]
        )

        # compute dual
        theta = diff
        theta /= max(1, self._dual_norm_slope(theta, self.alphas))
        d_obj = (norm(y) ** 2
                 - norm(y - theta * n_samples) ** 2) / (2 * n_samples)

        return dict(value=p_obj, duality_gap=p_obj - d_obj)

    def get_objective(self):
        return dict(
            X=self.X, y=self.y, alphas=self.alphas,
            fit_intercept=self.fit_intercept)

    def _dual_norm_slope(self, theta, alphas):
        Xtheta = np.sort(np.abs(self.X.T @ theta))[::-1]
        taus = 1 / np.cumsum(alphas)
        return np.max(np.cumsum(Xtheta) * taus)

    def _get_lambda_seq(self):
        randnorm = stats.norm(loc=0, scale=1)
        q = self.q
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, self.X.shape[1] + 1) * q / (2 * self.X.shape[1])
        )

        alpha_max = self._dual_norm_slope(
            (self.y - self.fit_intercept * np.mean(self.y)) / len(self.y),
            alphas_seq
        )
        return alpha_max * alphas_seq * self.reg
