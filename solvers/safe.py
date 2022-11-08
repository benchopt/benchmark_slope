from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from slopescreening.solver.parameters import SlopeParameters
    from slopescreening.solver.slope import slope_gp


class Solver(BaseSolver):
    name = "Safe"
    stopping_strategy = "iteration"
    install_cmd = "conda"
    requirements = [
        "pip:git+https://gitlab-research.centralesupelec.fr/2020elvirac/slope-screening@master"
    ]
    references = [
        "C. Elvira and C. Herzet, “Safe rules for the identification of zeros in the solutions of the SLOPE problem.” arXiv, Oct. 04, 2022. doi: 10.48550/arXiv.2110.11784."
    ]
    parameters = {
        "accelerated": [True, False],
    }

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.n_samples = len(y)
        self.fit_intercept = fit_intercept

        self.param = SlopeParameters()
        self.param.accelerated = False
        self.param.gap_stopping = 0
        self.param.eval_gap_it = 10
        self.param.verbose = False

    def skip(self, X, y, alphas, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle intercept fitting"

        if sparse.issparse(X):
            return True, f"{self.name} does not handle sparse design matrices"

        return False, None

    def run(self, it):
        self.param.accelerated = self.accelerated
        self.param.max_it = it

        self.coef_ = slope_gp(
            self.y, self.X, 1.0, self.alphas * self.n_samples, self.param
        )["sol"]

    def get_result(self):
        return np.hstack((np.array([0.0]), self.coef_))
