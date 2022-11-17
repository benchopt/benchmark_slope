from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY, SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from scipy import sparse

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages("SLOPE")


class Solver(BaseSolver):
    name = "rSLOPE"

    install_cmd = "conda"
    requirements = ["r-base", "rpy2", "r:r-slope", "r-matrix", "scipy"]
    references = [
        "M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candès, ",
        "“SLOPE – adaptive variable selection via convex optimization,” ",
        "Ann Appl Stat, vol. 9, no. 3, pp. 1103–1140, Sep. 2015, ",
        "doi: 10.1214/15-AOAS842.",
    ]
    support_sparse = True

    stopping_criterion = SufficientProgressCriterion(
        patience=5, eps=1e-38, strategy="tolerance"
    )

    def set_objective(self, X, y, alphas, fit_intercept):
        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
            X = X.tocoo()
            self.X = r_Matrix.sparseMatrix(
                i=robjects.IntVector(X.row + 1),
                j=robjects.IntVector(X.col + 1),
                x=robjects.FloatVector(X.data),
                dims=robjects.IntVector(X.shape),
            )
        else:
            self.X = X
        self.y, self.alphas = y, alphas
        self.fit_intercept = fit_intercept

        self.slope = robjects.r["SLOPE"]

    def run(self, tol):
        if tol == INFINITY:
            max_passes = 1
            tol = 1
        else:
            max_passes = 1_000_000

        fit_dict = {"lambda": self.alphas}

        self.fit = self.slope(
            self.X,
            self.y,
            intercept=self.fit_intercept,
            scale="none",
            alpha=1.0,
            center=False,
            max_passes=max_passes,
            tol_rel_gap=tol * 0.1,
            tol_infeas=tol,
            tol_rel_coef_change=tol,
            **fit_dict,
        )

    def get_result(self):
        results = dict(zip(self.fit.names, list(self.fit)))
        r_as = robjects.r["as"]
        coefs = np.array(r_as(results["coefficients"], "vector"))

        return coefs if self.fit_intercept else np.hstack((np.array([0.0]), coefs))
