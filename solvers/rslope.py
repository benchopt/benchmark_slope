from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY, SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from rpy2.robjects.packages import isinstalled
    from scipy import sparse

    # SLOPE is not packaged on conda-forge, so install it at import time from
    # r-universe (which serves prebuilt binaries for Windows, macOS, and Linux),
    # falling back to CRAN.
    if not isinstalled("SLOPE"):
        packages.importr("utils").install_packages(
            "SLOPE",
            repos=robjects.StrVector(
                ["https://jolars.r-universe.dev", "https://cloud.r-project.org"]
            ),
        )

    import_rpackages("SLOPE")


class Solver(BaseSolver):
    name = "rSLOPE"

    install_cmd = "conda"
    requirements = [
        "conda-forge::r-base",
        "conda-forge::r-matrix",
        "conda-forge::rpy2",
        "conda-forge::scipy",
    ]
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

        # Convert numpy inputs to R explicitly. rpy2 removed the global
        # numpy2ri.activate(), and a conversion context would also turn the
        # returned SLOPE object into a plain dict, so we convert at the boundary
        # and keep self.fit as an R object.
        X = numpy2ri.numpy2rpy(self.X) if isinstance(self.X, np.ndarray) else self.X

        self.fit = self.slope(
            X,
            numpy2ri.numpy2rpy(self.y),
            intercept=self.fit_intercept,
            scale="none",
            alpha=1.0,
            center=False,
            max_passes=max_passes,
            tol_rel_gap=tol * 0.1,
            tol_infeas=tol,
            tol_rel_coef_change=tol,
            **{"lambda": numpy2ri.numpy2rpy(self.alphas)},
        )

    def get_result(self):
        # SLOPE returns coefficients as a length-one list holding a sparse
        # dgCMatrix (one entry per penalty sequence). Unwrap the list, then
        # build the dense vector from the matrix slots directly. This avoids
        # R-side S4 coercion, which rpy2 does not dispatch reliably.
        coefs = self.fit.rx2("coefficients")
        if tuple(coefs.rclass) == ("list",):
            coefs = coefs[0]

        data = np.asarray(coefs.do_slot("x"))
        indices = np.asarray(coefs.do_slot("i"))
        indptr = np.asarray(coefs.do_slot("p"))
        dim = tuple(int(d) for d in coefs.do_slot("Dim"))
        beta = np.asarray(
            sparse.csc_matrix((data, indices, indptr), shape=dim).todense()
        ).ravel()

        beta = beta if self.fit_intercept else np.hstack((np.array([0.0]), beta))

        return dict(beta=beta)
