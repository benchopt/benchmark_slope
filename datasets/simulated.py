from benchopt import BaseDataset, safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    preprocess_data = import_ctx.import_from("utils", "preprocess_data")


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        "n_samples, n_features, n_signals, X_density": [
            (20_000, 1_000, 40, 1.0),
            (500, 200, 20, 1.0),
            (200, 20_000, 20, 1.0),
            (200, 2_000_000, 20, 0.001),
        ],
        "rho": [0, 0.80],
        "standardize": [True, False],
    }

    def __init__(
        self,
        n_samples=10,
        n_features=50,
        n_signals=5,
        X_density=1.0,
        rho=0,
        standardize=True,
        random_state=27,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_signals = n_signals
        self.X_density = X_density
        self.rho = rho
        self.standardize = standardize
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            rho=self.rho,
            density=self.n_signals / self.n_features,
            random_state=self.random_state,
            X_density=self.X_density,
        )

        X, y = preprocess_data(X, y, remove_zerovar=True, standardize=self.standardize)

        return dict(X=X, y=y)
