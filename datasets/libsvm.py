from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm

    from benchmark_utils import preprocess_data


class Dataset(BaseDataset):
    name = "libsvm"

    parameters = {
        "dataset": [
            "news20.binary",
            "rcv1.binary",
            "real-sim",
            "url",
            "YearPredictionMSD",
        ],
        "standardize": [True, False],
    }

    install_cmd = "conda"
    requirements = ["pip:libsvmdata", "scikit-learn"]

    def __init__(self, dataset="YearPredictionMSD", standardize=True):
        self.dataset = dataset
        self.standardize = standardize

    def get_data(self):
        X, y = fetch_libsvm(self.dataset)

        if self.standardize:
            X = VarianceThreshold().fit_transform(X)

            if sparse.issparse(X):
                X = MaxAbsScaler().fit_transform(X).tocsc()
            else:
                X = StandardScaler().fit_transform(X)

        return dict(X=X, y=y)
