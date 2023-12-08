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
    requirements = ["pip:libsvmdata", "sklearn"]

    def __init__(self, dataset="YearPredictionMSD", standardize=True):
        self.dataset = dataset
        self.standardize = standardize

    def get_data(self):
        X, y = fetch_libsvm(self.dataset)
        X, y = preprocess_data(X, y, remove_zerovar=True, standardize=self.standardize)

        return dict(X=X, y=y)
