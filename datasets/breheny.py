from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os

    import appdirs
    import numpy as np
    from download import download
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from scipy.sparse import csc_array
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def fetch_breheny(dataset: str):
    base_dir = appdirs.user_cache_dir("benchmark_lasso_path")

    path = os.path.join(base_dir, dataset + ".rds")

    # download raw data unless it is stored in data folder already
    if not os.path.isfile(path):
        url = "https://s3.amazonaws.com/pbreheny-data-sets/" + dataset + ".rds"
        download(url, path)

    read_rds = robjects.r["readRDS"]
    numpy2ri.activate()

    data = read_rds(path)
    X = data[0]
    y = data[1]

    density = np.sum(X != 0) / X.size

    if density <= 0.2:
        X = csc_array(X)

    return X, y


class Dataset(BaseDataset):
    name = "breheny"

    parameters = {
        "dataset": ["bcTCGA", "Rhee2006", "Scheetz2006"],
        "standardize": [True, False],
    }

    install_cmd = "conda"
    requirements = ["rpy2", "numpy", "scipy", "appdirs", "r", "pip:scikit-learn"]

    def __init__(self, dataset="bcTCGA", standardize=True):
        self.dataset = dataset
        self.standardize = standardize

    def get_data(self):
        X, y = fetch_breheny(self.dataset)

        if self.standardize:
            X = VarianceThreshold().fit_transform(X)

            if sparse.issparse(X):
                X = MaxAbsScaler().fit_transform(X).tocsc()
            else:
                X = StandardScaler().fit_transform(X)

        return dict(X=X, y=y)
