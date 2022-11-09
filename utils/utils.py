from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def preprocess_data(X, y=None, remove_zerovar=True, standardize=True):
    if remove_zerovar:
        X = VarianceThreshold().fit_transform(X)

    if standardize:
        if sparse.issparse(X):
            X = MaxAbsScaler().fit_transform(X).tocsc()
        else:
            X = StandardScaler().fit_transform(X)

    return X, y
