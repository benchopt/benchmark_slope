import ssl
import warnings
from contextlib import contextmanager
from threading import Lock
from urllib import request

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_dataset
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler


_https_context_lock = Lock()


def _is_certificate_verification_error(error):
    pending = [error]
    seen = set()

    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, ssl.SSLCertVerificationError):
            return True

        for related in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
            getattr(current, "reason", None),
        ):
            if isinstance(related, BaseException):
                pending.append(related)

    return False


@contextmanager
def _without_x509_strict():
    strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if not strict:
        yield
        return

    context = ssl.create_default_context()
    context.verify_flags &= ~strict
    opener = request.build_opener(request.HTTPSHandler(context=context))

    # The downloader does not accept an SSLContext and urllib caches its
    # opener, so this scoped override is needed for legacy trust anchors.
    with _https_context_lock:
        default_opener = request._opener
        request.install_opener(opener)
        try:
            yield
        finally:
            request._opener = default_opener


def _fetch_libsvm(dataset):
    try:
        return fetch_dataset(dataset)
    except RuntimeError as error:
        if not _is_certificate_verification_error(error):
            raise

    warnings.warn(
        "The LIBSVM server certificate failed strict X.509 validation; "
        "retrying with standard certificate and hostname verification.",
        RuntimeWarning,
        stacklevel=2,
    )
    with _without_x509_strict():
        return fetch_dataset(dataset)


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
    requirements = ["pip::libsvmdata>=0.5", "scikit-learn"]

    def __init__(self, dataset="YearPredictionMSD", standardize=True):
        super().__init__()
        self.dataset = dataset
        self.standardize = standardize

    def get_data(self):
        X, y = _fetch_libsvm(self.dataset)

        if self.standardize:
            X = VarianceThreshold().fit_transform(X)

            if sparse.issparse(X):
                X = MaxAbsScaler().fit_transform(X).tocsc()
            else:
                X = StandardScaler().fit_transform(X)

        return dict(X=X, y=y)
