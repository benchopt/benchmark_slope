import importlib.util
import ssl
from pathlib import Path
from urllib import request
from urllib.error import URLError

import pytest


MODULE_PATH = Path(__file__).parents[1] / "datasets" / "libsvm.py"
SPEC = importlib.util.spec_from_file_location("slope_libsvm_dataset", MODULE_PATH)
libsvm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(libsvm)


def certificate_download_error():
    try:
        raise ssl.SSLCertVerificationError(
            1,
            "certificate verify failed: Missing Subject Key Identifier",
        )
    except ssl.SSLCertVerificationError as error:
        return RuntimeError("Dataset fetching aborted."), URLError(error)


def test_fetch_retries_certificate_error_without_x509_strict(monkeypatch):
    strict = ssl.VERIFY_X509_STRICT
    calls = []
    create_default_context = ssl.create_default_context

    def default_https_context():
        context = create_default_context()
        context.verify_flags |= strict
        return context

    strict_context = default_https_context()
    strict_opener = request.build_opener(
        request.HTTPSHandler(context=strict_context)
    )

    def fetch_dataset(dataset):
        handler = next(
            handler
            for handler in request._opener.handlers
            if isinstance(handler, request.HTTPSHandler)
        )
        context = handler._context
        calls.append((dataset, context.verify_flags))
        if context.verify_flags & strict:
            error, cause = certificate_download_error()
            raise error from cause
        return "X", "y"

    monkeypatch.setattr(ssl, "create_default_context", default_https_context)
    monkeypatch.setattr(request, "_opener", strict_opener)
    monkeypatch.setattr(libsvm, "fetch_dataset", fetch_dataset)

    with pytest.warns(RuntimeWarning, match="failed strict X.509 validation"):
        assert libsvm._fetch_libsvm("rcv1.binary") == ("X", "y")
    assert calls[0][0] == calls[1][0] == "rcv1.binary"
    assert calls[0][1] & strict
    assert not calls[1][1] & strict
    assert calls[0][1] & ~strict == calls[1][1]
    assert request._opener is strict_opener


def test_fetch_does_not_retry_unrelated_errors(monkeypatch):
    calls = 0

    def fetch_dataset(dataset):
        nonlocal calls
        calls += 1
        raise RuntimeError("The dataset is unavailable.")

    monkeypatch.setattr(libsvm, "fetch_dataset", fetch_dataset)

    with pytest.raises(RuntimeError, match="dataset is unavailable"):
        libsvm._fetch_libsvm("rcv1.binary")

    assert calls == 1


def test_fetch_restores_opener_when_retry_fails(monkeypatch):
    calls = 0
    original_opener = request.build_opener()

    def fetch_dataset(dataset):
        nonlocal calls
        calls += 1
        error, cause = certificate_download_error()
        raise error from cause

    monkeypatch.setattr(request, "_opener", original_opener)
    monkeypatch.setattr(libsvm, "fetch_dataset", fetch_dataset)

    with pytest.warns(RuntimeWarning, match="failed strict X.509 validation"):
        with pytest.raises(RuntimeError, match="Dataset fetching aborted"):
            libsvm._fetch_libsvm("rcv1.binary")

    assert calls == 2
    assert request._opener is original_opener
