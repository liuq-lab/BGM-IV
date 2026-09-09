"""Frozen training-proxy PCA for the Vector PCA-only, no-frontend experiment.

Fit on training covariates only, then reuse the same mean/components/global
scale for every restart and evaluation draw. Never whiten individual scores.
Byte-identical copies live here and in bgm_iv/bgm_iv/proxy_transform.py.
Compatible with Python 3.8 and NumPy 1.18.4; no model-framework dependencies.
"""
import contextlib
import hashlib
import numbers
import os
import re
import warnings

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional in deployment environments
    threadpool_limits = None

VERSION = "vector-pca-2026-09-09"


def parse_pca_dim(value, vector_dim=784):
    """Normalize an explicit dimension or identity marker; reject bool/float."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("", "none", "null"):
            return None
        if not re.fullmatch(r"[+]?[0-9]+", value):
            raise ValueError("pca_dim must be an integer in 1..%d or none" % vector_dim)
        value = int(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("pca_dim must be an integer in 1..%d or none" % vector_dim)
    value = int(value)
    if not 1 <= value <= int(vector_dim):
        raise ValueError("pca_dim must be in 1..%d, got %r" % (vector_dim, value))
    return value


def arm_label(k):
    k = parse_pca_dim(k)
    return "none" if k is None else "pca%d" % k


@contextlib.contextmanager
def _single_threaded_blas():
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            yield
        return
    warnings.warn("WARNING: threadpoolctl unavailable; using BLAS environment limits, "
                  "which may not affect an already initialized BLAS runtime", RuntimeWarning)
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def proxy_sha1(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def rounded_sha1(a, decimals=4):
    return hashlib.sha1(np.ascontiguousarray(
        np.round(np.asarray(a, dtype=np.float64), decimals)).tobytes()).hexdigest()


def fit_pca(train_proxy, k):
    k = parse_pca_dim(k)
    x = np.asarray(train_proxy, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("training proxy must be a nonempty matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("training proxy must be finite")
    if k is None:
        return {"kind": "identity", "k": None, "hash": "identity", "version": VERSION}
    if k > min(x.shape):
        raise ValueError("pca_dim=%d exceeds min(n, d)=%d for training proxy %r" %
                         (k, min(x.shape), x.shape))
    mean = x.mean(axis=0)
    xc = x - mean
    raw_var = float(xc.var(axis=0).mean())
    with _single_threaded_blas():
        _, singular_values, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:k].copy()
    for row in comp:
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0:
            row *= -1
    scores = xc @ comp.T
    score_var = float(scores.var(axis=0).mean())
    if not np.isfinite(raw_var) or not np.isfinite(score_var) or raw_var <= 0 or score_var <= 0:
        raise ValueError("PCA global rescaling requires positive finite proxy/score variance")
    scale = float(np.sqrt(raw_var / score_var))
    return {"kind": "pca", "k": k, "mean": mean, "components": comp,
            "scale": scale, "singular_values": singular_values[:k].copy(),
            "raw_var": raw_var, "version": VERSION,
            "hash": "%s-%s-%.10e" % (_sha16(comp), _sha16(mean), scale)}


def apply_pca(t, proxy):
    x = np.asarray(proxy)
    if x.ndim != 2:
        raise ValueError("proxy must be a matrix")
    if t["kind"] == "identity":
        return np.array(x, dtype=np.float32, copy=True)
    if x.shape[1] != len(t["mean"]):
        raise ValueError("proxy width differs from fitted PCA input")
    return ((np.asarray(x, dtype=np.float64) - t["mean"]) @
            t["components"].T * t["scale"]).astype(np.float32)
