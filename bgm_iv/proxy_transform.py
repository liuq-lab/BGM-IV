import contextlib
import operator

import numpy as np
from threadpoolctl import threadpool_limits


def parse_pca_dim(value):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        k = operator.index(value)
    except TypeError:
        raise ValueError(f"pca_dim must be an integer in 1..784, got {value!r}") from None
    if not 1 <= k <= 784:
        raise ValueError(f"pca_dim must be an integer in 1..784, got {value!r}")
    return k


@contextlib.contextmanager
def _single_threaded_blas():
    with threadpool_limits(limits=1):
        yield


def fit_pca(train_proxy, k):
    k = parse_pca_dim(k)
    x = np.asarray(train_proxy, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("training proxy must be a nonempty matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("training proxy must be finite")
    if k > min(x.shape):
        raise ValueError("pca_dim=%d exceeds min(n, d)=%d for training proxy %r" %
                         (k, min(x.shape), x.shape))
    mean = x.mean(axis=0)
    xc = x - mean
    raw_var = float(xc.var(axis=0).mean())
    with _single_threaded_blas():
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
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
    return {"mean": mean, "components": comp, "scale": scale}


def apply_pca(t, proxy):
    x = np.asarray(proxy)
    if x.ndim != 2:
        raise ValueError("proxy must be a matrix")
    if x.shape[1] != len(t["mean"]):
        raise ValueError("proxy width differs from fitted PCA input")
    return ((np.asarray(x, dtype=np.float64) - t["mean"]) @
            t["components"].T * t["scale"]).astype(np.float32)
