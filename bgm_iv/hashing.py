"""SHA-256 digests of arrays, JSON payloads and network weights."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_array(value: Any, *, kind: str = "") -> str:
    """Digest of an ndarray (dtype, shape, C-order bytes), optionally prefixed by ``kind``."""
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError(f"{kind or 'array'} cannot have object dtype")
    digest = hashlib.sha256()
    if kind:
        digest.update(kind.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_json(name: str, payload: Mapping[str, Any]) -> str:
    """Digest of ``name`` and the canonical JSON encoding of ``payload``."""
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(name.encode("utf-8") + b"\0" + encoded).hexdigest()


def sha256_weights(network: Any) -> str:
    """Ordered digest over every weight of a Keras model (index, dtype, shape, bytes)."""
    digest = hashlib.sha256()
    weights = tuple(getattr(network, "weights", ()))
    digest.update(np.asarray([len(weights)], dtype="<i8").tobytes())
    for index, weight in enumerate(weights):
        array = np.ascontiguousarray(weight.numpy())
        digest.update(np.asarray([index], dtype="<i8").tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


__all__ = ["require_digest", "sha256_array", "sha256_json", "sha256_weights"]
