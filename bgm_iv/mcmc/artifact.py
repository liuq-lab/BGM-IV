"""Fail-closed persistence for complete MCMC production draw tensors."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Optional

import numpy as np


class MCMCDrawArtifactError(RuntimeError):
    """A production-draw artifact violated its persistence contract."""


_ARM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_arm_id(arm_id: Any) -> str:
    if not isinstance(arm_id, str):
        raise MCMCDrawArtifactError("arm_id must be a non-empty string")
    value = str(arm_id).strip()
    if not _ARM_ID.fullmatch(value):
        raise MCMCDrawArtifactError(
            "arm_id must contain only letters, numbers, dot, underscore or hyphen"
        )
    return value


def save_draw_artifact(
    draws: Any,
    *,
    artifact_root: Any,
    arm_id: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one ``[T,C,U,D]`` float32 draw artifact.

    Both files are first written into a staging directory on the destination
    filesystem.  The directory rename is the commit point, so a visible final
    artifact always contains a complete ``draws.npy`` and its hash manifest.
    Existing final artifacts are never overwritten.
    """

    latent = np.asarray(draws)
    if latent.ndim != 4:
        raise MCMCDrawArtifactError("draws must have shape [T,C,U,D]")
    if latent.dtype != np.dtype(np.float32):
        raise MCMCDrawArtifactError("draws must use float32 storage")
    if not np.all(np.isfinite(latent)):
        raise MCMCDrawArtifactError("draws must be finite")
    arm = _validate_arm_id(arm_id)
    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    identity_payload = {
        "arm_id": arm,
        "shape": [int(value) for value in latent.shape],
        "dtype": str(latent.dtype),
        "provenance": dict(provenance),
    }
    identity = hashlib.sha256(
        b"bgm-mcmc-draw-artifact\0" + _canonical_json(identity_payload)
    ).hexdigest()
    final_dir = root / f"{arm}--{identity[:16]}"
    if final_dir.exists():
        raise MCMCDrawArtifactError(
            f"refusing to overwrite existing draw artifact: {final_dir}"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{arm}.staging-", dir=str(root)))
    try:
        draws_path = staging / "draws.npy"
        with draws_path.open("wb") as handle:
            np.save(handle, latent, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        draws_hash = _sha256_file(draws_path)
        stat = draws_path.stat()
        manifest_core = {
            "schema_version": "bgm-mcmc-draw-artifact",
            "artifact_identity": identity,
            "arm_id": arm,
            "draw_file": "draws.npy",
            "draw_sha256": draws_hash,
            "draw_shape": [int(value) for value in latent.shape],
            "draw_dtype": str(latent.dtype),
            "draw_nbytes": int(latent.nbytes),
            "file_nbytes": int(stat.st_size),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": dict(provenance),
        }
        manifest_hash = hashlib.sha256(
            b"bgm-mcmc-draw-manifest\0" + _canonical_json(manifest_core)
        ).hexdigest()
        manifest = {**manifest_core, "manifest_sha256": manifest_hash}
        manifest_path = staging / "manifest.json"
        with manifest_path.open("wb") as handle:
            handle.write(_canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging.rename(final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        **manifest,
        "artifact_dir": str(final_dir),
        "draws_path": str(final_dir / "draws.npy"),
        "manifest_path": str(final_dir / "manifest.json"),
    }


def load_draw_artifact(
    manifest_path: Any,
    *,
    mmap_mode: Optional[str] = "r",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify and load a committed production-draw artifact.

    Verification covers the canonical manifest digest, draw-file digest,
    stored shape/dtype and both logical and physical byte counts.  The default
    memory mapping avoids loading multi-gigabyte production tensors into a
    second resident-memory copy during readout recovery.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise MCMCDrawArtifactError(f"draw artifact manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MCMCDrawArtifactError("draw artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise MCMCDrawArtifactError("draw artifact manifest must be a JSON object")
    if manifest.get("schema_version") != "bgm-mcmc-draw-artifact":
        raise MCMCDrawArtifactError("unsupported draw artifact schema")
    claimed_manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(claimed_manifest_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", claimed_manifest_hash
    ):
        raise MCMCDrawArtifactError("draw artifact manifest hash is missing")
    manifest_core = dict(manifest)
    manifest_core.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        b"bgm-mcmc-draw-manifest\0" + _canonical_json(manifest_core)
    ).hexdigest()
    if actual_manifest_hash != claimed_manifest_hash:
        raise MCMCDrawArtifactError("draw artifact manifest hash mismatch")

    draw_file = manifest.get("draw_file")
    if draw_file != "draws.npy":
        raise MCMCDrawArtifactError("draw artifact file name is invalid")
    draws_path = (path.parent / draw_file).resolve()
    if draws_path.parent != path.parent or not draws_path.is_file():
        raise MCMCDrawArtifactError("draw artifact file is missing")
    file_nbytes = manifest.get("file_nbytes")
    if isinstance(file_nbytes, bool) or not isinstance(file_nbytes, int):
        raise MCMCDrawArtifactError("draw artifact file size is invalid")
    if file_nbytes != int(draws_path.stat().st_size):
        raise MCMCDrawArtifactError("draw artifact file size mismatch")
    draw_sha256 = manifest.get("draw_sha256")
    if not isinstance(draw_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", draw_sha256
    ):
        raise MCMCDrawArtifactError("draw artifact SHA-256 is invalid")
    if _sha256_file(draws_path) != draw_sha256:
        raise MCMCDrawArtifactError("draw artifact SHA-256 mismatch")
    try:
        draws = np.load(draws_path, allow_pickle=False, mmap_mode=mmap_mode)
    except (OSError, ValueError) as exc:
        raise MCMCDrawArtifactError("draw artifact is not a valid NumPy array") from exc
    expected_shape = manifest.get("draw_shape")
    if (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in expected_shape
        )
    ):
        raise MCMCDrawArtifactError("draw artifact manifest shape is invalid")
    if [int(value) for value in draws.shape] != expected_shape:
        raise MCMCDrawArtifactError("draw artifact shape mismatch")
    if str(draws.dtype) != manifest.get("draw_dtype"):
        raise MCMCDrawArtifactError("draw artifact dtype mismatch")
    if draws.dtype != np.dtype(np.float32):
        raise MCMCDrawArtifactError("draw artifact must use float32 storage")
    draw_nbytes = manifest.get("draw_nbytes")
    if isinstance(draw_nbytes, bool) or not isinstance(draw_nbytes, int):
        raise MCMCDrawArtifactError("draw artifact logical byte count is invalid")
    if int(draws.nbytes) != draw_nbytes:
        raise MCMCDrawArtifactError("draw artifact logical byte count mismatch")
    manifest_arm = _validate_arm_id(manifest.get("arm_id"))
    if manifest_arm != manifest.get("arm_id"):
        raise MCMCDrawArtifactError("draw artifact arm_id is not canonical")
    identity_payload = {
        "arm_id": manifest_arm,
        "shape": expected_shape,
        "dtype": str(draws.dtype),
        "provenance": manifest.get("provenance"),
    }
    expected_identity = hashlib.sha256(
        b"bgm-mcmc-draw-artifact\0" + _canonical_json(identity_payload)
    ).hexdigest()
    if manifest.get("artifact_identity") != expected_identity:
        raise MCMCDrawArtifactError("draw artifact identity mismatch")
    verified = {
        **manifest,
        "artifact_dir": str(path.parent),
        "draws_path": str(draws_path),
        "manifest_path": str(path),
        "verified": True,
    }
    return draws, verified


__all__ = [
    "MCMCDrawArtifactError",
    "load_draw_artifact",
    "save_draw_artifact",
]
