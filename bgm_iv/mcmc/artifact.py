from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Optional

import numpy as np


class MCMCDrawArtifactError(RuntimeError):
    pass


_ARM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCHEMA = "bgm-mcmc-draw-artifact-v2"


def _validate_arm_id(arm_id: Any) -> str:
    if not isinstance(arm_id, str):
        raise MCMCDrawArtifactError("arm_id must be a non-empty string")
    value = str(arm_id).strip()
    if not _ARM_ID.fullmatch(value):
        raise MCMCDrawArtifactError(
            "arm_id must contain only letters, numbers, dot, underscore or hyphen"
        )
    return value


def _write_synced(path: Path, writer) -> None:
    with path.open("wb") as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())


def save_draw_artifact(
    draws: Any,
    *,
    artifact_root: Any,
    arm_id: str,
    settings: Mapping[str, Any],
    grid: Mapping[str, np.ndarray],
) -> dict[str, Any]:
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
    final_dir = root / arm
    if final_dir.exists():
        raise MCMCDrawArtifactError(
            f"refusing to overwrite existing draw artifact: {final_dir}"
        )

    manifest = {
        "schema_version": _SCHEMA,
        "arm_id": arm,
        "draw_shape": [int(value) for value in latent.shape],
        "draw_dtype": str(latent.dtype),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": json.loads(json.dumps(dict(settings), sort_keys=True, default=str)),
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{arm}.staging-", dir=str(root)))
    try:
        _write_synced(
            staging / "draws.npy",
            lambda handle: np.save(handle, latent, allow_pickle=False),
        )
        _write_synced(
            staging / "grid.npz",
            lambda handle: np.savez(
                handle, **{key: np.asarray(value) for key, value in grid.items()}
            ),
        )
        _write_synced(
            staging / "manifest.json",
            lambda handle: handle.write(
                json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8") + b"\n"
            ),
        )
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
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise MCMCDrawArtifactError(f"draw artifact manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MCMCDrawArtifactError("draw artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA:
        raise MCMCDrawArtifactError("unsupported draw artifact schema")
    draws_path = path.parent / "draws.npy"
    grid_path = path.parent / "grid.npz"
    if not draws_path.is_file() or not grid_path.is_file():
        raise MCMCDrawArtifactError("draw artifact files are missing")
    try:
        draws = np.load(draws_path, allow_pickle=False, mmap_mode=mmap_mode)
        with np.load(grid_path, allow_pickle=False) as stored:
            grid = {key: stored[key] for key in stored.files}
    except (OSError, ValueError) as exc:
        raise MCMCDrawArtifactError("draw artifact is not a valid NumPy archive") from exc
    if [int(value) for value in draws.shape] != manifest.get("draw_shape"):
        raise MCMCDrawArtifactError("draw artifact shape mismatch")
    if draws.dtype != np.dtype(np.float32) or str(draws.dtype) != manifest.get("draw_dtype"):
        raise MCMCDrawArtifactError("draw artifact must use float32 storage")
    loaded = {
        **manifest,
        "artifact_dir": str(path.parent),
        "draws_path": str(draws_path),
        "manifest_path": str(path),
    }
    return draws, grid, loaded


__all__ = [
    "MCMCDrawArtifactError",
    "load_draw_artifact",
    "save_draw_artifact",
]
