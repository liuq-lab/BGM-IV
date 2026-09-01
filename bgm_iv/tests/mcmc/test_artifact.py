"""Atomic MCMC production-draw artifact tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from bgm_iv.mcmc.artifact import (
    MCMCDrawArtifactError,
    load_draw_artifact,
    save_draw_artifact,
)


def test_draw_artifact_roundtrip_and_hash(tmp_path):
    draws = np.arange(5 * 4 * 3 * 2, dtype=np.float32).reshape(5, 4, 3, 2)
    artifact = save_draw_artifact(
        draws,
        artifact_root=tmp_path,
        arm_id="w2000-d5000",
        provenance={"checkpoint_identity": "checkpoint", "seed": 17},
    )

    loaded = np.load(artifact["draws_path"], allow_pickle=False)
    np.testing.assert_array_equal(loaded, draws)
    with open(artifact["draws_path"], "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == artifact["draw_sha256"]
    with open(artifact["manifest_path"], encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["draw_shape"] == [5, 4, 3, 2]
    assert manifest["draw_dtype"] == "float32"
    assert manifest["draw_sha256"] == artifact["draw_sha256"]
    assert not list(tmp_path.glob(".*.staging-*"))

    mapped, verified = load_draw_artifact(artifact["manifest_path"])
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, draws)
    assert verified["verified"] is True
    assert verified["manifest_sha256"] == artifact["manifest_sha256"]


def test_draw_artifact_refuses_overwrite_and_unsafe_arm(tmp_path):
    draws = np.ones((2, 4, 1, 1), np.float32)
    kwargs = {
        "artifact_root": tmp_path,
        "arm_id": "arm",
        "provenance": {"checkpoint_identity": "checkpoint"},
    }
    save_draw_artifact(draws, **kwargs)
    with pytest.raises(MCMCDrawArtifactError, match="overwrite"):
        save_draw_artifact(draws, **kwargs)
    with pytest.raises(MCMCDrawArtifactError, match="arm_id"):
        save_draw_artifact(
            draws,
            artifact_root=tmp_path,
            arm_id="../escape",
            provenance={},
        )


@pytest.mark.parametrize(
    "draws,message",
    [
        (np.ones((2, 4, 1), np.float32), "shape"),
        (np.ones((2, 4, 1, 1), np.float64), "float32"),
        (np.full((2, 4, 1, 1), np.nan, np.float32), "finite"),
    ],
)
def test_draw_artifact_rejects_invalid_tensor(tmp_path, draws, message):
    with pytest.raises(MCMCDrawArtifactError, match=message):
        save_draw_artifact(
            draws,
            artifact_root=tmp_path,
            arm_id="arm",
            provenance={},
        )


def test_draw_artifact_loader_rejects_manifest_and_draw_corruption(tmp_path):
    draws = np.ones((2, 4, 1, 1), np.float32)
    first = save_draw_artifact(
        draws,
        artifact_root=tmp_path,
        arm_id="manifest-corruption",
        provenance={"checkpoint_identity": "checkpoint"},
    )
    with open(first["manifest_path"], "r+", encoding="utf-8") as handle:
        manifest = json.load(handle)
        manifest["draw_shape"] = [3, 4, 1, 1]
        handle.seek(0)
        json.dump(manifest, handle)
        handle.truncate()
    with pytest.raises(MCMCDrawArtifactError, match="manifest hash mismatch"):
        load_draw_artifact(first["manifest_path"])

    second = save_draw_artifact(
        draws,
        artifact_root=tmp_path,
        arm_id="draw-corruption",
        provenance={"checkpoint_identity": "checkpoint"},
    )
    with open(second["draws_path"], "r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(MCMCDrawArtifactError, match="SHA-256 mismatch"):
        load_draw_artifact(second["manifest_path"])
