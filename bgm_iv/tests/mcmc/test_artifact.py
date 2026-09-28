from __future__ import annotations

import json

import numpy as np
import pytest

from bgm_iv.mcmc.artifact import (
    MCMCDrawArtifactError,
    load_draw_artifact,
    save_draw_artifact,
)

GRID = {
    "query_x": np.array([[0.1], [0.2]], np.float32),
    "truth": np.array([1.0, 2.0], np.float64),
}


def test_draw_artifact_roundtrip(tmp_path):
    draws = np.arange(5 * 4 * 3 * 2, dtype=np.float32).reshape(5, 4, 3, 2)
    artifact = save_draw_artifact(
        draws,
        artifact_root=tmp_path,
        arm_id="w2000-d5000",
        settings={"family": "demand", "seeds": {"pilot": 1, "production": 2}},
        grid=GRID,
    )

    loaded = np.load(artifact["draws_path"], allow_pickle=False)
    np.testing.assert_array_equal(loaded, draws)
    with open(artifact["manifest_path"], encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["draw_shape"] == [5, 4, 3, 2]
    assert manifest["draw_dtype"] == "float32"
    assert manifest["settings"]["seeds"] == {"pilot": 1, "production": 2}
    assert not list(tmp_path.glob(".*.staging-*"))

    mapped, grid, restored = load_draw_artifact(artifact["manifest_path"])
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, draws)
    assert set(grid) == set(GRID)
    for key, value in GRID.items():
        np.testing.assert_array_equal(grid[key], value)
        assert grid[key].dtype == value.dtype
    assert restored["settings"]["family"] == "demand"


def test_draw_artifact_refuses_overwrite_and_unsafe_arm(tmp_path):
    draws = np.ones((2, 4, 1, 1), np.float32)
    kwargs = {"artifact_root": tmp_path, "arm_id": "arm", "settings": {}, "grid": GRID}
    save_draw_artifact(draws, **kwargs)
    with pytest.raises(MCMCDrawArtifactError, match="overwrite"):
        save_draw_artifact(draws, **kwargs)
    with pytest.raises(MCMCDrawArtifactError, match="arm_id"):
        save_draw_artifact(draws, **{**kwargs, "arm_id": "../escape"})


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
            draws, artifact_root=tmp_path, arm_id="arm", settings={}, grid=GRID
        )


def test_draw_artifact_loader_rejects_inconsistent_or_missing_files(tmp_path):
    draws = np.ones((2, 4, 1, 1), np.float32)
    first = save_draw_artifact(
        draws, artifact_root=tmp_path, arm_id="shape", settings={}, grid=GRID
    )
    with open(first["manifest_path"], "r+", encoding="utf-8") as handle:
        manifest = json.load(handle)
        manifest["draw_shape"] = [3, 4, 1, 1]
        handle.seek(0)
        json.dump(manifest, handle)
        handle.truncate()
    with pytest.raises(MCMCDrawArtifactError, match="shape mismatch"):
        load_draw_artifact(first["manifest_path"])

    second = save_draw_artifact(
        draws, artifact_root=tmp_path, arm_id="missing", settings={}, grid=GRID
    )
    (tmp_path / "missing" / "grid.npz").unlink()
    with pytest.raises(MCMCDrawArtifactError, match="missing"):
        load_draw_artifact(second["manifest_path"])

    with pytest.raises(MCMCDrawArtifactError, match="does not exist"):
        load_draw_artifact(tmp_path / "nowhere" / "manifest.json")
