from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from bgm_iv.models.bgm_iv import BGM_IV, BGM_IV_Image
from bgm_iv.mcmc import inference
from bgm_iv.mcmc.inference import (
    FAMILY_RECIPES,
    MCMCInferenceError,
    run_mcmc_grid,
)
from bgm_iv.mcmc.readout import ReadoutError
from bgm_iv.mcmc.target import AffinePreprocessorSpec


RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_inference_runtime_"))


def _identity(dimension):
    return AffinePreprocessorSpec(
        mean=np.zeros(int(dimension), np.float32),
        scale=np.ones(int(dimension), np.float32),
    )


def _params(family):
    common = {
        "dataset": f"Inference_{family}",
        "output_dir": str(RUNTIME_ROOT),
        "save_model": False,
        "z_dims": [1, 1, 1, 1],
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [8, 8],
        "e_units": [8, 8],
        "f_units": [8, 4],
        "h_units": [8, 4],
        "dz_units": [8, 4],
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "structural_map_steps": 100,
        "structural_map_lr": 5e-4,
    }
    if family == "demand":
        common["v_dim"] = 2
    else:
        common["v_dim"] = 785
    return common


def _model_and_grid(family):
    params = _params(family)
    if family == "demand":
        model = BGM_IV(params, timestamp=family, random_seed=7)
        unique = np.array([[0.0, 1.0], [0.5, 2.0]], np.float32)
    else:
        model = BGM_IV_Image(params, timestamp=family, random_seed=7)
        image_a = np.full(784, 64.0, np.float32)
        image_b = np.full(784, 192.0, np.float32)
        unique = np.stack(
            [np.concatenate([[0.0], image_a]), np.concatenate([[0.5], image_b])]
        ).astype(np.float32)
    grid_v = unique[[0, 1, 0]]
    grid_x = np.array([[-0.2], [0.1], [0.7]], np.float32)
    truth = np.array([1.0, 2.0, 3.0], np.float64)
    return model, grid_x, grid_v, truth


_RECIPE = inference._recipe
SMOKE_COUNTS = {
    "production_num_chains": 4,
    "production_warmup_steps": 4,
    "production_draws": 6,
}


def _smoke_recipe(family, **counts):
    recipe = _RECIPE(family, **counts)
    recipe["pilot"].update(warmup_steps=3, segment_size=4)
    recipe["production"].update(
        initial_step_size=0.02,
        num_leapfrog_steps=2,
        target_accept_prob=0.8,
        trajectory_support=[1, 2],
        overdispersion_scale=0.3,
    )
    return recipe


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.setattr(inference, "_recipe", _smoke_recipe)


def test_family_recipes_pin_requested_production_settings():
    assert FAMILY_RECIPES == {
        "demand": (0.05, 5, (3, 5, 7)),
        "mnist_pixel": (0.02, 31, (7, 15, 31)),
    }
    demand = _RECIPE("demand", num_chains=4, warmup_steps=2000, draws=5000)
    assert demand["pilot"]["initial_step_size"] == 0.05
    assert demand["production"] == {
        "num_chains": 4,
        "warmup_steps": 2000,
        "segment_size": 5000,
        "initial_step_size": 0.05,
        "num_leapfrog_steps": 5,
        "target_accept_prob": 0.9,
        "trajectory_support": [3, 5, 7],
        "overdispersion_scale": 2.0,
    }
    mnist = _RECIPE("mnist_pixel", num_chains=4, warmup_steps=1000, draws=1000)
    assert mnist["pilot"]["initial_step_size"] == 0.02
    assert mnist["production"]["initial_step_size"] == 0.02
    assert mnist["production"]["num_leapfrog_steps"] == 31
    assert mnist["production"]["trajectory_support"] == [7, 15, 31]
    for recipe in (demand, mnist):
        assert recipe["pilot"]["warmup_steps"] == 400
        assert recipe["pilot"]["segment_size"] == 240
        assert recipe["pilot"]["num_leapfrog_steps"] == 5
        assert recipe["pilot"]["trajectory_support"] == [3, 5, 7]
        assert recipe["pilot"]["jitter_scale"] == 0.3
        assert recipe["production"]["target_accept_prob"] == 0.9
    with pytest.raises(MCMCInferenceError, match="unknown MCMC family"):
        _RECIPE("other", num_chains=4, warmup_steps=1, draws=1)


SEEDS = {"pilot": 11, "production": 12}


@pytest.mark.slow
@pytest.mark.parametrize("family", ["demand", "mnist_pixel"])
def test_all_families_run_one_full_grid_target_set(family, capsys, smoke):
    model, grid_x, grid_v, truth = _model_and_grid(family)
    result = run_mcmc_grid(
        model,
        family=family,
        grid_x_model=grid_x,
        grid_v_raw=grid_v,
        preprocessor=_identity(grid_v.shape[1]),
        truth_original_units=truth,
        truth_label="tiny-full-grid",
        outcome_shift=0.0,
        outcome_scale=1.0,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        seeds=SEEDS,
        **SMOKE_COUNTS,
    )
    output = capsys.readouterr().out
    assert "acceptance mean=" in output
    assert "structural MSE" not in output
    assert result["family"] == family
    assert result["grid"]["num_queries"] == 3
    assert result["grid"]["num_targets"] == 2
    assert result["seeds"] == SEEDS
    assert result["readout"]["num_components"] == 24
    assert set(result["readout"]["coverage"]) == {"0.9", "0.95", "0.99"}
    assert (
        result["readout"]["width90"]
        < result["readout"]["width95"]
        < result["readout"]["width99"]
    )
    assert set(result["readout"]) == {
        "schema_version",
        "num_targets",
        "num_queries",
        "num_chains",
        "draws_per_chain",
        "num_components",
        "coverage",
        "width90",
        "width95",
        "width99",
        "readout_seconds",
    }


def test_misaligned_grid_is_rejected(smoke):
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    with pytest.raises(MCMCInferenceError, match="equal rows"):
        run_mcmc_grid(
            model,
            family="demand",
            grid_x_model=grid_x,
            grid_v_raw=grid_v,
            preprocessor=_identity(2),
            truth_original_units=truth[:-1],
            truth_label="bad",
            outcome_shift=0.0,
            outcome_scale=1.0,
            treatment_transform={"shift": 0.0, "scale": 1.0},
            seeds=SEEDS,
            **SMOKE_COUNTS,
        )


@pytest.mark.slow
def test_production_overrides_and_draw_artifact(tmp_path, smoke):
    family = "demand"
    model, grid_x, grid_v, truth = _model_and_grid(family)
    result = run_mcmc_grid(
        model,
        family=family,
        grid_x_model=grid_x,
        grid_v_raw=grid_v,
        preprocessor=_identity(grid_v.shape[1]),
        truth_original_units=truth,
        truth_label="tiny-override-grid",
        outcome_shift=0.0,
        outcome_scale=1.0,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        seeds=SEEDS,
        production_num_chains=5,
        production_warmup_steps=3,
        production_draws=5,
        artifact_root=tmp_path,
        arm_id="w3-d5",
        progress=None,
    )

    assert result["recipe"]["production"]["warmup_steps"] == 3
    assert result["recipe"]["production"]["segment_size"] == 5
    assert result["readout"]["draws_per_chain"] == 5
    assert result["readout"]["num_chains"] == 5
    assert result["artifact"]["draw_shape"][0] == 5
    np.testing.assert_array_equal(
        np.load(result["artifact"]["draws_path"], allow_pickle=False).shape,
        (5, 5, 2, 4),
    )
    assert result["sampler"]["acceptance"]["per_chain"]


@pytest.mark.slow
def test_artifact_readout_skips_sampling_and_revalidates_context(
    tmp_path, monkeypatch, smoke
):
    family = "demand"
    model, grid_x, grid_v, truth = _model_and_grid(family)
    common = {
        "family": family,
        "grid_x_model": grid_x,
        "grid_v_raw": grid_v,
        "preprocessor": _identity(grid_v.shape[1]),
        "truth_original_units": truth,
        "truth_label": "tiny-artifact-grid",
        "outcome_shift": 0.0,
        "outcome_scale": 1.0,
        "treatment_transform": {"shift": 0.0, "scale": 1.0},
        "seeds": SEEDS,
        "source": {"checkpoint_timestamp": "tiny-artifact", "repeat_id": 0},
        "production_num_chains": 4,
        "production_warmup_steps": 3,
        "production_draws": 5,
        "progress": None,
    }
    sampled = run_mcmc_grid(
        model,
        **common,
        artifact_root=tmp_path,
        arm_id="w3-d5-readout",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling must not run in artifact readout mode")

    monkeypatch.setattr("bgm_iv.mcmc.inference._sample", forbidden)
    restored_common = dict(common)
    restored = run_mcmc_grid(
        model,
        **restored_common,
        readout_artifact_manifest=sampled["artifact"]["manifest_path"],
    )

    assert restored["pilot"]["skipped"] is True
    assert restored["pilot"]["reason"] == "readout_artifact_manifest"
    assert restored["timings"]["pilot_seconds"] == 0.0
    assert restored["timings"]["mcmc_seconds"] == 0.0
    assert restored["readout"]["coverage"] == sampled["readout"]["coverage"]
    for key in ("width90", "width95", "width99"):
        assert restored["readout"][key] == sampled["readout"][key]

    with pytest.raises(MCMCInferenceError, match="source checkpoint mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "source": {"checkpoint_timestamp": "other"}},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
        )

    with pytest.raises(MCMCInferenceError, match="seeds mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "seeds": {"pilot": 11, "production": 13}},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
        )

    with pytest.raises(MCMCInferenceError, match="production config mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "production_draws": 6},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
        )

    with pytest.raises(MCMCInferenceError, match="readout context mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "outcome_scale": 2.0},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
        )

    changed_truth = truth.copy()
    changed_truth[0] += 1.0
    with pytest.raises(MCMCInferenceError, match="grid truth mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "truth_original_units": changed_truth},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"production_warmup_steps": 0}, "positive integer"),
        ({"production_num_chains": 3}, "at least four chains"),
        ({"production_draws": True}, "positive integer"),
        ({"production_draws": 2.0}, "positive integer"),
        ({"artifact_root": "somewhere"}, "both be set"),
        ({"artifact_root": "somewhere", "arm_id": "bad/arm"}, "arm_id must contain"),
        ({"artifact_root": "somewhere", "arm_id": ".hidden"}, "arm_id must contain"),
        ({"artifact_root": "somewhere", "arm_id": "a" * 129}, "arm_id must contain"),
        (
            {"readout_artifact_manifest": "m.json", "artifact_root": "a", "arm_id": "a"},
            "cannot be combined",
        ),
    ],
)
def test_invalid_overrides_fail_before_sampling(monkeypatch, smoke, kwargs, message):
    model, grid_x, grid_v, truth = _model_and_grid("demand")

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling started before the options were checked")

    monkeypatch.setattr("bgm_iv.mcmc.inference._sample", forbidden)
    with pytest.raises(MCMCInferenceError, match=message):
        run_mcmc_grid(
            model,
            family="demand",
            grid_x_model=grid_x,
            grid_v_raw=grid_v,
            preprocessor=_identity(2),
            truth_original_units=truth,
            truth_label="bad-override",
            outcome_shift=0.0,
            outcome_scale=1.0,
            treatment_transform={"shift": 0.0, "scale": 1.0},
            seeds=SEEDS,
            progress=None,
            **{**SMOKE_COUNTS, **kwargs},
        )


def test_existing_artifact_destination_fails_before_sampling(monkeypatch, smoke, tmp_path):
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    (tmp_path / "taken").mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling started before the destination was checked")

    monkeypatch.setattr("bgm_iv.mcmc.inference._sample", forbidden)
    with pytest.raises(MCMCInferenceError, match="draw artifact already exists"):
        run_mcmc_grid(
            model,
            family="demand",
            grid_x_model=grid_x,
            grid_v_raw=grid_v,
            preprocessor=_identity(2),
            truth_original_units=truth,
            truth_label="taken-destination",
            outcome_shift=0.0,
            outcome_scale=1.0,
            treatment_transform={"shift": 0.0, "scale": 1.0},
            seeds=SEEDS,
            progress=None,
            artifact_root=tmp_path,
            arm_id="taken",
            **SMOKE_COUNTS,
        )


def _grid_kwargs(family, grid_x, grid_v, truth):
    return dict(
        family=family,
        grid_x_model=grid_x,
        grid_v_raw=grid_v,
        preprocessor=_identity(grid_v.shape[1]),
        truth_original_units=truth,
        truth_label="bad-grid",
        outcome_shift=0.0,
        outcome_scale=1.0,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        seeds=SEEDS,
        progress=None,
        **SMOKE_COUNTS,
    )


def test_mnist_grid_enforces_raw_pixel_support(smoke):
    model, grid_x, grid_v, truth = _model_and_grid("mnist_pixel")
    grid_v = grid_v.copy()
    grid_v[0, 20] = 300.0
    with pytest.raises(MCMCInferenceError, match=r"\[0,255\]"):
        run_mcmc_grid(model, **_grid_kwargs("mnist_pixel", grid_x, grid_v, truth))


def test_nonfinite_grid_is_rejected(smoke):
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    grid_v = grid_v.copy()
    grid_v[1, 0] = np.nan
    with pytest.raises(ReadoutError, match="finite"):
        run_mcmc_grid(model, **_grid_kwargs("demand", grid_x, grid_v, truth))


def test_preprocessor_width_and_recipe_must_match_the_model(smoke):
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    kwargs = _grid_kwargs("demand", grid_x, grid_v, truth)
    with pytest.raises(MCMCInferenceError, match="v_dim"):
        run_mcmc_grid(
            model,
            **{**kwargs, "grid_v_raw": np.zeros((3, 3), np.float32),
               "preprocessor": _identity(3)},
        )
    with pytest.raises(MCMCInferenceError, match="recipe does not match"):
        run_mcmc_grid(model, **{**kwargs, "family": "mnist_pixel"})
