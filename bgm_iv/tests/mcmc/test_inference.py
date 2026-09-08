"""Family recipes and tiny end-to-end full-grid MCMC inference."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest

from bgm_iv.models.bgm_iv import BGM_IV, BGM_IV_Image, BGM_IV_Vector
from bgm_iv.mcmc.inference import (
    FAMILY_RECIPES,
    MCMCConfig,
    MCMCInferenceError,
    derive_mcmc_seeds,
    run_mcmc_grid,
)
from bgm_iv.mcmc.readout import ReadoutConfig
from bgm_iv.mcmc.target import AffinePreprocessorSpec


RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_inference_runtime_"))


def _params(family):
    common = {
        "dataset": f"Inference_{family}",
        "output_dir": str(RUNTIME_ROOT),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [8, 8],
        "e_units": [8, 8],
        "f_units": [8, 4],
        "h_units": [8, 4],
        "dz_units": [8, 4],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "first_stage_warmup_epochs": 0,
        "sigma_time": 0.1,
        "sigma_y_softfloor": 0.1,
        "covariate_block_scale": "sum",
    }
    if family == "demand":
        common["v_dim"] = 2
    elif family in {"vector", "mnist_feature"}:
        common.update(v_dim=5, vector_dim=4, sigma_vector_softfloor=0.1)
    else:
        common["v_dim"] = 785
    return common


def _model_and_grid(family):
    params = _params(family)
    if family == "demand":
        model = BGM_IV(params, timestamp=family, random_seed=7)
        unique = np.array([[0.0, 1.0], [0.5, 2.0]], np.float32)
    elif family in {"vector", "mnist_feature"}:
        model = BGM_IV_Vector(params, timestamp=family, random_seed=7)
        unique = np.array(
            [[0.0, 0.1, -0.2, 0.3, -0.4], [0.5, -0.4, 0.3, -0.2, 0.1]],
            np.float32,
        )
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


def _smoke_recipe(family):
    return replace(
        FAMILY_RECIPES[family],
        pilot_warmup_steps=3,
        pilot_segment_size=4,
        production=MCMCConfig(
            num_chains=4,
            warmup_steps=4,
            segment_size=6,
            initial_step_size=0.02,
            num_leapfrog_steps=2,
            target_accept_prob=0.8,
            trajectory_support=(1, 2),
            overdispersion_scale=0.3,
        ),
        readout=ReadoutConfig(
            query_chunk=2,
            draw_chunk=5,
            bisection_iterations=20,
        ),
    )


def test_family_recipes_pin_requested_production_settings():
    assert set(FAMILY_RECIPES) == {
        "demand",
        "vector",
        "mnist_feature",
        "mnist_pixel",
    }
    assert FAMILY_RECIPES["demand"].production.segment_size == 12000
    assert tuple(FAMILY_RECIPES["demand"].production.trajectory_support) == (3, 5, 7)
    assert FAMILY_RECIPES["vector"].production.segment_size == 24000
    assert tuple(FAMILY_RECIPES["vector"].production.trajectory_support) == (7, 15, 31)
    assert tuple(FAMILY_RECIPES["mnist_pixel"].production.trajectory_support) == (7, 15, 31)
    assert tuple(FAMILY_RECIPES["mnist_feature"].production.trajectory_support) == (7, 15)
    assert FAMILY_RECIPES["mnist_pixel"].target_kind == "generalized_gibbs"
    assert all(
        recipe.readout.to_payload()["scoring_unit"] == "query"
        for recipe in FAMILY_RECIPES.values()
    )


def test_production_chain_count_allows_four_or_more_only():
    FAMILY_RECIPES["vector"].production.validate()
    replace(FAMILY_RECIPES["vector"].production, num_chains=5).validate()
    with pytest.raises(MCMCInferenceError, match="at least four chains"):
        replace(FAMILY_RECIPES["vector"].production, num_chains=3).validate()


def test_seeds_are_content_derived_and_stage_distinct():
    first = derive_mcmc_seeds("vector", 0, "checkpoint")
    assert first == derive_mcmc_seeds("vector", 0, "checkpoint")
    assert first != derive_mcmc_seeds("vector", 1, "checkpoint")
    assert first["pilot"] != first["production"]


@pytest.mark.slow
@pytest.mark.parametrize("family", ["demand", "vector", "mnist_feature", "mnist_pixel"])
def test_all_families_run_one_full_grid_target_set(family, capsys):
    model, grid_x, grid_v, truth = _model_and_grid(family)
    result = run_mcmc_grid(
        model,
        family=family,
        grid_x_model=grid_x,
        grid_v_raw=grid_v,
        preprocessor=AffinePreprocessorSpec.identity_map(grid_v.shape[1]),
        truth_original_units=truth,
        truth_label="tiny-full-grid",
        outcome_shift=0.0,
        outcome_scale=1.0,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        data_seed=0,
        checkpoint_identity=f"tiny-{family}",
        run_label=f"tiny-{family}",
        recipe=_smoke_recipe(family),
    )
    output = capsys.readouterr().out
    assert "acceptance mean=" in output
    assert result["schema_version"] == "bgm-mcmc-inference"
    assert result["grid"]["num_queries"] == 3
    assert result["grid"]["num_targets"] == 2
    assert result["readout"]["num_components"] == 24
    assert result["readout"]["config"]["scoring_unit"] == "query"
    assert set(result["readout"]["coverage"]) == {"0.5", "0.8", "0.95"}
    assert (
        result["readout"]["width50"]
        < result["readout"]["width80"]
        < result["readout"]["width95"]
    )
    assert "chain_mean_variance_penalty" in result["readout"]["sensitivity"]
    assert "u_corrected_mse" not in result["readout"]["sensitivity"]
    rendered = repr(result).lower()
    for removed in ("reportable", "certified", "wasserstein", "rhat", "iact"):
        assert removed not in rendered


def test_misaligned_grid_is_rejected():
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    with pytest.raises(MCMCInferenceError, match="equal rows"):
        run_mcmc_grid(
            model,
            family="demand",
            grid_x_model=grid_x,
            grid_v_raw=grid_v,
            preprocessor=AffinePreprocessorSpec.identity_map(2),
            truth_original_units=truth[:-1],
            truth_label="bad",
            outcome_shift=0.0,
            outcome_scale=1.0,
            treatment_transform={"shift": 0.0, "scale": 1.0},
            data_seed=0,
            checkpoint_identity="bad",
            run_label="bad",
            recipe=_smoke_recipe("demand"),
        )


@pytest.mark.slow
def test_ablation_overrides_save_before_prefix_readouts(tmp_path):
    family = "demand"
    model, grid_x, grid_v, truth = _model_and_grid(family)
    result = run_mcmc_grid(
        model,
        family=family,
        grid_x_model=grid_x,
        grid_v_raw=grid_v,
        preprocessor=AffinePreprocessorSpec.identity_map(grid_v.shape[1]),
        truth_original_units=truth,
        truth_label="tiny-prefix-grid",
        outcome_shift=0.0,
        outcome_scale=1.0,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        data_seed=0,
        checkpoint_identity="tiny-prefix",
        run_label="tiny-prefix",
        recipe=_smoke_recipe(family),
        production_num_chains=5,
        production_warmup_steps=3,
        production_draws=5,
        artifact_root=tmp_path,
        arm_id="w3-d5",
        readout_prefixes=[3, 4, 5],
        progress=None,
    )

    assert result["recipe"]["production"]["warmup_steps"] == 3
    assert result["recipe"]["production"]["segment_size"] == 5
    assert set(result["readouts"]) == {"3", "4", "5"}
    assert result["readout"] is result["readouts"]["5"]
    assert result["readouts"]["3"]["draws_per_chain"] == 3
    assert result["readouts"]["4"]["draws_per_chain"] == 4
    assert result["artifact"]["draw_shape"][0] == 5
    np.testing.assert_array_equal(
        np.load(result["artifact"]["draws_path"], allow_pickle=False).shape,
        (5, 5, 2, 4),
    )
    assert result["sampler"]["acceptance"]["per_chain"]


@pytest.mark.slow
def test_artifact_readout_skips_sampling_and_revalidates_context(
    tmp_path, monkeypatch
):
    family = "demand"
    model, grid_x, grid_v, truth = _model_and_grid(family)
    common = {
        "family": family,
        "grid_x_model": grid_x,
        "grid_v_raw": grid_v,
        "preprocessor": AffinePreprocessorSpec.identity_map(grid_v.shape[1]),
        "truth_original_units": truth,
        "truth_label": "tiny-artifact-grid",
        "outcome_shift": 0.0,
        "outcome_scale": 1.0,
        "treatment_transform": {"shift": 0.0, "scale": 1.0},
        "data_seed": 0,
        "checkpoint_identity": "tiny-artifact",
        "run_label": "tiny-artifact",
        "recipe": _smoke_recipe(family),
        "production_warmup_steps": 3,
        "production_draws": 5,
        "progress": None,
    }
    sampled = run_mcmc_grid(
        model,
        **common,
        artifact_root=tmp_path,
        arm_id="w3-d5-readout",
        readout_prefixes=[3, 5],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling must not run in artifact readout mode")

    monkeypatch.setattr(
        "bgm_iv.mcmc.inference.estimate_pilot_state_variance", forbidden
    )
    monkeypatch.setattr("bgm_iv.mcmc.inference.run_mcmc", forbidden)
    restored_common = {
        key: value
        for key, value in common.items()
        if key not in {"production_warmup_steps", "production_draws"}
    }
    restored = run_mcmc_grid(
        model,
        **restored_common,
        readout_artifact_manifest=sampled["artifact"]["manifest_path"],
        readout_prefixes=[3, 4, 5],
    )

    assert restored["mode"] == "artifact-readout"
    assert restored["pilot"]["skipped"] is True
    assert restored["timings"]["pilot_seconds"] == 0.0
    assert restored["timings"]["mcmc_seconds"] == 0.0
    assert set(restored["readouts"]) == {"3", "4", "5"}
    assert restored["readout"]["structural_mse_plugin"] == pytest.approx(
        sampled["readout"]["structural_mse_plugin"], abs=0.0
    )

    with pytest.raises(MCMCInferenceError, match="checkpoint identity mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "checkpoint_identity": "wrong-checkpoint"},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
            readout_prefixes=[5],
        )

    changed_truth = truth.copy()
    changed_truth[0] += 1.0
    with pytest.raises(MCMCInferenceError, match="truth_hash mismatch"):
        run_mcmc_grid(
            model,
            **{**restored_common, "truth_original_units": changed_truth},
            readout_artifact_manifest=sampled["artifact"]["manifest_path"],
            readout_prefixes=[5],
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"production_warmup_steps": 0}, "positive integer"),
        ({"production_num_chains": 3}, "at least four chains"),
        ({"production_draws": True}, "positive integer"),
        ({"production_draws": 5, "readout_prefixes": [6]}, "between"),
        ({"artifact_root": "somewhere"}, "both be set"),
    ],
)
def test_ablation_arguments_fail_before_sampling(kwargs, message):
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    with pytest.raises(MCMCInferenceError, match=message):
        run_mcmc_grid(
            model,
            family="demand",
            grid_x_model=grid_x,
            grid_v_raw=grid_v,
            preprocessor=AffinePreprocessorSpec.identity_map(2),
            truth_original_units=truth,
            truth_label="bad-ablation",
            outcome_shift=0.0,
            outcome_scale=1.0,
            treatment_transform={"shift": 0.0, "scale": 1.0},
            data_seed=0,
            checkpoint_identity="bad-ablation",
            run_label="bad-ablation",
            recipe=_smoke_recipe("demand"),
            progress=None,
            **kwargs,
        )
