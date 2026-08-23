"""Certification: recipes, seeds, aggregation, the batch run and an end-to-end smoke.

The smoke tiers run the full pilot -> production -> calibration -> aggregate
chain on a tiny untrained demand model with short chains; the gate thresholds
stay at production tier, so verdicts are asserted to be well-formed and
consistently aggregated, not scientifically meaningful.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV
from bgm_iv.mcmc.certify import (
    FAMILY_RECIPES,
    BatchOutcome,
    CertificationError,
    MCMCConfig,
    aggregate_batch_outcomes,
    certify_grid,
    derive_certification_seeds,
    run_mcmc,
)
from bgm_iv.mcmc.readout import build_query_table
from bgm_iv.mcmc.target import AffinePreprocessorSpec

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_certify_runtime_"))


def _tiny_demand_model(seed=907):
    params = {
        "dataset": "Certify_demand",
        "output_dir": str(RUNTIME_ROOT),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 2,
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
        "sigma_y_softfloor": 0.1,
        "mcmc_seed": 11,
    }
    return BGM_IV(params=params, timestamp=f"certify_{seed}", random_seed=seed)


def _smoke_grid(num_unique=3, prices_per_v=4, seed=41):
    rng = np.random.default_rng(seed)
    unique_v = rng.normal(size=(num_unique, 2)).astype(np.float32)
    grid_v = np.repeat(unique_v, prices_per_v, axis=0)
    grid_x = np.tile(np.linspace(-1.0, 1.0, prices_per_v, dtype=np.float32)[:, None], (num_unique, 1))
    truth = rng.normal(size=grid_x.shape[0]).astype(np.float64)
    return grid_x, grid_v, truth


# Loose precision: the smoke verdict is decided by the stationarity and
# numerical gates, which stay at production tier.
_SMOKE_CONFIG = MCMCConfig(
    num_chains=4, warmup_steps=200, segment_size=240, max_batch_size=8,
    absolute_halfwidth=1e6, relative_halfwidth=0.0, precision_reference_scale=1.0,
)


# --- recipes, seeds, aggregation --------------------------------------------


def test_family_recipes_pin_the_reported_settings():
    vector = FAMILY_RECIPES["vector"]
    assert vector.production.to_payload()["trajectory_support"] == [7, 15, 31]
    assert vector.production.num_leapfrog_steps == 31 and vector.production.segment_size == 24000
    assert vector.production.warmup_steps == 1600 and vector.production.target_accept_prob == 0.90
    assert vector.pilot_batch == 128
    feature = FAMILY_RECIPES["mnist_feature"]
    assert feature.production.to_payload()["trajectory_support"] == [7, 15]
    assert feature.production.num_leapfrog_steps == 15
    assert feature.production.warmup_leapfrog_matches_max_trajectory
    assert feature.pilot_batch == 64
    demand = FAMILY_RECIPES["demand"]
    assert demand.calibration_unit == "query" and demand.pilot_batch == 140
    assert demand.production.segment_size == 12000 and demand.production.initial_step_size == 0.05
    assert set(FAMILY_RECIPES) == {"demand", "vector", "mnist_feature"}
    hashes = {name: recipe.recipe_hash for name, recipe in FAMILY_RECIPES.items()}
    assert len(set(hashes.values())) == len(hashes)


def test_config_payload_records_warmup_trajectory_relation():
    config = MCMCConfig(num_leapfrog_steps=31, trajectory_support=(7, 15))
    payload = config.validate().to_payload()
    assert payload["warmup_leapfrog_steps"] == 31
    assert payload["warmup_leapfrog_matches_max_trajectory"] is False
    with pytest.raises(CertificationError):
        MCMCConfig(trajectory_support=(3, 3)).validate()


def test_seeds_are_content_derived_and_stage_distinct():
    a = derive_certification_seeds("vector", 0, "ckpt-a")
    assert a == derive_certification_seeds("vector", 0, "ckpt-a")
    assert a != derive_certification_seeds("vector", 0, "ckpt-b")
    assert a != derive_certification_seeds("vector", 1, "ckpt-a")
    assert a["pilot_run_seed"] != a["production_run_seed"]


def _fake_reportable(batch_index, ids, plugin_mse, mcse):
    metric = {
        "plugin_mse": plugin_mse, "u_corrected_mse": plugin_mse - 0.5, "integration_penalty": 0.5,
        "plugin_mcse": mcse, "max_iact": 3.0 + batch_index,
    }
    sampler = {
        "rank_rhat_max": 1.002 + 0.001 * batch_index, "folded_rhat_max": 1.003,
        "bulk_ess_min": 900.0 - batch_index, "tail_ess_min": 800.0,
    }
    return BatchOutcome(
        batch_index=batch_index, global_ids=ids, state="REPORTABLE",
        record={"metric": metric, "sampler": sampler, "block_len": 10, "max_iact": metric["max_iact"]},
        trace_summary={"divergences": 0, "diagnostic_unscored": {"plugin_mse_pooled": plugin_mse + 1.0}},
    )


def test_aggregate_weights_by_rows_and_pairs_on_certified_subset():
    rng = np.random.default_rng(0)
    unique_v = rng.normal(size=(4, 2)).astype(np.float32)
    inverse = np.array([0, 0, 1, 1, 2, 3, 3, 3])  # targets own 2, 2, 1, 3 rows
    table = build_query_table(rng.normal(size=(8, 1)).astype(np.float32), unique_v[inverse])
    truth = np.arange(8, dtype=np.float64)
    outcomes = [
        _fake_reportable(0, (0, 1), plugin_mse=10.0, mcse=1.0),  # 4 rows
        BatchOutcome(batch_index=1, global_ids=(2,), state="EXTEND_PRECISION", reason="r",
                     trace_summary={"divergences": 2, "diagnostic_unscored": {"plugin_mse_pooled": 99.0}}),
        _fake_reportable(2, (3,), plugin_mse=20.0, mcse=2.0),  # 3 rows
    ]
    agg = aggregate_batch_outcomes(outcomes, table, truth, {"map": truth + 1.0})
    assert agg["num_batches"] == 3 and agg["num_reportable"] == 2
    assert agg["state_counts"] == {"EXTEND_PRECISION": 1, "REPORTABLE": 2}
    assert agg["certified_num_queries"] == 7 and abs(agg["certified_query_fraction"] - 7 / 8) < 1e-12
    np.testing.assert_allclose(agg["certified_plugin_mse"], (4 * 10 + 3 * 20) / 7)
    np.testing.assert_allclose(agg["certified_pooled_mcse"], np.sqrt((4 * 1) ** 2 + (3 * 2) ** 2) / 7)
    assert agg["divergences_total"] == 2
    assert agg["rank_rhat_max"] == pytest.approx(1.004)
    assert agg["bulk_ess_min"] == 898.0 and agg["max_iact"] == 5.0
    np.testing.assert_allclose(agg["unscored_plugin_mse_all_batches"], (4 * 11 + 1 * 99 + 3 * 21) / 8)
    assert agg["paired"]["map"] == {"all_rows_mse": 1.0, "certified_subset_mse": 1.0}
    with pytest.raises(Exception, match="truth"):
        aggregate_batch_outcomes(outcomes, table, truth[:4], {"map": truth + 1.0})


def test_batch_outcome_carries_a_record_exactly_when_reportable():
    with pytest.raises(CertificationError):
        BatchOutcome(batch_index=0, global_ids=(0,), state="REPORTABLE")
    with pytest.raises(CertificationError):
        BatchOutcome(batch_index=0, global_ids=(0,), state="EXTEND_PRECISION", record={})
    with pytest.raises(CertificationError):
        BatchOutcome(batch_index=0, global_ids=(0,), state="UNKNOWN")


# --- the batch run -----------------------------------------------------------


def _run(model, run_seed=20260811):
    grid_x, grid_v, truth = _smoke_grid()
    return run_mcmc(
        model, grid_x, grid_v,
        preprocessor=AffinePreprocessorSpec.identity_map(2),
        truth_original_units=truth, truth_label="smoke_truth",
        outcome_shift=100.0, outcome_scale=3.5,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        run_seed=run_seed, run_label="run-smoke", config=_SMOKE_CONFIG,
    )


@pytest.mark.slow
def test_run_mcmc_classifies_every_batch_and_replays_at_the_same_seed():
    model = _tiny_demand_model()
    result = _run(model)
    assert result["schema_version"] == "bgm-mcmc-run"
    assert result["execution_environment"]["tensorflow_version"] == tf.__version__
    assert result["target"]["family"] == "demand"
    assert result["num_targets"] == 3 and result["num_queries"] == 12
    outcomes = result["batch_outcomes"]
    assert len(outcomes) == 1
    for outcome in outcomes:
        assert isinstance(outcome, BatchOutcome)
        assert outcome.trace_summary["numerical_anomalies"] == 0
        assert len(outcome.trace_summary["draws_hash"]) == 64
        if outcome.state == "REPORTABLE":
            assert outcome.record["reportability"] == "REPORTABLE"
            assert outcome.record["sampler"]["num_chains"] == 4
            assert outcome.record["metric"]["original_outcome_units"] is True
            assert outcome.record["identity"]["draws_hash"] == outcome.trace_summary["draws_hash"]
        else:
            assert outcome.record is None and outcome.reason

    second = _run(model)
    third = _run(model, run_seed=8)
    hashes = lambda r: [o.trace_summary["draws_hash"] for o in r["batch_outcomes"]]
    assert hashes(result) == hashes(second) and hashes(result) != hashes(third)
    assert [o.state for o in result["batch_outcomes"]] == [o.state for o in second["batch_outcomes"]]


def test_run_mcmc_rejects_bnn_and_misaligned_truth():
    model = _tiny_demand_model()
    grid_x, grid_v, truth = _smoke_grid()
    common = dict(
        preprocessor=AffinePreprocessorSpec.identity_map(2), truth_label="t",
        outcome_shift=0.0, outcome_scale=1.0, treatment_transform={"shift": 0.0, "scale": 1.0},
        run_seed=1, run_label="guard", config=_SMOKE_CONFIG,
    )
    with pytest.raises(CertificationError):
        run_mcmc(model, grid_x, grid_v, truth_original_units=truth[:-1], **common)
    model.params["use_bnn"] = True
    try:
        with pytest.raises(CertificationError):
            run_mcmc(model, grid_x, grid_v, truth_original_units=truth, **common)
    finally:
        model.params["use_bnn"] = False


# --- end to end ----------------------------------------------------------------


@pytest.mark.slow
def test_certify_grid_smoke_runs_the_full_chain():
    model = _tiny_demand_model()
    grid_x, grid_v, truth = _smoke_grid()
    recipe = replace(
        FAMILY_RECIPES["demand"],
        pilot_batch=2, pilot_warmup_steps=40, pilot_segment_size=24,
        production=replace(
            FAMILY_RECIPES["demand"].production,
            warmup_steps=60, segment_size=120, absolute_halfwidth=1e6, relative_halfwidth=0.0,
        ),
        calibration_num_draws=50,
    )
    paired = {"map": truth + 0.5, "encoder": truth - 0.5}
    result = certify_grid(
        model, family="demand", grid_x_model=grid_x, grid_v_raw=grid_v,
        preprocessor=AffinePreprocessorSpec.identity_map(2),
        truth_original_units=truth, truth_label="smoke_truth",
        outcome_shift=100.0, outcome_scale=3.5, treatment_transform={"shift": 0.0, "scale": 1.0},
        data_seed=0, checkpoint_identity="smoke", run_label="certify-smoke",
        paired_predictions=paired, recipe=recipe, progress=None,
    )
    assert result["schema_version"] == "bgm-certification"
    assert result["grid"]["num_targets"] == 3 and result["grid"]["num_rows"] == 12
    assert result["pipeline"]["config"]["max_batch_size"] == 2
    assert result["pipeline"]["target"]["family"] == "demand"
    assert len(result["batches"]) == 2
    states = {b["state"] for b in result["batches"]}
    assert states <= {"REPORTABLE", "EXTEND_PRECISION", "STATIONARITY_RESTART",
                      "NUMERICAL_RESTART", "MASS_UPGRADE_REQUIRED", "CONFIG_INVALID"}
    agg = result["aggregate"]
    assert agg["num_batches"] == 2
    assert agg["paired"]["map"]["all_rows_mse"] == pytest.approx(0.25)
    assert set(result["calibration"]["batch_states"]) == {"0", "1"}
    assert result["calibration"]["diagnostic_unscored"]["units"] == 12  # query unit (demand)
    for batch in result["batches"]:
        assert batch["num_queries"] in (8, 4)
        assert len(batch["trace_summary"]["draws_hash"]) == 64
        if batch["state"] == "REPORTABLE":
            assert batch["metric"]["plugin_mse"] > 0
    assert result["execution_environment"]["tensorflow_version"] == tf.__version__
    assert result["recipe_hash"] != FAMILY_RECIPES["demand"].recipe_hash


def test_certify_grid_rejects_misaligned_inputs():
    model = _tiny_demand_model(seed=5)
    with pytest.raises(CertificationError, match="equal row counts"):
        certify_grid(
            model, family="demand", grid_x_model=np.zeros((3, 1)), grid_v_raw=np.zeros((3, 2)),
            preprocessor=AffinePreprocessorSpec.identity_map(2), truth_original_units=np.zeros(2),
            truth_label="t", outcome_shift=0.0, outcome_scale=1.0,
            treatment_transform={"shift": 0.0, "scale": 1.0},
            data_seed=0, checkpoint_identity="c", run_label="r", progress=None,
        )
