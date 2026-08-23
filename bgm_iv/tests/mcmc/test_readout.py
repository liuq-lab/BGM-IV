"""Query table, chain-preserving readout and exact-mixture predictive calibration."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import ndtr, ndtri

from bgm_iv.models.bgm_iv import BGM_IV

from bgm_iv.mcmc.certify import BatchOutcome
from bgm_iv.mcmc.readout import (
    CalibrationConfig,
    FunctionalReadout,
    PredictiveCalibrationAccumulator,
    ReadoutError,
    batch_query_view,
    build_query_table,
    gaussian_mixture_quantiles,
)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_readout_runtime_"))


def _tiny_demand_model(seed: int = 611) -> BGM_IV:
    params = {
        "dataset": "Readout_demand",
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
    }
    return BGM_IV(params=params, timestamp=f"readout_{seed}", random_seed=seed)


def _grid(num_unique: int = 5, prices_per_v: int = 4, seed: int = 23):
    rng = np.random.default_rng(seed)
    unique_v = rng.normal(size=(num_unique, 2)).astype(np.float32)
    grid_v = np.repeat(unique_v, prices_per_v, axis=0)
    grid_x = rng.normal(size=(num_unique * prices_per_v, 1)).astype(np.float32)
    return grid_x, grid_v, unique_v


# --- query table and readout -------------------------------------------------


def test_query_table_keeps_first_occurrence_order_and_inverse():
    rng = np.random.default_rng(7)
    rows = rng.normal(size=(3, 2)).astype(np.float32)
    grid_v = np.stack([rows[2], rows[0], rows[2], rows[1], rows[0]], axis=0)
    grid_x = rng.normal(size=(5, 1)).astype(np.float32)
    table = build_query_table(grid_x, grid_v)
    np.testing.assert_array_equal(table.unique_v, np.stack([rows[2], rows[0], rows[1]], axis=0))
    np.testing.assert_array_equal(table.query_inverse, [0, 1, 0, 2, 1])
    np.testing.assert_array_equal(table.query_x, grid_x)
    np.testing.assert_array_equal(table.unique_v[table.query_inverse], grid_v)
    assert len(table.catalog_hash) == 64 and table.query_hash != table.catalog_hash


def test_functional_readout_matches_direct_outcome_head():
    model = _tiny_demand_model()
    grid_x, grid_v, _ = _grid()
    table = build_query_table(grid_x, grid_v)
    latent = np.random.default_rng(31).normal(size=(6, 4, table.num_targets, 4)).astype(np.float32)
    functional = FunctionalReadout(model, table, time_chunk=2, query_chunk=3)(latent)
    assert functional.shape == (6, 4, table.num_queries) and functional.dtype == np.float32
    for q in range(table.num_queries):
        z = latent[:, :, table.query_inverse[q], :].reshape(-1, 4)
        x = np.full((z.shape[0], 1), table.query_x[q, 0], np.float32)
        expected = model._outcome_output(tf.constant(z), tf.constant(x)).numpy()[:, 0].reshape(6, 4)
        np.testing.assert_allclose(functional[:, :, q], expected, rtol=0, atol=1e-6)


def test_functional_readout_detects_outcome_mutation():
    model = _tiny_demand_model()
    grid_x, grid_v, _ = _grid(num_unique=2, prices_per_v=2)
    table = build_query_table(grid_x, grid_v)
    readout = FunctionalReadout(model, table)
    weights = model.f_net.weights[0]
    original = weights.numpy()
    weights.assign(original + 1e-3)
    try:
        with pytest.raises(ReadoutError):
            readout.assert_runtime_identity()
        latent = np.random.default_rng(5).normal(size=(4, 4, table.num_targets, 4)).astype(np.float32)
        with pytest.raises(ReadoutError):
            readout(latent)
    finally:
        weights.assign(original)
    readout.assert_runtime_identity()


def test_batch_query_view_restricts_and_reindexes():
    grid_x, grid_v, unique_v = _grid(num_unique=5, prices_per_v=3)
    table = build_query_table(grid_x, grid_v)
    view = batch_query_view(table, (3, 1))
    np.testing.assert_array_equal(view.unique_v, unique_v[[3, 1]])
    keep = np.isin(table.query_inverse, [3, 1])
    np.testing.assert_array_equal(view.query_x, table.query_x[keep])
    np.testing.assert_array_equal(view.unique_v[view.query_inverse], table.unique_v[table.query_inverse[keep]])
    with pytest.raises(ReadoutError):
        batch_query_view(table, (9,))
    with pytest.raises(ReadoutError):
        batch_query_view(table, (1, 1))


def _duplicate_row_grid():
    rng = np.random.default_rng(1)
    unique_v = rng.normal(size=(3, 2)).astype(np.float32)
    # target 0 appears twice (rows 0 and 3), targets 1 and 2 once
    grid_v = np.stack([unique_v[0], unique_v[1], unique_v[2], unique_v[0]], axis=0)
    grid_x = np.array([[0.2], [-0.1], [0.4], [0.9]], np.float32)
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    return grid_x, grid_v, truth


def test_representative_rows_and_row_length_guard():
    grid_x, grid_v, truth = _duplicate_row_grid()
    table = build_query_table(grid_x, grid_v)
    rep = table.representative_rows
    np.testing.assert_array_equal(table.query_inverse[rep], np.arange(table.num_targets))
    np.testing.assert_array_equal(rep, [0, 1, 2])
    np.testing.assert_array_equal(table.query_rows_of_targets([0]), [0, 3])
    with pytest.raises(ReadoutError, match="representative_rows"):
        table.row_length_array(truth[: table.num_targets], "truth")


# --- predictive calibration --------------------------------------------------


def test_mixture_quantiles_are_exact():
    rng = np.random.default_rng(0)
    means = rng.normal(size=(3, 40)) * 3.0
    sds = np.exp(rng.normal(size=(3, 40)) * 0.3)
    probs = np.array([0.025, 0.1, 0.5, 0.9, 0.975])
    exact = gaussian_mixture_quantiles(means, sds, probs)
    samples = means[:, :, None] + sds[:, :, None] * rng.standard_normal((3, 40, 4000))
    empirical = np.quantile(samples.reshape(3, -1), probs, axis=1).T
    np.testing.assert_allclose(exact, empirical, atol=0.08)
    cdf = ndtr((exact[:, :, None] - means[:, None, :]) / sds[:, None, :]).mean(axis=2)
    np.testing.assert_allclose(cdf, np.broadcast_to(probs, cdf.shape), atol=1e-9)
    single = gaussian_mixture_quantiles(np.full((1, 5), 2.0), np.full((1, 5), 0.5), np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(single[0], 2.0 + 0.5 * ndtri([0.1, 0.5, 0.9]), atol=1e-9)


def _outcome_from_model(model, z, x):
    out = model._outcome_output(tf.constant(z, tf.float32), tf.constant(x, tf.float32))
    mu = np.asarray(out[:, 0], np.float64)
    sd = np.sqrt(np.asarray(model._continuous_sigma(out, sigma_key="sigma_y"), np.float64).reshape(-1))
    return mu, sd


def test_target_unit_scores_the_representative_row():
    model = _tiny_demand_model(seed=5)
    grid_x, grid_v, truth = _duplicate_row_grid()
    table = build_query_table(grid_x, grid_v)
    draws = np.random.default_rng(2).normal(size=(30, 4, 3, 4)).astype(np.float32)
    outcome = BatchOutcome(batch_index=0, global_ids=(0, 1, 2), state="STATIONARITY_RESTART", reason="x")
    acc = PredictiveCalibrationAccumulator(
        model, table, truth, outcome_shift=10.0, outcome_scale=2.0,
        config=CalibrationConfig(num_draws=120, scoring_unit="target"),
    )
    record = acc.add_batch(outcome, draws)
    assert record["units"] == 3
    # target 0 is scored at row 0 (x = 0.2, truth 1.0), not at row 3
    flat = draws.reshape(120, 3, 4)
    mu, sd = _outcome_from_model(model, flat[:, 0, :], np.full((120, 1), 0.2, np.float32))
    q = gaussian_mixture_quantiles((mu * 2.0 + 10.0)[None, :], (sd * 2.0)[None, :], np.array([0.1, 0.9]))[0]
    expected_cov80 = ndtr(q[1] - 1.0) - ndtr(q[0] - 1.0)
    assert abs(acc.channels["diagnostic"].coverage[0.8][0] - expected_cov80) < 1e-6
    summary = acc.summary()
    assert summary["units_calibrated"] == 0  # not REPORTABLE: certified channel stays empty
    assert summary["diagnostic_unscored"]["units"] == 3
    assert summary["config"]["quantile_method"].startswith("exact_gaussian_mixture")


def test_query_unit_covers_every_row_of_the_batch():
    model = _tiny_demand_model(seed=5)
    grid_x, grid_v, truth = _duplicate_row_grid()
    table = build_query_table(grid_x, grid_v)
    draws = np.random.default_rng(3).normal(size=(20, 4, 2, 4)).astype(np.float32)
    outcome = BatchOutcome(batch_index=0, global_ids=(0, 2), state="REPORTABLE", record={"metric": {}})
    acc = PredictiveCalibrationAccumulator(
        model, table, truth, outcome_shift=0.0, outcome_scale=1.0,
        config=CalibrationConfig(num_draws=80, scoring_unit="query"),
    )
    record = acc.add_batch(outcome, draws)
    assert record["units"] == 3  # rows 0, 2, 3
    summary = acc.summary()
    assert summary["units_calibrated"] == 3
    assert summary["diagnostic_unscored"]["units"] == 3
    assert set(summary["coverage_mean"]) == {"0.5", "0.8", "0.95"}
    with pytest.raises(ReadoutError, match="already calibrated"):
        acc.add_batch(outcome, draws)
