"""Full-grid all-draw structural MCMC readout."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import ndtr, ndtri

from bgm_iv.models.bgm_iv import BGM_IV
from bgm_iv.mcmc.readout import (
    FullGridReadout,
    ReadoutConfig,
    ReadoutError,
    build_query_table,
    gaussian_mixture_quantiles,
)


RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_readout_runtime_"))


def _tiny_model(seed: int = 611) -> BGM_IV:
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


def _duplicate_grid():
    unique = np.array([[0.1, -0.2], [0.5, 0.7]], np.float32)
    grid_v = unique[[0, 1, 0]]
    grid_x = np.array([[-0.5], [0.2], [0.8]], np.float32)
    truth = np.array([1.0, 2.0, 4.0], np.float64)
    return grid_x, grid_v, truth


def test_query_table_preserves_all_queries_and_deduplicates_targets():
    grid_x, grid_v, truth = _duplicate_grid()
    table = build_query_table(grid_x, grid_v)
    assert table.num_targets == 2 and table.num_queries == 3
    np.testing.assert_array_equal(table.query_inverse, [0, 1, 0])
    np.testing.assert_array_equal(table.unique_v[table.query_inverse], grid_v)
    np.testing.assert_array_equal(table.row_length_array(truth), truth)
    with pytest.raises(ReadoutError):
        table.row_length_array(truth[:2])


def test_gaussian_mixture_quantiles_are_exact():
    rng = np.random.default_rng(0)
    means = rng.normal(size=(3, 40)) * 3.0
    sds = np.exp(rng.normal(size=(3, 40)) * 0.3)
    probs = np.array([0.025, 0.1, 0.5, 0.9, 0.975])
    quantiles = gaussian_mixture_quantiles(means, sds, probs)
    cdf = ndtr(
        (quantiles[:, :, None] - means[:, None, :]) / sds[:, None, :]
    ).mean(axis=2)
    np.testing.assert_allclose(cdf, np.broadcast_to(probs, cdf.shape), atol=1e-9)
    single = gaussian_mixture_quantiles(
        np.full((1, 5), 2.0),
        np.full((1, 5), 0.5),
        np.array([0.1, 0.5, 0.9]),
    )
    np.testing.assert_allclose(
        single[0], 2.0 + 0.5 * ndtri([0.1, 0.5, 0.9]), atol=1e-9
    )


def test_full_grid_readout_uses_every_chain_draw_and_query():
    model = _tiny_model()
    grid_x, grid_v, truth = _duplicate_grid()
    table = build_query_table(grid_x, grid_v)
    latent = np.arange(3 * 4 * 2 * 4, dtype=np.float32).reshape(3, 4, 2, 4)
    latent = latent / 40.0 - 0.5
    result = FullGridReadout(
        model,
        table,
        truth,
        outcome_shift=10.0,
        outcome_scale=2.0,
        config=ReadoutConfig(query_chunk=2, draw_chunk=5, bisection_iterations=30),
    )(latent)
    assert result["num_queries"] == 3
    assert result["num_targets"] == 2
    assert result["num_chains"] == 4
    assert result["draws_per_chain"] == 3
    assert result["num_components"] == 12
    assert result["config"]["draw_usage"] == "all_post_warmup_draws"
    assert set(result["coverage"]) == {"0.5", "0.8", "0.95"}
    assert result["width50"] > 0.0
    assert result["width80"] > 0.0
    assert result["width95"] > 0.0
    assert result["width50"] < result["width80"] < result["width95"]

    flat = latent.reshape(12, 2, 4)
    expected_means = []
    chain_means = np.empty((4, 3), np.float64)
    for query in range(3):
        z = flat[:, table.query_inverse[query]]
        x = np.full((12, 1), grid_x[query, 0], np.float32)
        output = model._outcome_output(tf.constant(z), tf.constant(x))
        means = np.asarray(output[:, 0], np.float64) * 2.0 + 10.0
        expected_means.append(means.mean())
        for chain in range(4):
            chain_means[chain, query] = means[np.arange(12) % 4 == chain].mean()
    expected_plugin = np.mean((np.asarray(expected_means) - truth) ** 2)
    expected_penalty = np.mean(np.var(chain_means, axis=0, ddof=1) / 4.0)
    assert result["structural_mse_plugin"] == pytest.approx(expected_plugin)
    assert result["sensitivity"]["chain_mean_variance_penalty"] == pytest.approx(
        expected_penalty
    )
    assert result["sensitivity"]["penalty_fraction_of_plugin"] == pytest.approx(
        expected_penalty / expected_plugin
    )


def test_readout_rejects_nonfinite_draws_and_outcome_mutation():
    model = _tiny_model(seed=19)
    grid_x, grid_v, truth = _duplicate_grid()
    table = build_query_table(grid_x, grid_v)
    readout = FullGridReadout(
        model,
        table,
        truth,
        outcome_shift=0.0,
        outcome_scale=1.0,
        config=ReadoutConfig(query_chunk=2, draw_chunk=4, bisection_iterations=20),
    )
    bad = np.zeros((2, 4, 2, 4), np.float32)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ReadoutError, match="finite"):
        readout(bad)
    variable = model.f_net.trainable_variables[0]
    original = variable.numpy()
    variable.assign_add(tf.ones_like(variable) * 0.01)
    try:
        with pytest.raises(ReadoutError, match="changed"):
            readout.assert_runtime_identity()
    finally:
        variable.assign(original)
