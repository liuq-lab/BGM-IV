from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import ndtr, ndtri

from bgm_iv.models.bgm_iv import BGM_IV
from bgm_iv.mcmc import readout as readout_module
from bgm_iv.mcmc.readout import (
    FullGridReadout,
    ReadoutError,
    build_query_table,
    gaussian_mixture_quantiles,
)


RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_readout_runtime_"))


def _tiny_model(seed: int = 611) -> BGM_IV:
    params = {
        "dataset": "Readout_demand",
        "output_dir": str(RUNTIME_ROOT),
        "save_model": False,
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
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "structural_map_steps": 100,
        "structural_map_lr": 5e-4,
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
    model = _tiny_model()
    with pytest.raises(ReadoutError, match="one entry per query row"):
        FullGridReadout(model, table, truth[:2], outcome_shift=0.0, outcome_scale=1.0)


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


def test_readout_settings_are_the_fixed_constants():
    assert readout_module.LEVELS == (0.9, 0.95, 0.99)
    assert readout_module.TRUTH_NOISE_SD == 1.0
    assert readout_module.QUERY_CHUNK == 16
    assert readout_module.DRAW_CHUNK == 16384
    assert readout_module.BISECTION_ITERATIONS == 60


def test_full_grid_readout_uses_every_chain_draw_and_query(monkeypatch):
    monkeypatch.setattr(readout_module, "QUERY_CHUNK", 2)
    monkeypatch.setattr(readout_module, "DRAW_CHUNK", 5)
    monkeypatch.setattr(readout_module, "BISECTION_ITERATIONS", 30)
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
    )(latent)
    assert result["num_queries"] == 3
    assert result["num_targets"] == 2
    assert result["num_chains"] == 4
    assert result["draws_per_chain"] == 3
    assert result["num_components"] == 12
    assert set(result["coverage"]) == {"0.9", "0.95", "0.99"}
    assert result["width90"] > 0.0
    assert result["width95"] > 0.0
    assert result["width99"] > 0.0
    assert result["width90"] < result["width95"] < result["width99"]

    flat = latent.reshape(12, 2, 4)

    from scipy.optimize import brentq

    coverage = {level: [] for level in (0.9, 0.95, 0.99)}
    widths = {level: [] for level in (0.9, 0.95, 0.99)}
    for query in range(3):
        z = flat[:, table.query_inverse[query]]
        x = np.full((12, 1), grid_x[query, 0], np.float32)
        output = model._outcome_output(tf.constant(z), tf.constant(x))
        means = np.asarray(output[:, 0], np.float64) * 2.0 + 10.0
        sds = np.sqrt(
            np.asarray(model._continuous_sigma(output, sigma_key="sigma_y")).reshape(-1)
        ).astype(np.float64) * 2.0

        def quantile(prob):
            return brentq(
                lambda q: ndtr((q - means) / sds).mean() - prob,
                means.min() - 20 * sds.max(),
                means.max() + 20 * sds.max(),
                xtol=1e-12,
            )

        for level in coverage:
            lo, hi = quantile((1 - level) / 2), quantile(1 - (1 - level) / 2)
            coverage[level].append(ndtr(hi - truth[query]) - ndtr(lo - truth[query]))
            widths[level].append(hi - lo)
    for level, key in ((0.9, "0.9"), (0.95, "0.95"), (0.99, "0.99")):
        assert result["coverage"][key] == pytest.approx(np.mean(coverage[level]), abs=1e-6)
        assert result[f"width{int(round(level * 100))}"] == pytest.approx(
            np.mean(widths[level]), rel=1e-6
        )


def test_readout_rejects_nonfinite_draws(monkeypatch):
    monkeypatch.setattr(readout_module, "QUERY_CHUNK", 2)
    monkeypatch.setattr(readout_module, "DRAW_CHUNK", 4)
    model = _tiny_model(seed=19)
    grid_x, grid_v, truth = _duplicate_grid()
    table = build_query_table(grid_x, grid_v)
    readout = FullGridReadout(
        model,
        table,
        truth,
        outcome_shift=0.0,
        outcome_scale=1.0,
    )
    bad = np.zeros((2, 4, 2, 4), np.float32)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ReadoutError, match="finite"):
        readout(bad)
