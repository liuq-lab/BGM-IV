"""Image-representation export and preprocessing tests."""

import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.features import (
    export_image_representations,
    fit_feature_standardizer,
    image_encoder_features,
    within_cluster_floor_rule,
)
from bgm_iv.models.networks import DemandImageFeatureExtractor
from bgm_iv.mcmc.target import (
    FeaturePreprocessorSpec,
    PCAFeaturePreprocessorSpec,
    sha256_weights,
)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _fake_split(n, seed, v_dim=785):
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, 8, size=n)
    images = np.stack(
        [(rng.integers(0, 256, size=784) * (0.3 + 0.1 * lab)).clip(0, 255) for lab in labels]
    ).astype(np.float32)
    time = rng.uniform(0, 10, size=(n, 1)).astype(np.float32)
    parts = [time, images]
    if v_dim > 785:
        parts.append(rng.normal(size=(n, v_dim - 785)).astype(np.float32))
    return {
        "v": np.concatenate(parts, axis=1).astype(np.float32),
        "customer_group": labels.reshape(-1, 1).astype(np.float32),
    }


def test_trunk_features_are_inference_mode_and_duplicate_rows_agree():
    trunk = DemandImageFeatureExtractor(v_dim=785)
    trunk(tf.zeros((1, 785)), training=False)
    split = _fake_split(12, 0)
    v = np.concatenate([split["v"], split["v"][:3]], axis=0)  # rows 12..14 duplicate 0..2
    phi = image_encoder_features(trunk, v)
    assert phi.shape == (15, 64)
    np.testing.assert_array_equal(phi[12:15], phi[:3])
    again = image_encoder_features(trunk, v)
    np.testing.assert_array_equal(phi, again)  # dropout is off


def test_floor_rule_is_label_free_deterministic_and_bounded_below():
    rng = np.random.default_rng(4)
    labels_tr = np.repeat(np.arange(7), 20)
    labels_ho = np.repeat(np.arange(7), 10)
    centers = rng.normal(size=(7, 6)) * 3
    feat_tr = centers[labels_tr] + 0.2 * rng.normal(size=(140, 6))
    feat_ho = centers[labels_ho] + 0.2 * rng.normal(size=(70, 6))
    stats = fit_feature_standardizer(feat_tr)
    std_tr = (feat_tr - stats["mean"]) / stats["scale"]
    std_ho = (feat_ho - stats["mean"]) / stats["scale"]
    rule = within_cluster_floor_rule(std_tr, std_ho, factor=0.5, minimum=0.02, clusters=7)
    assert abs(rule["sigma_vector_softfloor"] - round(0.5 * rule["within_cluster_resid_sd_median"], 3)) < 1e-9
    assert rule["n_holdout"] == 70 and rule["clusters"] == 7
    # well-separated clusters: within-cluster sd is the 0.2 noise in standardized units
    within_class = (std_ho - np.stack([std_tr[labels_tr == c].mean(0) for c in range(7)])[labels_ho]).std(0, ddof=1)
    assert abs(rule["within_cluster_resid_sd_median"] - np.median(within_class)) < 0.05
    assert rule == within_cluster_floor_rule(std_tr, std_ho, factor=0.5, minimum=0.02, clusters=7)
    tiny = within_cluster_floor_rule(std_tr, std_ho, factor=1e-6, minimum=0.02, clusters=7)
    assert tiny["sigma_vector_softfloor"] == 0.02
    with pytest.raises(ValueError, match="clusters"):
        within_cluster_floor_rule(std_tr[:3], std_ho, clusters=7)


def test_export_hashes_standardizes_with_train_stats_and_passes_noise_through():
    trunk = DemandImageFeatureExtractor(v_dim=1000)
    trunk(tf.zeros((1, 1000)), training=False)
    train, grid, holdout = _fake_split(70, 1, 1000), _fake_split(21, 2, 1000), _fake_split(35, 3, 1000)
    export = export_image_representations(
        trunk=trunk, train=train, grid=grid, holdout=holdout, pixel_v_dim=1000, feature_map="egm"
    )
    assert export.train_v.shape == (70, 1 + 64 + 215)
    assert export.grid_v.shape == (21, 280)
    np.testing.assert_array_equal(export.train_v[:, :1], train["v"][:, :1])
    np.testing.assert_array_equal(export.train_v[:, 65:], train["v"][:, 785:])
    phi_std = export.train_v[:, 1:65]
    np.testing.assert_allclose(phi_std.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(phi_std.std(axis=0, ddof=1), 1.0, atol=2e-2)
    assert export.trunk_weights_sha256 == sha256_weights(trunk)
    assert export.hashes["phi_train_sha256"] != export.hashes["phi_grid_sha256"]
    payload = export.to_payload()
    assert payload["noise_dim"] == 215 and payload["feature_map"] == "egm"


def test_pca_control_export_has_no_trunk():
    train, grid, holdout = _fake_split(80, 5), _fake_split(14, 6), _fake_split(30, 7)
    export = export_image_representations(
        trunk=None, train=train, grid=grid, holdout=holdout, pixel_v_dim=785, feature_map="pca"
    )
    assert export.trunk_weights_sha256 is None
    assert export.pca["components"].shape == (64, 784)
    assert export.train_v.shape == (80, 65)


def test_feature_preprocessor_spec_identity_and_transform():
    trunk = DemandImageFeatureExtractor(v_dim=785)
    trunk(tf.zeros((1, 785)), training=False)
    split = _fake_split(16, 8)
    phi = image_encoder_features(trunk, split["v"])
    stats = fit_feature_standardizer(phi)
    spec = FeaturePreprocessorSpec(
        trunk=trunk, mean=stats["mean"], scale=stats["scale"], input_dimension=785,
        pixel_checkpoint={"output_dir": "x", "timestamp": "t0"},
    )
    assert spec.dimension == 785 and spec.model_dimension == 65
    model_v = spec.transform(split["v"])
    assert model_v.shape == (16, 65)
    np.testing.assert_array_equal(model_v[:, 0], split["v"][:, 0])
    np.testing.assert_allclose(model_v[:, 1:], (phi - stats["mean"]) / stats["scale"], rtol=1e-5, atol=1e-5)
    identity = spec.identity
    other_stats = FeaturePreprocessorSpec(
        trunk=trunk, mean=stats["mean"] + 1.0, scale=stats["scale"], input_dimension=785,
        pixel_checkpoint={"output_dir": "x", "timestamp": "t0"},
    )
    assert other_stats.identity != identity
    # mutate the trunk -> identity changes and the bound spec refuses to run
    trunk.weights[0].assign(trunk.weights[0] + 1e-3)
    fresh = FeaturePreprocessorSpec(
        trunk=trunk, mean=stats["mean"], scale=stats["scale"], input_dimension=785,
        pixel_checkpoint={"output_dir": "x", "timestamp": "t0"},
    )
    assert fresh.identity != identity
    with pytest.raises(RuntimeError, match="image encoder weights changed"):
        spec.transform(split["v"])


def test_pca_preprocessor_spec_matches_export():
    train, grid, holdout = _fake_split(80, 9), _fake_split(14, 10), _fake_split(30, 11)
    export = export_image_representations(
        trunk=None, train=train, grid=grid, holdout=holdout, pixel_v_dim=785, feature_map="pca"
    )
    spec = PCAFeaturePreprocessorSpec(
        components=export.pca["components"], pca_mean=export.pca["mean"],
        mean=export.stats["mean"], scale=export.stats["scale"], input_dimension=785,
    )
    np.testing.assert_allclose(spec.transform(grid["v"]), export.grid_v, rtol=1e-4, atol=1e-4)
    assert spec.model_dimension == 65
