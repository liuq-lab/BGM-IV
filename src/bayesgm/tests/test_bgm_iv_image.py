from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import main as main_module

from bayesgm.datasets import make_demand_design_grid, simulate_demand_design_iv
from bayesgm.datasets.simulator_image import (
    make_demand_design_mnist_grid,
    simulate_demand_design_mnist_iv,
)
from bayesgm.models.bgm_iv import BGM_IV_Image
from bayesgm.models.networks import DemandImageFeatureExtractor
import bayesgm.datasets.simulator_image as simulator_image_module

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _fake_mnist_arrays():
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    base_pattern = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28)
    for digit in range(10):
        for replica in range(4):
            train_images.append(((base_pattern + digit * 11 + replica) % 256).astype(np.uint8))
            train_labels.append(digit)
            test_images.append(((base_pattern + digit * 17 + replica + 5) % 256).astype(np.uint8))
            test_labels.append(digit)

    return {
        "train_images": np.stack(train_images, axis=0),
        "train_labels": np.asarray(train_labels, dtype=np.int64),
        "test_images": np.stack(test_images, axis=0),
        "test_labels": np.asarray(test_labels, dtype=np.int64),
    }


@pytest.fixture
def fake_mnist(monkeypatch):
    arrays = _fake_mnist_arrays()
    monkeypatch.setattr(simulator_image_module, "_load_mnist_arrays", lambda: arrays)
    return arrays


def _reference_attach_images(labels, images, image_labels, seed):
    rng = np.random.default_rng(seed)
    attached = []
    for label in np.asarray(labels, dtype=np.int64).reshape(-1):
        candidates = images[image_labels == label]
        attached.append(candidates[rng.choice(len(candidates))].reshape(1, -1))
    return np.concatenate(attached, axis=0).astype(np.float32)


def _make_image_params(output_dir):
    return {
        "dataset": "Sim_Demand_Design_Mnist_IV",
        "output_dir": str(output_dir),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 785,
        "w_dim": 1,
        "image_seed": 42,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [16, 16],
        "e_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "dz_units": [16, 8],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "first_stage_warmup_epochs": 0,
        "structural_map_steps": 2,
        "latent_pzv_weight": 0.5,
    }


def test_simulate_demand_design_mnist_iv_matches_reference_construction(fake_mnist):
    seed = 7
    train = simulate_demand_design_mnist_iv(n_samples=12, rho=0.5, seed=seed, v_dim=785)
    base = simulate_demand_design_iv(n_samples=12, rho=0.5, seed=seed)
    expected_images = _reference_attach_images(
        base["customer_group"].astype(np.int64).reshape(-1),
        fake_mnist["train_images"],
        fake_mnist["train_labels"],
        seed,
    )

    np.testing.assert_allclose(train["x"], base["x"], atol=1e-6)
    np.testing.assert_allclose(train["y"], base["y"], atol=1e-6)
    np.testing.assert_allclose(train["w"], base["w"], atol=1e-6)
    np.testing.assert_allclose(train["y_struct"], base["y_struct"], atol=1e-6)
    np.testing.assert_allclose(train["v"][:, :1], base["time"], atol=1e-6)
    np.testing.assert_allclose(train["v"][:, 1:], expected_images, atol=0.0)


def test_make_demand_design_mnist_grid_matches_reference_construction(fake_mnist):
    image_seed = 42
    grid = make_demand_design_mnist_grid(
        price_points=3,
        time_points=2,
        v_dim=785,
        image_seed=image_seed,
    )
    base_grid = make_demand_design_grid(
        price_points=3,
        time_points=2,
    )
    expected_images = _reference_attach_images(
        base_grid["customer_group"].astype(np.int64).reshape(-1),
        fake_mnist["test_images"],
        fake_mnist["test_labels"],
        image_seed,
    )

    np.testing.assert_allclose(grid["x"], base_grid["x"], atol=1e-6)
    np.testing.assert_allclose(grid["y_struct"], base_grid["y_struct"], atol=1e-6)
    np.testing.assert_allclose(grid["v"][:, :1], base_grid["time"], atol=1e-6)
    np.testing.assert_allclose(grid["v"][:, 1:], expected_images, atol=0.0)


def test_mnist_grid_image_seed_changes_only_images(fake_mnist):
    grid_a = make_demand_design_mnist_grid(price_points=2, time_points=2, v_dim=785, image_seed=42)
    grid_b = make_demand_design_mnist_grid(price_points=2, time_points=2, v_dim=785, image_seed=99)

    np.testing.assert_allclose(grid_a["x"], grid_b["x"], atol=0.0)
    np.testing.assert_allclose(grid_a["v"][:, :1], grid_b["v"][:, :1], atol=0.0)
    assert not np.allclose(grid_a["v"][:, 1:], grid_b["v"][:, 1:])


def test_mnist_simulator_rejects_too_small_v_dim(fake_mnist):
    with pytest.raises(ValueError, match="v_dim >= 785"):
        simulate_demand_design_mnist_iv(n_samples=8, rho=0.5, seed=1, v_dim=200)

    with pytest.raises(ValueError, match="v_dim >= 785"):
        make_demand_design_mnist_grid(price_points=2, time_points=2, v_dim=200, image_seed=42)


def test_mnist_simulator_appends_gaussian_covariates_without_changing_dgp(fake_mnist):
    seed = 11
    base = simulate_demand_design_mnist_iv(n_samples=10, rho=0.25, seed=seed, v_dim=785)
    noisy = simulate_demand_design_mnist_iv(n_samples=10, rho=0.25, seed=seed, v_dim=1000)

    assert base["v"].shape == (10, 785)
    assert noisy["v"].shape == (10, 1000)
    np.testing.assert_allclose(noisy["v"][:, :785], base["v"], atol=0.0)
    assert noisy["v"][:, 785:].shape == (10, 215)
    assert abs(float(np.mean(noisy["v"][:, 785:]))) < 0.5
    assert 0.5 < float(np.std(noisy["v"][:, 785:])) < 1.5

    for key in ("x", "y", "y_struct", "w", "time", "customer_group"):
        np.testing.assert_allclose(noisy[key], base[key], atol=1e-6)


def test_mnist_grid_appends_deterministic_gaussian_covariates(fake_mnist):
    grid_a = make_demand_design_mnist_grid(
        price_points=2,
        time_points=2,
        v_dim=1000,
        image_seed=42,
        noise_seed=7,
    )
    grid_b = make_demand_design_mnist_grid(
        price_points=2,
        time_points=2,
        v_dim=1000,
        image_seed=42,
        noise_seed=7,
    )
    grid_c = make_demand_design_mnist_grid(
        price_points=2,
        time_points=2,
        v_dim=1000,
        image_seed=42,
        noise_seed=8,
    )
    grid_base = make_demand_design_mnist_grid(
        price_points=2,
        time_points=2,
        v_dim=785,
        image_seed=42,
    )

    assert grid_a["v"].shape[1] == 1000
    np.testing.assert_allclose(grid_a["v"][:, :785], grid_base["v"], atol=0.0)
    np.testing.assert_allclose(grid_a["v"], grid_b["v"], atol=0.0)
    np.testing.assert_allclose(grid_a["v"][:, :785], grid_c["v"][:, :785], atol=0.0)
    assert not np.allclose(grid_a["v"][:, 785:], grid_c["v"][:, 785:])


def test_image_feature_extractor_splits_image_and_noise_blocks():
    extractor = DemandImageFeatureExtractor(v_dim=1000)
    data = tf.zeros((2, 1000), dtype=tf.float32)
    features = extractor(data, training=False)

    assert features.shape == (2, 97)


def test_bgm_iv_image_accepts_mnist_hd_v_dim(tmp_path):
    params = _make_image_params(tmp_path)
    params["v_dim"] = 1000
    model = BGM_IV_Image(params=params, random_seed=13)

    data_v = tf.zeros((2, 1000), dtype=tf.float32)
    data_z = model.e_net(data_v, training=False)
    decoded = model._decode_covariates(data_z, training=False)

    assert data_z.shape == (2, sum(params["z_dims"]))
    assert decoded["public_v"].shape == (2, 1000)
    assert decoded["image_probs"].shape == (2, 784)
    assert decoded["noise_mean"].shape == (2, 215)


def test_bgm_iv_image_smoke(fake_mnist, tmp_path):
    train = simulate_demand_design_mnist_iv(n_samples=24, rho=0.5, seed=3, v_dim=785)
    grid = make_demand_design_mnist_grid(price_points=3, time_points=2, v_dim=785, image_seed=42)
    train_std, grid_std, stats = main_module._standardize_demand_design_image_data(train, grid)

    params = _make_image_params(tmp_path)
    model = BGM_IV_Image(params=params, random_seed=13)
    model.fit(
        data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
        epochs=1,
        epochs_per_eval=1,
        batch_size=8,
        use_egm_init=True,
        egm_n_iter=0,
        egm_batches_per_eval=1,
        verbose=0,
        first_stage_warmup_epochs=0,
    )

    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
        data_z=None,
        nb_intervals=4,
    )
    assert causal_pre.shape == (4,)
    assert np.isfinite(float(mse_x))
    assert np.isfinite(float(mse_y))
    assert np.isfinite(float(mse_v))
    assert float(mse_v) < 100.0

    structural_pred = model.predict_structural(
        grid_std["x"],
        grid_std["v"],
        latent_method="map",
        map_steps=2,
    )
    structural_pred = main_module._inverse_transform(structural_pred, stats["y"])
    structural_mse = float(np.mean((grid["y_struct"] - structural_pred) ** 2))
    assert structural_pred.shape == grid["y_struct"].shape
    assert np.isfinite(structural_mse)


def test_image_model_rejects_alpha_v(fake_mnist, tmp_path):
    params = _make_image_params(tmp_path)
    params["alpha_v"] = 1.0

    with pytest.raises(ValueError, match="alpha_v"):
        BGM_IV_Image(params=params, random_seed=3)


def test_image_covariate_posterior_is_untempered(fake_mnist, tmp_path):
    train = simulate_demand_design_mnist_iv(n_samples=4, rho=0.5, seed=3, v_dim=785)
    data_v = tf.constant(train["v"][:2], dtype=tf.float32)
    data_z = tf.constant(
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.1, -0.3, 0.2]], dtype=tf.float32
    )

    params = _make_image_params(tmp_path)
    model = BGM_IV_Image(params=params, random_seed=5)

    loss_pv_z, _, _ = model._covariate_loss_terms(data_v, data_z, training=False)
    loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
    expected = -(loss_pv_z + loss_prior_z)
    actual = model.get_log_covariate_posterior(data_v, data_z)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_mnist_runner_helpers_use_mnist_slug():
    params = {
        "dataset": "Sim_Demand_Design_Mnist_IV",
        "n_samples": 1000,
        "rho": 0.5,
        "n_repeat": 2,
        "repeat_id": 1,
        "seed": 0,
        "run_seed": 1,
        "image_seed": 42,
        "noise_seed": 99,
        "z_dims": [2, 1, 1, 7],
        "v_dim": 1000,
        "w_dim": 1,
        "latent_pzv_weight": 0.5,
    }

    text = main_module._render_demand_design_run_config(params)
    assert "image_seed: 42" in text
    assert "noise_seed: 99" in text
    assert (
        main_module._demand_design_active_window_path(params).name
        == "outputs_dev_sim_demand_design_mnist_iv_active.md"
    )


def test_parallel_worker_dispatches_mnist_runner(monkeypatch):
    seen = {}

    def fake_run(run_params):
        seen["dataset"] = run_params["dataset"]
        seen["repeat_id"] = run_params["repeat_id"]
        print("mnist worker executed")
        return {
            "training_history": [],
            "training_history_text": "",
            "run_config_text": "mnist run",
            "ranges_text": "mnist ranges",
        }

    monkeypatch.setattr(main_module, "_run_single_demand_design_mnist_iv", fake_run)
    result = main_module._run_demand_design_parallel_worker(
        2,
        4,
        {
            "dataset": "Sim_Demand_Design_Mnist_IV",
            "n_samples": 64,
            "rho": 0.5,
            "n_repeat": 2,
            "repeat_id": 1,
            "run_seed": 1,
            "image_seed": 42,
            "v_dim": 785,
            "w_dim": 1,
        },
    )

    assert result["error"] is None
    assert seen == {"dataset": "Sim_Demand_Design_Mnist_IV", "repeat_id": 1}
    assert "Demand-design sweep run [2/4]" in result["stdout"]
    assert "mnist worker executed" in result["stdout"]
    assert result["repeat_outputs"]["run_config_text"] == "mnist run"
