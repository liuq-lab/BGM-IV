import numpy as np
import pytest
import tensorflow as tf
import main as main_module

from bgm_iv.datasets import make_demand_design_grid, simulate_demand_design_iv
from bgm_iv.datasets.simulator_image import (
    make_demand_design_mnist_grid,
    simulate_demand_design_mnist_iv,
)
from bgm_iv.models.bgm_iv import BGM_IV_Image
from bgm_iv.models.networks import DemandImageFeatureExtractor
import bgm_iv.datasets.simulator_image as simulator_image_module

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
        "save_model": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 785,
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [16, 16],
        "e_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "dz_units": [16, 8],
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "structural_map_steps": 2,
        "structural_map_lr": 5e-4,
    }


def _state_values(model):
    networks = (model.g_net, model.e_net, model.f_net, model.h_net, model.dz_net)
    values = [w.numpy().copy() for net in networks for w in net.weights]
    return values + [model.egm_sigma2_x_ema.numpy().copy()]


def test_image_model_state_checkpoint_is_strict_and_optimizer_free(tmp_path):
    params = _make_image_params(tmp_path)
    model = BGM_IV_Image(params=params, random_seed=13)
    model.egm_sigma2_x_ema.assign(0.321)
    expected = _state_values(model)
    checkpoint = model.save_model_state_checkpoint(
        tmp_path / "inference-state" / "ckpt"
    )
    variable_names = [name for name, _ in tf.train.list_variables(checkpoint)]
    assert not any("optimizer" in name.lower() for name in variable_names)
    assert not any("data_z" in name for name in variable_names)
    allowed_roots = (
        "g_net/",
        "e_net/",
        "f_net/",
        "h_net/",
        "dz_net/",
        "egm_sigma2_x_ema/",
        "save_counter/",
        "_CHECKPOINTABLE_OBJECT_GRAPH",
    )
    assert all(name.startswith(allowed_roots) for name in variable_names)

    restored = BGM_IV_Image(
        params=params,
        random_seed=99,
        auto_restore_checkpoint=False,
    )
    restored.restore_model_state_checkpoint(checkpoint)
    actual = _state_values(restored)
    assert len(actual) == len(expected)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_simulate_demand_design_mnist_iv_matches_reference_construction(fake_mnist):
    seed = 7
    train = simulate_demand_design_mnist_iv(n_samples=12, rho=0.5, seed=seed)
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
    np.testing.assert_allclose(train["v"][:, :1], base["v"][:, :1], atol=0.0)
    np.testing.assert_allclose(train["v"][:, 1:], expected_images, atol=0.0)


def test_make_demand_design_mnist_grid_matches_reference_construction(fake_mnist):
    image_seed = 42
    grid = make_demand_design_mnist_grid(image_seed=image_seed)
    base_grid = make_demand_design_grid()
    expected_images = _reference_attach_images(
        base_grid["customer_group"].astype(np.int64).reshape(-1),
        fake_mnist["test_images"],
        fake_mnist["test_labels"],
        image_seed,
    )

    np.testing.assert_allclose(grid["x"], base_grid["x"], atol=1e-6)
    np.testing.assert_allclose(grid["y_struct"], base_grid["y_struct"], atol=1e-6)
    np.testing.assert_allclose(grid["v"][:, :1], base_grid["v"][:, :1], atol=0.0)
    np.testing.assert_allclose(grid["v"][:, 1:], expected_images, atol=0.0)


def test_mnist_grid_image_seed_changes_only_images(fake_mnist):
    grid_a = make_demand_design_mnist_grid(image_seed=42)
    grid_b = make_demand_design_mnist_grid(image_seed=99)

    np.testing.assert_allclose(grid_a["x"], grid_b["x"], atol=0.0)
    np.testing.assert_allclose(grid_a["v"][:, :1], grid_b["v"][:, :1], atol=0.0)
    assert not np.allclose(grid_a["v"][:, 1:], grid_b["v"][:, 1:])


def test_image_networks_expect_time_plus_one_image():
    extractor = DemandImageFeatureExtractor(v_dim=785)
    features = extractor(tf.zeros((2, 785), dtype=tf.float32), training=False)
    assert features.shape == (2, 65)
    with pytest.raises(ValueError):
        DemandImageFeatureExtractor(v_dim=1000)


def test_bgm_iv_image_requires_v_dim_785(tmp_path):
    params = _make_image_params(tmp_path)
    params["v_dim"] = 1000
    with pytest.raises(ValueError, match="785"):
        BGM_IV_Image(params=params, random_seed=13)


def test_bgm_iv_image_smoke(fake_mnist, tmp_path):
    train = simulate_demand_design_mnist_iv(n_samples=24, rho=0.5, seed=3)
    grid = {key: value[::200] for key, value in make_demand_design_mnist_grid().items()}
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
        verbose=0,
    )

    mse_x, mse_y, mse_v = model.evaluate(
        data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
        data_z=None,
    )
    assert np.isfinite(float(mse_x))
    assert np.isfinite(float(mse_y))
    assert np.isfinite(float(mse_v))
    assert float(mse_v) < 100.0

    structural_pred = model.predict_structural(
        grid_std["x"],
        grid_std["v"],
        map_steps=2,
    )
    structural_pred = main_module._inverse_transform(structural_pred, stats["y"])
    structural_mse = float(np.mean((grid["y_struct"] - structural_pred) ** 2))
    assert structural_pred.shape == grid["y_struct"].shape
    assert np.isfinite(structural_mse)


def test_image_covariate_posterior_is_untempered(fake_mnist, tmp_path):
    train = simulate_demand_design_mnist_iv(n_samples=4, rho=0.5, seed=3)
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
        "run_seed": 1,
        "z_dims": [2, 1, 1, 2],
        "v_dim": 785,
        "w_dim": 1,
    }

    text = main_module._render_demand_design_run_config(params)
    assert "run_seed: 1" in text
    assert "z_dims: [2, 1, 1, 2]" in text
    assert main_module._build_demand_design_run_id(params).startswith(
        "sim_demand_design_mnist_iv_"
    )
