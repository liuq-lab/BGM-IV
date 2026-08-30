import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.datasets import (
    make_demand_design_grid,
    make_demand_design_vector_grid,
    simulate_demand_design_iv,
    simulate_demand_design_vector_iv,
)
from bgm_iv.models.bgm_iv import BGM_IV_Vector
from bgm_iv.models.networks import DemandVectorFeatureExtractor


def _make_vector_params(output_dir):
    return {
        "dataset": "Sim_Demand_Design_Vector_IV",
        "output_dir": str(output_dir),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "vector_dim": 784,
        "v_dim": 785,
        "w_dim": 1,
        "feature_seed": 42,
        "test_vector_seed": 42,
        "representation_sd": 0.5,
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
    }


def test_simulate_demand_design_vector_iv_preserves_dgp_and_replaces_s():
    seed = 7
    train = simulate_demand_design_vector_iv(
        n_samples=12,
        rho=0.5,
        seed=seed,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        representation_sd=0.5,
    )
    base = simulate_demand_design_iv(n_samples=12, rho=0.5, seed=seed)

    np.testing.assert_allclose(train["x"], base["x"], atol=1e-6)
    np.testing.assert_allclose(train["y"], base["y"], atol=1e-6)
    np.testing.assert_allclose(train["w"], base["w"], atol=1e-6)
    np.testing.assert_allclose(train["y_struct"], base["y_struct"], atol=1e-6)
    np.testing.assert_allclose(train["v"][:, :1], base["time"], atol=1e-6)
    assert train["v"].shape == (12, 785)
    assert not np.allclose(train["v"][:, 1], base["customer_group"].reshape(-1))


def test_make_demand_design_vector_grid_matches_base_grid_layout():
    grid = make_demand_design_vector_grid(
        price_points=3,
        time_points=2,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        test_vector_seed=42,
        representation_sd=0.5,
    )
    base_grid = make_demand_design_grid(price_points=3, time_points=2)

    np.testing.assert_allclose(grid["x"], base_grid["x"], atol=1e-6)
    np.testing.assert_allclose(grid["y_struct"], base_grid["y_struct"], atol=1e-6)
    np.testing.assert_allclose(grid["v"][:, :1], base_grid["time"], atol=1e-6)
    assert grid["v"].shape == (42, 785)


def test_vector_feature_extractor_output_shape():
    extractor = DemandVectorFeatureExtractor(v_dim=785, vector_dim=784)
    features = extractor(tf.zeros((2, 785), dtype=tf.float32), training=False)

    assert features.shape == (2, 65)


def test_bgm_iv_vector_accepts_v_dim(tmp_path):
    model = BGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)

    data_v = tf.zeros((2, 785), dtype=tf.float32)
    data_z = model.e_net(data_v, training=False)
    decoded = model.g_net(data_z, training=False)

    assert data_z.shape == (2, 4)
    assert decoded["public_v"].shape == (2, 785)
    assert decoded["vector_mean"].shape == (2, 784)


def test_vector_model_rejects_alpha_v(tmp_path):
    params = _make_vector_params(tmp_path)
    params["alpha_v"] = 1.0

    with pytest.raises(ValueError, match="alpha_v"):
        BGM_IV_Vector(params=params, random_seed=13)


def test_vector_covariate_posterior_is_untempered(tmp_path):
    model = BGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)
    data_v = tf.zeros((2, 785), dtype=tf.float32)
    data_z = tf.constant(
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.1, -0.3, 0.2]],
        dtype=tf.float32,
    )

    loss_pv_z, _, _ = model._covariate_loss_terms(
        data_v, data_z, training=False,
        block_scale=str(model.params["covariate_block_scale"]),
    )
    loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
    expected = -(loss_pv_z + loss_prior_z)
    actual = model.get_log_covariate_posterior(data_v, data_z)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_full_train_egm_score_matches_manual_gh_and_is_chunk_invariant(tmp_path):
    model = BGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)
    train = simulate_demand_design_vector_iv(
        n_samples=11,
        rho=0.5,
        seed=7,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        representation_sd=0.5,
    )
    data = (train["x"], train["y"], train["v"], train["w"])

    full = model.evaluate_egm_full_train_l2_y(data, chunk_size=None)
    chunked = model.evaluate_egm_full_train_l2_y(data, chunk_size=3)
    data_z = model.e_net(train["v"], training=False)
    data_z0, data_z1, data_z2 = model._split_z(data_z)
    h_output = model.h_net(
        tf.concat([data_z0, data_z2, train["w"]], axis=-1), training=False
    )
    mu_x = h_output[:, :1]
    sigma2_x = tf.minimum(
        model.egm_sigma2_x_ema * tf.ones_like(mu_x),
        float(model.params["egm_outcome_sigma_cap"]) ** 2,
    )
    x_nodes = (
        mu_x[None, :, :]
        + np.sqrt(2.0)
        * tf.sqrt(sigma2_x)[None, :, :]
        * model._egm_gh_t
    )
    means = []
    for node_index in range(int(model.params["egm_outcome_gh_nodes"])):
        f_output = model.f_net(
            tf.concat(
                [data_z0, data_z1, x_nodes[node_index]], axis=-1
            ),
            training=False,
        )[:, :1]
        means.append(model._egm_gh_w[node_index] * f_output)
    manual_mean = tf.add_n(means).numpy()
    manual = float(
        np.mean(
            np.square(train["y"].astype(np.float64) - manual_mean.astype(np.float64))
        )
    )

    assert full == pytest.approx(manual, rel=1e-6, abs=1e-8)
    assert chunked == pytest.approx(full, rel=1e-6, abs=1e-8)


def test_full_train_egm_score_does_not_mutate_model_or_ema(tmp_path):
    global_generator = tf.random.Generator.from_seed(20260830)
    tf.random.set_global_generator(global_generator)
    model = BGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)
    train = simulate_demand_design_vector_iv(
        n_samples=8,
        rho=0.5,
        seed=7,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        representation_sd=0.5,
    )
    data = (train["x"], train["y"], train["v"], train["w"])
    model.egm_sigma2_x_ema.assign(0.123)
    before_weights = [variable.numpy().copy() for variable in model.ckpt.g_net.variables]
    before_weights += [variable.numpy().copy() for variable in model.ckpt.e_net.variables]
    before_weights += [variable.numpy().copy() for variable in model.ckpt.f_net.variables]
    before_weights += [variable.numpy().copy() for variable in model.ckpt.h_net.variables]
    before_ema = float(model.egm_sigma2_x_ema.numpy())
    before_iterations = {
        name: int(optimizer.iterations.numpy())
        for name, optimizer in (
            ("g_pre", model.g_pre_optimizer),
            ("d_pre", model.d_pre_optimizer),
            ("g", model.g_optimizer),
            ("f", model.f_optimizer),
            ("h", model.h_optimizer),
            ("posterior", model.posterior_optimizer),
        )
    }
    numpy_state = np.random.get_state()
    tf_state = global_generator.state.numpy().copy()

    score = model.evaluate_egm_full_train_l2_y(data, chunk_size=3)

    after_weights = [variable.numpy() for variable in model.ckpt.g_net.variables]
    after_weights += [variable.numpy() for variable in model.ckpt.e_net.variables]
    after_weights += [variable.numpy() for variable in model.ckpt.f_net.variables]
    after_weights += [variable.numpy() for variable in model.ckpt.h_net.variables]
    assert np.isfinite(score)
    for before, after in zip(before_weights, after_weights):
        np.testing.assert_array_equal(before, after)
    assert float(model.egm_sigma2_x_ema.numpy()) == pytest.approx(before_ema)
    after_iterations = {
        name: int(optimizer.iterations.numpy())
        for name, optimizer in (
            ("g_pre", model.g_pre_optimizer),
            ("d_pre", model.d_pre_optimizer),
            ("g", model.g_optimizer),
            ("f", model.f_optimizer),
            ("h", model.h_optimizer),
            ("posterior", model.posterior_optimizer),
        )
    }
    assert after_iterations == before_iterations
    after_numpy_state = np.random.get_state()
    assert after_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(after_numpy_state[1], numpy_state[1])
    assert after_numpy_state[2:] == numpy_state[2:]
    np.testing.assert_array_equal(global_generator.state.numpy(), tf_state)


def test_vector_checkpoint_roundtrips_egm_sigma2_x_ema(tmp_path):
    params = _make_vector_params(tmp_path)
    params["save_model"] = True
    model = BGM_IV_Vector(params=params, timestamp="ema_roundtrip", random_seed=13)
    model.egm_sigma2_x_ema.assign(0.123)
    checkpoint = model.ckpt_manager.save(checkpoint_number=0)

    restored = BGM_IV_Vector(
        params=params, timestamp="ema_roundtrip", random_seed=99
    )
    status = restored.ckpt.restore(checkpoint)
    status.assert_existing_objects_matched()
    assert float(restored.egm_sigma2_x_ema.numpy()) == pytest.approx(0.123)


def test_egm_records_exact_requested_full_train_score_iterations(tmp_path):
    model = BGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)
    train = simulate_demand_design_vector_iv(
        n_samples=8,
        rho=0.5,
        seed=7,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        representation_sd=0.5,
    )
    data = (train["x"], train["y"], train["v"], train["w"])
    requested = tuple(range(10))
    scores = model.egm_init(
        data,
        egm_n_iter=9,
        batch_size=4,
        egm_batches_per_eval=1,
        verbose=0,
        score_iterations=requested,
        score_chunk_size=3,
    )
    assert [record["iteration"] for record in scores] == list(requested)
    assert all(np.isfinite(record["full_train_l2_loss_y"]) for record in scores)


def test_restored_pure_egm_checkpoint_continues_one_bgm_epoch(tmp_path):
    params = _make_vector_params(tmp_path)
    params["save_model"] = True
    train = simulate_demand_design_vector_iv(
        n_samples=8,
        rho=0.5,
        seed=7,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        representation_sd=0.5,
    )
    data = (train["x"], train["y"], train["v"], train["w"])
    candidate = BGM_IV_Vector(
        params=params, timestamp="phase_split", random_seed=13
    )
    candidate.egm_init(
        data,
        egm_n_iter=0,
        batch_size=4,
        egm_batches_per_eval=1,
        verbose=0,
        score_iterations=[0],
    )
    checkpoint = candidate.ckpt_manager.save(checkpoint_number=0)
    post_egm_history = list(candidate.training_history)

    restored = BGM_IV_Vector(
        params=params, timestamp="phase_split_restored", random_seed=99
    )
    status = restored.ckpt.restore(checkpoint)
    status.assert_existing_objects_matched()
    restored.training_history = post_egm_history
    history = restored.fit_bgm_from_egm(
        data,
        epochs=0,
        epochs_per_eval=1,
        batch_size=4,
        verbose=0,
    )
    assert history[0]["stage"] == "post_egm"
    assert history[-1]["stage"] == "epoch_eval"
    assert history[-1]["epoch"] == 0
