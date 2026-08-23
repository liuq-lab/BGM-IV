"""Outcome->particle coupling gamma: wiring contract on the live update.

particle objective = loss_pv_z + loss_prior_z + loss_px_z + gamma * loss_py_z
* gamma=0 and the boolean alias `stop_outcome_to_particles=True` give the
  bit-identical particle update (C-form);
* gamma=1 equals the joint objective;
* the outcome contribution to the particle gradient scales linearly in gamma;
* EGM (and hence the post-EGM state) is untouched by gamma.
"""

import numpy as np
import tensorflow as tf

from bgm_iv.datasets import simulate_demand_design_iv
from bgm_iv.models.bgm_iv import BGM_IV

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _params(tmp_path, **overrides):
    params = {
        "dataset": "GammaWiring",
        "output_dir": str(tmp_path),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 2,
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 1e-2,
        "g_units": [8, 8],
        "e_units": [8, 8],
        "f_units": [8, 4],
        "h_units": [8, 4],
        "dz_units": [8, 4],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "first_stage_warmup_epochs": 0,
        "deterministic_training": True,
    }
    params.update(overrides)
    return params


def _particle_gradient(model, train, seed=5):
    """Gradient of the particle objective w.r.t. the particles (fresh particles)."""
    tf.keras.utils.set_random_seed(seed)
    n = train["x"].shape[0]
    model.data_z = tf.Variable(
        np.random.default_rng(seed).normal(size=(n, sum(model.params["z_dims"]))).astype(np.float32)
    )
    idx = tf.constant(np.arange(n), dtype=tf.int32)
    x, y, v, w = (tf.constant(train[k], tf.float32) for k in ("x", "y", "v", "w"))
    with tf.GradientTape() as tape:
        z = tf.gather(model.data_z, idx, axis=0)
        g_output = model.g_net(z)
        mu_v = g_output[:, : model.params["v_dim"]]
        sigma_square_v = model._continuous_sigma(g_output, sigma_key="sigma_v")
        loss_pv = tf.reduce_mean(model._gaussian_nll(v, mu_v, sigma_square_v, event_dim=2))
        treatment_output = model._treatment_output(z, w)
        sigma_square_x = model._continuous_sigma(treatment_output, sigma_key="sigma_x")
        loss_px = tf.reduce_mean(model._gaussian_nll(x, treatment_output[:, :1], sigma_square_x, event_dim=1))
        loss_prior = tf.reduce_mean(tf.reduce_sum(z ** 2, axis=1) / 2.0)
        tf.keras.utils.set_random_seed(seed + 1)
        loss_py = -tf.reduce_mean(model._integrated_outcome_log_prob(z, w, y, n_samples=4))
        loss = loss_pv + loss_prior + loss_px + model._outcome_to_particles_weight() * loss_py
    gradient = tape.gradient(loss, [model.data_z])[0]
    return tf.convert_to_tensor(gradient).numpy()


def _run_one_update(model, train, seed=11):
    tf.keras.utils.set_random_seed(seed)
    n = train["x"].shape[0]
    model.data_z = tf.Variable(
        np.random.default_rng(seed).normal(size=(n, sum(model.params["z_dims"]))).astype(np.float32)
    )
    idx = np.arange(n)
    gamma = model._outcome_to_particles_weight()
    tf.keras.utils.set_random_seed(seed + 7)
    model.update_latent_variable_sgd(
        train["x"], train["y"], train["v"], train["w"], idx, include_outcome=gamma > 0.0
    )
    return model.data_z.numpy().copy()


def test_gamma_zero_and_boolean_alias_are_bit_identical(tmp_path):
    train = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=3)
    explicit = BGM_IV(params=_params(tmp_path, outcome_to_particles_weight=0.0), random_seed=13)
    alias = BGM_IV(params=_params(tmp_path, stop_outcome_to_particles=True), random_seed=13)
    for net in ("g_net", "e_net", "f_net", "h_net"):
        for a, b in zip(getattr(explicit, net).weights, getattr(alias, net).weights):
            b.assign(a)
    np.testing.assert_array_equal(_run_one_update(explicit, train), _run_one_update(alias, train))


def test_outcome_gradient_scales_linearly_in_gamma(tmp_path):
    train = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=3)
    models = {}
    for gamma in (0.0, 0.5, 1.0):
        models[gamma] = BGM_IV(params=_params(tmp_path, outcome_to_particles_weight=gamma), random_seed=13)
    for net in ("g_net", "e_net", "f_net", "h_net"):
        for a, b, c in zip(
            getattr(models[0.0], net).weights,
            getattr(models[0.5], net).weights,
            getattr(models[1.0], net).weights,
        ):
            b.assign(a)
            c.assign(a)
    g0 = _particle_gradient(models[0.0], train)
    g_half = _particle_gradient(models[0.5], train)
    g1 = _particle_gradient(models[1.0], train)
    outcome_part = g1 - g0
    assert np.linalg.norm(outcome_part) > 1e-6, "outcome term must move the particles"
    np.testing.assert_allclose(g_half - g0, 0.5 * outcome_part, rtol=1e-4, atol=1e-6)


def test_gamma_does_not_enter_egm(tmp_path):
    train = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=3)
    states = []
    for gamma in (0.0, 0.25):
        model = BGM_IV(params=_params(tmp_path, outcome_to_particles_weight=gamma), random_seed=13)
        model.training_history = []
        tf.keras.utils.set_random_seed(21)
        np.random.seed(21)
        model.egm_init(
            (train["x"], train["y"], train["v"], train["w"]),
            egm_n_iter=3,
            batch_size=8,
            egm_batches_per_eval=100,
            verbose=0,
        )
        states.append([w.numpy().copy() for w in model.g_net.weights + model.e_net.weights])
    for a, b in zip(*states):
        np.testing.assert_array_equal(a, b)
