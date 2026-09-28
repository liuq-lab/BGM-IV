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
        "save_model": False,
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
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "structural_map_steps": 100,
        "structural_map_lr": 1e-2,
    }
    params.update(overrides)
    return params


def _live_particle_gradient(model, train, monkeypatch, seed=5):
    n = train["x"].shape[0]
    model.data_z = tf.Variable(
        np.random.default_rng(seed).normal(size=(n, sum(model.params["z_dims"]))).astype(np.float32)
    )
    recorded = tf.Variable(tf.zeros_like(model.data_z))

    def record(grads_and_vars):
        (gradient, variable), = list(grads_and_vars)
        assert variable is model.data_z
        return recorded.assign(tf.convert_to_tensor(gradient))

    monkeypatch.setattr(model.posterior_optimizer, "apply_gradients", record)
    tf.keras.utils.set_random_seed(seed)
    model.update_latent_variable_sgd(
        *(tf.constant(train[k], tf.float32) for k in ("x", "y", "v", "w")),
        tf.constant(np.arange(n), dtype=tf.int32),
    )
    return recorded.numpy()


def test_live_latent_update_scales_outcome_gradient_linearly_in_gamma(tmp_path, monkeypatch):
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
    g0, g_half, g1 = (
        _live_particle_gradient(models[gamma], train, monkeypatch) for gamma in (0.0, 0.5, 1.0)
    )
    outcome_part = g1 - g0
    assert np.linalg.norm(outcome_part) > 1e-6, "outcome term must move the particles"
    np.testing.assert_allclose(g_half - g0, 0.5 * outcome_part, rtol=1e-4, atol=1e-6)


def test_gamma_does_not_enter_egm(tmp_path):
    train = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=3)
    states = []
    for gamma in (0.0, 0.25):
        model = BGM_IV(params=_params(tmp_path, outcome_to_particles_weight=gamma), random_seed=13)
        tf.keras.utils.set_random_seed(21)
        np.random.seed(21)
        model.egm_init(
            (train["x"], train["y"], train["v"], train["w"]),
            egm_n_iter=3,
            batch_size=8,
            verbose=0,
        )
        states.append([w.numpy().copy() for w in model.g_net.weights + model.e_net.weights])
    for a, b in zip(*states):
        np.testing.assert_array_equal(a, b)
