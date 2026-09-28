import numpy as np
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV, BGM_IV_Image
from bgm_iv.mcmc.target import latent_log_prob


def _image_params(**overrides):
    params = {
        "dataset": "Floor_image", "output_dir": ".", "save_model": False,
        "z_dims": [1, 1, 1, 1], "v_dim": 785, "w_dim": 1,
        "lr_theta": 5e-4, "lr_z": 5e-4, "g_units": [8, 8], "e_units": [8, 8],
        "f_units": [8, 4], "h_units": [8, 4], "dz_units": [8, 4], "lr": 5e-4,
        "g_d_freq": 1, "iv_mc_samples": 4, "eval_mc_samples": 4,
        "structural_map_steps": 100, "structural_map_lr": 5e-4,
    }
    params.update(overrides)
    return params


def _time_var_gradient_norm(model, z):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(
            model._decode_covariates(tf.constant(z), training=False)["time_var"]
        )
    grads = tape.gradient(loss, model.g_net.trainable_variables)
    return float(
        tf.add_n([tf.reduce_sum(tf.abs(g)) for g in grads if g is not None] or [tf.constant(0.0)])
    )


def test_fixed_time_scale_stops_the_time_variance_head():
    z = np.random.default_rng(3).normal(size=(5, 4)).astype(np.float32)
    fixed = BGM_IV_Image(params=_image_params(), random_seed=1)
    np.testing.assert_allclose(
        fixed._decode_covariates(z, training=False)["time_var"].numpy(), 0.01, rtol=1e-6
    )
    assert _time_var_gradient_norm(fixed, z) == 0.0


def test_image_covariate_posterior_is_the_event_sum_target():
    model = BGM_IV_Image(params=_image_params(), random_seed=1)
    rng = np.random.default_rng(4)
    v = np.concatenate(
        [rng.normal(size=(5, 1)), rng.uniform(0.0, 255.0, size=(5, 784))], axis=1
    ).astype(np.float32)
    z = rng.normal(size=(5, 4)).astype(np.float32)
    summed, _, _ = model._covariate_loss_terms(v, z, training=False)
    posterior = model.get_log_covariate_posterior(tf.constant(v), tf.constant(z)).numpy()
    prior = -0.5 * np.sum(z ** 2, axis=1)
    np.testing.assert_allclose(posterior, -(summed.numpy()) + prior, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(
        latent_log_prob(model, "mnist", tf.constant(v), tf.constant(z)).numpy(),
        posterior,
        rtol=1e-5,
        atol=1e-3,
    )


def _tiny_demand(seed=1):
    params = {
        "dataset": "Floor_demand", "output_dir": ".", "save_model": False,
        "z_dims": [1, 1, 1, 1], "v_dim": 2, "w_dim": 1,
        "lr_theta": 5e-4, "lr_z": 5e-4, "g_units": [8, 8], "e_units": [8, 8], "f_units": [8, 4],
        "h_units": [8, 4], "dz_units": [8, 4], "lr": 5e-4, "g_d_freq": 1,
        "iv_mc_samples": 8, "eval_mc_samples": 8,
        "structural_map_steps": 100, "structural_map_lr": 5e-4,
    }
    return BGM_IV(params=params, random_seed=seed)


def test_outcome_noise_floor_drives_training_likelihood_and_prediction_alike():
    model = _tiny_demand()
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(6, 4)), tf.float32)
    w = tf.constant(rng.normal(size=(6, 1)), tf.float32)
    y = tf.constant(rng.normal(size=(6, 1)), tf.float32)
    tf.random.set_seed(0)
    log_prob = model._integrated_outcome_log_prob(z, w, y, n_samples=8).numpy()

    tf.random.set_seed(0)
    x_samples = model._sample_treatment(z, w, n_samples=8, eps=1e-6)
    outputs = model._outcome_outputs_for_samples(z, x_samples)
    mu = outputs[:, :, :1]
    raw = tf.reshape(outputs, (-1, 2))
    sigma2 = tf.reshape(model._continuous_sigma(raw, sigma_key="sigma_y"), tf.shape(mu))
    assert np.all(np.sqrt(sigma2.numpy()) >= 0.1)
    np.testing.assert_allclose(
        np.sqrt(sigma2.numpy()).reshape(-1),
        0.1 + np.sqrt(tf.nn.softplus(raw[:, -1]).numpy() + 1e-6),
        rtol=1e-6,
    )
    log_terms = -((y[None] - mu) ** 2 / (2.0 * sigma2) + 0.5 * tf.math.log(sigma2))
    manual = tf.reduce_logsumexp(tf.squeeze(log_terms, -1), axis=0) - np.log(8.0)
    np.testing.assert_allclose(log_prob, manual.numpy(), rtol=1e-5, atol=1e-5)
