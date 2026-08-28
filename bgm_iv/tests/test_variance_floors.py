"""Soft variance floors: ``variance = (sigma_min + sqrt(learned))^2``.

The head stays learnable but bounded below; the fixed override stops its
gradients; the floor value (scalar or per-block) is part of the decoder
identity, and the resolved target reads the floored variance.
"""

import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV, BGM_IV_Vector
from bgm_iv.mcmc.target import AffinePreprocessorSpec, resolve_target, sha256_decoder


def _params(v_dim=65, vector_dim=64, **overrides):
    params = {
        "dataset": "Sim_Demand_Design_Vector_IV",
        "output_dir": ".",
        "v_dim": v_dim,
        "vector_dim": vector_dim,
        "w_dim": 1,
        "z_dims": [2, 1, 1, 2],
        "e_units": [16, 16],
        "g_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "dz_units": [16, 8],
        "lr": 2e-4,
        "lr_theta": 1e-4,
        "lr_z": 1e-4,
        "use_bnn": False,
        "use_z_rec": True,
        "kl_weight": 1e-4,
        "binary_treatment": False,
        "iv_mc_samples": 4,
        "save_model": False,
        "save_res": False,
        "covariate_block_scale": "sum",
    }
    params.update(overrides)
    return params


def _batch(model, n=5, seed=3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, int(model.params["v_dim"]))).astype(np.float32)
    z = rng.normal(size=(n, sum(model.params["z_dims"]))).astype(np.float32)
    return v, z


def _gradient_norm(model, z, key):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model._decode_covariates(tf.constant(z), training=False)[key])
    grads = tape.gradient(loss, model.g_net.trainable_variables)
    return float(tf.add_n([tf.reduce_sum(tf.abs(g)) for g in grads if g is not None] or [tf.constant(0.0)]))


@pytest.mark.parametrize("key,floor_param,fixed_param", [
    ("time_var", "sigma_time_softfloor", "sigma_time"),
    ("vector_var", "sigma_vector_softfloor", "sigma_vector"),
])
def test_soft_floor_bounds_keeps_gradients_and_excludes_fixed_override(key, floor_param, fixed_param):
    soft = BGM_IV_Vector(params=_params(**{floor_param: 0.4}), random_seed=1)
    _, z = _batch(soft)
    variance = soft._decode_covariates(z, training=False)[key].numpy()
    assert np.all(variance >= 0.4 ** 2 - 1e-7)
    assert np.std(variance) > 0.0  # learned component survives
    assert _gradient_norm(soft, z, key) > 0.0

    fixed = BGM_IV_Vector(params=_params(**{fixed_param: 0.4}), random_seed=1)
    np.testing.assert_allclose(fixed._decode_covariates(z, training=False)[key].numpy(), 0.16, rtol=1e-6)
    assert _gradient_norm(fixed, z, key) == 0.0

    both = BGM_IV_Vector(params=_params(**{fixed_param: 0.1, floor_param: 0.1}), random_seed=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        both._decode_covariates(z, training=False)


def test_floors_are_identity_relevant():
    base = BGM_IV_Vector(params=_params(), random_seed=1)
    time_soft = BGM_IV_Vector(params=_params(sigma_time_softfloor=0.1), random_seed=1)
    vector_soft = BGM_IV_Vector(params=_params(sigma_vector_softfloor=0.4), random_seed=1)
    vector_other = BGM_IV_Vector(params=_params(sigma_vector_softfloor=0.5), random_seed=1)
    hashes = {sha256_decoder(m) for m in (base, time_soft, vector_soft, vector_other)}
    assert len(hashes) == 4


def test_resolved_target_uses_floored_variances():
    model = BGM_IV_Vector(params=_params(sigma_vector_softfloor=0.4, sigma_time_softfloor=0.1), random_seed=1)
    v, z = _batch(model)
    resolved = resolve_target(model, AffinePreprocessorSpec.identity_map(65, name="vtilde"))
    lp = resolved.log_prob(tf.constant(v), tf.constant(z)).numpy()
    decoded = model._decode_covariates(z, training=False)
    time_var = decoded["time_var"].numpy().reshape(-1)
    vec_var = decoded["vector_var"].numpy()
    assert np.all(time_var >= 0.1 ** 2 - 1e-8) and np.all(vec_var >= 0.4 ** 2 - 1e-7)
    manual = (
        -0.5 * np.sum(z ** 2, axis=1)
        - ((v[:, 0] - decoded["time_mean"].numpy().reshape(-1)) ** 2 / (2 * time_var) + 0.5 * np.log(time_var))
        - np.sum((v[:, 1:] - decoded["vector_mean"].numpy()) ** 2 / (2 * vec_var) + 0.5 * np.log(vec_var), axis=1)
    )
    np.testing.assert_allclose(lp, manual, rtol=1e-4, atol=1e-3)


def test_per_block_floor_uses_vector_blocks_layout():
    model = BGM_IV_Vector(
        params=_params(v_dim=71, vector_dim=70, vector_blocks=[64, 6], sigma_vector_softfloor=[0.4, 0.1]),
        random_seed=1,
    )
    _, z = _batch(model)
    vector_var = model._decode_covariates(z, training=False)["vector_var"].numpy()
    assert np.all(vector_var[:, :64] >= 0.4 ** 2 - 1e-7)
    assert np.all(vector_var[:, 64:] >= 0.1 ** 2 - 1e-7)
    scalar = BGM_IV_Vector(params=_params(v_dim=71, vector_dim=70, sigma_vector_softfloor=0.4), random_seed=1)
    for a, b in zip(model.g_net.weights, scalar.g_net.weights):
        b.assign(a)
    scalar_var = scalar._decode_covariates(z, training=False)["vector_var"].numpy()
    # same learned head: the first block is floored identically, the second at
    # 0.1 instead of 0.4 and is therefore strictly smaller
    np.testing.assert_allclose(vector_var[:, :64], scalar_var[:, :64], rtol=1e-6)
    assert np.all(vector_var[:, 64:] < scalar_var[:, 64:])
    assert sha256_decoder(scalar) != sha256_decoder(model)

    bad = BGM_IV_Vector(params=_params(vector_blocks=[60, 10], sigma_vector_softfloor=[0.4, 0.1]), random_seed=1)
    with pytest.raises(ValueError, match="vector_blocks"):
        bad._decode_covariates(z[:, :], training=False)
    wrong_len = BGM_IV_Vector(params=_params(vector_blocks=[32, 32], sigma_vector_softfloor=[0.4]), random_seed=1)
    with pytest.raises(ValueError, match="list length"):
        wrong_len._decode_covariates(z, training=False)


def test_sum_is_the_default_block_scale_everywhere():
    model = BGM_IV_Vector(params=_params(), random_seed=1)
    del model.params["covariate_block_scale"]
    v, z = _batch(model)
    resolve_target(model, AffinePreprocessorSpec.identity_map(65))
    summed, _, _ = model._covariate_loss_terms(v, z, training=False)
    explicit, _, _ = model._covariate_loss_terms(v, z, training=False, block_scale="sum")
    np.testing.assert_allclose(summed.numpy(), explicit.numpy())
    posterior = model.get_log_covariate_posterior(tf.constant(v), tf.constant(z)).numpy()
    prior = -0.5 * np.sum(z ** 2, axis=1)
    np.testing.assert_allclose(posterior, -(explicit.numpy()) + prior, rtol=1e-5, atol=1e-4)


def _tiny_demand(sigma_y_softfloor, seed=1):
    params = {
        "dataset": "Floor_demand", "output_dir": ".", "save_res": False, "save_model": False,
        "binary_treatment": False, "use_bnn": False, "z_dims": [1, 1, 1, 1], "v_dim": 2, "w_dim": 1,
        "lr_theta": 5e-4, "lr_z": 5e-4, "g_units": [8, 8], "e_units": [8, 8], "f_units": [8, 4],
        "h_units": [8, 4], "dz_units": [8, 4], "kl_weight": 0.0, "lr": 5e-4, "g_d_freq": 1,
        "use_z_rec": True, "iv_mc_samples": 8, "eval_mc_samples": 8, "first_stage_warmup_epochs": 0,
        "sigma_y_softfloor": sigma_y_softfloor,
    }
    return BGM_IV(params=params, random_seed=seed)


def test_outcome_noise_floor_drives_training_likelihood_and_prediction_alike():
    """The IV pseudo-likelihood that trains f and the predictive noise used at
    readout must share one sigma_y: the soft floor enters both identically."""
    small, big = _tiny_demand(0.1), _tiny_demand(10.0)  # same seed -> same weights
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(6, 4)), tf.float32)
    w = tf.constant(rng.normal(size=(6, 1)), tf.float32)
    y = tf.constant(rng.normal(size=(6, 1)), tf.float32)
    tf.random.set_seed(0)
    lp_small = small._integrated_outcome_log_prob(z, w, y, n_samples=8).numpy()
    tf.random.set_seed(0)
    lp_big = big._integrated_outcome_log_prob(z, w, y, n_samples=8).numpy()
    assert not np.allclose(lp_small, lp_big)
    assert np.all(lp_small > lp_big)  # a 10x wider noise model is far less likely on unit-scale y

    # exact agreement with the inference-side noise model on the same treatment draws
    tf.random.set_seed(0)
    x_samples, _, _ = big._sample_treatment(z, w, n_samples=8, use_mean=False, eps=1e-6)
    outputs = big._outcome_outputs_for_samples(z, x_samples)
    mu = outputs[:, :, :1]
    sigma2 = tf.reshape(big._continuous_sigma(tf.reshape(outputs, (-1, 2)), sigma_key="sigma_y"), tf.shape(mu))
    assert np.all(np.sqrt(sigma2.numpy()) >= 10.0)
    log_terms = -((y[None] - mu) ** 2 / (2.0 * sigma2) + 0.5 * tf.math.log(sigma2))
    manual = tf.reduce_logsumexp(tf.squeeze(log_terms, -1), axis=0) - np.log(8.0)
    np.testing.assert_allclose(lp_big, manual.numpy(), rtol=1e-5, atol=1e-5)
