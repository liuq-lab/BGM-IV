from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV, BGM_IV_Image

from bgm_iv.mcmc.target import AffinePreprocessorSpec, latent_log_prob


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_target_runtime_"))


def _params(family: str):
    common = {
        "dataset": f"Target_{family}",
        "output_dir": str(RUNTIME_ROOT),
        "save_model": False,
        "z_dims": [1, 1, 1, 1],
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
    if family == "demand":
        common["v_dim"] = 2
    elif family == "mnist":
        common["v_dim"] = 785
    else:
        raise AssertionError(family)
    return common


def _make_model(family: str, seed: int = 413):
    cls = {"demand": BGM_IV, "mnist": BGM_IV_Image}[family]
    return cls(
        params=_params(family),
        timestamp=f"{family}_{seed}",
        random_seed=seed,
    )


def _observation(family: str, seed: int = 19):
    rng = np.random.default_rng(seed)
    if family == "mnist":
        return np.concatenate(
            [
                rng.normal(size=(3, 1)),
                rng.uniform(0.0, 255.0, size=(3, 784)),
            ],
            axis=1,
        ).astype(np.float32)
    return rng.normal(size=(3, 2)).astype(np.float32)


def independent_formula_oracle(model, family, data_v, data_z):
    data_v = tf.cast(data_v, tf.float32)
    data_z = tf.cast(data_z, tf.float32)

    def gaussian_score(observed, mean, variance):
        observed = tf.cast(observed, tf.float32)
        mean = tf.cast(mean, tf.float32)
        variance = tf.cast(variance, tf.float32)
        return -0.5 * (
            tf.square(observed - mean) / variance + tf.math.log(variance)
        )

    prior = -tf.reduce_sum(tf.square(data_z), axis=1) / 2.0
    if family == "demand":
        output = model.g_net(data_z, training=False)
        mean = output[:, : int(model.params["v_dim"])]
        variance = model._continuous_sigma(output, sigma_key="sigma_v", eps=1e-6)
        covariance = tf.reduce_sum(
            gaussian_score(data_v, mean, variance), axis=1
        )
        return prior + covariance

    decoded = model._decode_covariates(data_z, training=False)
    time = tf.reduce_sum(
        gaussian_score(
            data_v[:, :1], decoded["time_mean"], decoded["time_var"]
        ),
        axis=1,
    )
    image = tf.reduce_sum(
        -tf.nn.sigmoid_cross_entropy_with_logits(
            labels=data_v[:, 1:785] / 255.0,
            logits=decoded["image_logits"],
        ),
        axis=1,
    )
    return prior + time + image


def _value_and_gradient(function, z_value):
    z = tf.Variable(np.asarray(z_value, dtype=np.float32))
    with tf.GradientTape() as tape:
        value = function(z)
        objective = tf.reduce_sum(value)
    gradient = tape.gradient(objective, z)
    assert gradient is not None
    return np.asarray(value.numpy()), np.asarray(gradient.numpy())


@pytest.mark.parametrize("family", ["demand", "mnist"])
def test_values_and_latent_gradients_equal_live_getter_and_oracle(family):
    model = _make_model(family)
    data_v = tf.constant(_observation(family))
    z0 = np.random.default_rng(91).normal(size=(3, 4)).astype(np.float32)

    live_value, live_gradient = _value_and_gradient(
        lambda z: model.get_log_covariate_posterior(data_v, z), z0
    )
    new_value, new_gradient = _value_and_gradient(
        lambda z: latent_log_prob(model, family, data_v, z), z0
    )
    oracle_value, oracle_gradient = _value_and_gradient(
        lambda z: independent_formula_oracle(model, family, data_v, z), z0
    )
    np.testing.assert_allclose(new_value, live_value, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(new_gradient, live_gradient, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(oracle_value, live_value, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(oracle_gradient, live_gradient, rtol=2e-6, atol=2e-5)


def test_preprocessor_maps_raw_covariates_to_model_scale():
    preprocessor = AffinePreprocessorSpec(
        mean=np.array([0.25, 0.0], dtype=np.float32),
        scale=np.array([2.0, 1.0], dtype=np.float32),
    )
    assert preprocessor.dimension == 2
    raw_v = np.array([[0.5, -0.25]], dtype=np.float64)
    transformed = preprocessor.transform(raw_v)
    assert transformed.dtype == np.float32
    np.testing.assert_array_equal(transformed, np.array([[0.125, -0.25]], np.float32))
    with pytest.raises(ValueError, match="dimension"):
        preprocessor.transform(np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match="strictly positive"):
        AffinePreprocessorSpec(mean=np.zeros(2), scale=np.array([1.0, 0.0]))
