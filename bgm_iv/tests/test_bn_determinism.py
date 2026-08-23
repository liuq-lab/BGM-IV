"""Determinism contract for the only BatchNorm-bearing network (image decoder).

With `deterministic_training=True` every BatchNormalization layer must run
the non-fused kernel (FusedBatchNormGradV3 has no deterministic GPU
implementation).  The flip now happens at construction (before the checkpoint
restore / any tf.function trace), so restore-path consumers inherit it, and
the fused flags are part of the decoder identity hash.
"""

import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV_Image
from bgm_iv.mcmc.target import _batchnorm_fused_flags, sha256_decoder

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _params(tmp_path, **overrides):
    params = {
        "dataset": "Sim_Demand_Design_Mnist_IV",
        "output_dir": str(tmp_path),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
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
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "first_stage_warmup_epochs": 0,
    }
    params.update(overrides)
    return params


def _bn_layers(network):
    return [l for l in network.submodules if isinstance(l, tf.keras.layers.BatchNormalization)]


def test_deterministic_image_model_has_non_fused_batchnorm_at_construction(tmp_path):
    model = BGM_IV_Image(params=_params(tmp_path, deterministic_training=True), random_seed=3)
    layers = _bn_layers(model.g_net)
    assert len(layers) == 3
    assert all(layer.fused is False for layer in layers)
    assert _batchnorm_fused_flags(model.g_net) == [False, False, False]


def test_non_deterministic_image_model_keeps_fused_batchnorm(tmp_path):
    model = BGM_IV_Image(params=_params(tmp_path), random_seed=3)
    flags = _batchnorm_fused_flags(model.g_net)
    assert len(flags) == 3
    assert all(flag is not False for flag in flags)


def test_fused_flag_is_part_of_the_decoder_identity(tmp_path):
    deterministic = BGM_IV_Image(params=_params(tmp_path, deterministic_training=True), random_seed=3)
    plain = BGM_IV_Image(params=_params(tmp_path), random_seed=3)
    for a, b in zip(deterministic.g_net.weights, plain.g_net.weights):
        b.assign(a)
    assert sha256_decoder(deterministic) != sha256_decoder(plain)


def test_restored_deterministic_decoder_is_bitwise_repeatable(tmp_path):
    params = _params(tmp_path, deterministic_training=True, save_model=True)
    model = BGM_IV_Image(params=params, random_seed=3)
    model.ckpt_manager.save(0)
    timestamp = model.timestamp
    z = np.random.default_rng(0).normal(size=(4, 4)).astype(np.float32)
    outputs = []
    for _ in range(2):
        restored = BGM_IV_Image(params=params, timestamp=timestamp, random_seed=3)
        assert all(layer.fused is False for layer in _bn_layers(restored.g_net))
        with tf.GradientTape() as tape:
            decoded = restored._decode_covariates(tf.constant(z), training=False)
            loss = tf.reduce_sum(decoded["image_probs"])
        grads = tape.gradient(loss, restored.g_net.trainable_variables)
        outputs.append([g.numpy().copy() for g in grads if g is not None])
    for a, b in zip(*outputs):
        np.testing.assert_array_equal(a, b)
