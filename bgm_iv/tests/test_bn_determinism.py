import numpy as np
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV_Image

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _params(tmp_path, **overrides):
    params = {
        "dataset": "Sim_Demand_Design_Mnist_IV",
        "output_dir": str(tmp_path),
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
        "structural_map_steps": 100,
        "structural_map_lr": 5e-4,
    }
    params.update(overrides)
    return params


def _bn_layers(network):
    return [l for l in network.submodules if isinstance(l, tf.keras.layers.BatchNormalization)]


def test_image_model_has_non_fused_batchnorm_at_construction(tmp_path):
    model = BGM_IV_Image(params=_params(tmp_path), random_seed=3)
    layers = _bn_layers(model.g_net)
    assert len(layers) == 3
    assert all(layer.fused is False for layer in layers)


def test_restored_deterministic_decoder_is_bitwise_repeatable(tmp_path):
    params = _params(tmp_path, save_model=True)
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
