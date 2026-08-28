"""Resolved target: density equals the live getter and an independent formula,
blocks are event sums with unit powers, identities are bound and sensitive."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from bgm_iv.models.bgm_iv import BGM_IV, BGM_IV_Image, BGM_IV_Vector

from bgm_iv.mcmc.target import (
    AffinePreprocessorSpec,
    EvidenceBlockSpec,
    ModelTrainingProvenance,
    TargetSpec,
    independent_formula_oracle,
    model_training_provenance,
    resolve_target,
    sha256_decoder,
)


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="bgm_target_runtime_"))


def _params(family: str, block_scale: str = "sum"):
    common = {
        "dataset": f"Target_{family}_{block_scale}",
        "output_dir": str(RUNTIME_ROOT),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [8, 8],
        "e_units": [8, 8],
        "f_units": [8, 4],
        "h_units": [8, 4],
        "dz_units": [8, 4],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "first_stage_warmup_epochs": 0,
        "covariate_block_scale": block_scale,
    }
    if family == "demand":
        common["v_dim"] = 2
    elif family == "vector":
        common.update(v_dim=785, vector_dim=784)
    elif family == "mnist":
        common["v_dim"] = 787
    else:
        raise AssertionError(family)
    return common


def _make_model(family: str, block_scale: str = "sum", seed: int = 413):
    cls = {"demand": BGM_IV, "vector": BGM_IV_Vector, "mnist": BGM_IV_Image}[family]
    return cls(
        params=_params(family, block_scale),
        timestamp=f"{family}_{block_scale}_{seed}",
        random_seed=seed,
    )


def _observation(family: str, seed: int = 19):
    rng = np.random.default_rng(seed)
    if family == "mnist":
        return np.concatenate(
            [
                rng.normal(size=(3, 1)),
                rng.uniform(0.0, 255.0, size=(3, 784)),
                rng.normal(size=(3, 2)),
            ],
            axis=1,
        ).astype(np.float32)
    width = {"demand": 2, "vector": 785}[family]
    return rng.normal(size=(3, width)).astype(np.float32)


def _value_and_gradient(function, z_value):
    z = tf.Variable(np.asarray(z_value, dtype=np.float32))
    with tf.GradientTape() as tape:
        value = function(z)
        objective = tf.reduce_sum(value)
    gradient = tape.gradient(objective, z)
    assert gradient is not None
    return np.asarray(value.numpy()), np.asarray(gradient.numpy())


@pytest.mark.parametrize("family", ["demand", "vector", "mnist"])
def test_values_and_latent_gradients_equal_live_getter_and_oracle(family):
    model = _make_model(family)
    data_v = tf.constant(_observation(family))
    z0 = np.random.default_rng(91).normal(size=(3, 4)).astype(np.float32)
    preprocessor = AffinePreprocessorSpec.identity_map(
        int(model.params["v_dim"]), name=f"{family}-model-scale"
    )
    resolved = resolve_target(model, preprocessor)

    live_value, live_gradient = _value_and_gradient(
        lambda z: model.get_log_covariate_posterior(data_v, z), z0
    )
    new_value, new_gradient = _value_and_gradient(
        lambda z: resolved.log_prob(data_v, z), z0
    )
    oracle_value, oracle_gradient = _value_and_gradient(
        lambda z: independent_formula_oracle(model, data_v, z, global_power=1.0), z0
    )
    np.testing.assert_allclose(new_value, live_value, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(new_gradient, live_gradient, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(oracle_value, live_value, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(oracle_gradient, live_gradient, rtol=2e-6, atol=2e-5)


def test_blocks_are_event_sums_with_unit_powers():
    vector = resolve_target(_make_model("vector"), AffinePreprocessorSpec.identity_map(785))
    assert vector.spec.block_powers == {"prior_z": 1.0, "time": 1.0, "vector_proxy": 1.0}
    assert [block.event_size for block in vector.spec.blocks] == [4, 1, 784]
    assert all(block.event_reduction == "sum" for block in vector.spec.blocks)
    demand = resolve_target(_make_model("demand"), AffinePreprocessorSpec.identity_map(2))
    assert demand.spec.block_powers == {"prior_z": 1.0, "covariate_v": 1.0}
    image = resolve_target(
        _make_model("mnist"), AffinePreprocessorSpec.identity_map(787)
    )
    assert image.spec.target_kind == "generalized_gibbs"
    assert image.spec.block_powers == {
        "prior_z": 1.0,
        "time": 1.0,
        "pixels": 1.0,
        "noise": 1.0,
    }
    assert next(
        block for block in image.spec.blocks if block.name == "pixels"
    ).evidence_kind == "generalized_score"

    data_v = tf.constant(_observation("vector"))
    data_z = tf.constant(np.random.default_rng(5).normal(size=(3, 4)), tf.float32)
    base = vector.unpowered_event_sums(data_v, data_z)
    powered = vector.powered_blocks(data_v, data_z)
    for block in vector.spec.blocks:
        np.testing.assert_allclose(powered[block.name].numpy(), base[block.name].numpy())
    expected = sum(powered[block.name] for block in vector.spec.blocks)
    np.testing.assert_allclose(vector.log_prob(data_v, data_z), expected, rtol=1e-7)


def test_target_legality_is_fail_closed():
    demand = _make_model("demand")
    resolve_target(demand, AffinePreprocessorSpec.identity_map(2))
    with pytest.raises(ValueError, match="event-sum"):
        _make_model("vector", "mean")
    image = resolve_target(
        _make_model("mnist"), AffinePreprocessorSpec.identity_map(787)
    )
    assert image.spec.target_kind == "generalized_gibbs"
    with pytest.raises(ValueError, match="global_power=1"):
        resolve_target(demand, AffinePreprocessorSpec.identity_map(2), global_power=0.5)
    with pytest.raises(ValueError, match="dimension"):
        resolve_target(demand, AffinePreprocessorSpec.identity_map(3))


def test_bnn_and_non_unit_powers_are_rejected():
    prior = EvidenceBlockSpec(
        name="prior_z", role="prior", event_size=4, event_reduction="sum",
        power=1.0, evidence_kind="log_prior", observation_support="R^4",
    )
    with pytest.raises(ValueError, match="stochastic"):
        TargetSpec(
            family="demand", target_kind="model_posterior",
            blocks=(prior,), global_power=1.0,
            decoder_model_hash="model", preprocessor_hash="preprocessor",
            model_training_provenance_hash="training", use_bnn=True, dtype="float32",
        )
    with pytest.raises(ValueError, match="unit block powers"):
        TargetSpec(
            family="demand", target_kind="model_posterior",
            blocks=(prior, EvidenceBlockSpec(
                name="covariate_v", role="observation", event_size=2, event_reduction="sum",
                power=0.5, evidence_kind="log_likelihood", observation_support="R^2")),
            global_power=1.0, decoder_model_hash="model", preprocessor_hash="preprocessor",
            model_training_provenance_hash="training", use_bnn=False, dtype="float32",
        )


def test_model_and_preprocessor_hashes_are_bound_and_sensitive():
    model = _make_model("demand")
    identity_pre = AffinePreprocessorSpec.identity_map(2, name="raw-v")
    shifted_pre = AffinePreprocessorSpec(
        mean=np.array([0.25, 0.0], dtype=np.float32), scale=np.ones(2, dtype=np.float32), name="raw-v"
    )
    first = resolve_target(model, identity_pre)
    second = resolve_target(model, shifted_pre)
    assert first.spec.decoder_model_hash == sha256_decoder(model)
    assert first.spec.preprocessor_hash == identity_pre.identity
    assert first.spec.model_training_provenance_hash
    manifest = first.spec.manifest
    assert manifest["target_hash"] == first.spec.identity
    assert manifest["decoder_model_hash"] == first.spec.decoder_model_hash
    assert manifest["preprocessor_hash"] == first.spec.preprocessor_hash
    assert manifest["family"] == "demand"
    assert manifest["target_kind"] == "model_posterior"
    assert manifest["blocks"] == [block.payload() for block in first.spec.blocks]
    assert first.spec.identity != second.spec.identity
    assert identity_pre.identity != shifted_pre.identity

    data_z = tf.zeros((1, 4), tf.float32)
    raw_v = np.array([[0.5, -0.25]], dtype=np.float32)
    np.testing.assert_allclose(
        first.log_prob_raw(raw_v, data_z, identity_pre),
        first.log_prob(identity_pre.transform(raw_v), data_z),
    )
    with pytest.raises(ValueError, match="preprocessor hash"):
        first.log_prob_raw(raw_v, data_z, shifted_pre)


def test_decoder_hash_includes_image_batchnorm_state():
    model = _make_model("mnist")
    assert model.g_net.non_trainable_variables, "image decoder should expose BN state"
    variable = model.g_net.non_trainable_variables[0]
    original = np.asarray(variable.numpy()).copy()
    before = sha256_decoder(model)
    variable.assign_add(tf.ones_like(variable) * 0.125)
    after = sha256_decoder(model)
    variable.assign(original)
    assert after != before
    assert sha256_decoder(model) == before


def test_resolved_target_detects_decoder_mutation():
    model = _make_model("demand")
    target = resolve_target(model, AffinePreprocessorSpec.identity_map(2))
    target.assert_runtime_identity()
    variable = model.g_net.trainable_variables[0]
    original = np.asarray(variable.numpy()).copy()
    variable.assign_add(tf.ones_like(variable) * 0.01)
    with pytest.raises(RuntimeError, match="changed"):
        target.assert_runtime_identity()
    variable.assign(original)
    target.assert_runtime_identity()


def test_training_provenance_must_match_decoder_hash():
    model = _make_model("demand")
    assert model_training_provenance(model).decoder_model_hash == sha256_decoder(model)
    wrong = ModelTrainingProvenance(decoder_model_hash="not-the-live-model", data_mode="not_applicable")
    with pytest.raises(ValueError, match="does not match"):
        resolve_target(model, AffinePreprocessorSpec.identity_map(2), training_provenance=wrong)


def test_observation_validation_rejects_nonfinite_values():
    model = _make_model("demand")
    target = resolve_target(model, AffinePreprocessorSpec.identity_map(2))
    bad = _observation("demand")
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        target.validate_observation(bad)
    with pytest.raises(ValueError, match="shape"):
        target.validate_observation(np.zeros((3, 5), np.float32))


def test_mnist_observation_validation_enforces_raw_pixel_support():
    target = resolve_target(
        _make_model("mnist"), AffinePreprocessorSpec.identity_map(787)
    )
    bad = _observation("mnist")
    bad[0, 20] = 300.0
    with pytest.raises(ValueError, match=r"\[0,255\]"):
        target.validate_observation(bad)
