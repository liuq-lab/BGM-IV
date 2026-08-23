"""Frozen-batch HMC: exact replay, factorized axes, per-(chain, target)
numerical flags, fail-closed evaluator qualification, mass regularization."""

from __future__ import annotations

import hashlib
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

# Compiled tf.while_loop HMC is slow on the Metal plugin; CPU also keeps the
# same-runtime bitwise tests backend-independent.
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from bgm_iv.models.bgm_iv import BGM_IV

from bgm_iv.mcmc.diagnostics import chain_diagnostics
from bgm_iv.mcmc.sampler import (
    FrozenBatch,
    FrozenHMCError,
    FrozenVectorizedHMC,
    GaussianContextEvaluator,
    LatentPosteriorEvaluator,
    MassRegularization,
    ProductionConfig,
    TrajectoryPolicy,
    WarmupConfig,
    assert_axis_separable,
    overdispersed_initial_state,
    regularize_state_variance,
)


RUN_KEY = hashlib.sha256(b"frozen-vectorized-hmc-tests").hexdigest()
SEGMENT_FIELDS = (
    "pre_state",
    "post_state",
    "draws",
    "is_accepted",
    "log_accept_ratio",
    "energy_error",
    "has_nonfinite",
    "has_extreme_energy_error",
    "has_divergence",
    "has_numerical_anomaly",
    "trajectory_length",
    "seed",
    "step_size",
    "state_variance",
)
MODES = np.array([[-1.5, 0.4], [0.0, -0.8], [1.2, 2.0], [3.5, -2.5]], np.float32)
SCALES = np.array([[0.4, 0.5], [0.8, 0.65], [1.3, 1.0], [1.8, 2.1]], np.float32)


def _gaussian_context(locations, scales):
    loc = np.asarray(locations, np.float32)
    scale = np.asarray(scales, np.float32)
    return np.concatenate([loc, np.log(scale)], axis=1).astype(np.float32)


def _runner(*, latent_dim=2, max_batch=4, warmup=12, draws=10, support=(2, 3)):
    return FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(latent_dim),
        num_chains=4,
        max_batch_size=max_batch,
        warmup_config=WarmupConfig(
            warmup_steps=warmup, initial_step_size=0.08, num_leapfrog_steps=3
        ),
        production_config=ProductionConfig(
            segment_size=draws, trajectory_policy=TrajectoryPolicy(tuple(support))
        ),
    )


def _sample(runner, ids, modes, scales, batch_index, *, warmup_state=None):
    batch = FrozenBatch(tuple(ids), _gaussian_context(modes, scales), batch_index=batch_index)
    epoch = runner.bind_epoch(batch, run_seed=20260811, run_key=RUN_KEY, production_epoch=0)
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], np.float32)[:, None, None]
    initial = (modes[None] + offsets * scales[None]).astype(np.float32)
    warm = runner.warmup(epoch, initial_state=initial, state_variance=(scales**2).astype(np.float32))
    return runner.run_segment(
        epoch,
        segment_index=0,
        pre_state=warm.final_state,
        step_size=warm.step_size,
        state_variance=warm.state_variance,
    )


# --- replay and batching -----------------------------------------------------


def test_exact_replay_and_partial_batch_without_retrace():
    runner = _runner(max_batch=4, warmup=4, draws=4)
    context = _gaussian_context(
        [[0.0, 0.0], [1.0, -1.0], [-0.5, 0.25]], [[1.0, 1.0], [0.7, 1.2], [1.3, 0.8]]
    )
    batch = FrozenBatch((3, 8, 13), context, batch_index=2)
    epoch = runner.bind_epoch(batch, run_seed=77, run_key=RUN_KEY, production_epoch=0)
    state = np.zeros((4, 3, 2), np.float32)
    variance = np.ones((3, 2), np.float32)
    warm_a = runner.warmup(epoch, initial_state=state, state_variance=variance)
    warm_b = runner.warmup(epoch, initial_state=state, state_variance=variance)
    np.testing.assert_array_equal(warm_a.final_state, warm_b.final_state)
    np.testing.assert_array_equal(warm_a.step_size, warm_b.step_size)
    segments = [
        runner.run_segment(
            epoch, segment_index=0, pre_state=warm_a.final_state,
            step_size=warm_a.step_size, state_variance=variance,
        )
        for _ in range(2)
    ]
    for name in SEGMENT_FIELDS:
        np.testing.assert_array_equal(getattr(segments[0], name), getattr(segments[1], name))
    segment = segments[0]
    assert segment.draws.shape == (4, 4, 3, 2)
    assert set(np.unique(segment.trajectory_length)).issubset({2, 3})
    # one trajectory length per transition, common to all chains and targets
    assert all(np.unique(row).size == 1 for row in segment.trajectory_length)
    assert runner.warmup_tracing_count == 1 and runner.production_tracing_count == 1

    partial = FrozenBatch((21,), _gaussian_context([[2.0, -2.0]], [[0.9, 1.1]]), batch_index=3)
    partial_epoch = runner.bind_epoch(partial, run_seed=77, run_key=RUN_KEY, production_epoch=0)
    partial_warm = runner.warmup(
        partial_epoch,
        initial_state=np.zeros((4, 1, 2), np.float32),
        state_variance=np.ones((1, 2), np.float32),
    )
    partial_segment = runner.run_segment(
        partial_epoch, segment_index=0, pre_state=partial_warm.final_state,
        step_size=partial_warm.step_size, state_variance=partial_warm.state_variance,
    )
    assert partial_segment.draws.shape == (4, 4, 1, 2)
    assert runner.warmup_tracing_count == 1 and runner.production_tracing_count == 1
    assert partial_epoch.epoch_identity != epoch.epoch_identity
    assert batch.target_context.flags.writeable is False


def test_other_targets_do_not_change_a_target_stream():
    evaluator = GaussianContextEvaluator(2)
    context = _gaussian_context([[0.2, -0.1], [4.0, -3.0]], [[1.0, 0.8], [0.3, 1.7]])
    assert_axis_separable(
        evaluator, state=np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 10.0, context=context
    )
    runner = FrozenVectorizedHMC(
        evaluator=evaluator,
        num_chains=4,
        max_batch_size=2,
        warmup_config=WarmupConfig(warmup_steps=3),
        production_config=ProductionConfig(segment_size=5, trajectory_policy=TrajectoryPolicy((2, 4))),
    )
    batch = FrozenBatch((100, 200), context, batch_index=0)
    epoch = runner.bind_epoch(batch, run_seed=9, run_key=RUN_KEY, production_epoch=1)
    state_a = np.zeros((4, 2, 2), np.float32)
    state_b = state_a.copy()
    state_b[:, 1] = np.array([25.0, -31.0], np.float32)
    step = np.full((4, 2, 1), 0.1, np.float32)
    variance = np.ones((2, 2), np.float32)
    left = runner.run_segment(epoch, segment_index=0, pre_state=state_a, step_size=step, state_variance=variance)
    right = runner.run_segment(epoch, segment_index=0, pre_state=state_b, step_size=step, state_variance=variance)
    for name in SEGMENT_FIELDS[2:11]:
        np.testing.assert_array_equal(getattr(left, name)[:, :, 0], getattr(right, name)[:, :, 0])


@pytest.mark.slow
def test_permutation_and_rebatching_change_bits_not_the_law():
    runner = FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(2),
        num_chains=4,
        max_batch_size=4,
        warmup_config=WarmupConfig(warmup_steps=100, initial_step_size=0.1, num_leapfrog_steps=4),
        production_config=ProductionConfig(segment_size=1_400, trajectory_policy=TrajectoryPolicy((3, 5, 7))),
    )

    def assert_moments(draws, modes, scales, tolerance=0.10):
        mean = draws.mean((0, 1), dtype=np.float64)
        variance = draws.var((0, 1), dtype=np.float64)
        assert np.max(np.abs(mean - modes) / scales) < tolerance
        assert np.max(np.abs(variance / scales**2 - 1.0)) < 0.16

    ids = np.array([11, 23, 37, 41])
    full = _sample(runner, ids, MODES, SCALES, 0)
    assert_moments(full.draws, MODES, SCALES)
    order = np.array([2, 0, 3, 1])
    permuted = _sample(runner, ids[order], MODES[order], SCALES[order], 1)
    assert_moments(permuted.draws, MODES[order], SCALES[order])
    assert not np.array_equal(full.draws, permuted.draws[:, :, np.argsort(order)])
    pieces = []
    for batch_index, positions in enumerate((np.array([0, 1]), np.array([2, 3])), 2):
        part = _sample(runner, ids[positions], MODES[positions], SCALES[positions], batch_index)
        assert_moments(part.draws, MODES[positions], SCALES[positions], tolerance=0.13)
        pieces.append(part.draws)
    assert not np.array_equal(full.draws, np.concatenate(pieces, axis=2))
    assert runner.warmup_tracing_count == 1 and runner.production_tracing_count == 1


@pytest.mark.slow
def test_gaussian_law_passes_rhat_and_ess():
    locations = np.array([[0.5, -1.0], [-1.25, 0.75]], np.float32)
    scales = np.array([[0.7, 1.2], [1.1, 0.6]], np.float32)
    runner = _runner(max_batch=3, warmup=150, draws=1_200, support=(3, 5, 7))
    segment = _sample(runner, (17, 43), locations, scales, 0)
    assert not np.any(segment.has_divergence)
    diagnostics = chain_diagnostics(segment.draws)
    assert float(np.max(diagnostics["rhat_rank"])) < 1.02
    assert float(np.max(diagnostics["rhat_folded"])) < 1.02
    assert float(np.min(diagnostics["ess_bulk"])) > 250.0
    estimate = np.mean(segment.draws, axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(estimate, locations, atol=0.07, rtol=0.0)


# --- numerical flags ---------------------------------------------------------


def test_nonfinite_and_divergence_flags_are_per_chain_and_target():
    runner = _runner(max_batch=2, warmup=3, draws=2, support=(10,))
    context = _gaussian_context([[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0]] * 2)
    epoch = runner.bind_epoch(FrozenBatch((1, 2), context, batch_index=0), run_seed=4, run_key=RUN_KEY, production_epoch=0)
    step = np.full((4, 2, 1), 0.08, np.float32)
    step[:, 1, 0] = 1_000.0
    segment = runner.run_segment(
        epoch, segment_index=0, pre_state=np.ones((4, 2, 2), np.float32),
        step_size=step, state_variance=np.ones((2, 2), np.float32),
    )
    assert not np.any(segment.has_divergence[:, :, 0])
    assert np.all(segment.has_divergence[:, :, 1])
    assert np.all(segment.has_nonfinite[:, :, 1])
    assert np.all(segment.has_extreme_energy_error[:, :, 1])
    assert np.all(segment.has_numerical_anomaly[:, :, 1])
    assert np.all(np.isfinite(segment.draws))


def test_large_negative_delta_h_is_an_anomaly_not_a_divergence():
    runner = _runner(max_batch=1, warmup=2, draws=1, support=(1,))
    epoch = runner.bind_epoch(
        FrozenBatch((7,), _gaussian_context([[0.0, 0.0]], [[1.0, 1.0]]), batch_index=0),
        run_seed=4, run_key=RUN_KEY, production_epoch=1,
    )
    segment = runner.run_segment(
        epoch, segment_index=0, pre_state=np.full((4, 1, 2), 10_000.0, np.float32),
        step_size=np.full((4, 1, 1), 1.5, np.float32), state_variance=np.ones((1, 2), np.float32),
    )
    assert np.all(segment.energy_error < -1_000.0)
    assert not np.any(segment.has_nonfinite)
    assert not np.any(segment.has_divergence)
    assert np.all(segment.has_extreme_energy_error)
    assert np.all(segment.has_numerical_anomaly)


def test_step_size_adapts_per_chain_and_target_independently_of_padding():
    runner = FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(1),
        num_chains=4,
        max_batch_size=3,
        warmup_config=WarmupConfig(warmup_steps=30, initial_step_size=0.1, num_leapfrog_steps=3),
        production_config=ProductionConfig(segment_size=2),
    )
    context = np.array([[0.0, np.log(0.1)], [0.0, np.log(5.0)], [0.0, 0.0]], np.float32)
    mask = np.array([True, True, False])
    state = np.zeros((4, 3, 1), np.float32)
    variance = np.ones((3, 1), np.float32)
    seed = np.array([12, 34], np.int32)
    first = runner._warmup_graph(tf.constant(state), tf.constant(variance), tf.constant(context), tf.constant(mask), tf.constant(seed))
    padded_state, padded_context = state.copy(), context.copy()
    padded_state[:, 2] = 100.0
    padded_context[2] = np.array([99.0, 50.0], np.float32)
    second = runner._warmup_graph(tf.constant(padded_state), tf.constant(variance), tf.constant(padded_context), tf.constant(mask), tf.constant(seed))
    np.testing.assert_array_equal(first[0].numpy()[:, :2], second[0].numpy()[:, :2])
    np.testing.assert_array_equal(first[1].numpy()[:, :2], second[1].numpy()[:, :2])
    assert not np.allclose(first[1].numpy()[:, 0], first[1].numpy()[:, 1])

    production_seed = np.array([56, 78], np.int32)
    production_state_b = first[0].numpy().copy()
    production_state_b[:, 2] = 100.0
    production_a = runner._production_graph(first[0], first[1], tf.constant(variance), tf.constant(context), tf.constant(mask), tf.constant(production_seed))
    production_b = runner._production_graph(tf.constant(production_state_b), first[1], tf.constant(variance), tf.constant(padded_context), tf.constant(mask), tf.constant(production_seed))
    for left, right in zip(production_a, production_b):
        np.testing.assert_array_equal(left.numpy()[:, :, :2], right.numpy()[:, :, :2])


# --- evaluator qualification -------------------------------------------------


def test_contract_rejects_fewer_than_four_chains_and_oversized_batch():
    with pytest.raises(FrozenHMCError, match="num_chains"):
        FrozenVectorizedHMC(
            evaluator=GaussianContextEvaluator(2), num_chains=2, max_batch_size=2,
            warmup_config=WarmupConfig(warmup_steps=2), production_config=ProductionConfig(segment_size=2),
        )
    runner = _runner(max_batch=1, warmup=2, draws=2)
    batch = FrozenBatch((1, 2), _gaussian_context([[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0]] * 2), batch_index=0)
    with pytest.raises(FrozenHMCError, match="max_batch_size"):
        runner.bind_epoch(batch, run_seed=1, run_key=RUN_KEY, production_epoch=0)


def test_separability_check_rejects_state_context_and_training_bn_coupling():
    class CoupledEvaluator:
        axes_separable = True
        latent_dim = 2
        context_width = 1
        evaluator_identity = hashlib.sha256(b"coupled").hexdigest()

        @staticmethod
        def assert_runtime_identity():
            return None

        @staticmethod
        def evaluate(state, context):
            centered = state - tf.reduce_mean(state, axis=1, keepdims=True)
            return -0.5 * tf.reduce_sum(centered**2, axis=-1) + 0.0 * tf.reduce_sum(context)

    state = np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 10.0
    values = np.zeros((2, 1), np.float32)
    with pytest.raises(FrozenHMCError, match="unqualified generic evaluator"):
        FrozenVectorizedHMC(
            evaluator=CoupledEvaluator(), num_chains=4, max_batch_size=2,
            warmup_config=WarmupConfig(warmup_steps=2), production_config=ProductionConfig(segment_size=2),
        )
    with pytest.raises(FrozenHMCError, match="cross-axis"):
        assert_axis_separable(CoupledEvaluator(), state=state, context=values)

    class BatchNormEvaluator(CoupledEvaluator):
        evaluator_identity = hashlib.sha256(b"training-bn").hexdigest()
        norm = tf.keras.layers.BatchNormalization(center=False, scale=False)

        @classmethod
        def evaluate(cls, state, context):
            decoded = cls.norm(state, training=True)
            return -0.5 * tf.reduce_sum(decoded**2, axis=-1) + 0.0 * tf.reduce_sum(context)

    BatchNormEvaluator.norm(tf.constant(state), training=True)
    with pytest.raises(FrozenHMCError, match="cross-axis"):
        assert_axis_separable(BatchNormEvaluator(), state=state, context=values)

    class ContextCoupledEvaluator:
        axes_separable = True
        latent_dim = 1
        context_width = 1
        evaluator_identity = hashlib.sha256(b"context-coupled").hexdigest()

        @staticmethod
        def assert_runtime_identity():
            return None

        @staticmethod
        def evaluate(state, context):
            return -0.5 * tf.square(state[..., 0] - tf.reduce_mean(context[:, 0]))

    with pytest.raises(FrozenHMCError, match="cross-target context"):
        assert_axis_separable(ContextCoupledEvaluator(), state=np.zeros((4, 2, 1), np.float32), context=np.array([[0.0], [2.0]], np.float32))
    experimental = FrozenVectorizedHMC(
        evaluator=ContextCoupledEvaluator(), num_chains=4, max_batch_size=2,
        warmup_config=WarmupConfig(warmup_steps=2), production_config=ProductionConfig(segment_size=2),
        allow_unverified_evaluator=True,
    )
    with pytest.raises(FrozenHMCError, match="cross-target context"):
        experimental.bind_epoch(FrozenBatch((10, 11), np.array([[1.0], [3.0]], np.float32), batch_index=0), run_seed=1, run_key=RUN_KEY, production_epoch=0)

    class ContextIndependentEvaluator:
        @staticmethod
        def evaluate(state, context):
            del context
            return -0.5 * tf.reduce_sum(state**2, axis=-1)

    # A disconnected context gradient is the zero Jacobian, not a failure.
    assert_axis_separable(ContextIndependentEvaluator(), state=np.zeros((2, 2, 1), np.float32), context=np.array([[1.0], [3.0]], np.float32), padding_size=3)


def test_stochastic_target_is_rejected_before_sampling():
    class StochasticEvaluator:
        axes_separable = True
        latent_dim = 1
        context_width = 1
        evaluator_identity = hashlib.sha256(b"stochastic-target").hexdigest()

        @staticmethod
        def assert_runtime_identity():
            return None

        @staticmethod
        def evaluate(state, context):
            del context
            return -0.5 * tf.reduce_sum(state**2, axis=-1) + tf.random.uniform(tf.shape(state)[:2])

    runner = FrozenVectorizedHMC(
        evaluator=StochasticEvaluator(), num_chains=4, max_batch_size=1,
        warmup_config=WarmupConfig(warmup_steps=2), production_config=ProductionConfig(segment_size=2),
        allow_unverified_evaluator=True,
    )
    with pytest.raises(FrozenHMCError):
        runner.bind_epoch(FrozenBatch((1,), np.zeros((1, 1), np.float32), batch_index=0), run_seed=1, run_key=RUN_KEY, production_epoch=0)


def test_latent_posterior_evaluator_rejects_duck_typed_targets():
    class FakeResolved:
        model = SimpleNamespace(params={"z_dims": [1], "v_dim": 1})
        spec = SimpleNamespace(
            family="demand", use_bnn=False, dtype="float32", global_power=1.0,
            blocks=(SimpleNamespace(name="prior_z"), SimpleNamespace(name="covariate_v")),
            identity=hashlib.sha256(b"fake-resolved").hexdigest(),
            block_powers={"prior_z": 1.0, "covariate_v": 1.0},
            decoder_model_hash="fake-decoder", preprocessor_hash="fake-preprocessor",
            model_training_provenance_hash="fake-provenance",
        )

        @staticmethod
        def assert_runtime_identity():
            return None

        @staticmethod
        def log_prob(context, state):
            return -0.5 * tf.reduce_sum(state**2, axis=1) + tf.reduce_mean(context)

    with pytest.raises(FrozenHMCError, match="ResolvedTarget"):
        LatentPosteriorEvaluator(FakeResolved())


# --- mass matrix and initialization -----------------------------------------


def test_mass_regularization_caps_condition_and_keeps_dtype():
    raw = np.array([[1e-12, 1.0, 1e6], [0.5, 2.0, 8.0]], np.float32)
    regularized = regularize_state_variance(raw, MassRegularization(shrinkage=0.05, condition_cap=1e4))
    assert regularized.dtype == np.float32 and regularized.shape == raw.shape
    assert np.all(regularized > 0)
    assert np.all(regularized.max(axis=1) / regularized.min(axis=1) <= 1e4 * (1 + 1e-6))
    np.testing.assert_allclose(
        regularize_state_variance(raw[1:], MassRegularization(shrinkage=0.0, condition_cap=1e12)),
        raw[1:], rtol=1e-6,
    )
    with pytest.raises(ValueError):
        regularize_state_variance(np.array([[1.0, -1.0]]), MassRegularization())
    with pytest.raises(ValueError):
        MassRegularization(shrinkage=1.0).validate()


def test_overdispersed_initial_state_is_encoder_centred_and_seeded():
    params = {
        "dataset": "Init_demand", "output_dir": tempfile.mkdtemp(prefix="bgm_init_"),
        "save_res": False, "save_model": False, "binary_treatment": False, "use_bnn": False,
        "z_dims": [1, 1, 1, 1], "v_dim": 2, "w_dim": 1, "lr_theta": 5e-4, "lr_z": 5e-4,
        "g_units": [8, 8], "e_units": [8, 8], "f_units": [8, 4], "h_units": [8, 4], "dz_units": [8, 4],
        "kl_weight": 0.0, "lr": 5e-4, "g_d_freq": 1, "use_z_rec": True, "iv_mc_samples": 2,
        "eval_mc_samples": 2, "first_stage_warmup_epochs": 0,
    }
    model = BGM_IV(params=params, timestamp="init_1", random_seed=3)
    batch_v = np.random.default_rng(0).normal(size=(3, 2)).astype(np.float32)
    encoder = np.asarray(model.infer_latent_from_covariates(batch_v, method="encoder"), np.float32)
    variance = np.full((3, 4), 0.25, np.float32)
    kwargs = dict(num_chains=4, latent_dim=4, scale=2.0, variance=variance)
    first = overdispersed_initial_state(model, batch_v, epoch_identity="e1", **kwargs)
    assert first.shape == (4, 3, 4) and first.dtype == np.float32
    np.testing.assert_array_equal(first[0], encoder)
    assert all(not np.array_equal(first[c], encoder) for c in range(1, 4))
    np.testing.assert_array_equal(first, overdispersed_initial_state(model, batch_v, epoch_identity="e1", **kwargs))
    assert not np.array_equal(first, overdispersed_initial_state(model, batch_v, epoch_identity="e2", **kwargs))
    # dispersion is measured in mass-metric standard deviations
    spread = np.abs(first[1:] - encoder[None]).mean()
    assert 0.5 < spread < 2.0
