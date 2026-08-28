"""Frozen target-set HMC replay, acceptance logging and hard contracts."""

from __future__ import annotations

import hashlib
import tempfile

import numpy as np
import pytest
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from bgm_iv.models.bgm_iv import BGM_IV
from bgm_iv.mcmc.sampler import (
    FrozenBatch,
    FrozenHMCError,
    FrozenVectorizedHMC,
    GaussianContextEvaluator,
    MassRegularization,
    ProductionConfig,
    TrajectoryPolicy,
    WarmupConfig,
    assert_axis_separable,
    overdispersed_initial_state,
    regularize_state_variance,
)


RUN_KEY = hashlib.sha256(b"full-grid-hmc-tests").hexdigest()


def _context(locations, scales):
    loc = np.asarray(locations, np.float32)
    scale = np.asarray(scales, np.float32)
    return np.concatenate([loc, np.log(scale)], axis=1).astype(np.float32)


def _runner(max_targets=3, warmup=4, draws=5, support=(2, 3)):
    return FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(2),
        num_chains=4,
        max_batch_size=max_targets,
        warmup_config=WarmupConfig(
            warmup_steps=warmup,
            initial_step_size=0.08,
            num_leapfrog_steps=3,
        ),
        production_config=ProductionConfig(
            segment_size=draws,
            trajectory_policy=TrajectoryPolicy(tuple(support)),
        ),
    )


def _sample(runner, context):
    batch = FrozenBatch(tuple(range(len(context))), context, batch_index=0)
    epoch = runner.bind_epoch(
        batch, run_seed=77, run_key=RUN_KEY, production_epoch=0
    )
    state = np.zeros((4, len(context), 2), np.float32)
    variance = np.ones((len(context), 2), np.float32)
    warm = runner.warmup(
        epoch, initial_state=state, state_variance=variance
    )
    return runner.run_segment(
        epoch,
        segment_index=0,
        pre_state=warm.final_state,
        step_size=warm.step_size,
        state_variance=warm.state_variance,
    )


def test_exact_replay_retains_only_draws_and_acceptance_rates():
    context = _context(
        [[0.0, 0.0], [1.0, -1.0], [-0.5, 0.25]],
        [[1.0, 1.0], [0.7, 1.2], [1.3, 0.8]],
    )
    runner = _runner()
    first = _sample(runner, context)
    second = _sample(runner, context)
    for name in (
        "pre_state",
        "post_state",
        "draws",
        "acceptance_rate",
        "seed",
        "step_size",
        "state_variance",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.draws.shape == (5, 4, 3, 2)
    assert first.acceptance_rate.shape == (4, 3)
    assert np.all((0.0 <= first.acceptance_rate) & (first.acceptance_rate <= 1.0))
    for removed in (
        "is_accepted",
        "log_accept_ratio",
        "energy_error",
        "has_divergence",
        "trajectory_length",
    ):
        assert not hasattr(first, removed)
    assert runner.warmup_tracing_count == 1
    assert runner.production_tracing_count == 1


def test_other_targets_do_not_change_a_target_stream():
    evaluator = GaussianContextEvaluator(2)
    context = _context(
        [[0.2, -0.1], [4.0, -3.0]], [[1.0, 0.8], [0.3, 1.7]]
    )
    assert_axis_separable(
        evaluator,
        state=np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 10.0,
        context=context,
    )
    runner = FrozenVectorizedHMC(
        evaluator=evaluator,
        num_chains=4,
        max_batch_size=2,
        warmup_config=WarmupConfig(warmup_steps=3),
        production_config=ProductionConfig(
            segment_size=5, trajectory_policy=TrajectoryPolicy((2, 4))
        ),
    )
    batch = FrozenBatch((100, 200), context, batch_index=0)
    epoch = runner.bind_epoch(
        batch, run_seed=9, run_key=RUN_KEY, production_epoch=1
    )
    state_a = np.zeros((4, 2, 2), np.float32)
    state_b = state_a.copy()
    state_b[:, 1] = [25.0, -31.0]
    step = np.full((4, 2, 1), 0.1, np.float32)
    variance = np.ones((2, 2), np.float32)
    left = runner.run_segment(
        epoch,
        segment_index=0,
        pre_state=state_a,
        step_size=step,
        state_variance=variance,
    )
    right = runner.run_segment(
        epoch,
        segment_index=0,
        pre_state=state_b,
        step_size=step,
        state_variance=variance,
    )
    np.testing.assert_array_equal(left.draws[:, :, 0], right.draws[:, :, 0])
    np.testing.assert_array_equal(
        left.acceptance_rate[:, 0], right.acceptance_rate[:, 0]
    )


def test_oversized_target_set_and_fewer_chains_are_rejected():
    with pytest.raises(FrozenHMCError, match="num_chains"):
        FrozenVectorizedHMC(
            evaluator=GaussianContextEvaluator(2),
            num_chains=2,
            max_batch_size=2,
            warmup_config=WarmupConfig(warmup_steps=2),
            production_config=ProductionConfig(segment_size=2),
        )
    runner = _runner(max_targets=1)
    batch = FrozenBatch(
        (1, 2),
        _context([[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0]] * 2),
        batch_index=0,
    )
    with pytest.raises(FrozenHMCError, match="max_batch_size"):
        runner.bind_epoch(
            batch, run_seed=1, run_key=RUN_KEY, production_epoch=0
        )


def test_separability_and_stochasticity_checks_remain_fail_closed():
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
            del context
            centered = state - tf.reduce_mean(state, axis=1, keepdims=True)
            return -0.5 * tf.reduce_sum(centered**2, axis=-1)

    with pytest.raises(FrozenHMCError, match="cross-axis"):
        assert_axis_separable(
            CoupledEvaluator(),
            state=np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 10.0,
            context=np.zeros((2, 1), np.float32),
        )

    class StochasticEvaluator:
        axes_separable = True
        latent_dim = 1
        context_width = 1
        evaluator_identity = hashlib.sha256(b"stochastic").hexdigest()

        @staticmethod
        def assert_runtime_identity():
            return None

        @staticmethod
        def evaluate(state, context):
            del context
            return -0.5 * state[..., 0] ** 2 + tf.random.uniform(tf.shape(state)[:2])

    runner = FrozenVectorizedHMC(
        evaluator=StochasticEvaluator(),
        num_chains=4,
        max_batch_size=1,
        warmup_config=WarmupConfig(warmup_steps=2),
        production_config=ProductionConfig(segment_size=2),
        allow_unverified_evaluator=True,
    )
    with pytest.raises(FrozenHMCError):
        runner.bind_epoch(
            FrozenBatch((1,), np.zeros((1, 1), np.float32), batch_index=0),
            run_seed=1,
            run_key=RUN_KEY,
            production_epoch=0,
        )


def test_mass_regularization_and_overdispersed_initialization():
    raw = np.array([[1e-12, 1.0, 1e6], [0.5, 2.0, 8.0]], np.float32)
    regularized = regularize_state_variance(
        raw, MassRegularization(shrinkage=0.05, condition_cap=1e4)
    )
    assert regularized.dtype == np.float32
    assert np.all(
        regularized.max(axis=1) / regularized.min(axis=1) <= 1e4 * (1 + 1e-6)
    )

    params = {
        "dataset": "Init_demand",
        "output_dir": tempfile.mkdtemp(prefix="bgm_init_"),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 2,
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
    }
    model = BGM_IV(params=params, timestamp="init", random_seed=3)
    batch_v = np.random.default_rng(0).normal(size=(3, 2)).astype(np.float32)
    variance = np.full((3, 4), 0.25, np.float32)
    kwargs = dict(num_chains=4, latent_dim=4, scale=2.0, variance=variance)
    first = overdispersed_initial_state(
        model, batch_v, epoch_identity="e1", **kwargs
    )
    second = overdispersed_initial_state(
        model, batch_v, epoch_identity="e1", **kwargs
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 3, 4)
