from __future__ import annotations

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
    _INITIAL_STATE_KEY,
    _SEGMENT_KEY,
    _WARMUP_KEY,
    FrozenHMCError,
    FrozenVectorizedHMC,
    _seed_pair,
    _seed_sequence,
    overdispersed_initial_state,
    regularize_state_variance,
)


class GaussianContextEvaluator:
    def __init__(self, latent_dim: int):
        self.latent_dim = int(latent_dim)
        self.context_width = 2 * self.latent_dim

    def evaluate(self, state, context):
        location = context[:, : self.latent_dim]
        log_scale = context[:, self.latent_dim :]
        scale = tf.exp(log_scale)
        standardized = (state - location[None, :, :]) / scale[None, :, :]
        return -0.5 * tf.reduce_sum(tf.square(standardized), axis=-1) - tf.reduce_sum(
            log_scale, axis=-1
        )[None, :]


def _context(locations, scales):
    loc = np.asarray(locations, np.float32)
    scale = np.asarray(scales, np.float32)
    return np.concatenate([loc, np.log(scale)], axis=1).astype(np.float32)


def _runner(num_targets=3, warmup=4, draws=5, support=(2, 3), num_chains=4):
    return FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(2),
        num_chains=num_chains,
        num_targets=num_targets,
        warmup_steps=warmup,
        initial_step_size=0.08,
        num_leapfrog_steps=3,
        target_accept_prob=0.8,
        segment_size=draws,
        trajectory_support=support,
    )


def _sample(runner, context, run_seed=77):
    context = runner.check_target(context)
    variance = np.ones((len(context), 2), np.float32)
    final_state, step = runner.warmup(
        run_seed=run_seed,
        context=context,
        initial_state=np.zeros((4, len(context), 2), np.float32),
        state_variance=variance,
    )
    draws, acceptance = runner.run_segment(
        run_seed=run_seed,
        context=context,
        state=final_state,
        step_size=step,
        state_variance=variance,
    )
    return final_state, step, draws, acceptance


def test_exact_replay_retains_only_draws_and_acceptance_rates():
    context = _context(
        [[0.0, 0.0], [1.0, -1.0], [-0.5, 0.25]],
        [[1.0, 1.0], [0.7, 1.2], [1.3, 0.8]],
    )
    runner = _runner()
    first = _sample(runner, context)
    second = _sample(runner, context)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    _, step, draws, acceptance = first
    assert step.shape == (4, 3, 1)
    assert draws.shape == (5, 4, 3, 2)
    assert acceptance.shape == (4, 3)
    assert np.all((0.0 <= acceptance) & (acceptance <= 1.0))
    assert runner._warmup_graph.experimental_get_tracing_count() == 1
    assert runner._production_graph.experimental_get_tracing_count() == 1


def test_run_seed_selects_distinct_streams():
    context = _context([[0.0, 0.0], [1.0, -1.0]], [[1.0, 1.0], [0.7, 1.2]])
    runner = _runner(num_targets=2)
    first = _sample(runner, context, run_seed=77)
    other = _sample(runner, context, run_seed=78)
    assert not np.array_equal(first[2], other[2])


def test_stream_spawn_keys_and_seed_pairs_are_pinned():
    assert _WARMUP_KEY == (0, 0, 0)
    assert _SEGMENT_KEY == (1, 0, 0, 0)
    assert _INITIAL_STATE_KEY == (2, 0, 0)
    assert _seed_pair(77, *_WARMUP_KEY).tolist() == [-266163261, 1866658239]
    assert _seed_pair(77, *_SEGMENT_KEY).tolist() == [1590207201, -344951743]
    initial = np.random.default_rng(_seed_sequence(5, *_INITIAL_STATE_KEY))
    np.testing.assert_array_equal(
        initial.standard_normal(3),
        [-1.1767615443084933, -1.7289054713323528, 1.4099385943759353],
    )


def test_other_targets_do_not_change_a_target_stream():
    context = _context(
        [[0.2, -0.1], [4.0, -3.0]], [[1.0, 0.8], [0.3, 1.7]]
    )
    runner = FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(2),
        num_chains=4,
        num_targets=2,
        warmup_steps=3,
        initial_step_size=0.1,
        num_leapfrog_steps=5,
        target_accept_prob=0.8,
        segment_size=5,
        trajectory_support=(2, 4),
    )
    context = runner.check_target(context)
    state_a = np.zeros((4, 2, 2), np.float32)
    state_b = state_a.copy()
    state_b[:, 1] = [25.0, -31.0]
    step = np.full((4, 2, 1), 0.1, np.float32)
    variance = np.ones((2, 2), np.float32)
    kwargs = dict(run_seed=9, context=context, step_size=step, state_variance=variance)
    left_draws, left_acceptance = runner.run_segment(state=state_a, **kwargs)
    right_draws, right_acceptance = runner.run_segment(state=state_b, **kwargs)
    np.testing.assert_array_equal(left_draws[:, :, 0], right_draws[:, :, 0])
    np.testing.assert_array_equal(left_acceptance[:, 0], right_acceptance[:, 0])


def test_target_count_mismatch_and_fewer_chains_are_rejected():
    with pytest.raises(FrozenHMCError, match="num_chains"):
        _runner(num_targets=2, num_chains=2)
    runner = _runner(num_targets=1)
    context = _context([[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0]] * 2)
    with pytest.raises(FrozenHMCError, match="num_targets"):
        runner.check_target(context)
    bad = _context([[0.0, 0.0]], [[1.0, 1.0]])
    bad[0, 0] = np.nan
    with pytest.raises(FrozenHMCError, match="finite"):
        runner.check_target(bad)


def test_mass_regularization_and_overdispersed_initialization():
    raw = np.array([[1e-12, 1.0, 1e6], [0.5, 2.0, 8.0]], np.float32)
    regularized = regularize_state_variance(raw).astype(np.float32)
    assert np.all(
        regularized.max(axis=1) / regularized.min(axis=1) <= 1e4 * (1 + 1e-6)
    )

    params = {
        "dataset": "Init_demand",
        "output_dir": tempfile.mkdtemp(prefix="bgm_init_"),
        "save_model": False,
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
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "structural_map_steps": 100,
        "structural_map_lr": 5e-4,
    }
    model = BGM_IV(params=params, timestamp="init", random_seed=3)
    batch_v = np.random.default_rng(0).normal(size=(3, 2)).astype(np.float32)
    variance = np.full((3, 4), 0.25, np.float32)
    kwargs = dict(num_chains=4, latent_dim=4, scale=2.0, variance=variance)
    first = overdispersed_initial_state(model, batch_v, run_seed=5, **kwargs)
    second = overdispersed_initial_state(model, batch_v, run_seed=5, **kwargs)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 3, 4)
    np.testing.assert_array_equal(first[0], model.encoder_latent(batch_v))
    other = overdispersed_initial_state(model, batch_v, run_seed=6, **kwargs)
    assert not np.array_equal(first[1:], other[1:])
