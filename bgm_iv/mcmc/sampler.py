from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass

import tensorflow_probability as tfp

from .target import _model_family, latent_log_prob


class FrozenHMCError(RuntimeError):
    pass


class FrozenHMCNumericalError(FrozenHMCError):
    pass


def _int(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FrozenHMCError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise FrozenHMCError(f"{name} must be at least {minimum}")
    return result


# Changing these spawn keys changes every MCMC draw.
_WARMUP_KEY = (0, 0, 0)
_SEGMENT_KEY = (1, 0, 0, 0)
_INITIAL_STATE_KEY = (2, 0, 0)


def _seed_sequence(run_seed: int, *key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(run_seed), spawn_key=tuple(int(k) for k in key))


def _seed_pair(run_seed: int, *key: int) -> np.ndarray:
    state = _seed_sequence(run_seed, *key).generate_state(2, np.uint32)
    return state.view(np.int32).copy()


class LatentPosteriorEvaluator:
    def __init__(self, model: Any):
        self.model = model
        self.family = _model_family(model)
        self.latent_dim = int(sum(self.model.params["z_dims"]))
        self.context_width = int(self.model.params["v_dim"])

    def evaluate(self, state: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        c = tf.shape(state)[0]
        b = tf.shape(state)[1]
        flat_state = tf.reshape(state, [c * b, self.latent_dim])
        flat_context = tf.reshape(
            tf.tile(context[None, :, :], [c, 1, 1]),
            [c * b, self.context_width],
        )
        value = latent_log_prob(self.model, self.family, flat_context, flat_state)
        return tf.reshape(tf.convert_to_tensor(value, tf.float32), [c, b])


class FrozenVectorizedHMC:
    def __init__(
        self,
        *,
        evaluator: Any,
        num_chains: int,
        num_targets: int,
        warmup_steps: int,
        initial_step_size: float,
        num_leapfrog_steps: int,
        target_accept_prob: float,
        segment_size: int,
        trajectory_support: Sequence[int],
    ):
        self.evaluator = evaluator
        self.num_chains = _int("num_chains", num_chains, minimum=4)
        self.num_targets = _int("num_targets", num_targets, minimum=1)
        self.latent_dim = _int("evaluator.latent_dim", evaluator.latent_dim, minimum=1)
        self.context_width = _int(
            "evaluator.context_width", evaluator.context_width, minimum=1
        )
        self.warmup_steps = int(warmup_steps)
        self.initial_step_size = float(initial_step_size)
        self.num_leapfrog_steps = int(num_leapfrog_steps)
        self.target_accept_prob = float(target_accept_prob)
        self.segment_size = int(segment_size)
        self.trajectory_support = tuple(int(value) for value in trajectory_support)
        self._warmup_graph = self._build_warmup_graph()
        self._production_graph = self._build_production_graph()

    def check_target(self, target_context: Any) -> np.ndarray:
        context = np.asarray(target_context)
        if (
            context.dtype != np.dtype(np.float32)
            or context.ndim != 2
            or not np.all(np.isfinite(context))
        ):
            raise FrozenHMCError("target_context must be finite float32 [B,V]")
        if context.shape[0] != self.num_targets:
            raise FrozenHMCError("target count does not match num_targets")
        if context.shape[1] != self.context_width:
            raise FrozenHMCError("target context width does not match evaluator")
        context = np.ascontiguousarray(context.copy())
        context.setflags(write=False)
        return context

    def _build_warmup_graph(self):
        c, b, d, v = (
            self.num_chains,
            self.num_targets,
            self.latent_dim,
            self.context_width,
        )
        warmup_steps = self.warmup_steps

        @tf.function(
            input_signature=(
                tf.TensorSpec([c, b, d], tf.float32),
                tf.TensorSpec([b, d], tf.float32),
                tf.TensorSpec([b, v], tf.float32),
                tf.TensorSpec([2], tf.int32),
            ),
            autograph=False,
            reduce_retracing=True,
        )
        def warmup_graph(state, variance, context, seed):
            scale = tf.sqrt(variance)[None, :, :]
            initial_u = state / scale

            def target_u(value):
                return self.evaluator.evaluate(value * scale, context)

            step = tf.fill([c, b, 1], tf.cast(self.initial_step_size, tf.float32))
            base = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_u,
                step_size=step,
                num_leapfrog_steps=self.num_leapfrog_steps,
                store_parameters_in_results=True,
            )
            adaptation_steps = min(
                warmup_steps, max(1, int(round(0.8 * warmup_steps)))
            )
            adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=base,
                num_adaptation_steps=adaptation_steps,
                target_accept_prob=self.target_accept_prob,
                validate_args=True,
            )
            result = tfp.mcmc.sample_chain(
                num_results=1,
                num_burnin_steps=warmup_steps - 1,
                current_state=initial_u,
                kernel=adaptive,
                trace_fn=lambda *_: (),
                return_final_kernel_results=True,
                parallel_iterations=1,
                seed=seed,
            )
            return (
                result.all_states[0] * scale,
                result.final_kernel_results.new_step_size,
                result.final_kernel_results.step,
            )

        return warmup_graph

    def _build_production_graph(self):
        c, b, d, v = (
            self.num_chains,
            self.num_targets,
            self.latent_dim,
            self.context_width,
        )
        support_tuple = self.trajectory_support
        n_support = len(support_tuple)
        t_size = self.segment_size

        @tf.function(
            input_signature=(
                tf.TensorSpec([c, b, d], tf.float32),
                tf.TensorSpec([c, b, 1], tf.float32),
                tf.TensorSpec([b, d], tf.float32),
                tf.TensorSpec([b, v], tf.float32),
                tf.TensorSpec([2], tf.int32),
            ),
            autograph=False,
            reduce_retracing=True,
        )
        def production_graph(state, step, variance, context, seed):
            scale = tf.sqrt(variance)[None, :, :]
            initial_u = state / scale

            def target_u(value):
                return self.evaluator.evaluate(value * scale, context)

            kernels = tuple(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_u,
                    step_size=step,
                    num_leapfrog_steps=leapfrog,
                    store_parameters_in_results=True,
                )
                for leapfrog in support_tuple
            )
            draws_array = tf.TensorArray(tf.float32, t_size)
            accepted_counts = tf.zeros([c, b], tf.int32)

            def cond(index, *_):
                return index < t_size

            def body(index, current, draws, counts):
                selection_seed = tf.random.experimental.stateless_fold_in(
                    seed, 2 * index
                )
                support_index = tf.random.stateless_uniform(
                    [], selection_seed, minval=0, maxval=n_support, dtype=tf.int32
                )
                transition_seed = tf.random.experimental.stateless_fold_in(
                    seed, 2 * index + 1
                )

                def make_branch(kernel):
                    def branch():
                        previous = kernel.bootstrap_results(current)
                        next_state, kr = kernel.one_step(
                            current, previous, seed=transition_seed
                        )
                        return next_state, kr.is_accepted

                    return branch

                next_u, accepted = tf.switch_case(
                    support_index,
                    branch_fns=tuple(make_branch(kernel) for kernel in kernels),
                )
                return (
                    index + 1,
                    next_u,
                    draws.write(index, next_u * scale),
                    counts + tf.cast(accepted, tf.int32),
                )

            result = tf.while_loop(
                cond,
                body,
                loop_vars=(
                    tf.constant(0, tf.int32),
                    initial_u,
                    draws_array,
                    accepted_counts,
                ),
                parallel_iterations=1,
            )
            return result[2].stack(), result[3]

        return production_graph

    def _state(self, state: Any) -> np.ndarray:
        state = np.asarray(state, np.float32)
        if state.shape != (self.num_chains, self.num_targets, self.latent_dim):
            raise FrozenHMCError("state must be float32 [C,B,D] over every target")
        return state

    def warmup(
        self,
        *,
        run_seed: int,
        context: np.ndarray,
        initial_state: Any,
        state_variance: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state = self._state(initial_state)
        variance = np.asarray(state_variance, np.float32)
        seed = _seed_pair(run_seed, *_WARMUP_KEY)
        final, tuned, count = self._warmup_graph(
            tf.constant(state),
            tf.constant(variance),
            tf.constant(context),
            tf.constant(seed),
        )
        if int(count.numpy()) != self.warmup_steps:
            raise FrozenHMCError("warmup transition count mismatch")
        final_array = np.asarray(final.numpy(), np.float32)
        step_array = np.asarray(tuned.numpy(), np.float32)
        if (
            not np.all(np.isfinite(final_array))
            or not np.all(np.isfinite(step_array))
            or np.any(step_array <= 0.0)
        ):
            raise FrozenHMCNumericalError("warmup produced invalid state/step")
        return final_array, step_array

    def run_segment(
        self,
        *,
        run_seed: int,
        context: np.ndarray,
        state: Any,
        step_size: Any,
        state_variance: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state = self._state(state)
        step = np.asarray(step_size, np.float32)
        variance = np.asarray(state_variance, np.float32)
        seed = _seed_pair(run_seed, *_SEGMENT_KEY)
        draws_tensor, accepted_counts = self._production_graph(
            tf.constant(state),
            tf.constant(step),
            tf.constant(variance),
            tf.constant(context),
            tf.constant(seed),
        )
        draws = np.ascontiguousarray(draws_tensor.numpy()).astype(
            np.float32, copy=False
        )
        acceptance_rate = np.ascontiguousarray(accepted_counts.numpy()).astype(
            np.float64
        ) / float(self.segment_size)
        if not np.all(np.isfinite(draws)):
            raise FrozenHMCNumericalError("retained production state is non-finite")
        return draws, acceptance_rate


_MASS_SHRINKAGE = 0.05
_MASS_CONDITION_CAP = 1.0e4
_MASS_ABSOLUTE_FLOOR = 1.0e-8


def regularize_state_variance(raw_variance: Any) -> np.ndarray:
    raw = np.asarray(raw_variance, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) <= 0:
        raise ValueError("raw_variance must have non-empty shape [B,D]")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("raw_variance must be finite and strictly positive")
    safe = np.maximum(raw, _MASS_ABSOLUTE_FLOOR)
    log_raw = np.log(safe)
    center = np.mean(log_raw, axis=1, keepdims=True)
    shrunk = (1.0 - _MASS_SHRINKAGE) * log_raw + _MASS_SHRINKAGE * center
    half_range = 0.5 * np.log(_MASS_CONDITION_CAP)
    clipped = np.clip(shrunk, center - half_range, center + half_range)
    return np.exp(clipped)


def overdispersed_initial_state(
    model: Any,
    batch_v: np.ndarray,
    *,
    num_chains: int,
    latent_dim: int,
    scale: float,
    run_seed: int,
    variance: np.ndarray,
) -> np.ndarray:
    encoder_state = np.asarray(model.encoder_latent(batch_v), np.float32)
    if encoder_state.shape != (batch_v.shape[0], latent_dim):
        raise FrozenHMCError("encoder latent state has unexpected shape")
    rng = np.random.default_rng(_seed_sequence(run_seed, *_INITIAL_STATE_KEY))
    metric = np.sqrt(np.asarray(variance, np.float32))[None, :, :]
    chains = [encoder_state[None]]
    for _ in range(num_chains - 1):
        chains.append(
            encoder_state[None]
            + scale * metric * rng.standard_normal((1,) + encoder_state.shape)
        )
    state = np.concatenate(chains, axis=0).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise FrozenHMCError("initial state is non-finite")
    return state


__all__ = [
    "FrozenHMCError",
    "FrozenHMCNumericalError",
    "FrozenVectorizedHMC",
    "LatentPosteriorEvaluator",
    "overdispersed_initial_state",
    "regularize_state_variance",
]
