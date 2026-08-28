"""Frozen target-set vectorized HMC over the latent inference target.

One compiled graph serves the complete target catalog; chain and target axes
are factorized by construction, and the target set, epoch and seed are frozen
and hashed so a segment replays bitwise on the same device.
Importing this module turns TensorFloat-32 off and op determinism on.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass

import tensorflow_probability as tfp

from .target import ResolvedTarget, require_digest, sha256_array, sha256_json


class FrozenHMCError(RuntimeError):
    """Base error for a frozen target-set contract violation."""


class FrozenHMCNumericalError(FrozenHMCError):
    """An active retained state or target became non-finite."""


def _int(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FrozenHMCError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise FrozenHMCError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise FrozenHMCError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FrozenHMCError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise FrozenHMCError(f"{name} must be {qualifier}")
    return result


def _seed_pair(namespace: str, payload: Mapping[str, Any]) -> np.ndarray:
    raw = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(namespace.encode("utf-8") + b"\0" + raw).digest()
    return np.frombuffer(digest[:8], dtype="<i4").astype(np.int32, copy=True)


class GaussianContextEvaluator:
    """Independent diagonal-Gaussian targets encoded as ``[loc, log_scale]``."""

    axes_separable = True

    def __init__(self, latent_dim: int):
        self.latent_dim = _int("latent_dim", latent_dim, minimum=1)
        self.context_width = 2 * self.latent_dim
        self.evaluator_contract = {
            "evaluator_id": "analytic.diagonal_gaussian_context",
            "latent_dim": self.latent_dim,
            "context": "concat(location,log_scale)",
            "axes_separable": True,
            "dtype": "float32",
        }
        self.evaluator_identity = sha256_json("evaluator", self.evaluator_contract)

    def assert_runtime_identity(self) -> None:
        return None

    def evaluate(self, state: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        location = context[:, : self.latent_dim]
        log_scale = context[:, self.latent_dim :]
        scale = tf.exp(log_scale)
        standardized = (state - location[None, :, :]) / scale[None, :, :]
        return -0.5 * tf.reduce_sum(tf.square(standardized), axis=-1) - tf.reduce_sum(
            log_scale, axis=-1
        )[None, :]


class LatentPosteriorEvaluator:
    """Vectorizes a :class:`ResolvedTarget` log density over chain and target axes.

    The target owns the evidence blocks, decoder and preprocessor identity and
    runtime mutation checks; this wrapper only reshapes ``[C, B, D]`` states
    and ``[B, V]`` contexts into one flat ``log_prob`` call.
    """

    axes_separable = True

    def __init__(self, resolved_target: Any):
        if type(resolved_target) is not ResolvedTarget:
            raise FrozenHMCError("resolved_target must be a ResolvedTarget")
        self.resolved_target = resolved_target
        self.model = resolved_target.model
        self.spec = resolved_target.spec
        if str(self.spec.family) not in {"demand", "vector", "mnist"}:
            raise FrozenHMCError("unsupported target family")
        if bool(self.spec.use_bnn):
            raise FrozenHMCError("stochastic BNN target is not fixed")
        if str(self.spec.dtype) != "float32":
            raise FrozenHMCError("target dtype must be float32")
        if not math.isfinite(float(self.spec.global_power)) or float(
            self.spec.global_power
        ) <= 0.0:
            raise FrozenHMCError("target global_power must be positive")
        block_names = tuple(block.name for block in self.spec.blocks)
        if self.spec.family == "demand":
            expected = ("prior_z", "covariate_v")
        elif self.spec.family == "vector":
            expected = ("prior_z", "time", "vector_proxy")
        else:
            expected = ("prior_z", "time", "pixels")
            if int(self.model.extra_noise_dim) > 0:
                expected += ("noise",)
        if block_names != expected:
            raise FrozenHMCError("unexpected target block layout")
        self.latent_dim = int(sum(self.model.params["z_dims"]))
        self.context_width = int(self.model.params["v_dim"])
        self.evaluator_contract = {
            "evaluator_id": "bgm_iv.latent_posterior_batch",
            "resolved_target_identity": str(self.spec.identity),
            "family": str(self.spec.family),
            "target_kind": str(self.spec.target_kind),
            "global_power": float(self.spec.global_power),
            "block_powers": {
                str(name): float(value)
                for name, value in self.spec.block_powers.items()
            },
            "decoder_model_hash": str(self.spec.decoder_model_hash),
            "preprocessor_hash": str(self.spec.preprocessor_hash),
            "model_training_provenance_hash": str(
                self.spec.model_training_provenance_hash
            ),
            "axes_separable": True,
            "dtype": "float32",
        }
        self.evaluator_identity = sha256_json("evaluator", self.evaluator_contract)

    def assert_runtime_identity(self) -> None:
        self.resolved_target.assert_runtime_identity()

    def evaluate(self, state: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        c = tf.shape(state)[0]
        b = tf.shape(state)[1]
        flat_state = tf.reshape(state, [c * b, self.latent_dim])
        flat_context = tf.reshape(
            tf.tile(context[None, :, :], [c, 1, 1]),
            [c * b, self.context_width],
        )
        # Exact type qualification above excludes subclass overrides.
        value = ResolvedTarget.log_prob(
            self.resolved_target, flat_context, flat_state
        )
        return tf.reshape(tf.convert_to_tensor(value, tf.float32), [c, b])


@dataclass(frozen=True)
class TrajectoryPolicy:
    support: Tuple[int, ...] = (3, 5, 7)
    policy_id: str = "target_set_common_stateless_uniform_support"

    def validate(self) -> "TrajectoryPolicy":
        support = tuple(_int("trajectory length", value, minimum=1) for value in self.support)
        if not support or len(support) != len(set(support)):
            raise FrozenHMCError("trajectory support must be non-empty and unique")
        if self.policy_id != "target_set_common_stateless_uniform_support":
            raise FrozenHMCError("unsupported trajectory policy")
        object.__setattr__(self, "support", support)
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "policy_id": self.policy_id,
            "support": list(self.support),
            "selection": "one_state_independent_L_per_target_set_transition",
            "state_dependent": False,
        }


@dataclass(frozen=True)
class WarmupConfig:
    warmup_steps: int = 500
    adaptation_fraction: float = 0.8
    initial_step_size: float = 0.1
    num_leapfrog_steps: int = 5
    target_accept_prob: float = 0.8

    def validate(self) -> "WarmupConfig":
        _int("warmup_steps", self.warmup_steps, minimum=1)
        _int("num_leapfrog_steps", self.num_leapfrog_steps, minimum=1)
        _finite_float("initial_step_size", self.initial_step_size, positive=True)
        fraction = _finite_float("adaptation_fraction", self.adaptation_fraction)
        accept = _finite_float("target_accept_prob", self.target_accept_prob)
        if not 0.0 < fraction <= 1.0:
            raise FrozenHMCError("adaptation_fraction must lie in (0,1]")
        if not 0.0 < accept < 1.0:
            raise FrozenHMCError("target_accept_prob must lie in (0,1)")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "warmup_steps": int(self.warmup_steps),
            "adaptation_fraction": float(self.adaptation_fraction),
            "initial_step_size": float(self.initial_step_size),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "target_accept_prob": float(self.target_accept_prob),
        }


@dataclass(frozen=True)
class ProductionConfig:
    segment_size: int = 500
    trajectory_policy: TrajectoryPolicy = TrajectoryPolicy()

    def validate(self) -> "ProductionConfig":
        _int("segment_size", self.segment_size, minimum=1)
        self.trajectory_policy.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "segment_size": int(self.segment_size),
            "trajectory_policy": self.trajectory_policy.to_dict(),
            "adaptation": False,
        }


@dataclass(frozen=True)
class FrozenBatch:
    global_ids: Tuple[int, ...]
    target_context: np.ndarray
    batch_index: int

    def __post_init__(self) -> None:
        ids = tuple(_int("global_id", value) for value in self.global_ids)
        if not ids or len(ids) != len(set(ids)):
            raise FrozenHMCError("global_ids must be non-empty and unique")
        context = np.asarray(self.target_context)
        if (
            context.dtype != np.dtype(np.float32)
            or context.ndim != 2
            or context.shape[0] != len(ids)
            or not np.all(np.isfinite(context))
        ):
            raise FrozenHMCError("target_context must be finite float32 [B,V]")
        context = np.ascontiguousarray(context.copy())
        context.setflags(write=False)
        object.__setattr__(self, "global_ids", ids)
        object.__setattr__(self, "target_context", context)
        object.__setattr__(self, "batch_index", _int("batch_index", self.batch_index))

    @property
    def active_size(self) -> int:
        return len(self.global_ids)


@dataclass(frozen=True)
class FrozenEpoch:
    batch: FrozenBatch
    run_seed: int
    run_key: str
    production_epoch: int
    qualification_hash: str
    epoch_identity: str


@dataclass(frozen=True)
class WarmupResult:
    epoch_identity: str
    global_ids: Tuple[int, ...]
    final_state: np.ndarray             # [C,B,D]
    step_size: np.ndarray               # [C,B,1]
    state_variance: np.ndarray          # [B,D]
    seed: np.ndarray                    # [2]


@dataclass(frozen=True)
class ProductionSegment:
    epoch_identity: str
    replay_key: str
    global_ids: Tuple[int, ...]
    segment_index: int
    pre_state: np.ndarray               # [C,B,D]
    post_state: np.ndarray              # [C,B,D]
    draws: np.ndarray                   # [T,C,B,D]
    acceptance_rate: np.ndarray         # [C,B], logging only
    seed: np.ndarray                    # [2]
    step_size: np.ndarray               # [C,B,1]
    state_variance: np.ndarray          # [B,D]


class FrozenVectorizedHMC:
    """Batched HMC with one compiled warmup graph and one production graph per evaluator.

    Warmup adapts the step size per (chain, target); production runs a fixed
    kernel whose trajectory length is drawn per transition from
    ``TrajectoryPolicy.support``.  Every random stream is a stateless function
    of the frozen epoch identity, so a segment replays bitwise on the same
    device.
    """

    def __init__(
        self,
        *,
        evaluator: Any,
        num_chains: int,
        max_batch_size: int,
        warmup_config: WarmupConfig,
        production_config: ProductionConfig,
        allow_unverified_evaluator: bool = False,
    ):
        self.evaluator = evaluator
        self.num_chains = _int("num_chains", num_chains, minimum=4)
        self.max_batch_size = _int("max_batch_size", max_batch_size, minimum=1)
        self.latent_dim = _int("evaluator.latent_dim", evaluator.latent_dim, minimum=1)
        self.context_width = _int(
            "evaluator.context_width", evaluator.context_width, minimum=1
        )
        if not isinstance(allow_unverified_evaluator, bool):
            raise FrozenHMCError("allow_unverified_evaluator must be bool")
        self.allow_unverified_evaluator = allow_unverified_evaluator
        qualified_types = {
            GaussianContextEvaluator,
            LatentPosteriorEvaluator,
        }
        if type(evaluator) not in qualified_types and not allow_unverified_evaluator:
            raise FrozenHMCError(
                "unqualified generic evaluator is fail-closed; use a built-in "
                "pointwise evaluator or set allow_unverified_evaluator"
            )
        if not bool(getattr(evaluator, "axes_separable", False)):
            raise FrozenHMCError("target evaluator must declare target/chain separability")
        try:
            evaluator_source = inspect.getsource(type(evaluator).evaluate)
        except (OSError, TypeError):
            if not allow_unverified_evaluator:
                raise FrozenHMCError(
                    "qualified evaluator source identity is unavailable"
                )
            evaluator_source = "unavailable-experimental-source"
        self.evaluator_qualification_contract = {
            "scheme": "exact_type_plus_locality_jacobian_permutation_padding",
            "evaluator_type": (
                f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"
            ),
            "evaluate_source_hash": hashlib.sha256(
                evaluator_source.encode("utf-8")
            ).hexdigest(),
            "exact_builtin_type": type(evaluator) in qualified_types,
            "experimental_override": bool(allow_unverified_evaluator),
        }
        self.evaluator_qualification_contract_hash = sha256_json("evaluator-qualification",
            self.evaluator_qualification_contract
        )
        self.warmup_config = warmup_config.validate()
        self.production_config = production_config.validate()
        self.kernel_contract = {
            "runner_id": "frozen_vectorized_hmc",
            "tensorflow_version": tf.__version__,
            "tensorflow_probability_version": tfp.__version__,
            "dtype": "float32",
            "tensorfloat32": False,
            "num_chains": self.num_chains,
            "max_batch_size": self.max_batch_size,
            "latent_dim": self.latent_dim,
            "context_width": self.context_width,
            "evaluator_identity": evaluator.evaluator_identity,
            "unverified_evaluator_override": bool(allow_unverified_evaluator),
            "evaluator_qualification_contract": (
                self.evaluator_qualification_contract
            ),
            "evaluator_qualification_contract_hash": (
                self.evaluator_qualification_contract_hash
            ),
            "axes": "chain_and_target_factorized",
            "partial_batch": "active_prefix_zero_padding_mask",
            "mass_parameterization": (
                "linear_whitening_z_equals_sqrt_state_variance_times_u"
            ),
            "rng_contract": "one_stateless_seed_per_frozen_target_set_segment",
            "replay_contract": "same_runtime_device_frozen_membership_order",
            "warmup": self.warmup_config.to_dict(),
            "production": self.production_config.to_dict(),
        }
        self.kernel_identity = sha256_json("kernel", self.kernel_contract)
        self._warmup_graph = self._build_warmup_graph()
        self._production_graph = self._build_production_graph()

    def _qualify_batch_evaluator(self, batch: FrozenBatch) -> str:
        """Run the separability check on this batch and return its identity."""

        if batch.active_size >= 2:
            probe_context = np.ascontiguousarray(batch.target_context[:2])
        else:
            first = np.asarray(batch.target_context[0], np.float32)
            delta = np.linspace(
                0.03125, 0.0625, self.context_width, dtype=np.float32
            )
            probe_context = np.stack([first, first + delta], axis=0)
        probe_state = np.linspace(
            -0.375,
            0.625,
            2 * 2 * self.latent_dim,
            dtype=np.float32,
        ).reshape(2, 2, self.latent_dim)
        self.evaluator.assert_runtime_identity()
        assert_axis_separable(
            self.evaluator,
            state=probe_state,
            context=probe_context,
            padding_size=3,
        )
        self.evaluator.assert_runtime_identity()
        return sha256_json("batch-qualification",
            {
                "scheme": self.evaluator_qualification_contract["scheme"],
                "qualification_contract_hash": (
                    self.evaluator_qualification_contract_hash
                ),
                "evaluator_identity": self.evaluator.evaluator_identity,
                "probe_state_hash": sha256_array(probe_state),
                "probe_context_hash": sha256_array(probe_context),
                "padding_size": 3,
                "result": "passed",
            }
        )

    def bind_epoch(
        self,
        batch: FrozenBatch,
        *,
        run_seed: int,
        run_key: str,
        production_epoch: int,
    ) -> FrozenEpoch:
        self._validate_batch(batch)
        seed = _int("run_seed", run_seed)
        try:
            key = require_digest("run_key", run_key.lower() if isinstance(run_key, str) else run_key)
        except ValueError as exc:
            raise FrozenHMCError(str(exc)) from exc
        epoch = _int("production_epoch", production_epoch)
        qualification_hash = self._qualify_batch_evaluator(batch)
        identity = sha256_json("epoch",
            {
                "contract": "frozen_batch_epoch",
                "global_ids": list(batch.global_ids),
                "target_context_hash": sha256_array(batch.target_context),
                "batch_index": batch.batch_index,
                "run_seed": seed,
                "run_key": key,
                "production_epoch": epoch,
                "kernel_identity": self.kernel_identity,
                "qualification_hash": qualification_hash,
            }
        )
        return FrozenEpoch(batch, seed, key, epoch, qualification_hash, identity)

    def _validate_batch(self, batch: FrozenBatch) -> None:
        if batch.active_size > self.max_batch_size:
            raise FrozenHMCError("batch exceeds max_batch_size")
        if batch.target_context.shape[1] != self.context_width:
            raise FrozenHMCError("batch context width does not match evaluator")

    def _pad_context(self, batch: FrozenBatch) -> Tuple[np.ndarray, np.ndarray]:
        context = np.zeros(
            (self.max_batch_size, self.context_width), dtype=np.float32
        )
        context[: batch.active_size] = batch.target_context
        mask = np.zeros(self.max_batch_size, dtype=bool)
        mask[: batch.active_size] = True
        return context, mask

    def _validate_state_variance(
        self, state: Any, variance: Any, *, step: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        z = np.asarray(state)
        if z.dtype != np.dtype(np.float32) or z.ndim != 3:
            raise FrozenHMCError("state must be exact float32 [C,B,D]")
        if z.shape[0] != self.num_chains or z.shape[2] != self.latent_dim:
            raise FrozenHMCError("state chain/latent shape mismatch")
        if not np.all(np.isfinite(z)):
            raise FrozenHMCError("state must be finite")
        mass = np.asarray(variance)
        if (
            mass.dtype != np.dtype(np.float32)
            or mass.shape != (z.shape[1], self.latent_dim)
            or not np.all(np.isfinite(mass))
            or np.any(mass <= 0.0)
        ):
            raise FrozenHMCError("state_variance must be positive float32 [B,D]")
        step_array = None
        if step is not None:
            step_array = np.asarray(step)
            if (
                step_array.dtype != np.dtype(np.float32)
                or step_array.shape != (self.num_chains, z.shape[1], 1)
                or not np.all(np.isfinite(step_array))
                or np.any(step_array <= 0.0)
            ):
                raise FrozenHMCError("step_size must be positive float32 [C,B,1]")
        return (
            np.ascontiguousarray(z),
            np.ascontiguousarray(mass),
            None if step_array is None else np.ascontiguousarray(step_array),
        )

    def _padded_arrays(
        self,
        batch: FrozenBatch,
        state: np.ndarray,
        variance: np.ndarray,
        step: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        b = batch.active_size
        padded_state = np.zeros(
            (self.num_chains, self.max_batch_size, self.latent_dim), np.float32
        )
        padded_state[:, :b] = state
        padded_variance = np.ones(
            (self.max_batch_size, self.latent_dim), np.float32
        )
        padded_variance[:b] = variance
        padded_step = None
        if step is not None:
            padded_step = np.full(
                (self.num_chains, self.max_batch_size, 1),
                float(self.warmup_config.initial_step_size),
                np.float32,
            )
            padded_step[:, :b] = step
        context, mask = self._pad_context(batch)
        return padded_state, padded_variance, padded_step, context, mask

    def _assert_active_target_valid(
        self, state: np.ndarray, context: np.ndarray
    ) -> None:
        z = tf.constant(state, tf.float32)
        values = tf.constant(context, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z)
            first = tf.convert_to_tensor(
                self.evaluator.evaluate(z, values), tf.float32
            )
            total = tf.reduce_sum(first)
        gradient = tape.gradient(total, z)
        second = tf.convert_to_tensor(
            self.evaluator.evaluate(z, values), tf.float32
        )
        if tuple(first.shape) != (state.shape[0], state.shape[1]):
            raise FrozenHMCError("evaluator must map [C,B,D] to [C,B]")
        if gradient is None or tuple(gradient.shape) != tuple(state.shape):
            raise FrozenHMCError("evaluator must provide a state gradient")
        if not np.array_equal(first.numpy(), second.numpy()):
            raise FrozenHMCError("target evaluator is stochastic at fixed inputs")
        if not bool(tf.reduce_all(tf.math.is_finite(first)).numpy()) or not bool(
            tf.reduce_all(tf.math.is_finite(gradient)).numpy()
        ):
            raise FrozenHMCNumericalError(
                "active target or gradient is non-finite at retained state"
            )

    def _masked_target(
        self, state: tf.Tensor, context: tf.Tensor, active_mask: tf.Tensor
    ) -> tf.Tensor:
        active = self.evaluator.evaluate(state, context)
        padding = -0.5 * tf.reduce_sum(tf.square(state), axis=-1)
        return tf.where(active_mask[None, :], active, padding)

    def _build_warmup_graph(self):
        c, m, d, v = (
            self.num_chains,
            self.max_batch_size,
            self.latent_dim,
            self.context_width,
        )
        cfg = self.warmup_config

        @tf.function(
            input_signature=(
                tf.TensorSpec([c, m, d], tf.float32),
                tf.TensorSpec([m, d], tf.float32),
                tf.TensorSpec([m, v], tf.float32),
                tf.TensorSpec([m], tf.bool),
                tf.TensorSpec([2], tf.int32),
            ),
            autograph=False,
            reduce_retracing=True,
        )
        def warmup_graph(state, variance, context, mask, seed):
            scale = tf.sqrt(variance)[None, :, :]
            initial_u = state / scale

            def target_u(value):
                return self._masked_target(value * scale, context, mask)

            step = tf.fill([c, m, 1], tf.cast(cfg.initial_step_size, tf.float32))
            base = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_u,
                step_size=step,
                num_leapfrog_steps=int(cfg.num_leapfrog_steps),
                store_parameters_in_results=True,
            )
            adaptation_steps = min(
                int(cfg.warmup_steps),
                max(1, int(round(cfg.adaptation_fraction * cfg.warmup_steps))),
            )
            adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=base,
                num_adaptation_steps=adaptation_steps,
                target_accept_prob=float(cfg.target_accept_prob),
                validate_args=True,
            )
            result = tfp.mcmc.sample_chain(
                num_results=1,
                num_burnin_steps=int(cfg.warmup_steps) - 1,
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
        c, m, d, v = (
            self.num_chains,
            self.max_batch_size,
            self.latent_dim,
            self.context_width,
        )
        cfg = self.production_config
        support_tuple = tuple(cfg.trajectory_policy.support)
        n_support = len(support_tuple)
        t_size = int(cfg.segment_size)

        @tf.function(
            input_signature=(
                tf.TensorSpec([c, m, d], tf.float32),
                tf.TensorSpec([c, m, 1], tf.float32),
                tf.TensorSpec([m, d], tf.float32),
                tf.TensorSpec([m, v], tf.float32),
                tf.TensorSpec([m], tf.bool),
                tf.TensorSpec([2], tf.int32),
            ),
            autograph=False,
            reduce_retracing=True,
        )
        def production_graph(state, step, variance, context, mask, seed):
            scale = tf.sqrt(variance)[None, :, :]
            initial_u = state / scale

            def target_u(value):
                return self._masked_target(value * scale, context, mask)

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
            accepted_counts = tf.zeros([c, m], tf.int32)

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

    @property
    def warmup_tracing_count(self) -> int:
        return int(self._warmup_graph.experimental_get_tracing_count())

    @property
    def production_tracing_count(self) -> int:
        return int(self._production_graph.experimental_get_tracing_count())

    def warmup(
        self,
        epoch: FrozenEpoch,
        *,
        initial_state: Any,
        state_variance: Any,
    ) -> WarmupResult:
        self._validate_batch(epoch.batch)
        state, variance, _ = self._validate_state_variance(
            initial_state, state_variance
        )
        if state.shape[1] != epoch.batch.active_size:
            raise FrozenHMCError("state B does not match frozen batch")
        padded_state, padded_variance, _, context, mask = self._padded_arrays(
            epoch.batch, state, variance
        )
        seed = _seed_pair(
            "bgm-iv-frozen-batch-warmup",
            {"epoch_identity": epoch.epoch_identity},
        )
        self.evaluator.assert_runtime_identity()
        self._assert_active_target_valid(state, epoch.batch.target_context)
        final, tuned, count = self._warmup_graph(
            tf.constant(padded_state),
            tf.constant(padded_variance),
            tf.constant(context),
            tf.constant(mask),
            tf.constant(seed),
        )
        self.evaluator.assert_runtime_identity()
        if int(count.numpy()) != int(self.warmup_config.warmup_steps):
            raise FrozenHMCError("warmup transition count mismatch")
        b = epoch.batch.active_size
        final_array = np.asarray(final.numpy()[:, :b], np.float32)
        step_array = np.asarray(tuned.numpy()[:, :b], np.float32)
        if (
            not np.all(np.isfinite(final_array))
            or not np.all(np.isfinite(step_array))
            or np.any(step_array <= 0.0)
        ):
            raise FrozenHMCNumericalError("warmup produced invalid active state/step")
        self._assert_active_target_valid(final_array, epoch.batch.target_context)
        return WarmupResult(
            epoch_identity=epoch.epoch_identity,
            global_ids=epoch.batch.global_ids,
            final_state=final_array,
            step_size=step_array,
            state_variance=variance.copy(),
            seed=seed,
        )

    def run_segment(
        self,
        epoch: FrozenEpoch,
        *,
        segment_index: int,
        pre_state: Any,
        step_size: Any,
        state_variance: Any,
    ) -> ProductionSegment:
        self._validate_batch(epoch.batch)
        segment = _int("segment_index", segment_index)
        state, variance, step = self._validate_state_variance(
            pre_state, state_variance, step=step_size
        )
        assert step is not None
        if state.shape[1] != epoch.batch.active_size:
            raise FrozenHMCError("state B does not match frozen batch")
        padded_state, padded_variance, padded_step, context, mask = self._padded_arrays(
            epoch.batch, state, variance, step
        )
        assert padded_step is not None
        seed = _seed_pair(
            "bgm-iv-frozen-batch-production-segment",
            {
                "epoch_identity": epoch.epoch_identity,
                "segment_index": segment,
            },
        )
        replay_key = sha256_json("segment",
            {
                "epoch_identity": epoch.epoch_identity,
                "segment_index": segment,
                "pre_state": sha256_array(state),
                "step_size": sha256_array(step),
                "state_variance": sha256_array(variance),
                "seed": sha256_array(seed),
            }
        )
        self.evaluator.assert_runtime_identity()
        self._assert_active_target_valid(state, epoch.batch.target_context)
        outputs = self._production_graph(
            tf.constant(padded_state),
            tf.constant(padded_step),
            tf.constant(padded_variance),
            tf.constant(context),
            tf.constant(mask),
            tf.constant(seed),
        )
        self.evaluator.assert_runtime_identity()
        draws_padded, accepted_counts = outputs
        b = epoch.batch.active_size
        draws = np.ascontiguousarray(draws_padded.numpy()[:, :, :b]).astype(
            np.float32, copy=False
        )
        acceptance_rate = (
            np.ascontiguousarray(accepted_counts.numpy()[:, :b]).astype(np.float64)
            / float(self.production_config.segment_size)
        )
        if not np.all(np.isfinite(draws)):
            raise FrozenHMCNumericalError("active retained production state is non-finite")
        self._assert_active_target_valid(draws[-1], epoch.batch.target_context)
        self.evaluator.assert_runtime_identity()
        return ProductionSegment(
            epoch_identity=epoch.epoch_identity,
            replay_key=replay_key,
            global_ids=epoch.batch.global_ids,
            segment_index=segment,
            pre_state=state.copy(),
            post_state=draws[-1].copy(),
            draws=draws,
            acceptance_rate=acceptance_rate,
            seed=seed,
            step_size=step.copy(),
            state_variance=variance.copy(),
        )


def assert_axis_separable(
    evaluator: Any,
    *,
    state: np.ndarray,
    context: np.ndarray,
    atol: float = 1e-6,
    padding_size: Optional[int] = None,
) -> None:
    """Check that an evaluator factorizes over chains and targets.

    For each output ``logpi[c,b]`` the derivatives with respect to another
    chain's or target's state and another target's context must vanish,
    reordering target rows must be equivariant, and the zero padding row
    must give a finite value and gradient.
    """

    state_array = np.asarray(state, dtype=np.float32)
    context_array = np.asarray(context, dtype=np.float32)
    if state_array.ndim != 3 or context_array.ndim != 2:
        raise FrozenHMCError("separability check requires [C,B,D] and [B,V]")
    if state_array.shape[1] != context_array.shape[0]:
        raise FrozenHMCError("state/context B mismatch in separability check")
    z = tf.constant(state_array)
    values = tf.constant(context_array)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        tape.watch(values)
        output = evaluator.evaluate(z, values)
    state_jacobian_tensor = tape.jacobian(output, z)
    context_jacobian_tensor = tape.jacobian(output, values)
    del tape
    if state_jacobian_tensor is None:
        raise FrozenHMCError("target evaluator has no state Jacobian")
    state_jacobian = state_jacobian_tensor.numpy()
    if context_jacobian_tensor is None:
        # A target independent of context is local: every context derivative
        # is exactly zero.  TensorFlow represents this as no gradient.
        context_jacobian = np.zeros(
            tuple(output.shape) + tuple(values.shape), dtype=np.float32
        )
    else:
        context_jacobian = context_jacobian_tensor.numpy()
    c, b = output.shape
    for out_chain in range(int(c)):
        for out_target in range(int(b)):
            state_block = state_jacobian[out_chain, out_target]
            state_mask = np.ones(state_block.shape[:2], dtype=bool)
            state_mask[out_chain, out_target] = False
            if np.any(np.abs(state_block[state_mask]) > atol):
                raise FrozenHMCError("target evaluator has cross-axis state dependence")
            context_block = context_jacobian[out_chain, out_target]
            context_mask = np.ones(context_block.shape[0], dtype=bool)
            context_mask[out_target] = False
            if np.any(np.abs(context_block[context_mask]) > atol):
                raise FrozenHMCError(
                    "target evaluator has cross-target context dependence"
                )

    permutation = np.arange(int(b) - 1, -1, -1)
    permuted = evaluator.evaluate(
        tf.gather(z, permutation, axis=1), tf.gather(values, permutation, axis=0)
    ).numpy()
    if not np.allclose(
        permuted, output.numpy()[:, permutation], atol=atol, rtol=0.0
    ):
        raise FrozenHMCError("target evaluator is not target-permutation equivariant")

    padded_b = int(b) if padding_size is None else _int(
        "padding_size", padding_size, minimum=int(b)
    )
    zero_state = tf.zeros([int(c), padded_b, state_array.shape[2]], tf.float32)
    zero_context = tf.zeros([padded_b, context_array.shape[1]], tf.float32)
    with tf.GradientTape() as padding_tape:
        padding_tape.watch(zero_state)
        zero_output = evaluator.evaluate(zero_state, zero_context)
        zero_total = tf.reduce_sum(zero_output)
    zero_gradient = padding_tape.gradient(zero_total, zero_state)
    if zero_gradient is None or not bool(
        tf.reduce_all(tf.math.is_finite(zero_output)).numpy()
    ) or not bool(tf.reduce_all(tf.math.is_finite(zero_gradient)).numpy()):
        raise FrozenHMCError("zero padding target/gradient is not finite")


# --- mass matrix -------------------------------------------------------------


@dataclass(frozen=True)
class MassRegularization:
    """Log-shrinkage and condition cap applied to a pilot variance estimate."""

    shrinkage: float = 0.05
    condition_cap: float = 1.0e4
    absolute_floor: float = 1.0e-8

    def validate(self) -> "MassRegularization":
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError("shrinkage must lie in [0, 1)")
        if not np.isfinite(self.condition_cap) or self.condition_cap < 1.0:
            raise ValueError("condition_cap must be finite and at least one")
        if not np.isfinite(self.absolute_floor) or self.absolute_floor <= 0.0:
            raise ValueError("absolute_floor must be finite and positive")
        return self


def regularize_state_variance(
    raw_variance: Any, config: MassRegularization
) -> np.ndarray:
    """Log-shrink and determinant-centered condition-cap a positive diagonal."""

    cfg = config.validate()
    raw = np.asarray(raw_variance, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) <= 0:
        raise ValueError("raw_variance must have non-empty shape [B,D]")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_variance contains non-finite values")
    if np.any(raw <= 0.0):
        raise ValueError("raw_variance must be strictly positive")
    safe = np.maximum(raw, cfg.absolute_floor)
    log_raw = np.log(safe)
    center = np.mean(log_raw, axis=1, keepdims=True)
    shrunk = (1.0 - cfg.shrinkage) * log_raw + cfg.shrinkage * center
    half_range = 0.5 * np.log(cfg.condition_cap)
    clipped = np.clip(shrunk, center - half_range, center + half_range)
    regularized = np.exp(clipped)
    if not np.all(np.isfinite(regularized)) or np.any(regularized <= 0.0):
        raise RuntimeError("regularization failed to produce a positive diagonal")
    condition = np.max(regularized, axis=1) / np.min(regularized, axis=1)
    if np.any(condition > cfg.condition_cap * (1.0 + 1e-12)):
        raise RuntimeError("regularized diagonal violates condition cap")
    input_dtype = np.asarray(raw_variance).dtype
    output_dtype = np.float64 if input_dtype == np.float64 else np.float32
    return regularized.astype(output_dtype, copy=False)


# --- initialization ----------------------------------------------------------


def _content_seed(namespace: str, payload: Mapping[str, Any]) -> np.random.Generator:
    digest = hashlib.sha256(
        (namespace + "\0" + repr(sorted(payload.items()))).encode("utf-8")
    ).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "little")))


def overdispersed_initial_state(
    model: Any,
    batch_v: np.ndarray,
    *,
    num_chains: int,
    latent_dim: int,
    scale: float,
    epoch_identity: str,
    variance: np.ndarray,
) -> np.ndarray:
    """Chain 0 starts at the encoder basin; the other chains at seeded
    overdispersed points around it, ``scale * sqrt(variance)`` away per
    dimension, so dispersion is measured in posterior standard deviations."""

    encoder_state = np.asarray(
        model.infer_latent_from_covariates(batch_v, method="encoder"), np.float32
    )
    if encoder_state.shape != (batch_v.shape[0], latent_dim):
        raise FrozenHMCError("encoder latent state has unexpected shape")
    rng = _content_seed("bgm-mcmc-overdispersed-init", {"epoch": epoch_identity})
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
    "FrozenBatch",
    "FrozenEpoch",
    "FrozenHMCError",
    "FrozenHMCNumericalError",
    "FrozenVectorizedHMC",
    "GaussianContextEvaluator",
    "LatentPosteriorEvaluator",
    "MassRegularization",
    "ProductionConfig",
    "ProductionSegment",
    "TrajectoryPolicy",
    "WarmupConfig",
    "WarmupResult",
    "assert_axis_separable",
    "overdispersed_initial_state",
    "regularize_state_variance",
]
