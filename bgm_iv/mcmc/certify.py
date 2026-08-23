"""Certified structural MCMC: pilot, production, calibration, aggregation.

``certify_grid`` is the single entry point.  Per benchmark family it pins a
recipe and runs: an identity-mass pilot on every batch whose within-chain
variance, regularized, becomes the frozen diagonal mass; production with four
overdispersed chains under a fixed kernel, gated per batch by
:func:`diagnostics.score_batch`; streaming predictive calibration; and a
cross-batch aggregate weighted by the evaluation rows each batch owns, with
the paired point readouts scored on exactly the certified subset.  All stage
seeds derive from ``(family, data_seed, checkpoint_identity)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import platform
import socket
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from .diagnostics import (
    BatchIdentity,
    GateError,
    OutcomeTransform,
    PrecisionPolicy,
    score_batch,
)
from .readout import (
    CalibrationConfig,
    FunctionalReadout,
    PredictiveCalibrationAccumulator,
    StructuralQueryTable,
    batch_query_view,
    build_query_table,
)
from .sampler import (
    FrozenBatch,
    FrozenEpoch,
    FrozenHMCError,
    FrozenVectorizedHMC,
    LatentPosteriorEvaluator,
    MassRegularization,
    ProductionConfig,
    ProductionSegment,
    TrajectoryPolicy,
    WarmupConfig,
    overdispersed_initial_state,
    regularize_state_variance,
)
from .target import AffinePreprocessorSpec, resolve_target, sha256_array, sha256_json


class CertificationError(RuntimeError):
    """A certification contract violation."""


GATE_STATES = (
    "CONFIG_INVALID",
    "NUMERICAL_RESTART",
    "MASS_UPGRADE_REQUIRED",
    "STATIONARITY_RESTART",
    "EXTEND_PRECISION",
    "REPORTABLE",
)


@dataclass(frozen=True)
class MCMCConfig:
    """Sampler and precision settings of one run.

    ``num_leapfrog_steps`` is the warmup trajectory length on which the step
    size is adapted; production transitions draw their length from
    ``trajectory_support``.  The family recipes keep
    ``num_leapfrog_steps == max(trajectory_support)``.
    """

    num_chains: int = 4
    warmup_steps: int = 600
    segment_size: int = 600
    max_batch_size: int = 140
    initial_step_size: float = 0.1
    num_leapfrog_steps: int = 5
    target_accept_prob: float = 0.8
    trajectory_support: Sequence[int] = (3, 5, 7)
    max_energy_diff: float = 1000.0
    overdispersion_scale: float = 2.0
    absolute_halfwidth: float = 0.0
    relative_halfwidth: float = 0.05
    precision_reference_scale: float = 1.0
    time_chunk: int = 64
    query_chunk: int = 512

    def validate(self) -> "MCMCConfig":
        support = tuple(int(v) for v in self.trajectory_support)
        if not support or len(support) != len(set(support)) or min(support) < 1:
            raise CertificationError("trajectory_support must be non-empty, unique, positive")
        if int(self.num_leapfrog_steps) < 1 or int(self.warmup_steps) < 1:
            raise CertificationError("warmup settings must be positive")
        if int(self.segment_size) < 12 or int(self.num_chains) < 4:
            raise CertificationError("production needs >= 4 chains and >= 12 draws")
        return self

    @property
    def warmup_leapfrog_steps(self) -> int:
        return int(self.num_leapfrog_steps)

    @property
    def warmup_leapfrog_matches_max_trajectory(self) -> bool:
        return int(self.num_leapfrog_steps) == max(int(v) for v in self.trajectory_support)

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_chains": int(self.num_chains),
            "warmup_steps": int(self.warmup_steps),
            "segment_size": int(self.segment_size),
            "max_batch_size": int(self.max_batch_size),
            "initial_step_size": float(self.initial_step_size),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "warmup_leapfrog_steps": int(self.num_leapfrog_steps),
            "warmup_leapfrog_matches_max_trajectory": bool(
                self.warmup_leapfrog_matches_max_trajectory
            ),
            "target_accept_prob": float(self.target_accept_prob),
            "trajectory_support": [int(v) for v in self.trajectory_support],
            "max_energy_diff": float(self.max_energy_diff),
            "overdispersion_scale": float(self.overdispersion_scale),
            "absolute_halfwidth": float(self.absolute_halfwidth),
            "relative_halfwidth": float(self.relative_halfwidth),
            "precision_reference_scale": float(self.precision_reference_scale),
        }


@dataclass(frozen=True)
class BatchOutcome:
    """One batch's classified result.

    ``state`` is one of :data:`GATE_STATES`; ``record`` is present exactly
    when the state is ``REPORTABLE``.  A failed gate carries the reason and
    never a structural score.
    """

    batch_index: int
    global_ids: tuple[int, ...]
    state: str
    reason: str = ""
    record: Optional[Mapping[str, Any]] = None
    trace_summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in GATE_STATES:
            raise CertificationError(f"unknown gate state: {self.state}")
        if (self.record is not None) != (self.state == "REPORTABLE"):
            raise CertificationError(
                "record must be present exactly for REPORTABLE batches"
            )

    @property
    def num_targets(self) -> int:
        return len(self.global_ids)


def execution_environment() -> dict[str, Any]:
    """Device and build record attached to every certification artifact.

    Bit-level reproducibility holds only on the same GPU model, so the device
    is what makes two runs comparable or not.
    """

    gpus = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:  # pragma: no cover - device-dependent
            details = {}
        capability = details.get("compute_capability")
        gpus.append(
            {
                "name": details.get("device_name"),
                "compute_capability": (
                    None if capability is None else [int(v) for v in capability]
                ),
            }
        )
    try:
        build = dict(tf.sysconfig.get_build_info())
    except Exception:  # pragma: no cover - build-dependent
        build = {}
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpus": gpus,
    }


def _classify_gate_failure(error: Exception) -> str:
    """Map a scoring rejection to its gate state.

    ``score_batch`` raises ``GateError`` for malformed or non-stationary
    batches and ``ValueError("scientific score forbidden: <STATE>: ...")``
    when the gate blocks the record; both carry the state name.  A
    ``ValueError`` without a state is a defect and is re-raised.
    """

    message = str(error)
    for state in GATE_STATES:
        if state in message:
            return state
    if isinstance(error, GateError):
        return "CONFIG_INVALID"
    raise error


_SPOT_CHECK_DRAWS = 64


def _assert_scoreable_segment(
    runner: FrozenVectorizedHMC,
    epoch: FrozenEpoch,
    segment: ProductionSegment,
    evaluator_identity: str,
) -> None:
    """Trace invariants that make a production segment eligible for scoring.

    The target must be unchanged since the epoch was bound, the divergence
    and anomaly flags must be exactly the kernel's energy rule, every
    transition must use one trajectory length from the policy support, the
    post-state must be the last draw, and the frozen target must evaluate
    the retained draws deterministically (a fixed subsample, evaluated twice).
    """

    runner.evaluator.assert_runtime_identity()
    if runner.evaluator.evaluator_identity != evaluator_identity:
        raise CertificationError("target evaluator identity changed during the batch")
    energy = np.asarray(segment.energy_error)
    log_ratio = np.asarray(segment.log_accept_ratio)
    finite = np.isfinite(energy) & np.isfinite(log_ratio)
    if not np.array_equal(energy[finite], -log_ratio[finite]):
        raise CertificationError("energy_error must equal -log_accept_ratio")
    if not np.array_equal(np.isneginf(log_ratio), np.isposinf(energy)):
        raise CertificationError("infinite energy and acceptance entries disagree")
    nonfinite = np.asarray(segment.has_nonfinite, bool)
    if np.any(np.isneginf(log_ratio) & ~nonfinite):
        raise CertificationError("-Inf acceptance without a non-finite proposal flag")
    threshold = float(runner.production_config.max_energy_diff)
    expected_extreme = np.abs(energy) > threshold
    expected_divergence = nonfinite | (energy > threshold)
    expected_anomaly = nonfinite | expected_extreme
    if not np.array_equal(np.asarray(segment.has_extreme_energy_error, bool), expected_extreme):
        raise CertificationError("extreme-energy flags do not follow the kernel rule")
    if not np.array_equal(np.asarray(segment.has_divergence, bool), expected_divergence):
        raise CertificationError("divergence flags do not follow the kernel rule")
    if not np.array_equal(np.asarray(segment.has_numerical_anomaly, bool), expected_anomaly):
        raise CertificationError("numerical-anomaly flags do not follow the kernel rule")
    trajectory = np.asarray(segment.trajectory_length)
    support = np.asarray(runner.production_config.trajectory_policy.support)
    if not np.all(np.isin(trajectory, support)):
        raise CertificationError("trajectory length outside the policy support")
    if not np.array_equal(trajectory, np.broadcast_to(trajectory[:, :1, :1], trajectory.shape)):
        raise CertificationError("trajectory length must be common to all chains and targets")
    if not np.array_equal(np.asarray(segment.post_state), np.asarray(segment.draws)[-1]):
        raise CertificationError("post_state is not the last retained draw")
    draws = np.asarray(segment.draws)
    t_size, c_size, b_size, d_size = draws.shape
    index = np.unique(np.linspace(0, t_size - 1, min(_SPOT_CHECK_DRAWS, t_size)).astype(np.int64))
    state = tf.constant(draws[index].reshape(index.size * c_size, b_size, d_size))
    context = tf.constant(epoch.batch.target_context)
    first = np.asarray(runner.evaluator.evaluate(state, context).numpy(), np.float32)
    second = np.asarray(runner.evaluator.evaluate(state, context).numpy(), np.float32)
    if first.shape != (index.size * c_size, b_size) or not np.all(np.isfinite(first)):
        raise CertificationError("retained draws do not evaluate to finite target values")
    if not np.array_equal(first, second):
        raise CertificationError("target evaluator is not deterministic on retained draws")
    runner.evaluator.assert_runtime_identity()


def run_mcmc(
    model: Any,
    grid_x_model: Any,
    grid_v_raw: Any,
    *,
    preprocessor: AffinePreprocessorSpec,
    truth_original_units: Any,
    truth_label: str,
    outcome_shift: float,
    outcome_scale: float,
    treatment_transform: Mapping[str, float],
    run_seed: int,
    run_label: str,
    config: MCMCConfig = MCMCConfig(),
    state_variance: Optional[Any] = None,
    global_power: float = 1.0,
    batch_callback: Optional[Callable[[BatchOutcome, np.ndarray], Any]] = None,
) -> dict[str, Any]:
    """Sample and gate every batch of the evaluation grid.

    Parameters
    ----------
    grid_x_model, grid_v_raw
        Treatment column in model scale and covariate grid in the raw scale
        declared by ``preprocessor``.
    truth_original_units
        Benchmark truth per grid row, in original outcome units.
    state_variance
        Optional frozen diagonal mass ``[U, D]`` over the catalog (identity
        when omitted), fixed before warmup so production stays non-adaptive.
    batch_callback
        ``callback(outcome, draws)`` per batch that produced draws, so
        streaming consumers never hold the whole grid's draws in memory.
    """

    config = config.validate()
    if bool(model.params.get("use_bnn", False)):
        raise CertificationError("stochastic BNN checkpoints have no fixed target")
    resolved = resolve_target(
        model,
        preprocessor,
        global_power=float(global_power),
    )
    grid_v_model = preprocessor.transform(np.asarray(grid_v_raw, np.float32))
    table = build_query_table(grid_x_model, grid_v_model)
    truth = np.asarray(truth_original_units, np.float64).reshape(-1)
    if truth.shape[0] != table.num_queries:
        raise CertificationError("truth vector must align with the query table")

    evaluator = LatentPosteriorEvaluator(resolved)
    runner = FrozenVectorizedHMC(
        evaluator=evaluator,
        allow_unverified_evaluator=False,
        num_chains=config.num_chains,
        max_batch_size=min(config.max_batch_size, table.num_targets),
        warmup_config=WarmupConfig(
            warmup_steps=config.warmup_steps,
            initial_step_size=config.initial_step_size,
            num_leapfrog_steps=config.num_leapfrog_steps,
            target_accept_prob=config.target_accept_prob,
        ),
        production_config=ProductionConfig(
            segment_size=config.segment_size,
            trajectory_policy=TrajectoryPolicy(tuple(config.trajectory_support)),
            max_energy_diff=config.max_energy_diff,
        ),
    )
    readout = FunctionalReadout(
        model,
        table,
        time_chunk=config.time_chunk,
        query_chunk=config.query_chunk,
    )
    run_key = hashlib.sha256(
        ("bgm-mcmc-run\0" + str(run_label)).encode("utf-8")
    ).hexdigest()
    outcome_transform = OutcomeTransform(
        shift=float(outcome_shift), scale=float(outcome_scale)
    )
    precision = PrecisionPolicy(
        absolute_halfwidth=config.absolute_halfwidth,
        relative_halfwidth=config.relative_halfwidth,
        reference_scale=config.precision_reference_scale,
    )
    truth_hash = sha256_json(
        "benchmark-truth",
        {"label": str(truth_label), "values_hash": sha256_array(truth)},
    )

    latent_dim = int(sum(model.params["z_dims"]))
    if state_variance is None:
        catalog_variance = np.ones((table.num_targets, latent_dim), np.float32)
    else:
        catalog_variance = np.asarray(state_variance, np.float32)
        if catalog_variance.shape != (table.num_targets, latent_dim):
            raise CertificationError("state_variance must be [num_targets, latent_dim]")
        if not np.all(np.isfinite(catalog_variance)) or np.any(
            catalog_variance <= 0.0
        ):
            raise CertificationError("state_variance must be finite and positive")
    outcomes: list[BatchOutcome] = []
    callback_records: list[Any] = []
    max_b = min(config.max_batch_size, table.num_targets)
    for batch_index, start in enumerate(range(0, table.num_targets, max_b)):
        ids = tuple(range(start, min(start + max_b, table.num_targets)))
        batch_view = batch_query_view(table, ids)
        frozen = FrozenBatch(ids, table.unique_v[list(ids)], batch_index=batch_index)
        epoch = runner.bind_epoch(
            frozen,
            run_seed=int(run_seed),
            run_key=run_key,
            production_epoch=0,
        )
        evaluator_identity = runner.evaluator.evaluator_identity
        variance = np.ascontiguousarray(catalog_variance[list(ids)])
        initial = overdispersed_initial_state(
            model,
            frozen.target_context,
            num_chains=config.num_chains,
            latent_dim=latent_dim,
            scale=config.overdispersion_scale,
            epoch_identity=epoch.epoch_identity,
            variance=variance,
        )
        try:
            warm = runner.warmup(
                epoch, initial_state=initial, state_variance=variance
            )
            segment = runner.run_segment(
                epoch,
                segment_index=0,
                pre_state=warm.final_state,
                step_size=warm.step_size,
                state_variance=warm.state_variance,
            )
        except FrozenHMCError as error:
            outcomes.append(
                BatchOutcome(
                    batch_index=batch_index,
                    global_ids=ids,
                    state="NUMERICAL_RESTART",
                    reason=f"runner failure: {error}",
                )
            )
            continue

        # --- bind the draws to the target and kernel that produced them ---
        _assert_scoreable_segment(runner, epoch, segment, evaluator_identity)
        draws = np.asarray(segment.draws)
        draws_hash = sha256_array(draws, kind="posterior-draws")
        kernel_hash = sha256_json(
            "production-kernel",
            {
                "runner_kernel_identity": runner.kernel_identity,
                "epoch_identity": epoch.epoch_identity,
                "replay_key": segment.replay_key,
                "step_size_hash": sha256_array(segment.step_size),
                "state_variance_hash": sha256_array(segment.state_variance),
            },
        )
        identity = BatchIdentity(
            target_hash=resolved.spec.identity,
            decoder_hash=resolved.spec.decoder_model_hash,
            preprocessor_hash=resolved.spec.preprocessor_hash,
            evaluator_identity=evaluator_identity,
            kernel_hash=kernel_hash,
            outcome_hash=readout.outcome_hash,
            draws_hash=draws_hash,
            ordered_target_values=frozen.target_context,
            ordered_query_values=batch_view.ordered_query_values,
            query_inverse=batch_view.query_inverse,
        )
        batch_readout = FunctionalReadout(
            model,
            batch_view,
            time_chunk=config.time_chunk,
            query_chunk=config.query_chunk,
        )
        keep = np.isin(table.query_inverse, np.asarray(ids))
        batch_truth = truth[keep]
        trace_summary = {
            "accept_rate_mean": float(segment.is_accepted.mean()),
            "accept_rate_min_chain": float(
                segment.is_accepted.mean(axis=0).min()
            ),
            "divergences": int(segment.has_divergence.sum()),
            "numerical_anomalies": int(segment.has_numerical_anomaly.sum()),
            "max_abs_energy_error": float(
                np.max(np.abs(np.asarray(segment.energy_error, np.float64)))
            ),
            "draws_hash": draws_hash,
            "num_queries": int(batch_truth.shape[0]),
        }
        # --- readout and gate ---
        functional = np.asarray(batch_readout(draws))
        # Diagnostic only, never a score: raw plugin readout with per-chain
        # spread, recorded for every batch regardless of the verdict.
        try:
            chain_means = (
                float(outcome_transform.shift)
                + float(outcome_transform.scale) * functional.mean(axis=0)
            )  # [C, Q]
            pooled_diag = chain_means.mean(axis=0)
            trace_summary["diagnostic_unscored"] = {
                "plugin_mse_pooled": float(np.mean((batch_truth - pooled_diag) ** 2)),
                "plugin_mse_per_chain": [
                    float(np.mean((batch_truth - chain_means[c]) ** 2))
                    for c in range(chain_means.shape[0])
                ],
                "cross_chain_mean_spread_median": float(
                    np.median(chain_means.max(axis=0) - chain_means.min(axis=0))
                ),
            }
        except Exception as diag_error:  # must never mask the verdict
            trace_summary["diagnostic_unscored"] = {"error": repr(diag_error)}
        try:
            record = score_batch(
                latent_draws=draws,
                functional_draws_model_units=functional,
                truth_original_units=batch_truth,
                accepted=segment.is_accepted,
                log_accept_ratio=segment.log_accept_ratio,
                energy_error=segment.energy_error,
                has_nonfinite=segment.has_nonfinite,
                divergence=segment.has_divergence,
                numerical_anomaly=segment.has_numerical_anomaly,
                trajectory_length=segment.trajectory_length,
                identity=identity,
                outcome_transform=outcome_transform,
                precision_policy=precision,
            )
        except ValueError as error:
            outcome = BatchOutcome(
                batch_index=batch_index,
                global_ids=ids,
                state=_classify_gate_failure(error),
                reason=str(error),
                trace_summary=trace_summary,
            )
        else:
            outcome = BatchOutcome(
                batch_index=batch_index,
                global_ids=ids,
                state="REPORTABLE",
                record=record,
                trace_summary=trace_summary,
            )
        outcomes.append(outcome)
        if batch_callback is not None:
            callback_records.append(batch_callback(outcome, draws))

    reportable = [o for o in outcomes if o.state == "REPORTABLE"]
    return {
        "schema_version": "bgm-mcmc-run",
        "target": resolved.spec.manifest,
        "config": config.to_payload(),
        "run_seed": int(run_seed),
        "run_key": run_key,
        "catalog_hash": table.catalog_hash,
        "query_hash": table.query_hash,
        "truth_label": str(truth_label),
        "truth_hash": truth_hash,
        "treatment_transform": {
            "shift": float(treatment_transform["shift"]),
            "scale": float(treatment_transform["scale"]),
        },
        "num_targets": table.num_targets,
        "num_queries": table.num_queries,
        "batch_outcomes": outcomes,
        "num_reportable_batches": len(reportable),
        "all_reportable": len(reportable) == len(outcomes),
        "batch_callback_records": callback_records,
        "execution_environment": execution_environment(),
    }


# --- family recipes ----------------------------------------------------------


@dataclass(frozen=True)
class FamilyRecipe:
    """Certification recipe of one benchmark family."""

    name: str
    pilot_batch: int
    calibration_unit: str               # "target" | "query"
    production: MCMCConfig
    pilot_warmup_steps: int = 400
    pilot_initial_step_size: float = 0.02
    pilot_leapfrog_steps: int = 5
    pilot_segment_size: int = 240
    pilot_trajectory_support: Sequence[int] = (3, 5, 7)
    pilot_jitter_scale: float = 0.3
    mass_regularization: MassRegularization = MassRegularization()
    calibration_num_draws: int = 2000

    def validate(self) -> "FamilyRecipe":
        if self.calibration_unit not in {"target", "query"}:
            raise CertificationError("calibration_unit must be 'target' or 'query'")
        if int(self.pilot_batch) < 1:
            raise CertificationError("pilot_batch must be positive")
        self.production.validate()
        self.mass_regularization.validate()
        return self

    def production_config(self, *, num_targets: int, truth_variance: float) -> MCMCConfig:
        return replace(
            self.production,
            max_batch_size=min(int(self.pilot_batch), int(num_targets)),
            precision_reference_scale=float(truth_variance),
        ).validate()

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pilot_batch": int(self.pilot_batch),
            "calibration_unit": self.calibration_unit,
            "pilot": {
                "warmup_steps": int(self.pilot_warmup_steps),
                "initial_step_size": float(self.pilot_initial_step_size),
                "num_leapfrog_steps": int(self.pilot_leapfrog_steps),
                "segment_size": int(self.pilot_segment_size),
                "trajectory_support": [int(v) for v in self.pilot_trajectory_support],
                "jitter_scale": float(self.pilot_jitter_scale),
                "target_accept_prob": 0.9,
                "mass_estimator": "per_chain_within_variance_mean_over_chains_ddof1",
                "mass_regularization": {
                    "shrinkage": float(self.mass_regularization.shrinkage),
                    "condition_cap": float(self.mass_regularization.condition_cap),
                    "absolute_floor": float(self.mass_regularization.absolute_floor),
                },
            },
            "production": self.production.to_payload(),
            "calibration_num_draws": int(self.calibration_num_draws),
        }

    @property
    def recipe_hash(self) -> str:
        return sha256_json("certification-recipe", self.to_payload())


_VECTOR_PRODUCTION = MCMCConfig(
    num_chains=4,
    warmup_steps=1600,
    segment_size=24000,
    max_batch_size=128,
    initial_step_size=0.02,
    num_leapfrog_steps=31,
    target_accept_prob=0.90,
    trajectory_support=(7, 15, 31),
    max_energy_diff=1000.0,
    overdispersion_scale=2.0,
    absolute_halfwidth=0.0,
    relative_halfwidth=0.05,
)

FAMILY_RECIPES: dict[str, FamilyRecipe] = {
    # Demand (2-d covariates): all 140 (time, group) targets in one batch;
    # calibration per query row because each target serves 20 price rows.
    "demand": FamilyRecipe(
        name="demand",
        pilot_batch=140,
        calibration_unit="query",
        pilot_initial_step_size=0.05,
        production=replace(
            _VECTOR_PRODUCTION,
            warmup_steps=800,
            segment_size=12000,
            max_batch_size=140,
            initial_step_size=0.05,
            num_leapfrog_steps=5,
            trajectory_support=(3, 5, 7),
        ),
    ),
    # Vector proxy (785-d): full grid, 128 targets per batch, warmup length
    # 31 with production trajectories (7, 15, 31).
    "vector": FamilyRecipe(
        name="vector",
        pilot_batch=128,
        calibration_unit="target",
        production=_VECTOR_PRODUCTION,
    ),
    # MNIST feature model: vector family on (time, phi), 64 targets per
    # batch, trajectories capped at 15 with the warmup length matched.
    "mnist_feature": FamilyRecipe(
        name="mnist_feature",
        pilot_batch=64,
        calibration_unit="target",
        production=replace(
            _VECTOR_PRODUCTION,
            max_batch_size=64,
            num_leapfrog_steps=15,
            trajectory_support=(7, 15),
        ),
    ),
}


def derive_certification_seeds(family: str, data_seed: int, checkpoint_identity: str) -> dict[str, int]:
    """Content-derived integer seeds for every certification stage."""

    def _seed(stage: str) -> int:
        digest = hashlib.sha256(
            f"bgm-certification-seed\0{family}\0{int(data_seed)}\0{checkpoint_identity}\0{stage}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "little") % (2**31 - 1)

    return {
        "pilot_run_seed": _seed("pilot"),
        "production_run_seed": _seed("production"),
    }


def estimate_pilot_state_variance(
    model: Any,
    resolved: Any,
    table: StructuralQueryTable,
    recipe: FamilyRecipe,
    *,
    pilot_run_seed: int,
    run_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identity-mass pilot over the catalog -> regularized diagonal mass."""

    evaluator = LatentPosteriorEvaluator(resolved)
    latent_dim = int(sum(int(v) for v in model.params["z_dims"]))
    batch_size = min(int(recipe.pilot_batch), table.num_targets)
    runner = FrozenVectorizedHMC(
        evaluator=evaluator,
        allow_unverified_evaluator=False,
        num_chains=4,
        max_batch_size=batch_size,
        warmup_config=WarmupConfig(
            warmup_steps=int(recipe.pilot_warmup_steps),
            initial_step_size=float(recipe.pilot_initial_step_size),
            num_leapfrog_steps=int(recipe.pilot_leapfrog_steps),
            target_accept_prob=0.9,
        ),
        production_config=ProductionConfig(
            segment_size=int(recipe.pilot_segment_size),
            trajectory_policy=TrajectoryPolicy(tuple(int(v) for v in recipe.pilot_trajectory_support)),
            max_energy_diff=1000.0,
        ),
    )
    pooled = np.empty((table.num_targets, latent_dim), np.float64)
    divergences = 0
    failures = []
    started = time.time()
    for batch_index, start in enumerate(range(0, table.num_targets, batch_size)):
        ids = tuple(range(start, min(start + batch_size, table.num_targets)))
        frozen = FrozenBatch(ids, table.unique_v[list(ids)], batch_index=batch_index)
        epoch = runner.bind_epoch(
            frozen,
            run_seed=int(pilot_run_seed),
            run_key=hashlib.sha256(f"bgm-certification-pilot\0{run_label}\0{batch_index}".encode()).hexdigest(),
            production_epoch=0,
        )
        variance = np.ones((frozen.active_size, latent_dim), np.float32)
        initial = overdispersed_initial_state(
            model,
            frozen.target_context,
            num_chains=4,
            latent_dim=latent_dim,
            scale=float(recipe.pilot_jitter_scale),
            epoch_identity=epoch.epoch_identity,
            variance=variance,
        )
        try:
            warm = runner.warmup(epoch, initial_state=initial, state_variance=variance)
            segment = runner.run_segment(
                epoch,
                segment_index=0,
                pre_state=warm.final_state,
                step_size=warm.step_size,
                state_variance=warm.state_variance,
            )
        except FrozenHMCError as error:
            failures.append({"batch_index": batch_index, "error": str(error)})
            pooled[list(ids)] = 1.0
            continue
        pooled[list(ids)] = np.var(segment.draws, axis=0, ddof=1).mean(axis=0)
        divergences += int(segment.has_divergence.sum())
    if not np.all(np.isfinite(pooled)) or np.any(pooled <= 0.0):
        raise CertificationError("pilot variance estimate is not positive/finite")
    regularized = regularize_state_variance(pooled, recipe.mass_regularization).astype(np.float32)
    info = {
        "divergences": int(divergences),
        "runner_failures": failures,
        "batch_size": int(batch_size),
        "num_batches": int(np.ceil(table.num_targets / batch_size)),
        "raw_variance_median": float(np.median(pooled)),
        "regularized_variance_median": float(np.median(regularized)),
        "regularized_variance_hash": sha256_array(regularized),
        "seconds": float(time.time() - started),
    }
    return regularized, info


def _batch_record(outcome: BatchOutcome, num_queries: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "batch_index": int(outcome.batch_index),
        "num_targets": int(outcome.num_targets),
        "num_queries": int(num_queries),
        "state": str(outcome.state),
        "reason": str(outcome.reason)[:400],
        "trace_summary": dict(outcome.trace_summary),
    }
    if outcome.record is not None:
        record["metric"] = dict(outcome.record["metric"])
        record["sampler"] = dict(outcome.record["sampler"])
        record["block_len"] = int(outcome.record["block_len"])
        record["max_iact"] = float(outcome.record["max_iact"])
    return record


def aggregate_batch_outcomes(
    outcomes: Sequence[BatchOutcome],
    table: StructuralQueryTable,
    truth_rows: Any,
    paired_predictions: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Cross-batch aggregate; weights are the evaluation rows each batch owns.

    Certified quantities pool REPORTABLE batches only.  The paired point
    readouts (MAP / encoder) are scored on exactly the certified rows AND on
    all rows, so the MCMC-vs-MAP comparison is a same-subset read-off.
    """

    truth = np.asarray(table.row_length_array(truth_rows, "truth"), np.float64).reshape(-1)
    rows_of = {
        int(o.batch_index): table.query_rows_of_targets(o.global_ids) for o in outcomes
    }
    states = [str(o.state) for o in outcomes]
    reportable = [o for o in outcomes if o.state == "REPORTABLE"]
    weights = np.asarray([rows_of[int(o.batch_index)].size for o in reportable], np.float64)
    total_queries = int(table.num_queries)
    certified_rows = (
        np.concatenate([rows_of[int(o.batch_index)] for o in reportable])
        if reportable
        else np.zeros(0, np.int64)
    )
    metrics = [o.record["metric"] for o in reportable]
    samplers = [o.record["sampler"] for o in reportable]

    def _weighted(key: str) -> Optional[float]:
        if not metrics:
            return None
        return float(np.sum(weights * np.asarray([m[key] for m in metrics])) / np.sum(weights))

    pooled_mcse = None
    if metrics:
        pooled_mcse = float(
            np.sqrt(np.sum((weights * np.asarray([m["plugin_mcse"] for m in metrics])) ** 2))
            / np.sum(weights)
        )
    aggregate: dict[str, Any] = {
        "num_batches": len(outcomes),
        "num_reportable": len(reportable),
        "state_counts": {state: states.count(state) for state in sorted(set(states))},
        "certified_query_fraction": float(certified_rows.size / total_queries) if total_queries else 0.0,
        "certified_target_fraction": float(
            sum(o.num_targets for o in reportable) / max(1, table.num_targets)
        ),
        "certified_num_queries": int(certified_rows.size),
        "certified_plugin_mse": _weighted("plugin_mse"),
        "certified_u_corrected_mse": _weighted("u_corrected_mse"),
        "certified_integration_penalty": _weighted("integration_penalty"),
        "certified_pooled_mcse": pooled_mcse,
        "certified_pooled_halfwidth95": None if pooled_mcse is None else float(1.96 * pooled_mcse),
        "rank_rhat_max": max((float(s["rank_rhat_max"]) for s in samplers), default=None),
        "folded_rhat_max": max((float(s["folded_rhat_max"]) for s in samplers), default=None),
        "bulk_ess_min": min((float(s["bulk_ess_min"]) for s in samplers), default=None),
        "tail_ess_min": min((float(s["tail_ess_min"]) for s in samplers), default=None),
        "max_iact": max((float(m["max_iact"]) for m in metrics), default=None),
        "divergences_total": int(
            sum(int(o.trace_summary.get("divergences", 0)) for o in outcomes)
        ),
        "numerical_anomalies_total": int(
            sum(int(o.trace_summary.get("numerical_anomalies", 0)) for o in outcomes)
        ),
        "unscored_plugin_mse_all_batches": None,
        "paired": {},
    }
    diag = [
        (rows_of[int(o.batch_index)].size, o.trace_summary.get("diagnostic_unscored", {}).get("plugin_mse_pooled"))
        for o in outcomes
    ]
    diag = [(w, v) for w, v in diag if v is not None]
    if diag:
        aggregate["unscored_plugin_mse_all_batches"] = float(
            sum(w * v for w, v in diag) / sum(w for w, _ in diag)
        )
    for name, prediction in (paired_predictions or {}).items():
        prediction = np.asarray(table.row_length_array(prediction, name), np.float64).reshape(-1)
        aggregate["paired"][name] = {
            "all_rows_mse": float(np.mean((truth - prediction) ** 2)),
            "certified_subset_mse": (
                float(np.mean((truth[certified_rows] - prediction[certified_rows]) ** 2))
                if certified_rows.size
                else None
            ),
        }
    return aggregate


def certify_grid(
    model: Any,
    *,
    family: str,
    grid_x_model: Any,
    grid_v_raw: Any,
    preprocessor: Any,
    truth_original_units: Any,
    truth_label: str,
    outcome_shift: float,
    outcome_scale: float,
    treatment_transform: Mapping[str, float],
    data_seed: int,
    checkpoint_identity: str,
    run_label: str,
    paired_predictions: Optional[Mapping[str, Any]] = None,
    recipe: Optional[FamilyRecipe] = None,
    calibration: bool = True,
    truth_noise_sd: float = 1.0,
    progress: Optional[Callable[[str], None]] = print,
) -> dict[str, Any]:
    """Pilot -> production -> calibration -> aggregate for one evaluation grid.

    ``grid_x_model`` / ``grid_v_raw`` / ``truth_original_units`` are the rows
    of the evaluation grid; ``preprocessor`` maps ``grid_v_raw`` to model covariates and is hashed
    into the target identity.
    """

    started = time.time()
    recipe = (FAMILY_RECIPES[family] if recipe is None else recipe).validate()
    if family not in FAMILY_RECIPES and recipe is None:
        raise CertificationError(f"unknown certification family {family!r}")
    if bool(model.params.get("use_bnn", False)):
        raise CertificationError("stochastic BNN checkpoints have no fixed target")
    grid_x_model = np.asarray(grid_x_model, np.float32).reshape(-1, 1)
    grid_v_raw = np.asarray(grid_v_raw, np.float32)
    truth = np.asarray(truth_original_units, np.float64).reshape(-1)
    if truth.shape[0] != grid_x_model.shape[0] or grid_v_raw.shape[0] != grid_x_model.shape[0]:
        raise CertificationError("grid_x_model, grid_v_raw and truth must have equal row counts")
    seeds = derive_certification_seeds(family, data_seed, checkpoint_identity)
    resolved = resolve_target(model, preprocessor, global_power=1.0)
    grid_v_model = preprocessor.transform(grid_v_raw)
    table = build_query_table(grid_x_model, grid_v_model)
    table.row_length_array(truth, "truth")
    config = recipe.production_config(num_targets=table.num_targets, truth_variance=float(np.var(truth)))
    if progress:
        progress(
            f"[certify {family}] catalog {table.num_targets} targets / {table.num_queries} rows, "
            f"batches of {config.max_batch_size}, T={config.segment_size}"
        )

    variance, pilot_info = estimate_pilot_state_variance(
        model, resolved, table, recipe, pilot_run_seed=seeds["pilot_run_seed"], run_label=run_label
    )
    if progress:
        progress(
            f"[certify {family}] pilot done: divergences={pilot_info['divergences']} "
            f"median var {pilot_info['regularized_variance_median']:.4g} ({pilot_info['seconds']:.0f}s)"
        )

    accumulator = None
    if calibration:
        accumulator = PredictiveCalibrationAccumulator(
            model,
            table,
            truth,
            outcome_shift=float(outcome_shift),
            outcome_scale=float(outcome_scale),
            config=CalibrationConfig(
                num_draws=int(recipe.calibration_num_draws),
                truth_noise_sd=float(truth_noise_sd),
                scoring_unit=recipe.calibration_unit,
            ),
        )

    def _callback(outcome: BatchOutcome, draws: np.ndarray):
        record = None
        if accumulator is not None:
            record = accumulator.add_batch(outcome, draws)
        if progress:
            progress(
                f"[certify {family}] batch {outcome.batch_index} -> {outcome.state} "
                f"(div {outcome.trace_summary.get('divergences')}, "
                f"acc {outcome.trace_summary.get('accept_rate_mean', float('nan')):.3f})"
            )
        return record

    production_started = time.time()
    pipeline = run_mcmc(
        model,
        grid_x_model,
        grid_v_raw,
        preprocessor=preprocessor,
        truth_original_units=truth,
        truth_label=str(truth_label),
        outcome_shift=float(outcome_shift),
        outcome_scale=float(outcome_scale),
        treatment_transform=dict(treatment_transform),
        run_seed=int(seeds["production_run_seed"]),
        run_label=str(run_label),
        config=config,
        state_variance=variance,
        batch_callback=_callback,
    )
    production_seconds = time.time() - production_started
    outcomes = list(pipeline["batch_outcomes"])
    rows_of = {int(o.batch_index): table.query_rows_of_targets(o.global_ids) for o in outcomes}
    aggregate = aggregate_batch_outcomes(outcomes, table, truth, paired_predictions)
    aggregate["pilot_divergences"] = int(pilot_info["divergences"])
    result = {
        "schema_version": "bgm-certification",
        "family": str(family),
        "recipe": recipe.to_payload(),
        "recipe_hash": recipe.recipe_hash,
        "checkpoint_identity": str(checkpoint_identity),
        "data_seed": int(data_seed),
        "seeds": seeds,
        "run_label": str(run_label),
        "grid": {
            "num_rows": int(table.num_queries),
            "num_targets": int(table.num_targets),
            "truth_label": str(truth_label),
            "truth_hash": pipeline["truth_hash"],
            "catalog_hash": pipeline["catalog_hash"],
            "query_hash": pipeline["query_hash"],
            "representative_rows_hash": sha256_array(table.representative_rows),
            "preprocessor_identity": str(preprocessor.identity),
        },
        "pipeline": {
            "schema_version": pipeline["schema_version"],
            "config": pipeline["config"],
            "config_hash": sha256_json("mcmc-config", pipeline["config"]),
            "run_seed": pipeline["run_seed"],
            "run_key": pipeline["run_key"],
            "target": pipeline["target"],
            "target_hash": pipeline["target"]["target_hash"],
            "num_reportable_batches": pipeline["num_reportable_batches"],
            "all_reportable": pipeline["all_reportable"],
        },
        "pilot": pilot_info,
        "batches": [_batch_record(o, rows_of[int(o.batch_index)].size) for o in outcomes],
        "aggregate": aggregate,
        "calibration": accumulator.summary() if accumulator is not None else None,
        "execution_environment": pipeline["execution_environment"],
        "timings": {
            "pilot_seconds": float(pilot_info["seconds"]),
            "production_seconds": float(production_seconds),
            "total_seconds": float(time.time() - started),
        },
    }
    if progress:
        agg = result["aggregate"]
        progress(
            f"[certify {family}] {agg['num_reportable']}/{agg['num_batches']} REPORTABLE; "
            f"certified plugin MSE {agg['certified_plugin_mse']} "
            f"(query fraction {agg['certified_query_fraction']:.3f}), "
            f"{result['timings']['total_seconds']:.0f}s"
        )
    return result


__all__ = [
    "FAMILY_RECIPES",
    "GATE_STATES",
    "BatchOutcome",
    "CertificationError",
    "FamilyRecipe",
    "MCMCConfig",
    "aggregate_batch_outcomes",
    "certify_grid",
    "derive_certification_seeds",
    "estimate_pilot_state_variance",
    "execution_environment",
    "run_mcmc",
]
