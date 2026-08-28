"""Full-grid ungated structural MCMC inference.

Each family runs one all-target pilot, one all-target production chain set and
one all-draw query-level readout.  Statistical reportability gates are
deliberately absent; malformed targets or non-finite retained states still
fail the complete run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import platform
import socket
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from .readout import FullGridReadout, ReadoutConfig, build_query_table
from .sampler import (
    FrozenBatch,
    FrozenVectorizedHMC,
    LatentPosteriorEvaluator,
    MassRegularization,
    ProductionConfig,
    TrajectoryPolicy,
    WarmupConfig,
    overdispersed_initial_state,
    regularize_state_variance,
)
from .target import AffinePreprocessorSpec, resolve_target, sha256_array, sha256_json


class MCMCInferenceError(RuntimeError):
    """A full-grid MCMC inference contract violation."""


@dataclass(frozen=True)
class MCMCConfig:
    num_chains: int = 4
    warmup_steps: int = 600
    segment_size: int = 600
    initial_step_size: float = 0.1
    num_leapfrog_steps: int = 5
    target_accept_prob: float = 0.8
    trajectory_support: Sequence[int] = (3, 5, 7)
    overdispersion_scale: float = 2.0

    def validate(self) -> "MCMCConfig":
        support = tuple(int(value) for value in self.trajectory_support)
        if int(self.num_chains) != 4:
            raise MCMCInferenceError("production recipes require exactly four chains")
        if int(self.warmup_steps) < 1 or int(self.segment_size) < 1:
            raise MCMCInferenceError("warmup and retained draw counts must be positive")
        if not support or len(support) != len(set(support)) or min(support) < 1:
            raise MCMCInferenceError("trajectory_support must be unique and positive")
        if int(self.num_leapfrog_steps) < 1:
            raise MCMCInferenceError("num_leapfrog_steps must be positive")
        if not np.isfinite(self.initial_step_size) or self.initial_step_size <= 0:
            raise MCMCInferenceError("initial_step_size must be positive")
        if not 0.0 < float(self.target_accept_prob) < 1.0:
            raise MCMCInferenceError("target_accept_prob must lie in (0,1)")
        if not np.isfinite(self.overdispersion_scale) or self.overdispersion_scale <= 0:
            raise MCMCInferenceError("overdispersion_scale must be positive")
        return self

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_chains": int(self.num_chains),
            "warmup_steps": int(self.warmup_steps),
            "segment_size": int(self.segment_size),
            "initial_step_size": float(self.initial_step_size),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "target_accept_prob": float(self.target_accept_prob),
            "trajectory_support": [int(v) for v in self.trajectory_support],
            "overdispersion_scale": float(self.overdispersion_scale),
        }


@dataclass(frozen=True)
class FamilyRecipe:
    name: str
    target_kind: str
    production: MCMCConfig
    pilot_warmup_steps: int = 400
    pilot_initial_step_size: float = 0.02
    pilot_leapfrog_steps: int = 5
    pilot_segment_size: int = 240
    pilot_trajectory_support: Sequence[int] = (3, 5, 7)
    pilot_jitter_scale: float = 0.3
    mass_regularization: MassRegularization = MassRegularization()
    readout: ReadoutConfig = ReadoutConfig()

    def validate(self) -> "FamilyRecipe":
        if self.target_kind not in {"model_posterior", "generalized_gibbs"}:
            raise MCMCInferenceError("unknown target_kind")
        if int(self.pilot_warmup_steps) < 1 or int(self.pilot_segment_size) < 2:
            raise MCMCInferenceError("pilot lengths must be positive")
        self.production.validate()
        self.mass_regularization.validate()
        self.readout.validate()
        return self

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_kind": self.target_kind,
            "pilot": {
                "warmup_steps": int(self.pilot_warmup_steps),
                "initial_step_size": float(self.pilot_initial_step_size),
                "num_leapfrog_steps": int(self.pilot_leapfrog_steps),
                "segment_size": int(self.pilot_segment_size),
                "trajectory_support": [int(v) for v in self.pilot_trajectory_support],
                "jitter_scale": float(self.pilot_jitter_scale),
                "mass_estimator": "per_chain_within_variance_mean_over_chains_ddof1",
            },
            "production": self.production.to_payload(),
            "readout": self.readout.to_payload(),
        }

    @property
    def recipe_hash(self) -> str:
        return sha256_json("mcmc-inference-recipe", self.to_payload())


_VECTOR_PRODUCTION = MCMCConfig(
    num_chains=4,
    warmup_steps=1600,
    segment_size=24000,
    initial_step_size=0.02,
    num_leapfrog_steps=31,
    target_accept_prob=0.90,
    trajectory_support=(7, 15, 31),
    overdispersion_scale=2.0,
)

FAMILY_RECIPES: dict[str, FamilyRecipe] = {
    "demand": FamilyRecipe(
        name="demand",
        target_kind="model_posterior",
        pilot_initial_step_size=0.05,
        production=replace(
            _VECTOR_PRODUCTION,
            warmup_steps=800,
            segment_size=12000,
            initial_step_size=0.05,
            num_leapfrog_steps=5,
            trajectory_support=(3, 5, 7),
        ),
    ),
    "vector": FamilyRecipe(
        name="vector",
        target_kind="model_posterior",
        production=_VECTOR_PRODUCTION,
    ),
    "mnist_feature": FamilyRecipe(
        name="mnist_feature",
        target_kind="model_posterior",
        production=replace(
            _VECTOR_PRODUCTION,
            num_leapfrog_steps=15,
            trajectory_support=(7, 15),
        ),
    ),
    "mnist_pixel": FamilyRecipe(
        name="mnist_pixel",
        target_kind="generalized_gibbs",
        production=_VECTOR_PRODUCTION,
    ),
}


def execution_environment() -> dict[str, Any]:
    gpus = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:  # pragma: no cover - device dependent
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
    except Exception:  # pragma: no cover - build dependent
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


def derive_mcmc_seeds(
    family: str, data_seed: int, checkpoint_identity: str
) -> dict[str, int]:
    def derive(stage: str) -> int:
        digest = hashlib.sha256(
            f"bgm-mcmc-seed\0{family}\0{int(data_seed)}\0"
            f"{checkpoint_identity}\0{stage}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "little") % (2**31 - 1)

    return {"pilot": derive("pilot"), "production": derive("production")}


def _make_runner(
    resolved: Any,
    *,
    num_targets: int,
    num_chains: int,
    warmup_steps: int,
    initial_step_size: float,
    num_leapfrog_steps: int,
    target_accept_prob: float,
    segment_size: int,
    trajectory_support: Sequence[int],
) -> FrozenVectorizedHMC:
    return FrozenVectorizedHMC(
        evaluator=LatentPosteriorEvaluator(resolved),
        allow_unverified_evaluator=False,
        num_chains=int(num_chains),
        max_batch_size=int(num_targets),
        warmup_config=WarmupConfig(
            warmup_steps=int(warmup_steps),
            initial_step_size=float(initial_step_size),
            num_leapfrog_steps=int(num_leapfrog_steps),
            target_accept_prob=float(target_accept_prob),
        ),
        production_config=ProductionConfig(
            segment_size=int(segment_size),
            trajectory_policy=TrajectoryPolicy(
                tuple(int(value) for value in trajectory_support)
            ),
        ),
    )


def estimate_pilot_state_variance(
    model: Any,
    resolved: Any,
    table: Any,
    recipe: FamilyRecipe,
    *,
    run_seed: int,
    run_label: str,
    progress: Optional[Callable[[str], None]] = print,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One all-target pilot followed by diagonal-mass regularization."""

    started = time.time()
    latent_dim = int(sum(int(value) for value in model.params["z_dims"]))
    runner = _make_runner(
        resolved,
        num_targets=table.num_targets,
        num_chains=4,
        warmup_steps=recipe.pilot_warmup_steps,
        initial_step_size=recipe.pilot_initial_step_size,
        num_leapfrog_steps=recipe.pilot_leapfrog_steps,
        target_accept_prob=0.9,
        segment_size=recipe.pilot_segment_size,
        trajectory_support=recipe.pilot_trajectory_support,
    )
    ids = tuple(range(table.num_targets))
    frozen = FrozenBatch(ids, table.unique_v, batch_index=0)
    epoch = runner.bind_epoch(
        frozen,
        run_seed=int(run_seed),
        run_key=hashlib.sha256(
            f"bgm-mcmc-pilot\0{run_label}".encode()
        ).hexdigest(),
        production_epoch=0,
    )
    identity_variance = np.ones((table.num_targets, latent_dim), np.float32)
    initial = overdispersed_initial_state(
        model,
        frozen.target_context,
        num_chains=4,
        latent_dim=latent_dim,
        scale=float(recipe.pilot_jitter_scale),
        epoch_identity=epoch.epoch_identity,
        variance=identity_variance,
    )
    warm = runner.warmup(
        epoch, initial_state=initial, state_variance=identity_variance
    )
    segment = runner.run_segment(
        epoch,
        segment_index=0,
        pre_state=warm.final_state,
        step_size=warm.step_size,
        state_variance=warm.state_variance,
    )
    if progress:
        per_chain = np.mean(segment.acceptance_rate, axis=1)
        progress(
            "[mcmc pilot] acceptance mean="
            f"{float(np.mean(segment.acceptance_rate)):.3f}, "
            f"min={float(np.min(segment.acceptance_rate)):.3f}, "
            f"per-chain={np.round(per_chain, 3).tolist()}"
        )
    raw = np.var(segment.draws, axis=0, ddof=1).mean(axis=0)
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise MCMCInferenceError("pilot variance estimate is not positive/finite")
    regularized = regularize_state_variance(
        raw, recipe.mass_regularization
    ).astype(np.float32)
    return regularized, {
        "seconds": float(time.time() - started),
        "raw_variance_median": float(np.median(raw)),
        "regularized_variance_median": float(np.median(regularized)),
        "regularized_variance_hash": sha256_array(regularized),
    }


def run_mcmc(
    model: Any,
    table: Any,
    *,
    preprocessor: AffinePreprocessorSpec,
    run_seed: int,
    run_label: str,
    config: MCMCConfig,
    state_variance: np.ndarray,
    progress: Optional[Callable[[str], None]] = print,
) -> dict[str, Any]:
    """Run one all-target warmup and production segment."""

    config = config.validate()
    if bool(model.params.get("use_bnn", False)):
        raise MCMCInferenceError("stochastic BNN checkpoints have no fixed target")
    resolved = resolve_target(model, preprocessor, global_power=1.0)
    if resolved.spec.family == "mnist":
        expected_kind = "generalized_gibbs"
    else:
        expected_kind = "model_posterior"
    if resolved.spec.target_kind != expected_kind:
        raise MCMCInferenceError("resolved target kind does not match the model family")
    runner = _make_runner(
        resolved,
        num_targets=table.num_targets,
        num_chains=config.num_chains,
        warmup_steps=config.warmup_steps,
        initial_step_size=config.initial_step_size,
        num_leapfrog_steps=config.num_leapfrog_steps,
        target_accept_prob=config.target_accept_prob,
        segment_size=config.segment_size,
        trajectory_support=config.trajectory_support,
    )
    ids = tuple(range(table.num_targets))
    frozen = FrozenBatch(ids, table.unique_v, batch_index=0)
    run_key = hashlib.sha256(
        ("bgm-mcmc-run\0" + str(run_label)).encode()
    ).hexdigest()
    epoch = runner.bind_epoch(
        frozen,
        run_seed=int(run_seed),
        run_key=run_key,
        production_epoch=0,
    )
    latent_dim = int(sum(int(value) for value in model.params["z_dims"]))
    variance = np.asarray(state_variance, np.float32)
    if variance.shape != (table.num_targets, latent_dim):
        raise MCMCInferenceError("state_variance must cover every target")
    initial = overdispersed_initial_state(
        model,
        frozen.target_context,
        num_chains=config.num_chains,
        latent_dim=latent_dim,
        scale=config.overdispersion_scale,
        epoch_identity=epoch.epoch_identity,
        variance=variance,
    )
    started = time.time()
    warm = runner.warmup(epoch, initial_state=initial, state_variance=variance)
    segment = runner.run_segment(
        epoch,
        segment_index=0,
        pre_state=warm.final_state,
        step_size=warm.step_size,
        state_variance=warm.state_variance,
    )
    seconds = time.time() - started
    if progress:
        per_chain = np.mean(segment.acceptance_rate, axis=1)
        progress(
            "[mcmc production] acceptance mean="
            f"{float(np.mean(segment.acceptance_rate)):.3f}, "
            f"min={float(np.min(segment.acceptance_rate)):.3f}, "
            f"per-chain={np.round(per_chain, 3).tolist()}"
        )
    return {
        "draws": segment.draws,
        "seconds": float(seconds),
        "run_seed": int(run_seed),
        "run_key": run_key,
        "target": resolved.spec.manifest,
        "config": config.to_payload(),
    }


def run_mcmc_grid(
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
    recipe: Optional[FamilyRecipe] = None,
    truth_noise_sd: float = 1.0,
    progress: Optional[Callable[[str], None]] = print,
) -> dict[str, Any]:
    """Pilot, production and all-draw readout on the complete query grid."""

    started = time.time()
    if recipe is None:
        try:
            recipe = FAMILY_RECIPES[str(family)]
        except KeyError:
            raise MCMCInferenceError(f"unknown MCMC family {family!r}") from None
    recipe = recipe.validate()
    grid_x = np.asarray(grid_x_model, np.float32).reshape(-1, 1)
    grid_v_raw = np.asarray(grid_v_raw, np.float32)
    truth = np.asarray(truth_original_units, np.float64).reshape(-1)
    if grid_v_raw.ndim != 2 or not (
        grid_x.shape[0] == grid_v_raw.shape[0] == truth.shape[0]
    ):
        raise MCMCInferenceError("grid_x, grid_v and truth must have equal rows")
    grid_v_model = preprocessor.transform(grid_v_raw)
    table = build_query_table(grid_x, grid_v_model)
    seeds = derive_mcmc_seeds(family, data_seed, checkpoint_identity)
    resolved = resolve_target(model, preprocessor, global_power=1.0)
    if resolved.spec.target_kind != recipe.target_kind:
        raise MCMCInferenceError("family recipe target_kind mismatch")
    if progress:
        progress(
            f"[mcmc {family}] {table.num_targets} targets / "
            f"{table.num_queries} full-grid queries"
        )
    variance, pilot = estimate_pilot_state_variance(
        model,
        resolved,
        table,
        recipe,
        run_seed=seeds["pilot"],
        run_label=run_label,
        progress=progress,
    )
    production = run_mcmc(
        model,
        table,
        preprocessor=preprocessor,
        run_seed=seeds["production"],
        run_label=run_label,
        config=recipe.production,
        state_variance=variance,
        progress=progress,
    )
    readout_started = time.time()
    readout_config = replace(
        recipe.readout, truth_noise_sd=float(truth_noise_sd)
    )
    readout = FullGridReadout(
        model,
        table,
        truth,
        outcome_shift=float(outcome_shift),
        outcome_scale=float(outcome_scale),
        config=readout_config,
    )(production["draws"])
    uq_seconds = time.time() - readout_started
    result = {
        "schema_version": "bgm-mcmc-inference",
        "family": str(family),
        "target_kind": str(recipe.target_kind),
        "recipe": recipe.to_payload(),
        "recipe_hash": recipe.recipe_hash,
        "checkpoint_identity": str(checkpoint_identity),
        "data_seed": int(data_seed),
        "seeds": seeds,
        "run_label": str(run_label),
        "grid": {
            "num_queries": int(table.num_queries),
            "num_targets": int(table.num_targets),
            "truth_label": str(truth_label),
            "truth_hash": sha256_array(truth),
            "catalog_hash": table.catalog_hash,
            "query_hash": table.query_hash,
            "preprocessor_identity": str(preprocessor.identity),
        },
        "sampler": {
            "config": production["config"],
            "config_hash": sha256_json("mcmc-config", production["config"]),
            "run_seed": production["run_seed"],
            "run_key": production["run_key"],
            "target": production["target"],
            "target_hash": production["target"]["target_hash"],
        },
        "pilot": pilot,
        "readout": readout,
        "treatment_transform": {
            "shift": float(treatment_transform["shift"]),
            "scale": float(treatment_transform["scale"]),
        },
        "execution_environment": execution_environment(),
        "timings": {
            "pilot_seconds": float(pilot["seconds"]),
            "mcmc_seconds": float(production["seconds"]),
            "uq_seconds": float(uq_seconds),
            "total_seconds": float(time.time() - started),
        },
    }
    if progress:
        progress(
            f"[mcmc {family}] structural MSE "
            f"{readout['structural_mse_plugin']:.6f}; "
            f"cov95 {readout['coverage']['0.95']:.6f}; "
            f"width80 {readout['width80']:.6f}"
        )
    return result


__all__ = [
    "FAMILY_RECIPES",
    "FamilyRecipe",
    "MCMCConfig",
    "MCMCInferenceError",
    "derive_mcmc_seeds",
    "estimate_pilot_state_variance",
    "execution_environment",
    "run_mcmc",
    "run_mcmc_grid",
]
