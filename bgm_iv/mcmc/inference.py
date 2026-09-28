from __future__ import annotations

import json
import platform
import socket
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from .artifact import (
    MCMCDrawArtifactError,
    _validate_arm_id,
    load_draw_artifact,
    save_draw_artifact,
)
from .readout import TRUTH_NOISE_SD, FullGridReadout, build_query_table
from .sampler import (
    FrozenVectorizedHMC,
    LatentPosteriorEvaluator,
    overdispersed_initial_state,
    regularize_state_variance,
)
from .target import _model_family


class MCMCInferenceError(RuntimeError):
    pass


FAMILY_RECIPES: dict[str, tuple[float, int, tuple[int, ...]]] = {
    "demand": (0.05, 5, (3, 5, 7)),
    "mnist_pixel": (0.02, 31, (7, 15, 31)),
}


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise MCMCInferenceError(f"{name} must be a positive integer")
    return int(value)


def _recipe(
    family: str, *, num_chains: int, warmup_steps: int, draws: int
) -> dict[str, Any]:
    try:
        step_size, leapfrog_steps, support = FAMILY_RECIPES[str(family)]
    except KeyError:
        raise MCMCInferenceError(f"unknown MCMC family {family!r}") from None
    num_chains = _positive_int("production_num_chains", num_chains)
    if num_chains < 4:
        raise MCMCInferenceError("production recipes require at least four chains")
    return {
        "name": str(family),
        "pilot": {
            "warmup_steps": 400,
            "initial_step_size": float(step_size),
            "num_leapfrog_steps": 5,
            "segment_size": 240,
            "trajectory_support": [3, 5, 7],
            "jitter_scale": 0.3,
            "mass_estimator": "per_chain_within_variance_mean_over_chains_ddof1",
        },
        "production": {
            "num_chains": num_chains,
            "warmup_steps": _positive_int("production_warmup_steps", warmup_steps),
            "segment_size": _positive_int("production_draws", draws),
            "initial_step_size": float(step_size),
            "num_leapfrog_steps": int(leapfrog_steps),
            "target_accept_prob": 0.9,
            "trajectory_support": [int(value) for value in support],
            "overdispersion_scale": 2.0,
        },
    }


def execution_environment() -> dict[str, Any]:
    gpus = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:
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
    except Exception:
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


_RECIPE_BY_MODEL_FAMILY = {"demand": "demand", "mnist": "mnist_pixel"}


def _sample(
    model: Any,
    context: np.ndarray,
    *,
    run_seed: int,
    num_chains: int,
    warmup_steps: int,
    initial_step_size: float,
    num_leapfrog_steps: int,
    target_accept_prob: float,
    segment_size: int,
    trajectory_support: Sequence[int],
    jitter_scale: float,
    variance: np.ndarray,
    label: str,
    progress: Optional[Callable[[str], None]],
) -> tuple[np.ndarray, np.ndarray]:
    runner = FrozenVectorizedHMC(
        evaluator=LatentPosteriorEvaluator(model),
        num_chains=int(num_chains),
        num_targets=int(context.shape[0]),
        warmup_steps=int(warmup_steps),
        initial_step_size=float(initial_step_size),
        num_leapfrog_steps=int(num_leapfrog_steps),
        target_accept_prob=float(target_accept_prob),
        segment_size=int(segment_size),
        trajectory_support=tuple(int(value) for value in trajectory_support),
    )
    context = runner.check_target(context)
    initial = overdispersed_initial_state(
        model,
        context,
        num_chains=int(num_chains),
        latent_dim=runner.latent_dim,
        scale=float(jitter_scale),
        run_seed=int(run_seed),
        variance=variance,
    )
    final_state, step = runner.warmup(
        run_seed=int(run_seed),
        context=context,
        initial_state=initial,
        state_variance=variance,
    )
    draws, acceptance = runner.run_segment(
        run_seed=int(run_seed),
        context=context,
        state=final_state,
        step_size=step,
        state_variance=variance,
    )
    if progress:
        per_chain = np.mean(acceptance, axis=1)
        progress(
            f"[mcmc {label}] acceptance mean="
            f"{float(np.mean(acceptance)):.3f}, "
            f"min={float(np.min(acceptance)):.3f}, "
            f"per-chain={np.round(per_chain, 3).tolist()}"
        )
    return draws, acceptance


def _grid_arrays(table: Any, truth: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "query_x": np.asarray(table.query_x),
        "query_inverse": np.asarray(table.query_inverse),
        "unique_v": np.asarray(table.unique_v),
        "truth": np.asarray(truth),
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
    seeds: Mapping[str, int],
    production_num_chains: int,
    production_warmup_steps: int,
    production_draws: int,
    source: Optional[Mapping[str, Any]] = None,
    artifact_root: Optional[Any] = None,
    arm_id: Optional[str] = None,
    readout_artifact_manifest: Optional[Any] = None,
    progress: Optional[Callable[[str], None]] = print,
) -> dict[str, Any]:
    started = time.time()
    seeds = {"pilot": int(seeds["pilot"]), "production": int(seeds["production"])}
    source = json.loads(json.dumps(dict(source or {}), sort_keys=True, default=str))
    recipe = _recipe(
        family,
        num_chains=production_num_chains,
        warmup_steps=production_warmup_steps,
        draws=production_draws,
    )
    pilot_config = recipe["pilot"]
    config = recipe["production"]
    loaded_draws = None
    loaded_grid = None
    loaded_artifact = None
    artifact_load_seconds = 0.0
    if readout_artifact_manifest is not None:
        if artifact_root is not None or arm_id is not None:
            raise MCMCInferenceError(
                "readout_artifact_manifest cannot be combined with artifact output"
            )
        artifact_load_started = time.time()
        try:
            loaded_draws, loaded_grid, loaded_artifact = load_draw_artifact(
                readout_artifact_manifest
            )
        except MCMCDrawArtifactError as exc:
            raise MCMCInferenceError(str(exc)) from exc
        artifact_load_seconds = time.time() - artifact_load_started
    total_draws = int(config["segment_size"])
    if readout_artifact_manifest is None and (
        (artifact_root is None) != (arm_id is None)
    ):
        raise MCMCInferenceError(
            "artifact_root and arm_id must either both be set or both be omitted"
        )
    if arm_id is not None:
        try:
            _validate_arm_id(arm_id)
        except MCMCDrawArtifactError as exc:
            raise MCMCInferenceError(str(exc)) from exc
        destination = Path(artifact_root).expanduser().resolve() / arm_id
        if destination.exists():
            raise MCMCInferenceError(f"draw artifact already exists: {destination}")
    grid_x = np.asarray(grid_x_model, np.float32).reshape(-1, 1)
    grid_v_raw = np.asarray(grid_v_raw, np.float32)
    truth = np.asarray(truth_original_units, np.float64).reshape(-1)
    if grid_v_raw.ndim != 2 or not (
        grid_x.shape[0] == grid_v_raw.shape[0] == truth.shape[0]
    ):
        raise MCMCInferenceError("grid_x, grid_v and truth must have equal rows")
    if _RECIPE_BY_MODEL_FAMILY[_model_family(model)] != str(family):
        raise MCMCInferenceError("family recipe does not match the model")
    if preprocessor.dimension != int(model.params["v_dim"]):
        raise MCMCInferenceError("preprocessor dimension does not match model v_dim")
    grid_v_model = preprocessor.transform(grid_v_raw)
    if str(family) == "mnist_pixel" and (
        np.any(grid_v_model[:, 1:785] < 0.0) or np.any(grid_v_model[:, 1:785] > 255.0)
    ):
        raise MCMCInferenceError("MNIST pixel values must lie in raw scale [0,255]")
    table = build_query_table(grid_x, grid_v_model)
    grid_arrays = _grid_arrays(table, truth)
    readout_context = {
        "truth_label": str(truth_label),
        "outcome_shift": float(outcome_shift),
        "outcome_scale": float(outcome_scale),
        "truth_noise_sd": float(TRUTH_NOISE_SD),
        "treatment_transform": {
            "shift": float(treatment_transform["shift"]),
            "scale": float(treatment_transform["scale"]),
        },
    }

    if loaded_artifact is not None:
        settings = loaded_artifact["settings"]

        def require_equal(name: str, actual: Any, expected: Any) -> None:
            if actual != expected:
                raise MCMCInferenceError(f"draw artifact {name} mismatch")

        require_equal("family", settings.get("family"), str(family))
        require_equal("source checkpoint", settings.get("source"), source)
        require_equal("seeds", settings.get("seeds"), seeds)
        require_equal("production config", settings.get("production_config"), config)
        require_equal("readout context", settings.get("readout_context"), readout_context)
        if set(loaded_grid) != set(grid_arrays):
            raise MCMCInferenceError("draw artifact grid is incomplete")
        for name, expected in grid_arrays.items():
            stored = loaded_grid[name]
            if stored.shape != expected.shape or not np.array_equal(stored, expected):
                raise MCMCInferenceError(f"draw artifact grid {name} mismatch")
        latent_dim = int(sum(int(value) for value in model.params["z_dims"]))
        expected_shape = (
            total_draws,
            int(config["num_chains"]),
            int(table.num_targets),
            latent_dim,
        )
        require_equal(
            "draw shape",
            tuple(int(value) for value in loaded_draws.shape),
            expected_shape,
        )
    if progress:
        progress(
            f"[mcmc {family}] {table.num_targets} targets / "
            f"{table.num_queries} full-grid queries"
        )
    if loaded_artifact is None:
        latent_dim = int(sum(int(value) for value in model.params["z_dims"]))
        pilot_started = time.time()
        pilot_draws, _ = _sample(
            model,
            table.unique_v,
            run_seed=seeds["pilot"],
            num_chains=4,
            warmup_steps=pilot_config["warmup_steps"],
            initial_step_size=pilot_config["initial_step_size"],
            num_leapfrog_steps=pilot_config["num_leapfrog_steps"],
            target_accept_prob=0.9,
            segment_size=pilot_config["segment_size"],
            trajectory_support=pilot_config["trajectory_support"],
            jitter_scale=pilot_config["jitter_scale"],
            variance=np.ones((table.num_targets, latent_dim), np.float32),
            label="pilot",
            progress=progress,
        )
        raw = np.var(pilot_draws, axis=0, ddof=1).mean(axis=0)
        if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
            raise MCMCInferenceError("pilot variance estimate is not positive/finite")
        variance = regularize_state_variance(raw).astype(np.float32)
        pilot = {
            "seconds": float(time.time() - pilot_started),
            "raw_variance_median": float(np.median(raw)),
            "regularized_variance_median": float(np.median(variance)),
        }
        production_started = time.time()
        draws, acceptance = _sample(
            model,
            table.unique_v,
            run_seed=seeds["production"],
            num_chains=config["num_chains"],
            warmup_steps=config["warmup_steps"],
            initial_step_size=config["initial_step_size"],
            num_leapfrog_steps=config["num_leapfrog_steps"],
            target_accept_prob=config["target_accept_prob"],
            segment_size=config["segment_size"],
            trajectory_support=config["trajectory_support"],
            jitter_scale=config["overdispersion_scale"],
            variance=variance,
            label="production",
            progress=progress,
        )
        per_chain = np.mean(acceptance, axis=1)
        production = {
            "draws": draws,
            "seconds": float(time.time() - production_started),
            "run_seed": int(seeds["production"]),
            "config": config,
            "acceptance": {
                "mean": float(np.mean(acceptance)),
                "minimum": float(np.min(acceptance)),
                "per_chain": [float(value) for value in per_chain],
            },
        }
    else:
        pilot = {
            "skipped": True,
            "seconds": 0.0,
            "reason": "readout_artifact_manifest",
        }
        production = {
            "draws": loaded_draws,
            "seconds": 0.0,
            "run_seed": int(seeds["production"]),
            "config": config,
            "acceptance": loaded_artifact["settings"].get("production_acceptance"),
        }
        if progress:
            progress(
                f"[mcmc {family}] checked draw artifact; skipping pilot and production"
            )

    artifact = loaded_artifact
    artifact_seconds = float(artifact_load_seconds)
    if artifact_root is not None:
        artifact_started = time.time()
        try:
            artifact = save_draw_artifact(
                production["draws"],
                artifact_root=artifact_root,
                arm_id=str(arm_id),
                settings={
                    "family": str(family),
                    "source": source,
                    "seeds": seeds,
                    "recipe": recipe,
                    "production_config": production["config"],
                    "production_acceptance": production["acceptance"],
                    "readout_context": readout_context,
                },
                grid=grid_arrays,
            )
        except MCMCDrawArtifactError as exc:
            raise MCMCInferenceError(str(exc)) from exc
        artifact_seconds = time.time() - artifact_started
        if progress:
            progress(
                f"[mcmc {family}] saved {total_draws} draws/chain to "
                f"{artifact['artifact_dir']}"
            )

    readout_runner = FullGridReadout(
        model,
        table,
        truth,
        outcome_shift=float(outcome_shift),
        outcome_scale=float(outcome_scale),
    )
    readout_started = time.time()
    readout = readout_runner(production["draws"])
    uq_seconds = time.time() - readout_started
    readout["readout_seconds"] = float(uq_seconds)
    if progress:
        progress(
            f"[mcmc {family}] cov95 {readout['coverage']['0.95']:.6f}; "
            f"width95 {readout['width95']:.6f}"
        )
    result = {
        "family": str(family),
        "recipe": recipe,
        "seeds": seeds,
        "grid": {
            "num_queries": int(table.num_queries),
            "num_targets": int(table.num_targets),
            "truth_label": str(truth_label),
        },
        "sampler": {
            "run_seed": production["run_seed"],
            "acceptance": production["acceptance"],
        },
        "pilot": pilot,
        "readout": readout,
        "artifact": artifact,
        "treatment_transform": {
            "shift": float(treatment_transform["shift"]),
            "scale": float(treatment_transform["scale"]),
        },
        "timings": {
            "pilot_seconds": float(pilot["seconds"]),
            "mcmc_seconds": float(production["seconds"]),
            "artifact_seconds": float(artifact_seconds),
            "uq_seconds": float(uq_seconds),
            "total_seconds": float(time.time() - started),
        },
    }
    return result


__all__ = [
    "FAMILY_RECIPES",
    "MCMCInferenceError",
    "execution_environment",
    "run_mcmc_grid",
]
