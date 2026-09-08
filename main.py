import argparse
import contextlib
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import gc
import hashlib
import io
import multiprocessing
import os
import numpy as np
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import json
from itertools import product
import yaml
from bgm_iv.models import (
    BGM_IV,
    BGM_IV_Image,
    BGM_IV_Vector,
)
from bgm_iv.datasets import (
    simulate_demand_design_iv,
    make_demand_design_grid,
    simulate_demand_design_mnist_iv,
    make_demand_design_mnist_grid,
    simulate_demand_design_vector_iv,
    make_demand_design_vector_grid,
)
import tensorflow as tf
from bgm_iv.hashing import sha256_array, sha256_json, sha256_weights
from bgm_iv.egm_multistart import (
    EGM_SELECTION_CRITERION,
    EGM_SELECTOR_VERSION,
    derive_multistart_seeds,
    make_candidate_manifest,
    score_evaluation_iterations,
    select_candidate_by_criterion,
    validate_multistart_config,
    verify_manifest_hash,
)


def _load_mcmc():
    """Import the MCMC inference package on first use.

    ``bgm_iv.mcmc.sampler`` disables TensorFloat-32 and enables op determinism
    at import time; loading it lazily keeps training numerics unchanged and
    applies the sampler's settings only once MCMC starts.
    """
    from bgm_iv.mcmc import inference as inference_module
    from bgm_iv.mcmc import target as target_module

    global FAMILY_RECIPES, run_mcmc_grid, AffinePreprocessorSpec, execution_environment
    FAMILY_RECIPES = inference_module.FAMILY_RECIPES
    run_mcmc_grid = inference_module.run_mcmc_grid
    AffinePreprocessorSpec = target_module.AffinePreprocessorSpec
    execution_environment = inference_module.execution_environment
    return inference_module


class _LazyName:
    """Module-level placeholder resolved by `_load_mcmc()` on first call."""

    def __init__(self, name):
        self._name = name

    def _resolve(self):
        _load_mcmc()
        return globals()[self._name]

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __getitem__(self, item):
        return self._resolve()[item]

    def __contains__(self, item):
        return item in self._resolve()

    def __iter__(self):
        return iter(self._resolve())


FAMILY_RECIPES = _LazyName("FAMILY_RECIPES")
run_mcmc_grid = _LazyName("run_mcmc_grid")
AffinePreprocessorSpec = _LazyName("AffinePreprocessorSpec")
execution_environment = _LazyName("execution_environment")


_DEMAND_DESIGN_DATASET_META = {
    "Sim_Demand_Design_IV": {
        "title": "Sim_Demand_Design_IV",
        "slug": "sim_demand_design_iv",
        "config_name": "Sim_Demand_Design_IV.yaml",
        "seed_key": "seed",
        "uses_rho": True,
    },
    "Sim_Demand_Design_Mnist_IV": {
        "title": "Sim_Demand_Design_Mnist_IV",
        "slug": "sim_demand_design_mnist_iv",
        "config_name": "Sim_Demand_Design_Mnist_IV.yaml",
        "seed_key": "image_seed",
        "uses_rho": True,
    },
    "Sim_Demand_Design_Vector_IV": {
        "title": "Sim_Demand_Design_Vector_IV",
        "slug": "sim_demand_design_vector_iv",
        "config_name": "Sim_Demand_Design_Vector_IV.yaml",
        "seed_key": "feature_seed",
        "uses_rho": True,
    },
}


_FIXED_BENCHMARK_DEFAULTS = {
    "seed": 0,
    "price_points": 20,
    "time_points": 20,
    "w_dim": 1,
    "fit_use_progress_bar": False,
    "covariate_block_scale": "sum",
}


_DATASET_FIXED_BENCHMARK_DEFAULTS = {
    "Sim_Demand_Design_IV": {
        "v_dim": 2,
    },
    "Sim_Demand_Design_Mnist_IV": {
        "image_seed": 42,
        "noise_seed": 42,
    },
    "Sim_Demand_Design_Vector_IV": {
        "vector_dim": 784,
        "v_dim": 785,
        "feature_seed": 42,
        "test_vector_seed": 42,
    },
}


_DATASET_OPTIONAL_BENCHMARK_DEFAULTS = {
    "Sim_Demand_Design_Mnist_IV": {
        "v_dim": 785,
    },
}


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to one of the supported BGM-IV config files.",
    )
    parser.add_argument(
        '-t',
        '--num_tasks',
        type=int,
        default=1,
        help='Number of parallel demand-design tasks to run (default: 1).',
    )
    parser.add_argument(
        "--mcmc-only",
        dest="mcmc_only",
        type=str,
        default=None,
        metavar="TIMESTAMP",
        help=(
            "Skip training: restore the checkpoint with this timestamp from the "
            "run's output_dir (same yaml, scalar n_samples/rho, --repeat-id) and "
            "run the structural evaluation and full-grid MCMC inference on it."
        ),
    )
    parser.add_argument(
        "--repeat-id",
        dest="repeat_id",
        type=int,
        default=None,
        help=(
            "Run only this repeat of the sweep (one Slurm array task per repeat); "
            "with --mcmc-only it is the repeat whose checkpoint is restored."
        ),
    )
    parser.add_argument(
        "--mcmc-production-warmup-steps",
        dest="mcmc_production_warmup_steps",
        type=int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-production-draws",
        dest="mcmc_production_draws",
        type=int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-artifact-root",
        dest="mcmc_artifact_root",
        type=str,
        default=None,
        metavar="DIR",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-arm-id",
        dest="mcmc_arm_id",
        type=str,
        default=None,
        metavar="ID",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-readout-prefixes",
        dest="mcmc_readout_prefixes",
        type=str,
        default=None,
        metavar="N[,N...]",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-readout-artifact",
        dest="mcmc_readout_artifact",
        type=str,
        default=None,
        metavar="MANIFEST",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-reference-map-mse",
        dest="mcmc_reference_map_mse",
        type=float,
        default=None,
        metavar="FLOAT",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcmc-reference-encoder-mse",
        dest="mcmc_reference_encoder_mse",
        type=float,
        default=None,
        metavar="FLOAT",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one config entry (YAML value syntax), e.g. --set n_samples=5000 "
            "--set rho=0.5; repeatable.  Lets one yaml serve every Slurm array cell."
        ),
    )
    return parser


_MCMC_INFERENCE_OPTIONS_KEY = "_mcmc_inference_options"
_MCMC_CONFIG_FIELDS = {
    "mcmc_num_chains": "production_num_chains",
    "mcmc_production_warmup_steps": "production_warmup_steps",
    "mcmc_production_draws": "production_draws",
}


def _positive_mcmc_config_integer(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"`{name}` must be an integer >= {minimum}")
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"`{name}` must be an integer >= {minimum}")
    return result


def _apply_mcmc_inference_config(params):
    """Move public YAML MCMC controls into the inference-only namespace.

    Keeping these values under an underscore-prefixed key prevents a sampling
    budget change from altering the training-manifest/checkpoint identity.
    """
    present = [name for name in _MCMC_CONFIG_FIELDS if name in params]
    if not present:
        return params
    if len(present) != len(_MCMC_CONFIG_FIELDS):
        missing = sorted(set(_MCMC_CONFIG_FIELDS) - set(present))
        raise ValueError(
            "MCMC YAML production controls must be provided together; "
            f"missing {missing}"
        )
    options = dict(params.get(_MCMC_INFERENCE_OPTIONS_KEY) or {})
    for public_name, private_name in _MCMC_CONFIG_FIELDS.items():
        minimum = 4 if public_name == "mcmc_num_chains" else 1
        options[private_name] = _positive_mcmc_config_integer(
            public_name, params.pop(public_name), minimum=minimum
        )
    params[_MCMC_INFERENCE_OPTIONS_KEY] = options
    return params


def _parse_mcmc_readout_prefixes(value):
    """Normalize the private ablation CLI's comma-separated draw prefixes."""
    if value is None:
        return None
    pieces = [piece.strip() for piece in str(value).split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--mcmc-readout-prefixes requires comma-separated integers")
    try:
        prefixes = sorted({int(piece) for piece in pieces})
    except ValueError as exc:
        raise ValueError(
            "--mcmc-readout-prefixes requires comma-separated integers"
        ) from exc
    if any(prefix <= 0 for prefix in prefixes):
        raise ValueError("--mcmc-readout-prefixes values must be positive")
    return prefixes


def _apply_mcmc_inference_cli(params, args):
    """Validate and attach inference-only MCMC ablation options.

    The options live under one underscore-prefixed key, so they are excluded
    from the model-training manifest and cannot change checkpoint identity.
    """
    raw = {
        "production_warmup_steps": args.mcmc_production_warmup_steps,
        "production_draws": args.mcmc_production_draws,
        "artifact_root": args.mcmc_artifact_root,
        "arm_id": args.mcmc_arm_id,
        "readout_prefixes": args.mcmc_readout_prefixes,
        "readout_artifact_manifest": args.mcmc_readout_artifact,
        "reference_map_mse": args.mcmc_reference_map_mse,
        "reference_encoder_mse": args.mcmc_reference_encoder_mse,
    }
    if all(value is None for value in raw.values()):
        return params
    if _mcmc_only_timestamp(params) is None:
        raise ValueError("MCMC ablation options require --mcmc-only TIMESTAMP")

    warmup = raw["production_warmup_steps"]
    draws = raw["production_draws"]
    if (warmup is None) != (draws is None):
        raise ValueError(
            "--mcmc-production-warmup-steps and --mcmc-production-draws "
            "must be provided together"
        )
    if warmup is not None and int(warmup) <= 0:
        raise ValueError("--mcmc-production-warmup-steps must be positive")
    if draws is not None and int(draws) <= 0:
        raise ValueError("--mcmc-production-draws must be positive")

    artifact_root = raw["artifact_root"]
    arm_id = raw["arm_id"]
    if (artifact_root is None) != (arm_id is None):
        raise ValueError(
            "--mcmc-artifact-root and --mcmc-arm-id must be provided together"
        )
    if artifact_root is not None and not str(artifact_root).strip():
        raise ValueError("--mcmc-artifact-root must be non-empty")
    if arm_id is not None:
        arm_id = str(arm_id).strip()
        if not arm_id:
            raise ValueError("--mcmc-arm-id must be non-empty")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        if any(character not in allowed for character in arm_id):
            raise ValueError(
                "--mcmc-arm-id may contain only letters, digits, '.', '_' and '-'"
            )

    prefixes = _parse_mcmc_readout_prefixes(raw["readout_prefixes"])
    readout_artifact = raw["readout_artifact_manifest"]
    if readout_artifact is not None:
        readout_artifact = str(readout_artifact).strip()
        if not readout_artifact:
            raise ValueError("--mcmc-readout-artifact must be non-empty")
        if warmup is not None or artifact_root is not None:
            raise ValueError(
                "--mcmc-readout-artifact cannot be combined with production "
                "warmup/draws or artifact root/arm options"
            )
    if prefixes is not None:
        if draws is None and readout_artifact is None:
            raise ValueError(
                "--mcmc-readout-prefixes requires production draws or "
                "--mcmc-readout-artifact"
            )
        if draws is not None and prefixes[-1] > int(draws):
            raise ValueError(
                "--mcmc-readout-prefixes cannot exceed --mcmc-production-draws"
            )

    reference_metrics = {}
    for name, value in (
        ("map", raw["reference_map_mse"]),
        ("encoder", raw["reference_encoder_mse"]),
    ):
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"--mcmc-reference-{name}-mse must be finite and non-negative"
                )
            reference_metrics[name] = value

    existing = dict(params.get(_MCMC_INFERENCE_OPTIONS_KEY) or {})
    params[_MCMC_INFERENCE_OPTIONS_KEY] = {
        **existing,
        "production_num_chains": existing.get("production_num_chains"),
        "production_warmup_steps": (
            existing.get("production_warmup_steps")
            if warmup is None else int(warmup)
        ),
        "production_draws": (
            existing.get("production_draws") if draws is None else int(draws)
        ),
        "artifact_root": None if artifact_root is None else str(artifact_root).strip(),
        "arm_id": arm_id,
        "readout_prefixes": prefixes,
        "readout_artifact_manifest": readout_artifact,
        "reference_metrics": reference_metrics,
    }
    return params


def _apply_config_overrides(params, overrides):
    """Apply ``--set KEY=VALUE`` entries in order; values are parsed as YAML."""
    for item in overrides:
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}.")
        params[key] = yaml.safe_load(value)
    return params


def _get_demand_design_dataset_meta(params):
    dataset = params.get("dataset", "Sim_Demand_Design_IV")
    if dataset not in _DEMAND_DESIGN_DATASET_META:
        raise ValueError(f"Unsupported demand-design dataset: {dataset}")
    return _DEMAND_DESIGN_DATASET_META[dataset]


def _coerce_fixed_benchmark_value(field_name, value, default):
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "false"}:
                    return lowered == "true"
            raise ValueError
        if isinstance(default, int) and not isinstance(default, bool):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value).strip()
    except (TypeError, ValueError):
        raise ValueError(
            f"`{field_name}` is fixed to {default!r} for the paper benchmarks; got {value!r}."
        ) from None
    return value


def _apply_fixed_benchmark_defaults(params, defaults):
    for field_name, default in defaults.items():
        if field_name in params:
            value = _coerce_fixed_benchmark_value(
                field_name,
                params[field_name],
                default,
            )
            if value != default:
                raise ValueError(
                    f"`{field_name}` is fixed to {default!r} for the paper benchmarks; "
                    f"got {params[field_name]!r}."
                )
        params[field_name] = default


def _apply_optional_benchmark_defaults(params, defaults):
    for field_name, default in defaults.items():
        if field_name not in params:
            params[field_name] = default
        else:
            params[field_name] = _coerce_fixed_benchmark_value(
                field_name,
                params[field_name],
                default,
            )


def _apply_demand_design_benchmark_defaults(params):
    """Inject fixed paper-benchmark defaults before run artifacts are created."""
    dataset = params.get("dataset", "Sim_Demand_Design_IV")
    _get_demand_design_dataset_meta(params)
    if "alpha_v" in params:
        raise ValueError(
            "`alpha_v` is no longer supported; test-time MAP uses the untempered "
            "covariate posterior."
        )
    if "mcmc_seed" in params:
        raise ValueError(
            "`mcmc_seed` is no longer supported; full-grid MCMC derives and "
            "records pilot/production seeds from run_seed and checkpoint identity."
        )
    if dataset == "Sim_Demand_Design_IV" and "noise_seed" in params:
        raise ValueError(
            "`noise_seed` is no longer supported for Sim_Demand_Design_IV; "
            "low-dimensional demand uses only `(time, customer_group)` covariates."
        )

    _apply_fixed_benchmark_defaults(params, _FIXED_BENCHMARK_DEFAULTS)
    _apply_fixed_benchmark_defaults(
        params,
        _DATASET_FIXED_BENCHMARK_DEFAULTS.get(dataset, {}),
    )
    _apply_optional_benchmark_defaults(
        params,
        _DATASET_OPTIONAL_BENCHMARK_DEFAULTS.get(dataset, {}),
    )
    if dataset == "Sim_Demand_Design_Mnist_IV" and int(params["v_dim"]) < 785:
        raise ValueError(
            "`v_dim` must be >= 785 for Sim_Demand_Design_Mnist_IV; "
            f"got {params['v_dim']!r}."
        )

    normalized = validate_multistart_config(
        params, mcmc_only=_is_mcmc_only(params)
    )
    params.clear()
    params.update(normalized)


def _demand_design_uses_rho(params):
    return bool(_get_demand_design_dataset_meta(params).get("uses_rho", True))


def _configure_tensorflow_threads(intra_op_threads=None, inter_op_threads=None):
    if intra_op_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_op_threads))
    if inter_op_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_op_threads))


def _configure_tensorflow_devices(
    use_gpu=False, gpu_slot=None, verbose=True, strict_memory_growth=False
):
    gpus = tf.config.list_physical_devices("GPU")

    if use_gpu:
        if not gpus:
            if strict_memory_growth:
                raise RuntimeError(
                    "TensorFlow GPU was required, but no GPU was detected"
                )
            if verbose:
                print(
                    "TensorFlow GPU requested but no GPU was detected. Falling back to CPU."
                )
            return
        selected_gpus = list(gpus)
        if gpu_slot is not None:
            gpu_slot = int(gpu_slot)
            if gpu_slot < 0 or gpu_slot >= len(gpus):
                raise ValueError(
                    f"Requested GPU slot {gpu_slot} but only {len(gpus)} visible GPU(s) exist."
                )
            selected_gpus = [gpus[gpu_slot]]
            if list(tf.config.get_visible_devices("GPU")) != selected_gpus:
                tf.config.set_visible_devices(selected_gpus, "GPU")
        for gpu in selected_gpus:
            try:
                memory_growth_enabled = bool(
                    tf.config.experimental.get_memory_growth(gpu)
                )
            except (RuntimeError, ValueError):
                if strict_memory_growth:
                    raise
                memory_growth_enabled = False
            if memory_growth_enabled:
                continue
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                if strict_memory_growth:
                    raise
            if strict_memory_growth and not tf.config.experimental.get_memory_growth(gpu):
                raise RuntimeError("TensorFlow GPU memory growth was not enabled")
        if verbose:
            if gpu_slot is None:
                print(f"TensorFlow GPU enabled with {len(selected_gpus)} device(s).")
            else:
                print(f"TensorFlow GPU enabled for worker slot {gpu_slot}.")
        return

    if tf.config.get_visible_devices("GPU"):
        tf.config.set_visible_devices([], "GPU")
    if verbose:
        print("TensorFlow GPU disabled. Using CPU only.")


def _resolve_num_tasks(params):
    num_tasks = params.get("num_tasks", 1)
    if isinstance(num_tasks, (list, tuple)):
        raise ValueError("`num_tasks` must be a positive integer, not a list.")
    num_tasks = int(num_tasks)
    if num_tasks < 1:
        raise ValueError("`num_tasks` must be >= 1.")
    return num_tasks


def _supports_parallel_demand_design(params):
    return params.get("dataset") in _DEMAND_DESIGN_DATASET_META


def _is_parallel_demand_design_run(params):
    return _supports_parallel_demand_design(params) and _resolve_num_tasks(params) > 1


def _resolve_parallel_gpu_slots(params):
    num_tasks = _resolve_num_tasks(params)
    if not bool(params.get("use_gpu", False)):
        return tuple(None for _ in range(num_tasks))
    visible_gpu_count = len(tf.config.list_physical_devices("GPU"))
    if visible_gpu_count == 0:
        raise ValueError(
            "`use_gpu: true` with `-t > 1` requires at least one visible GPU, but TensorFlow detected none."
        )
    if num_tasks > visible_gpu_count:
        raise ValueError(
            f"`num_tasks={num_tasks}` exceeds the number of visible GPU devices ({visible_gpu_count})."
        )
    return tuple(range(num_tasks))


def _fit_standardizer(data):
    mean = np.mean(data, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(data, axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def _transform(data, stats):
    return ((data - stats["mean"]) / stats["scale"]).astype(np.float32)


def _inverse_transform(data, stats):
    return (data * stats["scale"] + stats["mean"]).astype(np.float32)


def _summarize_ranges(train):
    print(_render_observed_ranges(train))


def _print_demand_design_run_config(params):
    print(_render_demand_design_run_config(params))


def _relative_markdown_link(from_path, target):
    return os.path.relpath(str(target), start=str(Path(from_path).parent))


def _render_demand_design_run_config(params):
    dataset_meta = _get_demand_design_dataset_meta(params)
    lines = ["Demand-design run config:"]
    keys = [
        "n_samples",
        "n_repeat",
        "repeat_id",
        "seed",
        "run_seed",
        dataset_meta["seed_key"],
        "z_dims",
        "v_dim",
        "w_dim",
        "treatment_dim",
        "treatment_feature_dim",
        "vector_dim",
        "feature_seed",
        "test_vector_seed",
        "representation_sd",
        # recipe provenance
        "outcome_to_particles_weight",
        "covariate_block_scale",
        "sigma_v",
        "sigma_v_softfloor",
        "sigma_time",
        "sigma_time_softfloor",
        "sigma_vector",
        "sigma_vector_softfloor",
        "vector_blocks",
        "sigma_y_softfloor",
        "deterministic_training",
        "training_grid_monitor",
        "egm_num_warm_starts",
        "structural_methods",
        "mcmc_family",
        "holdout_seed_offset",
    ]
    if _demand_design_uses_rho(params):
        keys.insert(1, "rho")
    seen_keys = set()
    for key in keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if key in params:
            lines.append(f"  {key}: {params.get(key)}")
    inference_options = params.get(_MCMC_INFERENCE_OPTIONS_KEY) or {}
    for label, key in (
        ("mcmc_num_chains", "production_num_chains"),
        ("mcmc_production_warmup_steps", "production_warmup_steps"),
        ("mcmc_production_draws", "production_draws"),
    ):
        if inference_options.get(key) is not None:
            lines.append(f"  {label}: {inference_options[key]}")
    if (
        params.get("dataset") == "Sim_Demand_Design_Mnist_IV"
        and int(params.get("v_dim", 785)) > 785
        and "noise_seed" in params
    ):
        lines.append(f"  noise_seed: {params.get('noise_seed')}")
    if "dsprite_data_dir" in params:
        lines.append(f"  dsprite_data_dir: {params.get('dsprite_data_dir')}")
    return "\n".join(lines)


def _render_observed_ranges(train):
    lines = ["Observed data ranges before normalization:"]
    for key in ("x", "y", "v", "w"):
        data = np.asarray(train[key], dtype=np.float32)
        lines.append(
            f"  {key}: min={float(np.min(data)):.4f}, max={float(np.max(data)):.4f}, "
            f"mean={float(np.mean(data)):.4f}, std={float(np.std(data)):.4f}"
        )
    return "\n".join(lines)


def _get_training_history_structural_keys(history):
    return sorted(
        {
            key
            for record in history
            for key in record
            if key.startswith("structural_mse_")
        }
    )


def _render_training_history(history):
    if not history:
        return ""

    structural_keys = _get_training_history_structural_keys(history)
    lines = ["Training metric history"]
    header = (
        f"{'stage':<12} {'epoch':>6} {'outcome':>8} "
        f"{'mse_x':>12} {'mse_y':>12} {'mse_v':>12}"
    )
    for key in structural_keys:
        method = key.removeprefix("structural_mse_")
        header += f" {method:>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for record in history:
        epoch = "-" if record["epoch"] is None else str(record["epoch"])
        row = (
            f"{record['stage']:<12} {epoch:>6} {str(record['include_outcome']):>8} "
            f"{record['mse_x']:>12.6f} {record['mse_y']:>12.6f} {record['mse_v']:>12.6f}"
        )
        for key in structural_keys:
            value = record.get(key)
            row += f" {('-' if value is None else f'{value:.6f}'):>14}"
        lines.append(row)
    return "\n".join(lines)


def _build_demand_design_run_timestamp(now=None):
    now = datetime.now() if now is None else now
    return now.strftime("%Y-%m-%d_%H-%M-%S-%f")


def _build_demand_design_run_id(params, now=None):
    dataset_meta = _get_demand_design_dataset_meta(params)
    return f"{dataset_meta['slug']}_{_build_demand_design_run_timestamp(now)}"


def _demand_design_src_root():
    return Path(__file__).resolve().parent


def _demand_design_logs_dir():
    return _demand_design_src_root() / "logs"


def _demand_design_dumps_dir():
    return _demand_design_src_root() / "dumps"


def _demand_design_active_window_path(params):
    slug = _get_demand_design_dataset_meta(params)["slug"]
    return _demand_design_logs_dir() / f"outputs_dev_{slug}_active.md"


def _build_demand_design_combo_dir_name(params):
    if not _demand_design_uses_rho(params):
        return (
            f"n_samples:{int(params['n_samples'])}"
            f"-v_dim:{int(params['v_dim'])}"
            f"-w_dim:{int(params['w_dim'])}"
        )
    name = (
        f"n_samples:{int(params['n_samples'])}"
        f"-rho:{_format_demand_design_sweep_value(params['rho'])}"
        f"-v_dim:{int(params['v_dim'])}"
    )
    if params.get("_sweep_gamma"):
        name += f"-gamma:{_format_demand_design_sweep_value(float(params['outcome_to_particles_weight']))}"
    return name


def _resolve_source_config_path(params):
    source = params.get("_config_source_path")
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    return source_path


def _clean_dumpable_params(params):
    return {key: value for key, value in params.items() if not str(key).startswith("_")}


def _copy_demand_design_config_snapshot(params, run_root):
    source_path = _resolve_source_config_path(params)
    dataset_meta = _get_demand_design_dataset_meta(params)
    destination_name = source_path.name if source_path is not None else dataset_meta["config_name"]
    destination = run_root / destination_name
    if source_path is not None and source_path.exists():
        shutil.copyfile(source_path, destination)
    else:
        destination.write_text(
            yaml.safe_dump(_clean_dumpable_params(params), sort_keys=False),
            encoding="utf-8",
        )
    return destination


def _csv_writer_append_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if not write_header:
        with path.open(newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if list(header) != list(fieldnames):
            raise RuntimeError(
                f"existing CSV schema differs from the current schema: {path}"
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_results_rows(history, repeat_id):
    rows = []
    for key in _get_training_history_structural_keys(history):
        method = key.removeprefix("structural_mse_")
        method_records = [record for record in history if key in record]
        if not method_records:
            continue
        result_record = method_records[-1]
        rows.append(
            {
                "repeat_id": int(repeat_id),
                "method": method,
                "stage": result_record["stage"],
                "epoch": result_record["epoch"],
                "include_outcome": result_record["include_outcome"],
                "mse_x": result_record["mse_x"],
                "mse_y": result_record["mse_y"],
                "mse_v": result_record["mse_v"],
                "structural_mse": result_record[key],
            }
        )
    return rows


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# results.csv: one row per repeat with headline structural MSEs, full-grid
# MCMC interval calibration and compact reproducibility provenance.
_FINAL_RESULT_COLUMNS = (
    "repeat_id",
    "run_seed",
    "stage",
    "epoch",
    "include_outcome",
    "mse_x",
    "mse_y",
    "mse_v",
    "structural_mse_map",
    "structural_mse_encoder",
    "structural_mse_mcmc",
    "mcmc_cov50",
    "mcmc_cov80",
    "mcmc_cov95",
    "mcmc_width50",
    "mcmc_width80",
    "mcmc_width95",
    "mcmc_family",
    "mcmc_num_targets",
    "mcmc_num_queries",
    "mcmc_num_chains",
    "mcmc_draws_per_chain",
    "mcmc_seconds",
    "mcmc_uq_seconds",
    "holdout_iv_mse_map",
    "holdout_iv_mse_encoder",
    "checkpoint_timestamp",
    "checkpoint_path",
    "checkpoint_identity",
    "weights_hash_g",
    "weights_hash_e",
    "weights_hash_f",
    "weights_hash_h",
    "outcome_to_particles_weight",
    "covariate_block_scale",
    "sigma_v",
    "sigma_v_softfloor",
    "sigma_time",
    "sigma_time_softfloor",
    "sigma_vector_softfloor",
    "sigma_y_softfloor",
    "deterministic_training",
    "training_grid_monitor",
    "egm_num_warm_starts",
    "egm_selector_version",
    "egm_selection_criterion",
    "egm_selected_candidate_id",
    "egm_selected_criterion",
    "egm_selected_train_iv_map",
    "egm_selected_train_iv_encoder",
    "egm_selected_train_mse_y",
    "egm_selected_egm_tail_rank",
    "egm_candidate_criteria_hash",
    "egm_selection_manifest_hash",
    "device_name",
    "hostname",
    "sigma_vector_softfloor_source",
    "mcmc_only",
)


def _blank(value):
    return "" if value is None else value


def _build_final_results_row(params, history, final_results, provenance=None):
    """One headline result row per repeat."""
    final_results = final_results or {}
    provenance = provenance or {}
    last = history[-1] if history else {}
    row = {column: "" for column in _FINAL_RESULT_COLUMNS}
    row.update(
        {
            "repeat_id": int(params.get("repeat_id", 0)),
            "run_seed": _blank(params.get("run_seed", params.get("seed"))),
            "stage": last.get("stage", ""),
            "epoch": last.get("epoch", ""),
            "include_outcome": last.get("include_outcome", ""),
            "mse_x": last.get("mse_x", ""),
            "mse_y": last.get("mse_y", ""),
            "mse_v": last.get("mse_v", ""),
        }
    )
    if "map" in final_results:
        row["structural_mse_map"] = final_results["map"]
    elif "structural_mse_map" in last:
        row["structural_mse_map"] = last["structural_mse_map"]
    if "encoder" in final_results:
        row["structural_mse_encoder"] = final_results["encoder"]
    for key in ("holdout_iv_mse_map", "holdout_iv_mse_encoder"):
        if key in final_results:
            row[key] = final_results[key]
    mcmc = final_results.get("_mcmc")
    if mcmc is not None:
        readout = mcmc["readout"]
        coverage = readout["coverage"]
        row.update(
            {
                "structural_mse_mcmc": readout["structural_mse_plugin"],
                "mcmc_cov50": coverage["0.5"],
                "mcmc_cov80": coverage["0.8"],
                "mcmc_cov95": coverage["0.95"],
                "mcmc_width50": readout["width50"],
                "mcmc_width80": readout["width80"],
                "mcmc_width95": readout["width95"],
                "mcmc_family": mcmc["family"],
                "mcmc_num_targets": mcmc["grid"]["num_targets"],
                "mcmc_num_queries": mcmc["grid"]["num_queries"],
                "mcmc_num_chains": readout["num_chains"],
                "mcmc_draws_per_chain": readout["draws_per_chain"],
                "mcmc_seconds": mcmc["timings"]["mcmc_seconds"],
                "mcmc_uq_seconds": mcmc["timings"]["uq_seconds"],
            }
        )
    for key in (
        "checkpoint_timestamp",
        "checkpoint_path",
        "checkpoint_identity",
        "device_name",
    ):
        row[key] = _blank(provenance.get(key))
    row["hostname"] = _blank((provenance.get("execution_environment") or {}).get("hostname"))
    for name in ("g", "e", "f", "h"):
        row[f"weights_hash_{name}"] = _blank((provenance.get("weights") or {}).get(name))
    for key in (
        "outcome_to_particles_weight",
        "covariate_block_scale",
        "sigma_v",
        "sigma_v_softfloor",
        "sigma_time",
        "sigma_time_softfloor",
        "sigma_vector_softfloor",
        "sigma_y_softfloor",
        "deterministic_training",
        "training_grid_monitor",
        "egm_num_warm_starts",
    ):
        if key in params:
            row[key] = _blank(params.get(key))
    if "outcome_to_particles_weight" not in params and "resolved_gamma" in provenance:
        row["outcome_to_particles_weight"] = provenance["resolved_gamma"]
    multistart = provenance.get("egm_multistart") or {}
    for key in (
        "egm_selector_version",
        "egm_selection_criterion",
        "egm_selected_candidate_id",
        "egm_selected_criterion",
        "egm_selected_train_iv_map",
        "egm_selected_train_iv_encoder",
        "egm_selected_train_mse_y",
        "egm_selected_egm_tail_rank",
        "egm_candidate_criteria_hash",
        "egm_selection_manifest_hash",
    ):
        if key in multistart:
            row[key] = _blank(multistart.get(key))
    row["sigma_vector_softfloor_source"] = _blank(provenance.get("sigma_vector_softfloor_source"))
    if "sigma_vector_softfloor_value" in provenance:
        row["sigma_vector_softfloor"] = provenance["sigma_vector_softfloor_value"]
    row["mcmc_only"] = _blank(provenance.get("mcmc_only"))
    return row


def _persist_demand_design_repeat_outputs(
    run_root,
    run_index,
    total_runs,
    params,
    run_config_text,
    ranges_text,
    training_history,
    final_results=None,
    provenance=None,
):
    combo_dir = run_root / _build_demand_design_combo_dir_name(params)
    combo_dir.mkdir(parents=True, exist_ok=True)
    final_results = final_results or {}

    _csv_writer_append_rows(
        combo_dir / "results.csv",
        _FINAL_RESULT_COLUMNS,
        [_build_final_results_row(params, training_history, final_results, provenance)],
    )
    mcmc = final_results.get("_mcmc")
    if provenance is not None or mcmc is not None:
        repeat_id = int(params.get("repeat_id", 0))
        timestamp = (provenance or {}).get("checkpoint_timestamp", "unknown")
        record_dir = combo_dir / "records"
        record_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "bgm-repeat-record",
            "repeat_id": repeat_id,
            "run_config_text": run_config_text,
            "provenance": provenance,
            "final_results": {
                key: value for key, value in final_results.items() if not str(key).startswith("_")
            },
            "training_evaluate": final_results.get("_training_evaluate"),
            "mcmc": mcmc,
            "training_history": training_history,
        }
        with (record_dir / f"repeat{repeat_id}_{timestamp}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(record, handle, indent=1, default=_json_default)


def _render_demand_design_active_window(
    active_path,
    status,
    params,
    run_root,
    command,
    run_output,
):
    dataset_meta = _get_demand_design_dataset_meta(params)
    source_path = _resolve_source_config_path(params)
    source_link = str(source_path) if source_path is not None else str(
        params.get("_config_source_path", f"configs/{dataset_meta['config_name']}")
    )
    if source_path is not None:
        source_link = f"[{source_path.name}]({_relative_markdown_link(active_path, source_path)})"
    dumps_link = f"[{run_root.name}]({_relative_markdown_link(active_path, run_root)})"
    body = run_output.rstrip() or "Waiting for run output..."
    return (
        f"# Active Window: {dataset_meta['title']}\n\n"
        f"- status: {status}\n"
        "- target: `bgm_iv`\n"
        f"- source config: {source_link}\n"
        f"- dumps root: {dumps_link}\n"
        f"- command: `{command}`\n\n"
        "## Run Output\n\n"
        "```text\n"
        f"{body}\n"
        "```\n"
    )


class _DemandDesignActiveWindowController:
    def __init__(self, active_path, params, run_root):
        self.active_path = active_path
        self.params = params
        self.run_root = run_root
        self.command = shlex.join([str(arg) for arg in sys.argv])
        self.status = "running"
        self._chunks = []
        self._last_render = 0.0

    def append(self, text):
        if not text:
            return
        self._chunks.append(text)
        self.render()

    def set_status(self, status):
        self.status = status
        self.render(force=True)

    def render(self, force=False):
        now = time.monotonic()
        if not force and (now - self._last_render) < 0.75:
            return
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(
            _render_demand_design_active_window(
                self.active_path,
                self.status,
                self.params,
                self.run_root,
                self.command,
                "".join(self._chunks),
            ),
            encoding="utf-8",
        )
        self._last_render = now


class _ActiveWindowStream:
    def __init__(self, underlying, controller):
        self.underlying = underlying
        self.controller = controller
        self.encoding = getattr(underlying, "encoding", "utf-8")

    def write(self, text):
        written = self.underlying.write(text)
        self.controller.append(text)
        return written

    def flush(self):
        self.underlying.flush()
        self.controller.render()

    def isatty(self):
        isatty = getattr(self.underlying, "isatty", None)
        return bool(isatty()) if callable(isatty) else False


def _normalize_demand_design_sweep_values(values, field_name, coercer):
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError(f"`{field_name}` must not be an empty list.")
        return tuple(coercer(value) for value in values)
    return (coercer(values),)


def _format_demand_design_sweep_value(value):
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _resolve_demand_design_repeat_count(params):
    n_repeat = params.get("n_repeat", 1)
    if isinstance(n_repeat, (list, tuple)):
        raise ValueError("`n_repeat` must be a positive integer, not a list.")
    n_repeat = int(n_repeat)
    if n_repeat < 1:
        raise ValueError("`n_repeat` must be >= 1.")
    return n_repeat


def _resolve_demand_design_run_seed(params, repeat_id):
    return int(params.get("seed", 0)) + int(repeat_id)


def _make_demand_design_sweep_output_dir(
    base_output_dir, n_samples, repeat_id, rho=None, gamma=None
):
    parts = [f"n_samples={n_samples}"]
    if rho is not None:
        parts.append(f"rho={_format_demand_design_sweep_value(rho)}")
    if gamma is not None:
        parts.append(f"gamma={_format_demand_design_sweep_value(gamma)}")
    parts.append(f"repeat={repeat_id}")
    return f"{base_output_dir}/sweeps/" + "__".join(parts)


def _iter_demand_design_sweep_runs(params):
    n_samples_values = _normalize_demand_design_sweep_values(
        params.get("n_samples", 5000),
        "n_samples",
        int,
    )
    rho_values = (
        _normalize_demand_design_sweep_values(
            params.get("rho", 0.5),
            "rho",
            float,
        )
        if _demand_design_uses_rho(params)
        else (None,)
    )
    n_repeat = _resolve_demand_design_repeat_count(params)
    gamma_raw = params.get("outcome_to_particles_weight")
    gamma_axis = isinstance(gamma_raw, (list, tuple))
    gamma_values = (
        _normalize_demand_design_sweep_values(
            gamma_raw, "outcome_to_particles_weight", float
        )
        if gamma_axis
        else (None,)
    )
    combinations = tuple(product(n_samples_values, rho_values, gamma_values))
    total_runs = len(combinations) * n_repeat

    for run_index, (n_samples, rho, gamma) in enumerate(combinations, start=1):
        for repeat_id in range(n_repeat):
            run_params = dict(params)
            run_params["n_samples"] = n_samples
            if rho is None:
                run_params.pop("rho", None)
            else:
                run_params["rho"] = rho
            if gamma is not None:
                run_params["outcome_to_particles_weight"] = gamma
                run_params["_sweep_gamma"] = True
            run_params["n_repeat"] = n_repeat
            run_params["repeat_id"] = repeat_id
            run_params["run_seed"] = _resolve_demand_design_run_seed(
                run_params,
                repeat_id,
            )
            global_run_index = (run_index - 1) * n_repeat + repeat_id + 1
            if total_runs > 1 and (run_params.get("save_model") or run_params.get("save_res")):
                run_params["output_dir"] = _make_demand_design_sweep_output_dir(
                    str(run_params.get("output_dir", ".")),
                    n_samples,
                    repeat_id,
                    rho=rho,
                    gamma=gamma,
                )
            yield global_run_index, total_runs, run_params


def _materialize_demand_design_sweep_runs(params):
    return list(_iter_demand_design_sweep_runs(params))


def _render_demand_design_sweep_banner(run_index, total_runs, params):
    if _demand_design_uses_rho(params):
        details = (
            f"n_samples={params['n_samples']}, rho={params['rho']}, "
            f"repeat={params['repeat_id']}"
        )
    else:
        details = f"n_samples={params['n_samples']}, repeat={params['repeat_id']}"
    return (
        f"Demand-design sweep run [{run_index}/{total_runs}]: "
        f"{details}"
    )


def _print_demand_design_sweep_banner(run_index, total_runs, params):
    print(f"\n{_render_demand_design_sweep_banner(run_index, total_runs, params)}")


def _standardize_demand_design_data(train, grid):
    stats = {key: _fit_standardizer(train[key]) for key in ("x", "y", "v", "w")}
    train_std = {
        "x": _transform(train["x"], stats["x"]),
        "y": _transform(train["y"], stats["y"]),
        "v": _transform(train["v"], stats["v"]),
        "w": _transform(train["w"], stats["w"]),
        "y_struct": train["y_struct"],
    }
    grid_std = {
        "x": _transform(grid["x"], stats["x"]),
        "v": _transform(grid["v"], stats["v"]),
        "y_struct": grid["y_struct"],
    }
    return train_std, grid_std, stats


def _fixed_standardizer(mean, scale):
    return {
        "mean": np.array([[mean]], dtype=np.float32),
        "scale": np.array([[scale]], dtype=np.float32),
    }


def _standardize_demand_design_image_data(train, grid=None):
    stats = {
        "x": _fixed_standardizer(17.779, 3.7),
        "y": _fixed_standardizer(-292.1, 158.0),
    }
    train_std = {
        "x": _transform(train["x"], stats["x"]),
        "y": _transform(train["y"], stats["y"]),
        "v": train["v"].astype(np.float32),
        "w": train["w"].astype(np.float32),
        "y_struct": train["y_struct"].astype(np.float32),
    }
    grid_std = None
    if grid is not None:
        grid_std = {
            "x": _transform(grid["x"], stats["x"]),
            "v": grid["v"].astype(np.float32),
            "y_struct": grid["y_struct"].astype(np.float32),
        }
    return train_std, grid_std, stats


def _normalize_method_list(methods, field_name):
    if isinstance(methods, str):
        methods = [methods]
    elif methods is None:
        methods = []

    normalized = []
    for method in methods:
        method = str(method).strip()
        if not method:
            raise ValueError(f"`{field_name}` must not contain empty method names.")
        if method not in normalized:
            normalized.append(method)

    if not normalized:
        raise ValueError(f"`{field_name}` must contain at least one method.")
    return tuple(normalized)


def _require_map_only_method_list(params, field_name):
    methods = _normalize_method_list(params.get(field_name, ["map"]), field_name)
    if methods != ("map",):
        raise ValueError(f"`{field_name}` must be exactly ['map']; got {list(methods)}.")
    params[field_name] = ["map"]
    return methods


def _require_map_only_method(params, field_name):
    method = str(params.get(field_name, "map")).strip()
    if not method:
        raise ValueError(f"`{field_name}` must not be empty; allowed value is 'map'.")
    if method != "map":
        raise ValueError(f"`{field_name}` must be 'map'; got {method!r}.")
    params[field_name] = "map"
    return method


# Structural readouts reported per repeat:
#   map      MAP refinement of the encoder latent under p(z | v)  (paired point readout)
#   encoder  amortized encoder latent e(v)                         (point readout)
#   mcmc     full-grid posterior/generalized-Gibbs integral
_ALLOWED_STRUCTURAL_METHODS = ("map", "encoder", "mcmc")


def _require_structural_method_list(params, field_name):
    methods = _normalize_method_list(params.get(field_name, ["map"]), field_name)
    invalid = [m for m in methods if m not in _ALLOWED_STRUCTURAL_METHODS]
    if invalid:
        raise ValueError(
            f"`{field_name}` supports {list(_ALLOWED_STRUCTURAL_METHODS)}; "
            f"got {invalid}."
        )
    if "map" not in methods:
        raise ValueError(
            f"`{field_name}` must include 'map' (the paired point readout); "
            f"got {list(methods)}."
        )
    params[field_name] = list(methods)
    return methods


def _validate_map_only_structural_config(params):
    _require_structural_method_list(params, "structural_methods")
    _require_map_only_method_list(params, "training_structural_methods")
    _require_map_only_method(params, "training_structural_monitor_method")
    _require_map_only_method(params, "structural_latent_method")


def _resolve_structural_methods(params):
    return _require_structural_method_list(params, "structural_methods")


def _resolve_training_monitor_methods(params, structural_methods):
    training_methods = _require_map_only_method_list(
        params, "training_structural_methods"
    )
    monitor_method = _require_map_only_method(
        params, "training_structural_monitor_method"
    )
    return monitor_method, training_methods


def _make_structural_monitor_callback(
    grid_x,
    grid_v,
    y_true,
    latent_method="map",
    y_stats=None,
    additional_methods=None,
):
    methods = [latent_method]
    for method in additional_methods or ():
        if method not in methods:
            methods.append(method)

    def callback(model, stage, epoch, metrics):
        results = {"structural_latent_method": latent_method}
        for method in methods:
            data_y_pred = model.predict_structural(
                grid_x,
                grid_v,
                latent_method=method,
            )
            if y_stats is not None:
                data_y_pred = _inverse_transform(data_y_pred, y_stats)
            structural_mse = float(np.mean((y_true - data_y_pred) ** 2))
            results[f"structural_mse_{method}"] = structural_mse
        results["structural_mse"] = results[f"structural_mse_{latent_method}"]
        return results

    return callback


def _maybe_structural_monitor_callback(params, grid_x, grid_v, y_true, y_stats=None):
    """Test-grid monitor during training ONLY when `training_grid_monitor` is on.

    The paper runs are grid-blind (fairness condition: no evaluation-grid
    quantity is computed before the final readout); the switch is recorded in
    the run config and in every results row.
    """
    if not bool(params.get("training_grid_monitor", False)):
        return None
    structural_methods = _resolve_structural_methods(params)
    monitor_method, training_methods = _resolve_training_monitor_methods(
        params,
        structural_methods,
    )
    additional = [method for method in training_methods if method != monitor_method]
    return _make_structural_monitor_callback(
        grid_x,
        grid_v,
        y_true,
        latent_method=monitor_method,
        y_stats=y_stats,
        additional_methods=additional,
    )


def _print_training_history(history):
    text = _render_training_history(history)
    if text:
        print(f"\n{text}")


_MODEL_CLASS_BY_DATASET = {
    "Sim_Demand_Design_IV": BGM_IV,
    "Sim_Demand_Design_Mnist_IV": BGM_IV_Image,
    "Sim_Demand_Design_Vector_IV": BGM_IV_Vector,
}

_MCMC_FAMILY_BY_DATASET = {
    "Sim_Demand_Design_IV": "demand",
    "Sim_Demand_Design_Mnist_IV": "mnist_pixel",
    "Sim_Demand_Design_Vector_IV": "vector",
}


def _model_class_for_dataset(dataset):
    try:
        return _MODEL_CLASS_BY_DATASET[dataset]
    except KeyError:
        raise ValueError(f"Unsupported demand-design dataset: {dataset}") from None


def _model_random_seed(params):
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    return run_seed if bool(params.get("deterministic_training", False)) else None


_MCMC_ONLY_TIMESTAMP_KEY = "_mcmc_only_timestamp"


def _mcmc_only_timestamp(params):
    """Return a validated restore timestamp or ``None`` for training mode."""
    if _MCMC_ONLY_TIMESTAMP_KEY not in params:
        return None
    raw_timestamp = params[_MCMC_ONLY_TIMESTAMP_KEY]
    if not isinstance(raw_timestamp, str):
        raise ValueError("--mcmc-only requires a non-empty TIMESTAMP")
    timestamp = raw_timestamp.strip()
    if not timestamp:
        raise ValueError("--mcmc-only requires a non-empty TIMESTAMP")
    return timestamp


def _is_mcmc_only(params):
    return _mcmc_only_timestamp(params) is not None


def _uses_egm_multistart(params):
    return (
        not _is_mcmc_only(params)
        and int(params.get("egm_num_warm_starts", 1)) > 1
    )


def _validate_egm_multistart_run_shape(params):
    """Require one concrete outer cell per multistart coordinator process."""
    if not _uses_egm_multistart(params):
        return
    if int(params.get("num_tasks", 1)) != 1:
        raise ValueError("EGM multistart requires -t 1 (one GPU per outer cell)")
    for field in ("n_samples", "rho"):
        if isinstance(params.get(field), (list, tuple)):
            raise ValueError(
                f"EGM multistart requires scalar {field}; use --set {field}=..."
            )
    n_repeat = int(params.get("n_repeat", 1))
    if n_repeat > 1 and params.get("_only_repeat_id") is None:
        raise ValueError(
            "EGM multistart requires one repeat; pass --repeat-id or set n_repeat=1"
        )


def _initialize_egm_candidate_worker(use_gpu):
    """Configure one isolated EGM candidate process."""
    _configure_tensorflow_threads(1, 1)
    _configure_tensorflow_devices(
        bool(use_gpu),
        gpu_slot=0 if use_gpu else None,
        strict_memory_growth=bool(use_gpu),
    )
    if use_gpu and not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("EGM multistart requested GPU but the worker sees no GPU")


def _wait_for_egm_candidate_start_barrier(
    barrier_dir, candidate_id, expected_workers, timeout_seconds=300
):
    """Synchronize candidate starts through auditable ready files."""
    barrier_path = Path(barrier_dir)
    barrier_path.mkdir(parents=True, exist_ok=True)
    ready_path = barrier_path / f"candidate_{int(candidate_id):02d}.ready.json"
    ready_payload = {
        "candidate_id": int(candidate_id),
        "worker_pid": os.getpid(),
        "ready_at": datetime.utcnow().isoformat(timespec="microseconds") + "Z",
    }
    temporary_ready_path = ready_path.with_suffix(ready_path.suffix + ".tmp")
    temporary_ready_path.write_text(
        json.dumps(ready_payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_ready_path, ready_path)
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        ready_files = list(barrier_path.glob("candidate_*.ready.json"))
        if len(ready_files) == int(expected_workers):
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in ready_files]
            pids = {int(payload["worker_pid"]) for payload in payloads}
            candidate_ids = {int(payload["candidate_id"]) for payload in payloads}
            if len(pids) != int(expected_workers):
                raise RuntimeError("EGM start barrier did not receive unique worker PIDs")
            if candidate_ids != set(range(int(expected_workers))):
                raise RuntimeError("EGM start barrier candidate IDs are incomplete")
            return ready_payload
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"EGM start barrier timed out with {len(ready_files)}/"
                f"{expected_workers} workers ready"
            )
        time.sleep(0.1)


def _evaluate_training_iv_criterion(model, train, *, y_raw, y_stats):
    """Training-set IV-moment residual of a trained model (the selection rule).

    Same statistic as ``_evaluate_holdout_criterion`` -- the observed outcome
    against ``E[f(X, z) | w, z]`` with ``z`` inferred from ``v`` alone -- but
    evaluated on the training rows the model was fitted on, so no additional
    data enters the selection.  ``y_raw`` is the observed training outcome in
    original units; ``train`` holds the (model-space) covariates.
    """
    observed = np.asarray(y_raw, np.float64).reshape(-1)
    if observed.shape[0] != np.asarray(train["v"]).shape[0]:
        raise ValueError("criterion outcome rows must match the training rows")
    scores = _evaluate_holdout_criterion(
        model,
        {"v": train["v"], "w": train["w"], "y": observed},
        y_stats=y_stats,
    )
    return {
        "train_iv_map": float(scores["holdout_iv_mse_map"]),
        "train_iv_encoder": float(scores["holdout_iv_mse_encoder"]),
    }


def _run_egm_candidate_worker(
    candidate_id,
    params,
    train,
    *,
    init_seed,
    schedule_seed,
    post_egm_seed,
    criterion_seed,
    criterion_y_raw,
    criterion_y_stats,
    evaluation_iterations,
    candidate_root,
    data_hash,
    config_hash,
    code_commit,
    barrier_dir,
    expected_workers,
):
    """Train one warm start end to end in a spawned process.

    Stage 1 (EGM): initialize with ``init_seed``, train with the shared
    ``schedule_seed`` and record the tail-window score (diagnostic only).
    Stage 2 (BGM): from the persisted EGM state, rebuild a fresh model under
    the shared ``post_egm_seed`` exactly as the single-continuation parent used
    to, and run the BGM stage.
    Stage 3 (criterion): under ``criterion_seed`` evaluate the training fit and
    the training-set IV-moment residual (MAP and encoder readouts) and persist
    the post-BGM state.  The parent selects the candidate with the smallest
    MAP-readout residual and reports every readout from that one model.
    """
    candidate_root_path = Path(candidate_root)
    candidate_root_path.mkdir(parents=True, exist_ok=True)
    stdout_path = candidate_root_path / "candidate.stdout.log"
    stderr_path = candidate_root_path / "candidate.stderr.log"
    started_at = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
    try:
        _wait_for_egm_candidate_start_barrier(
            barrier_dir,
            candidate_id,
            expected_workers,
        )
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_handle, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_handle, contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
            candidate_params = dict(params)
            candidate_params["output_dir"] = str(candidate_root)
            candidate_params["save_model"] = True
            candidate_params["save_res"] = False
            run_seed = int(
                candidate_params.get("run_seed", candidate_params.get("seed", 0))
            )
            model_cls = _model_class_for_dataset(candidate_params["dataset"])
            model = model_cls(
                params=candidate_params,
                timestamp=f"egm_candidate_{int(candidate_id):02d}",
                random_seed=int(init_seed),
                auto_restore_checkpoint=False,
            )
            # Initialization is already materialized.  Reset every subsequent
            # stochastic training stream to the common schedule seed so only
            # network initialization differs between candidates.
            tf.keras.utils.set_random_seed(int(schedule_seed))
            np.random.seed(int(schedule_seed))
            score_history = model.egm_init(
                data=(train["x"], train["y"], train["v"], train["w"]),
                egm_n_iter=int(candidate_params.get("fit_egm_n_iter", 10000)),
                batch_size=int(candidate_params.get("fit_batch_size", 32)),
                egm_batches_per_eval=int(
                    candidate_params.get("fit_egm_batches_per_eval", 500)
                ),
                verbose=1,
                evaluation_callback=None,
                score_iterations=evaluation_iterations,
            )
            checkpoint_path = model.save_model_state_checkpoint(
                candidate_root_path / "egm-transition" / "ckpt"
            )
            checkpoint_hash = _checkpoint_files_hash(checkpoint_path)
            checkpoint_weight_hash = sha256_json(
                "egm-candidate-network-weights", _model_weight_hashes(model)
            )
            egm_training_history = json.loads(
                json.dumps(model.training_history, sort_keys=True, default=str)
            )
            scores = [record["full_train_l2_loss_y"] for record in score_history]
            egm_finite = len(scores) == len(evaluation_iterations) and all(
                np.isfinite(float(value)) for value in scores
            )
            logical_gpus = [
                device.name for device in tf.config.list_logical_devices("GPU")
            ]
            physical_devices = [
                f"{device.device_type}:{device.name}"
                for device in tf.config.list_physical_devices()
            ]
            device_names = logical_gpus or ["cpu"]
            device_hash = sha256_json(
                "egm-candidate-device",
                {
                    "tensorflow_version": tf.__version__,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "physical_devices": physical_devices,
                    "logical_training_devices": device_names,
                },
            )
            # Release the EGM-stage graph before the BGM continuation.
            del model
            tf.keras.backend.clear_session()
            gc.collect()

            # --- Stage 2: BGM continuation from the persisted EGM state. ------
            # Identical to the historical single-continuation parent: a fresh
            # model (fresh optimizers, no particles) under the shared
            # post-EGM seed, strict restore of the candidate's EGM weights, then
            # the shared post-EGM streams are restarted right before BGM.
            bgm_started = time.time()
            tf.keras.utils.set_random_seed(int(post_egm_seed))
            np.random.seed(int(post_egm_seed))
            model = model_cls(
                params=candidate_params,
                timestamp=f"bgm_candidate_{int(candidate_id):02d}",
                random_seed=int(post_egm_seed),
                auto_restore_checkpoint=False,
            )
            model.restore_model_state_checkpoint(checkpoint_path)
            restored_weight_hash = sha256_json(
                "egm-candidate-network-weights", _model_weight_hashes(model)
            )
            if restored_weight_hash != checkpoint_weight_hash:
                raise RuntimeError(
                    f"candidate {int(candidate_id)} EGM checkpoint weight identity mismatch"
                )
            tf.keras.utils.set_random_seed(int(post_egm_seed))
            np.random.seed(int(post_egm_seed))
            model.training_history = list(egm_training_history)
            model.egm_score_history = [
                {
                    "iteration": int(iteration),
                    "full_train_l2_loss_y": float(score),
                }
                for iteration, score in zip(evaluation_iterations, scores)
            ]
            model.fit_bgm_from_egm(
                data=(train["x"], train["y"], train["v"], train["w"]),
                epochs=int(candidate_params.get("fit_epochs", 100)),
                epochs_per_eval=int(candidate_params.get("fit_epochs_per_eval", 10)),
                batch_size=int(candidate_params.get("fit_batch_size", 32)),
                verbose=1,
                first_stage_warmup_epochs=int(
                    candidate_params.get("fit_first_stage_warmup_epochs", 30)
                ),
                evaluation_callback=None,
                initialize_latents_from_encoder=True,
                write_params=True,
            )
            bgm_seconds = time.time() - bgm_started

            # --- Stage 3: training-side criterion under its own seed. --------
            tf.keras.utils.set_random_seed(int(criterion_seed))
            np.random.seed(int(criterion_seed))
            _, train_mse_x, train_mse_y, train_mse_v = model.evaluate(
                data=(train["x"], train["y"], train["v"], train["w"]),
                data_z=None,
                nb_intervals=int(candidate_params.get("nb_intervals", 20)),
            )
            criterion = _evaluate_training_iv_criterion(
                model,
                train,
                y_raw=criterion_y_raw,
                y_stats=criterion_y_stats,
            )
            print(
                f"Candidate {int(candidate_id)} post-BGM criterion: "
                f"train_iv_map={criterion['train_iv_map']:.6f} "
                f"train_iv_encoder={criterion['train_iv_encoder']:.6f} "
                f"train_mse_y={float(train_mse_y):.6f}"
            )
            bgm_checkpoint_path = model.save_model_state_checkpoint(
                candidate_root_path / "bgm-final" / "ckpt"
            )
            bgm_checkpoint_hash = _checkpoint_files_hash(bgm_checkpoint_path)
            bgm_checkpoint_weight_hash = sha256_json(
                "bgm-candidate-network-weights", _model_weight_hashes(model)
            )
            criterion_finite = all(
                np.isfinite(float(value))
                for value in (
                    criterion["train_iv_map"],
                    criterion["train_iv_encoder"],
                    train_mse_x,
                    train_mse_y,
                    train_mse_v,
                )
            )
            if egm_finite and criterion_finite:
                status, failure_reason = "completed", None
            elif not egm_finite:
                status, failure_reason = (
                    "nonfinite_score",
                    "non-finite full-training EGM score",
                )
            else:
                status, failure_reason = (
                    "nonfinite_criterion",
                    "non-finite post-BGM training criterion",
                )
            finished_at = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
            manifest = make_candidate_manifest(
                candidate_id=int(candidate_id),
                init_seed=int(init_seed),
                schedule_seed=int(schedule_seed),
                run_seed=run_seed,
                evaluation_iterations=evaluation_iterations,
                full_train_l2_loss_y=scores,
                status=status,
                failure_reason=failure_reason,
                data_hash=data_hash,
                config_hash=config_hash,
                code_commit=code_commit,
                checkpoint_path=str(checkpoint_path),
                checkpoint_hash=checkpoint_hash,
                checkpoint_weight_hash=checkpoint_weight_hash,
                started_at=started_at,
                finished_at=finished_at,
                worker_pid=os.getpid(),
                device_names=device_names,
                device_hash=device_hash,
                bgm_checkpoint_path=str(bgm_checkpoint_path),
                bgm_checkpoint_hash=bgm_checkpoint_hash,
                bgm_checkpoint_weight_hash=bgm_checkpoint_weight_hash,
                criterion_seed=int(criterion_seed),
                train_iv_map=float(criterion["train_iv_map"]),
                train_iv_encoder=float(criterion["train_iv_encoder"]),
                train_mse_x=float(train_mse_x),
                train_mse_y=float(train_mse_y),
                train_mse_v=float(train_mse_v),
                bgm_seconds=float(bgm_seconds),
            )
            manifest_path = candidate_root_path / "candidate_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            training_history = json.loads(
                json.dumps(model.training_history, sort_keys=True, default=str)
            )
        return {
            "candidate_id": int(candidate_id),
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "training_history": training_history,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": None,
        }
    except Exception as exc:
        with stderr_path.open("a", encoding="utf-8") as stderr_handle:
            traceback.print_exc(file=stderr_handle)
        return {
            "candidate_id": int(candidate_id),
            "manifest": None,
            "manifest_path": None,
            "training_history": [],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    finally:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()


def _fit_demand_design_model_multistart(params, train, *, criterion_data=None):
    """Run every warm start through EGM and BGM concurrently, then select one.

    Selection is the deterministic argmin of the training-set IV-moment
    residual under the MAP readout (``train_iv_map``), computed by each
    candidate worker after its BGM stage.  ``criterion_data`` must provide the
    observed training outcome in original units (``y_raw``) and the outcome
    standardizer (``y_stats``; ``None`` when training happens in original
    units) so the criterion is the same statistic for every benchmark.
    """
    normalized = validate_multistart_config(params)
    observed_train = {
        key: np.asarray(train[key], np.float32)
        for key in ("x", "y", "v", "w")
    }
    num_starts = int(normalized["egm_num_warm_starts"])
    if num_starts <= 1:
        raise ValueError("multistart runner requires egm_num_warm_starts > 1")
    if int(normalized.get("num_tasks", 1)) != 1:
        raise ValueError("EGM multistart requires outer num_tasks=1 per GPU")
    if not isinstance(criterion_data, Mapping) or "y_raw" not in criterion_data:
        raise ValueError(
            "EGM multistart requires criterion_data={'y_raw': ..., 'y_stats': ...}"
        )
    criterion_y_raw = np.asarray(criterion_data["y_raw"], np.float64).reshape(-1)
    if criterion_y_raw.shape[0] != observed_train["y"].shape[0]:
        raise ValueError("criterion_data['y_raw'] must have one row per training row")
    criterion_y_stats = criterion_data.get("y_stats")

    dataset = str(normalized["dataset"])
    n_samples = int(normalized.get("n_samples", len(train["x"])))
    rho = float(normalized.get("rho", 0.5))
    repeat_id = int(normalized.get("repeat_id", 0))
    run_seed = int(normalized.get("run_seed", normalized.get("seed", 0)))
    seeds = derive_multistart_seeds(
        dataset,
        n_samples,
        rho,
        repeat_id,
        num_starts,
        run_seed=run_seed,
    )
    evaluation_iterations = score_evaluation_iterations(
        int(normalized.get("fit_egm_n_iter", 10000)),
        int(normalized.get("fit_egm_batches_per_eval", 500)),
    )
    data_hashes = _training_data_hashes(observed_train)
    data_hash = sha256_json("egm-multistart-training-data", data_hashes)
    config_hash = sha256_json(
        "egm-multistart-training-config", _manifest_params(normalized)
    )
    code_commit = _code_commit() or "uncommitted"
    bundle_id = (
        f"n={n_samples}_rho={format(rho, '.17g')}_repeat={repeat_id}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    )
    bundle_root = (
        Path(str(normalized.get("output_dir", ".")))
        / "egm_multistart"
        / bundle_id
    )
    bundle_root.mkdir(parents=True, exist_ok=False)
    barrier_dir = bundle_root / "start_barrier"

    print(
        f"Launching {num_starts} warm starts concurrently on one device; every "
        f"candidate runs EGM and BGM, then selector={EGM_SELECTOR_VERSION} "
        f"picks the smallest {EGM_SELECTION_CRITERION}."
    )
    spawn_context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=num_starts,
        mp_context=spawn_context,
        initializer=_initialize_egm_candidate_worker,
        initargs=(bool(normalized.get("use_gpu", False)),),
    ) as executor:
        futures = []
        for candidate_id, init_seed in enumerate(seeds["init_seeds"]):
            candidate_root = bundle_root / f"candidate_{candidate_id:02d}"
            futures.append(
                executor.submit(
                    _run_egm_candidate_worker,
                    candidate_id,
                    normalized,
                    observed_train,
                    init_seed=int(init_seed),
                    schedule_seed=int(seeds["schedule_seed"]),
                    post_egm_seed=int(seeds["post_egm_seed"]),
                    criterion_seed=int(seeds["criterion_seeds"][candidate_id]),
                    criterion_y_raw=criterion_y_raw,
                    criterion_y_stats=criterion_y_stats,
                    evaluation_iterations=evaluation_iterations,
                    candidate_root=str(candidate_root),
                    data_hash=data_hash,
                    config_hash=config_hash,
                    code_commit=code_commit,
                    barrier_dir=str(barrier_dir),
                    expected_workers=num_starts,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["candidate_id"])
    if [item["candidate_id"] for item in results] != list(range(num_starts)):
        raise RuntimeError("EGM candidate result IDs are incomplete or duplicated")
    for result in results:
        candidate_id = int(result["candidate_id"])
        if result.get("error"):
            error = result["error"]
            raise RuntimeError(
                f"EGM candidate {candidate_id} failed with "
                f"{error['type']}: {error['message']}"
            )
        print(
            f"EGM candidate {candidate_id} completed; "
            f"log={result['stdout_path']}"
        )
        if not verify_manifest_hash(
            result["manifest"],
            hash_field="candidate_manifest_hash",
            namespace="egm-candidate-manifest",
        ):
            raise RuntimeError(
                f"EGM candidate {candidate_id} manifest hash mismatch"
            )
        manifest = result["manifest"]
        expected_fields = {
            "candidate_id": candidate_id,
            "init_seed": int(seeds["init_seeds"][candidate_id]),
            "schedule_seed": int(seeds["schedule_seed"]),
            "run_seed": run_seed,
            "evaluation_iterations": list(evaluation_iterations),
            "data_hash": data_hash,
            "config_hash": config_hash,
            "code_commit": code_commit,
        }
        mismatches = [
            key for key, value in expected_fields.items()
            if manifest.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"EGM candidate {candidate_id} identity mismatch: {mismatches}"
            )
        if manifest.get("status") not in {
            "completed",
            "nonfinite_score",
            "nonfinite_criterion",
        }:
            raise RuntimeError(f"EGM candidate {candidate_id} has invalid status")
        if bool(normalized.get("use_gpu", False)) and not any(
            "GPU" in str(name).upper() for name in manifest.get("device_names", [])
        ):
            raise RuntimeError(f"EGM candidate {candidate_id} did not run on GPU")
        for stage, path_key, hash_key, weight_key in (
            ("EGM", "checkpoint_path", "checkpoint_hash", "checkpoint_weight_hash"),
            (
                "BGM",
                "bgm_checkpoint_path",
                "bgm_checkpoint_hash",
                "bgm_checkpoint_weight_hash",
            ),
        ):
            checkpoint_path = manifest.get(path_key)
            if not checkpoint_path or not Path(str(checkpoint_path) + ".index").is_file():
                raise RuntimeError(
                    f"{stage} candidate {candidate_id} checkpoint is missing"
                )
            if _checkpoint_files_hash(checkpoint_path) != manifest.get(hash_key):
                raise RuntimeError(
                    f"{stage} candidate {candidate_id} checkpoint file hash mismatch"
                )
            if not manifest.get(weight_key):
                raise RuntimeError(
                    f"{stage} candidate {candidate_id} checkpoint weight hash is missing"
                )
        if int(manifest.get("criterion_seed") or -1) != int(
            seeds["criterion_seeds"][candidate_id]
        ):
            raise RuntimeError(
                f"candidate {candidate_id} criterion seed differs from the derived one"
            )
        if not manifest.get("device_hash"):
            raise RuntimeError(f"EGM candidate {candidate_id} device hash is missing")
        if manifest.get("data_hash") != data_hash:
            raise RuntimeError("EGM candidates do not share one training-data hash")

    if len({result["manifest"]["device_hash"] for result in results}) != 1:
        raise RuntimeError("EGM candidates do not share one device identity hash")

    candidate_criteria = {
        result["candidate_id"]: result["manifest"].get(EGM_SELECTION_CRITERION)
        for result in results
    }
    egm_tail_scores = {
        result["candidate_id"]: result["manifest"].get("tail_mean_score")
        for result in results
    }
    selection = select_candidate_by_criterion(
        candidate_criteria, egm_tail_scores=egm_tail_scores
    )
    selection_path = bundle_root / "selection_manifest.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected_id = int(selection["selected_candidate_id"])
    selected = next(item for item in results if item["candidate_id"] == selected_id)
    selected_checkpoint = str(selected["manifest"]["bgm_checkpoint_path"])
    if _checkpoint_files_hash(selected_checkpoint) != selected["manifest"]["bgm_checkpoint_hash"]:
        raise RuntimeError("selected BGM checkpoint file hash mismatch")

    # Candidate workers have exited and released their contexts. Configure the
    # parent only now, then restore the selected post-BGM state into the one
    # model that every structural readout is reported from.  No further
    # training happens in the parent.
    _configure_tensorflow_devices(
        bool(normalized.get("use_gpu", False)),
        strict_memory_growth=bool(normalized.get("use_gpu", False)),
    )
    if bool(normalized.get("use_gpu", False)) and not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("EGM multistart requested GPU but the parent sees no GPU")
    tf.keras.utils.set_random_seed(int(seeds["post_egm_seed"]))
    np.random.seed(int(seeds["post_egm_seed"]))
    model_cls = _model_class_for_dataset(dataset)
    model = model_cls(
        params=normalized,
        random_seed=int(seeds["post_egm_seed"]),
        auto_restore_checkpoint=False,
    )
    model.restore_model_state_checkpoint(selected_checkpoint)
    restored_weight_hash = sha256_json(
        "bgm-candidate-network-weights", _model_weight_hashes(model)
    )
    if restored_weight_hash != selected["manifest"]["bgm_checkpoint_weight_hash"]:
        raise RuntimeError("selected BGM checkpoint weight identity mismatch")
    # Model construction materializes fresh variables and Gaussian_sampler
    # historically resets NumPy.  Restore is complete now; restart the shared
    # post-EGM streams so the parent-side evaluation is reproducible.
    tf.keras.utils.set_random_seed(int(seeds["post_egm_seed"]))
    np.random.seed(int(seeds["post_egm_seed"]))
    model.training_history = list(selected.get("training_history") or [])
    model.egm_score_history = [
        {
            "iteration": int(iteration),
            "full_train_l2_loss_y": float(score),
        }
        for iteration, score in zip(
            selected["manifest"]["evaluation_iterations"],
            selected["manifest"]["full_train_l2_loss_y"],
        )
    ]
    model.egm_multistart_provenance = {
        "egm_num_warm_starts": num_starts,
        "egm_selector_version": EGM_SELECTOR_VERSION,
        "egm_selection_criterion": EGM_SELECTION_CRITERION,
        "egm_selected_candidate_id": selected_id,
        "egm_selected_criterion": float(selection["selected_criterion"]),
        "egm_selected_train_iv_map": float(selected["manifest"]["train_iv_map"]),
        "egm_selected_train_iv_encoder": float(
            selected["manifest"]["train_iv_encoder"]
        ),
        "egm_selected_train_mse_y": float(selected["manifest"]["train_mse_y"]),
        "egm_selected_egm_tail_rank": selection.get("selected_egm_tail_rank"),
        "egm_candidate_criteria_hash": sha256_json(
            "egm-candidate-criteria",
            {"candidate_criteria": selection["candidate_criteria"]},
        ),
        "egm_selection_manifest_hash": selection["selection_manifest_hash"],
        "egm_selection_manifest_path": str(selection_path),
        "egm_candidate_manifest_hashes": [
            result["manifest"]["candidate_manifest_hash"] for result in results
        ],
        "data_hash": data_hash,
        "config_hash": config_hash,
        "run_seed": run_seed,
        "init_seeds": [int(value) for value in seeds["init_seeds"]],
        "schedule_seed": int(seeds["schedule_seed"]),
        "post_egm_seed": int(seeds["post_egm_seed"]),
        "criterion_seeds": [int(value) for value in seeds["criterion_seeds"]],
        "uses_validation": False,
        "uses_holdout": False,
        "uses_test_grid": False,
    }
    print(
        "Multistart selected candidate "
        f"{selected_id} with {EGM_SELECTION_CRITERION}="
        f"{float(selection['selected_criterion']):.6f} "
        f"(EGM tail rank {selection.get('selected_egm_tail_rank')})."
    )
    return model


def _fit_demand_design_model(
    params, train, evaluation_callback=None, criterion_data=None
):
    if _uses_egm_multistart(params):
        if evaluation_callback is not None:
            raise ValueError("EGM multistart selection cannot receive a grid callback")
        return _fit_demand_design_model_multistart(
            params, train, criterion_data=criterion_data
        )
    model_cls = _model_class_for_dataset(params["dataset"])
    random_seed = _model_random_seed(params)
    model = model_cls(
        params=params,
        random_seed=random_seed,
        auto_restore_checkpoint=False,
    )
    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=int(params.get("fit_epochs", 100)),
        epochs_per_eval=int(params.get("fit_epochs_per_eval", 10)),
        batch_size=int(params.get("fit_batch_size", 32)),
        use_egm_init=True,
        egm_n_iter=int(params.get("fit_egm_n_iter", 10000)),
        egm_batches_per_eval=int(params.get("fit_egm_batches_per_eval", 500)),
        verbose=1,
        first_stage_warmup_epochs=int(params.get("fit_first_stage_warmup_epochs", 30)),
        evaluation_callback=evaluation_callback,
    )
    return model


def _restore_demand_design_model(params, timestamp, *, train=None, manifest_extra=None):
    """Strictly restore a pinned optimizer-free inference-state checkpoint."""
    timestamp = str(timestamp).strip()
    if not timestamp:
        raise ValueError("--mcmc-only requires a non-empty TIMESTAMP")
    manifest = _load_training_manifest(params, timestamp)
    model_cls = _model_class_for_dataset(params["dataset"])
    random_seed = _model_random_seed(params)
    model = model_cls(
        params=params,
        timestamp=timestamp,
        random_seed=random_seed,
        auto_restore_checkpoint=False,
    )
    if str(getattr(model, "timestamp", "")) != str(timestamp):
        raise RuntimeError("restored model timestamp differs from the requested one")
    state_checkpoint, state_payload = _resolve_inference_state_checkpoint(
        model, manifest
    )
    model.restore_model_state_checkpoint(state_checkpoint)
    if _model_state_hashes(model) != state_payload.get("state_hashes"):
        raise RuntimeError("restored inference-state identity mismatch")
    if _checkpoint_identity(model) != state_payload.get("checkpoint_identity"):
        raise RuntimeError("restored checkpoint identity mismatch")
    model.training_history = []
    manifest = _verify_training_manifest(model, params, train, manifest_extra)
    multistart = (manifest.get("notes") or {}).get("egm_multistart")
    if multistart:
        model.egm_multistart_provenance = multistart
    return model


def _fit_or_restore_demand_design_model(
    params,
    train,
    evaluation_callback=None,
    manifest_extra=None,
    manifest_notes=None,
    criterion_data=None,
):
    """Train (or restore for ``--mcmc-only``) the one model of an outer cell.

    ``criterion_data`` carries the observed training outcome in original units
    and its standardizer; the multistart runner uses it to compute the
    training-set IV-moment selection criterion for every warm start.
    """
    timestamp = _mcmc_only_timestamp(params)
    if timestamp is not None:
        print(f"Restoring checkpoint {timestamp} (mcmc-only; no training) ...")
        return _restore_demand_design_model(
            params, timestamp, train=train, manifest_extra=manifest_extra
        )
    model = _fit_demand_design_model(
        params,
        train,
        evaluation_callback=evaluation_callback,
        criterion_data=criterion_data,
    )
    multistart = getattr(model, "egm_multistart_provenance", None)
    if multistart:
        merged_notes = dict(manifest_notes or {})
        merged_notes["egm_multistart"] = multistart
        manifest_notes = merged_notes
    _write_training_manifest(model, params, train, manifest_extra, manifest_notes)
    return model


# --- training manifest -------------------------------------------------------
# Written next to every saved checkpoint; mcmc-only refuses a checkpoint
# whose manifest does not match the run that tries to use it.

_MANIFEST_EXCLUDED_KEYS = frozenset(
    {
        "output_dir",
        "save_res",
        "save_model",
        "use_gpu",
        "num_tasks",
        "n_repeat",
        "structural_methods",
        "mcmc_family",
        "training_grid_monitor",
        "training_structural_methods",
        "training_structural_monitor_method",
        "nb_intervals",
    }
)


def _manifest_params(params):
    """Training-relevant parameters as resolved by the model, JSON-canonical
    (run-control keys dropped)."""
    kept = {
        key: value
        for key, value in params.items()
        if not str(key).startswith("_") and key not in _MANIFEST_EXCLUDED_KEYS
    }
    # Preserve exact compatibility with manifests written before multistart:
    # an absent field and the normalized legacy default of one warm start
    # describe the same estimator and must hash identically.
    if int(kept.get("egm_num_warm_starts", 1)) == 1:
        kept.pop("egm_num_warm_starts", None)
    return json.loads(json.dumps(kept, sort_keys=True, default=str))


def _training_data_hashes(train):
    return {
        key: sha256_array(np.asarray(train[key], np.float32))
        for key in ("x", "y", "v", "w")
        if key in train
    }


def _code_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _training_manifest_path(model):
    return Path(model.checkpoint_path) / "manifest.json"


def _load_training_manifest(params, timestamp):
    path = (
        Path(str(params.get("output_dir", ".")))
        / "checkpoints"
        / str(params["dataset"])
        / str(timestamp)
        / "manifest.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint {path.parent} has no training manifest; it cannot be evaluated"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_training_manifest(
    model,
    params,
    train,
    extra=None,
    notes=None,
    inference_state=None,
):
    payload = {
        "schema_version": "bgm-training-manifest",
        "dataset": str(params["dataset"]),
        "checkpoint_timestamp": str(model.timestamp),
        "checkpoint_identity": _checkpoint_identity(model),
        "weights": _model_weight_hashes(model),
        "params": _manifest_params(model.params),
        "data": _training_data_hashes(train),
        "extra": json.loads(json.dumps(extra or {}, sort_keys=True, default=str)),
        "notes": json.loads(json.dumps(notes or {}, sort_keys=True, default=str)),
        "inference_state": json.loads(
            json.dumps(inference_state or {}, sort_keys=True, default=str)
        ),
        "code_commit": _code_commit(),
        "execution_environment": execution_environment(),
    }
    payload["params_hash"] = sha256_json("training-manifest-params", payload["params"])
    return payload


def _write_training_manifest(model, params, train, extra=None, notes=None):
    """Write the manifest once; a second write for the same checkpoint must agree."""
    if not bool(params.get("save_model")):
        return None
    path = _training_manifest_path(model)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        inference_state = existing.get("inference_state")
        if not isinstance(inference_state, dict) or not inference_state:
            raise RuntimeError(
                f"training manifest has no supported inference state: {path}"
            )
        _, resolved_state = _resolve_inference_state_checkpoint(model, existing)
        if _model_state_hashes(model) != resolved_state.get("state_hashes"):
            raise RuntimeError(f"training inference state differs from model: {path}")
        payload = _build_training_manifest(
            model,
            params,
            train,
            extra,
            notes,
            inference_state=inference_state,
        )
        comparable = {
            key: existing.get(key)
            for key in (
                "checkpoint_identity",
                "weights",
                "params_hash",
                "data",
                "extra",
                "notes",
                "code_commit",
                "inference_state",
            )
        }
        if comparable != {key: payload[key] for key in comparable}:
            raise RuntimeError(f"training manifest already exists and differs: {path}")
        return existing
    inference_state = _save_inference_state_checkpoint(model)
    payload = _build_training_manifest(
        model,
        params,
        train,
        extra,
        notes,
        inference_state=inference_state,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def _verify_training_manifest(model, params, train=None, extra=None):
    path = _training_manifest_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint {model.checkpoint_path} has no training manifest; it cannot be evaluated"
        )
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    current_params = _manifest_params(model.params)
    expected = {
        "dataset": str(params["dataset"]),
        "checkpoint_timestamp": str(model.timestamp),
        "checkpoint_identity": _checkpoint_identity(model),
        "weights": _model_weight_hashes(model),
        "params_hash": sha256_json("training-manifest-params", current_params),
    }
    if train is not None:
        expected["data"] = _training_data_hashes(train)
    if extra is not None:
        expected["extra"] = json.loads(json.dumps(extra, sort_keys=True, default=str))
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if "params_hash" in mismatches:
        recorded = manifest.get("params") or {}
        differing = sorted(
            key for key in set(recorded) | set(current_params)
            if recorded.get(key) != current_params.get(key)
        )
        mismatches[mismatches.index("params_hash")] = f"params{differing}"
    if mismatches:
        raise RuntimeError(
            f"training manifest mismatch for {model.checkpoint_path}: {mismatches}"
        )
    return manifest


def _resolve_mcmc_family(params):
    family = params.get("mcmc_family") or _MCMC_FAMILY_BY_DATASET.get(params["dataset"])
    if family is None:
        raise ValueError(f"{params['dataset']!r} has no MCMC family")
    family = str(family)
    if family not in FAMILY_RECIPES:
        raise ValueError(
            f"`mcmc_family` must be one of {sorted(FAMILY_RECIPES)}; got {family!r}."
        )
    return family


def _model_weight_hashes(model):
    return {
        name: sha256_weights(getattr(model, f"{name}_net"))
        for name in ("g", "e", "f", "h")
    }


def _checkpoint_files_hash(checkpoint_prefix):
    """Hash the immutable TensorFlow files belonging to one checkpoint prefix."""
    prefix = Path(checkpoint_prefix)
    paths = sorted(prefix.parent.glob(prefix.name + ".*"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"checkpoint files are missing for prefix {prefix}")
    files = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "suffix": path.name[len(prefix.name) :],
                "size": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    return sha256_json("tensorflow-checkpoint-files", {"files": files})


def _model_state_hashes(model):
    """Hashes for every value in the optimizer-free model-state checkpoint."""
    return {
        **_model_weight_hashes(model),
        "dz": sha256_weights(model.dz_net),
        "egm_sigma2_x_ema": sha256_array(model.egm_sigma2_x_ema.numpy()),
    }


def _save_inference_state_checkpoint(model):
    checkpoint_root = Path(model.checkpoint_path)
    checkpoint_prefix = Path(
        model.save_model_state_checkpoint(
            checkpoint_root / "inference-state" / "ckpt"
        )
    )
    relative_prefix = checkpoint_prefix.relative_to(checkpoint_root)
    payload = {
        "checkpoint_kind": "inference-state",
        "relative_prefix": relative_prefix.as_posix(),
        "checkpoint_hash": _checkpoint_files_hash(checkpoint_prefix),
        "checkpoint_identity": _checkpoint_identity(model),
        "state_hashes": _model_state_hashes(model),
    }
    payload["inference_state_hash"] = sha256_json(
        "bgm-inference-state", payload
    )
    return payload


def _resolve_inference_state_checkpoint(model, manifest):
    payload = manifest.get("inference_state")
    if not isinstance(payload, dict):
        raise RuntimeError(
            "checkpoint manifest has no supported inference-state checkpoint"
        )
    if payload.get("checkpoint_kind") != "inference-state":
        raise RuntimeError("checkpoint manifest has an invalid inference-state kind")
    recorded_hash = payload.get("inference_state_hash")
    unhashed = dict(payload)
    unhashed.pop("inference_state_hash", None)
    if recorded_hash != sha256_json("bgm-inference-state", unhashed):
        raise RuntimeError("inference-state manifest hash mismatch")

    relative_prefix = Path(str(payload.get("relative_prefix", "")))
    if (
        not relative_prefix.parts
        or relative_prefix.is_absolute()
        or ".." in relative_prefix.parts
    ):
        raise RuntimeError("inference-state checkpoint path is not relative and safe")
    checkpoint_root = Path(model.checkpoint_path).resolve()
    checkpoint_prefix = (checkpoint_root / relative_prefix).resolve()
    if checkpoint_root not in checkpoint_prefix.parents:
        raise RuntimeError("inference-state checkpoint escapes its checkpoint root")
    if not Path(str(checkpoint_prefix) + ".index").is_file():
        raise FileNotFoundError(
            f"inference-state checkpoint is missing: {checkpoint_prefix}"
        )
    if _checkpoint_files_hash(checkpoint_prefix) != payload.get("checkpoint_hash"):
        raise RuntimeError("inference-state checkpoint file hash mismatch")
    return checkpoint_prefix, payload


def _checkpoint_identity(model):
    weights = _model_weight_hashes(model)
    return sha256_json(
        "bgm-checkpoint-identity",
        {"timestamp": str(model.timestamp), **weights},
    )


_PROVENANCE_PARAM_KEYS = (
    "outcome_to_particles_weight",
    "covariate_block_scale",
    "sigma_v",
    "sigma_v_softfloor",
    "sigma_time",
    "sigma_time_softfloor",
    "sigma_vector",
    "sigma_vector_softfloor",
    "vector_blocks",
    "sigma_y_softfloor",
    "deterministic_training",
    "training_grid_monitor",
    "egm_num_warm_starts",
    "structural_methods",
    "mcmc_family",
    "z_dims",
    "v_dim",
    "vector_dim",
    "holdout_seed_offset",
)


def _run_provenance(params, model, extra=None):
    environment = execution_environment()
    gpus = environment.get("gpus") or []
    provenance = {
        "schema_version": "bgm-run-provenance",
        "dataset": params.get("dataset"),
        "repeat_id": int(params.get("repeat_id", 0)),
        "run_seed": int(params.get("run_seed", params.get("seed", 0))),
        "checkpoint_timestamp": str(getattr(model, "timestamp", "")),
        "checkpoint_path": str(getattr(model, "checkpoint_path", "")),
        "checkpoint_identity": _checkpoint_identity(model),
        "weights": _model_weight_hashes(model),
        "params": {
            key: params.get(key) for key in _PROVENANCE_PARAM_KEYS if key in params
        },
        "resolved_gamma": float(model._outcome_to_particles_weight()),
        "device_name": (gpus[0]["name"] if gpus else "cpu"),
        "execution_environment": environment,
        "mcmc_only": _is_mcmc_only(params),
    }
    multistart = getattr(model, "egm_multistart_provenance", None)
    if multistart:
        provenance["egm_multistart"] = json.loads(
            json.dumps(multistart, sort_keys=True, default=str)
        )
    if extra:
        provenance.update(extra)
    return provenance


def _standardizer_scalars(stats):
    if stats is None:
        return 0.0, 1.0
    return (
        float(np.asarray(stats["mean"]).reshape(-1)[0]),
        float(np.asarray(stats["scale"]).reshape(-1)[0]),
    )


def _mcmc_context(
    params,
    *,
    grid_x_model,
    grid_v_raw,
    truth_rows,
    truth_label,
    preprocessor,
    x_stats,
    y_stats,
):
    return {
        "params": params,
        "family": _resolve_mcmc_family(params),
        "grid_x_model": grid_x_model,
        "grid_v_raw": grid_v_raw,
        "truth_rows": truth_rows,
        "truth_label": truth_label,
        "preprocessor": preprocessor,
        "x_stats": x_stats,
        "y_stats": y_stats,
    }


def _run_structural_mcmc(
    model,
    params,
    *,
    family,
    grid_x_model,
    grid_v_raw,
    truth_rows,
    truth_label,
    preprocessor,
    x_stats,
    y_stats,
):
    """Run ungated all-draw MCMC on the complete evaluation grid."""
    grid_x_full = np.asarray(grid_x_model, np.float32).reshape(-1, 1)
    grid_v_full = np.asarray(grid_v_raw, np.float32)
    truth_full = np.asarray(truth_rows, np.float64).reshape(-1)
    y_shift, y_scale = _standardizer_scalars(y_stats)
    x_shift, x_scale = _standardizer_scalars(x_stats)
    repeat_id = int(params.get("repeat_id", 0))
    inference_options = params.get(_MCMC_INFERENCE_OPTIONS_KEY) or {}
    record = run_mcmc_grid(
        model,
        family=family,
        grid_x_model=grid_x_full,
        grid_v_raw=grid_v_full,
        preprocessor=preprocessor,
        truth_original_units=truth_full,
        truth_label=str(truth_label),
        outcome_shift=y_shift,
        outcome_scale=y_scale,
        treatment_transform={"shift": x_shift, "scale": x_scale},
        data_seed=int(params.get("run_seed", params.get("seed", 0))),
        checkpoint_identity=_checkpoint_identity(model),
        run_label=f"{params['dataset']}|repeat{repeat_id}|{model.timestamp}|{family}",
        production_num_chains=inference_options.get("production_num_chains"),
        production_warmup_steps=inference_options.get("production_warmup_steps"),
        production_draws=inference_options.get("production_draws"),
        artifact_root=inference_options.get("artifact_root"),
        arm_id=inference_options.get("arm_id"),
        readout_prefixes=inference_options.get("readout_prefixes"),
        readout_artifact_manifest=inference_options.get(
            "readout_artifact_manifest"
        ),
    )
    reference_metrics = inference_options.get("reference_metrics") or {}
    if reference_metrics:
        record["reference_metrics"] = dict(reference_metrics)
    return record


def _evaluate_structural_methods(
    model,
    grid_x,
    grid_v,
    y_true,
    methods,
    y_stats=None,
    mcmc_context=None,
):
    results = {}
    for method in methods:
        if method == "mcmc":
            if mcmc_context is None:
                raise ValueError("`mcmc` needs a full-grid inference context")
            record = _run_structural_mcmc(model, **mcmc_context)
            mse = float(record["readout"]["structural_mse_plugin"])
            results["mcmc"] = mse
            results["_mcmc"] = record
            print(
                f"Structural MSE [mcmc] = {mse:.6f} "
                f"[{record['grid']['num_queries']} full-grid queries, "
                f"{record['readout']['num_components']} retained draws]"
            )
            continue
        data_y_pred = model.predict_structural(
            grid_x,
            grid_v,
            latent_method=method,
        )
        if y_stats is not None:
            data_y_pred = _inverse_transform(data_y_pred, y_stats)
        mse = float(np.mean((y_true - data_y_pred) ** 2))
        results[method] = mse
        print(f"Structural MSE [{method}] = {mse:.6f}")
    return results


def _evaluate_holdout_criterion(model, holdout, y_stats=None):
    """Training-side held-out instrument-moment MSE (the gamma selection criterion).

    Held-out rows are a fresh draw of the TRAIN split (simulation seed
    run_seed + holdout_seed_offset).  The score compares the OBSERVED outcome
    with the model prediction integrated over the treatment model given the
    instrument, E[f(X, z) | w, v] with z inferred from v alone; under
    instrument validity E[Y | w, v] = E[g0 | w, v], so no structural ground
    truth enters.  The rows never touch the evaluation grid and the score is
    reported, never used for selection inside a run.
    """
    observed = np.asarray(holdout["y"], np.float64).reshape(-1)
    data_w = tf.constant(np.asarray(holdout["w"], np.float32).reshape(observed.shape[0], -1))
    out = {}
    for method in ("map", "encoder"):
        data_z = model.infer_latent_from_covariates(holdout["v"], method=method)
        prediction = model._integrated_outcome_mean(
            tf.constant(np.asarray(data_z, np.float32)), data_w, sample_y=False
        ).numpy()
        if y_stats is not None:
            prediction = _inverse_transform(prediction, y_stats)
        out[f"holdout_iv_mse_{method}"] = float(
            np.mean((observed - np.asarray(prediction, np.float64).reshape(-1)) ** 2)
        )
        print(f"Held-out IV-moment MSE [{method}] = {out[f'holdout_iv_mse_{method}']:.6f}")
    return out


def _resolve_holdout_settings(params):
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    holdout_seed = run_seed + int(params.get("holdout_seed_offset", 1000))
    holdout_n = params.get("holdout_n_samples") or params.get("n_samples", 5000)
    return holdout_seed, int(holdout_n)


def _finalize_demand_design_run(
    params,
    model,
    *,
    train_model,
    grid_x_model,
    grid_v_model,
    grid_truth,
    y_stats,
    methods,
    mcmc_context,
    holdout,
    run_config_text,
    ranges_text,
    space_label,
    extra_provenance=None,
):
    training_history = getattr(model, "training_history", [])
    training_history_text = _render_training_history(training_history)
    if training_history_text:
        print(f"\n{training_history_text}")
    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train_model["x"], train_model["y"], train_model["v"], train_model["w"]),
        data_z=None,
        nb_intervals=int(params.get("nb_intervals", 20)),
    )
    print(
        "Training evaluate:",
        causal_pre.shape,
        f"MSE_x={float(mse_x):.4f}",
        f"MSE_y={float(mse_y):.4f}",
        f"MSE_v={float(mse_v):.4f}",
    )
    results = _evaluate_structural_methods(
        model,
        grid_x_model,
        grid_v_model,
        grid_truth,
        methods=methods,
        y_stats=y_stats,
        mcmc_context=mcmc_context,
    )
    if holdout is not None:
        results.update(_evaluate_holdout_criterion(model, holdout, y_stats=y_stats))
    results["_training_evaluate"] = {
        "mse_x": float(mse_x),
        "mse_y": float(mse_y),
        "mse_v": float(mse_v),
    }
    provenance = _run_provenance(params, model, extra_provenance)
    print(f"\nStructural MSE summary ({space_label})")
    for method in methods:
        value = results.get(method)
        print(f"  {method}: {value:.6f}")
    print(
        f"  checkpoint {provenance['checkpoint_timestamp']} "
        f"gamma={provenance['resolved_gamma']} device={provenance['device_name']}"
    )
    return {
        "training_history": training_history,
        "training_history_text": training_history_text,
        "run_config_text": run_config_text,
        "ranges_text": ranges_text,
        "final_results": results,
        "provenance": provenance,
    }


def _run_single_demand_design_iv(params):
    """Run one demand-design IV experiment with concrete scalar settings."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    n_samples = int(params.get("n_samples", 5000))
    rho = float(params.get("rho", 0.5))
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    train = simulate_demand_design_iv(n_samples=n_samples, rho=rho, seed=run_seed)
    grid = make_demand_design_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
    )
    holdout_seed, holdout_n = _resolve_holdout_settings(params)
    holdout = simulate_demand_design_iv(n_samples=holdout_n, rho=rho, seed=holdout_seed)
    ranges_text = _render_observed_ranges(train)
    print(ranges_text)

    methods = _resolve_structural_methods(params)
    normalize_before_training = bool(params.get("normalize_before_training", True))

    if normalize_before_training:
        print("\nNormalized-space experiment")
        train_std, grid_std, stats = _standardize_demand_design_data(train, grid)
        holdout_model = {
            "v": _transform(holdout["v"], stats["v"]),
            "w": _transform(holdout["w"], stats["w"]),
            "y": holdout["y"],
        }
        preprocessor = AffinePreprocessorSpec(
            mean=np.asarray(stats["v"]["mean"], np.float32).reshape(-1),
            scale=np.asarray(stats["v"]["scale"], np.float32).reshape(-1),
            name="demand_v_standardizer",
        )
        model = _fit_or_restore_demand_design_model(
            params,
            train_std,
            evaluation_callback=_maybe_structural_monitor_callback(
                params, grid_std["x"], grid_std["v"], grid["y_struct"], y_stats=stats["y"]
            ),
            criterion_data={"y_raw": train["y"], "y_stats": stats["y"]},
        )
        mcmc_context = None
        if "mcmc" in methods:
            mcmc_context = _mcmc_context(
                params,
                grid_x_model=grid_std["x"],
                grid_v_raw=grid["v"],
                truth_rows=grid["y_struct"],
                truth_label="demand_design_grid_y_struct",
                preprocessor=preprocessor,
                x_stats=stats["x"],
                y_stats=stats["y"],
            )
        return _finalize_demand_design_run(
            params,
            model,
            train_model=train_std,
            grid_x_model=grid_std["x"],
            grid_v_model=grid_std["v"],
            grid_truth=grid["y_struct"],
            y_stats=stats["y"],
            methods=methods,
            mcmc_context=mcmc_context,
            holdout=holdout_model,
            run_config_text=run_config_text,
            ranges_text=ranges_text,
            space_label="DFIV-compatible original outcome space",
        )

    print("\nOriginal-space experiment")
    model = _fit_or_restore_demand_design_model(
        params,
        train,
        evaluation_callback=_maybe_structural_monitor_callback(
            params, grid["x"], grid["v"], grid["y_struct"]
        ),
        criterion_data={"y_raw": train["y"], "y_stats": None},
    )
    mcmc_context = None
    if "mcmc" in methods:
        mcmc_context = _mcmc_context(
            params,
            grid_x_model=grid["x"],
            grid_v_raw=grid["v"],
            truth_rows=grid["y_struct"],
            truth_label="demand_design_grid_y_struct",
            preprocessor=AffinePreprocessorSpec.identity_map(
                int(params["v_dim"]), name="demand_raw_v"
            ),
            x_stats=None,
            y_stats=None,
        )
    return _finalize_demand_design_run(
        params,
        model,
        train_model=train,
        grid_x_model=grid["x"],
        grid_v_model=grid["v"],
        grid_truth=grid["y_struct"],
        y_stats=None,
        methods=methods,
        mcmc_context=mcmc_context,
        holdout={"v": holdout["v"], "w": holdout["w"], "y": holdout["y"]},
        run_config_text=run_config_text,
        ranges_text=ranges_text,
        space_label="original space",
    )


def _run_single_demand_design_mnist_iv(params):
    """Run one MNIST (pixel-likelihood) demand-design IV experiment."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    v_dim = int(params.get("v_dim", 785))
    n_samples = int(params.get("n_samples", 5000))
    rho = float(params.get("rho", 0.5))
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    train = simulate_demand_design_mnist_iv(
        n_samples=n_samples, rho=rho, seed=run_seed, v_dim=v_dim
    )
    grid = make_demand_design_mnist_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
        v_dim=v_dim,
        image_seed=int(params.get("image_seed", 42)),
        noise_seed=int(params.get("noise_seed", 42)),
    )
    holdout_seed, holdout_n = _resolve_holdout_settings(params)
    holdout = simulate_demand_design_mnist_iv(
        n_samples=holdout_n, rho=rho, seed=holdout_seed, v_dim=v_dim
    )
    ranges_text = _render_observed_ranges(train)
    print(ranges_text)

    methods = _resolve_structural_methods(params)
    print("\nDFIV-style normalized-space experiment")
    train_std, grid_std, stats = _standardize_demand_design_image_data(train, grid)
    holdout_model = {
        "v": holdout["v"].astype(np.float32),
        "w": holdout["w"].astype(np.float32),
        "y": holdout["y"],
    }
    model = _fit_or_restore_demand_design_model(
        params,
        train_std,
        evaluation_callback=_maybe_structural_monitor_callback(
            params, grid_std["x"], grid_std["v"], grid["y_struct"], y_stats=stats["y"]
        ),
        criterion_data={"y_raw": train["y"], "y_stats": stats["y"]},
    )
    mcmc_context = None
    if "mcmc" in methods:
        mcmc_context = _mcmc_context(
            params,
            grid_x_model=grid_std["x"],
            grid_v_raw=grid_std["v"],
            truth_rows=grid["y_struct"],
            truth_label="mnist_demand_design_grid_y_struct",
            preprocessor=AffinePreprocessorSpec.identity_map(v_dim, name="mnist_raw_v"),
            x_stats=stats["x"],
            y_stats=stats["y"],
        )
    return _finalize_demand_design_run(
        params,
        model,
        train_model=train_std,
        grid_x_model=grid_std["x"],
        grid_v_model=grid_std["v"],
        grid_truth=grid["y_struct"],
        y_stats=stats["y"],
        methods=methods,
        mcmc_context=mcmc_context,
        holdout=holdout_model,
        run_config_text=run_config_text,
        ranges_text=ranges_text,
        space_label="DFIV-compatible original outcome space",
    )


def _run_single_demand_design_vector_iv(params):
    """Run one vector-proxy demand-design IV experiment with concrete settings."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    v_dim = int(params.get("v_dim", 785))
    vector_dim = int(params.get("vector_dim", 784))
    n_samples = int(params.get("n_samples", 5000))
    rho = float(params.get("rho", 0.5))
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    simulate_kwargs = dict(
        v_dim=v_dim,
        vector_dim=vector_dim,
        feature_seed=int(params.get("feature_seed", 42)),
        representation_sd=float(params.get("representation_sd", 0.5)),
    )
    train = simulate_demand_design_vector_iv(
        n_samples=n_samples, rho=rho, seed=run_seed, **simulate_kwargs
    )
    ranges_text = _render_observed_ranges(train)
    print(ranges_text)

    methods = _resolve_structural_methods(params)
    print("\nDFIV-style normalized-space experiment")
    multistart = _uses_egm_multistart(params)
    if multistart:
        # The EGM bundle sees only the complete training sample.  Test-grid and
        # additional holdout rows do not exist until selection and the single
        # BGM continuation have both completed.
        train_std, _, stats = _standardize_demand_design_image_data(train, None)
        model = _fit_or_restore_demand_design_model(
            params,
            train_std,
            evaluation_callback=None,
            criterion_data={"y_raw": train["y"], "y_stats": stats["y"]},
        )
        grid = make_demand_design_vector_grid(
            price_points=int(params.get("price_points", 20)),
            time_points=int(params.get("time_points", 20)),
            test_vector_seed=int(params.get("test_vector_seed", 42)),
            **simulate_kwargs,
        )
        _, grid_std, _ = _standardize_demand_design_image_data(train, grid)
        holdout_seed, holdout_n = _resolve_holdout_settings(params)
        holdout = simulate_demand_design_vector_iv(
            n_samples=holdout_n, rho=rho, seed=holdout_seed, **simulate_kwargs
        )
    else:
        grid = make_demand_design_vector_grid(
            price_points=int(params.get("price_points", 20)),
            time_points=int(params.get("time_points", 20)),
            test_vector_seed=int(params.get("test_vector_seed", 42)),
            **simulate_kwargs,
        )
        holdout_seed, holdout_n = _resolve_holdout_settings(params)
        holdout = simulate_demand_design_vector_iv(
            n_samples=holdout_n, rho=rho, seed=holdout_seed, **simulate_kwargs
        )
        train_std, grid_std, stats = _standardize_demand_design_image_data(train, grid)
        model = _fit_or_restore_demand_design_model(
            params,
            train_std,
            evaluation_callback=_maybe_structural_monitor_callback(
                params,
                grid_std["x"],
                grid_std["v"],
                grid["y_struct"],
                y_stats=stats["y"],
            ),
            criterion_data={"y_raw": train["y"], "y_stats": stats["y"]},
        )
    holdout_model = {
        "v": holdout["v"].astype(np.float32),
        "w": holdout["w"].astype(np.float32),
        "y": holdout["y"],
    }
    mcmc_context = None
    if "mcmc" in methods:
        mcmc_context = _mcmc_context(
            params,
            grid_x_model=grid_std["x"],
            grid_v_raw=grid_std["v"],
            truth_rows=grid["y_struct"],
            truth_label="vector_demand_design_grid_y_struct",
            preprocessor=AffinePreprocessorSpec.identity_map(v_dim, name="vector_raw_v"),
            x_stats=stats["x"],
            y_stats=stats["y"],
        )
    return _finalize_demand_design_run(
        params,
        model,
        train_model=train_std,
        grid_x_model=grid_std["x"],
        grid_v_model=grid_std["v"],
        grid_truth=grid["y_struct"],
        y_stats=stats["y"],
        methods=methods,
        mcmc_context=mcmc_context,
        holdout=holdout_model,
        run_config_text=run_config_text,
        ranges_text=ranges_text,
        space_label="DFIV-compatible original outcome space",
    )


def _select_demand_design_single_run_fn(dataset):
    if dataset == "Sim_Demand_Design_IV":
        return _run_single_demand_design_iv
    if dataset == "Sim_Demand_Design_Mnist_IV":
        return _run_single_demand_design_mnist_iv
    if dataset == "Sim_Demand_Design_Vector_IV":
        return _run_single_demand_design_vector_iv
    raise ValueError(f"Unsupported demand-design dataset: {dataset}")


def _normalize_repeat_outputs(run_params, repeat_outputs):
    if repeat_outputs is None:
        return {
            "training_history": [],
            "training_history_text": "",
            "run_config_text": _render_demand_design_run_config(run_params),
            "ranges_text": "",
            "final_results": {},
        }
    repeat_outputs.setdefault("final_results", {})
    return repeat_outputs


def _initialize_parallel_demand_design_worker(use_gpu, gpu_slot):
    _configure_tensorflow_threads(intra_op_threads=1, inter_op_threads=1)
    _configure_tensorflow_devices(use_gpu=use_gpu, gpu_slot=gpu_slot, verbose=False)


def _run_demand_design_parallel_worker(run_index, total_runs, run_params):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            _print_demand_design_sweep_banner(run_index, total_runs, run_params)
            repeat_outputs = _select_demand_design_single_run_fn(run_params["dataset"])(run_params)
            repeat_outputs = _normalize_repeat_outputs(run_params, repeat_outputs)
    except Exception as exc:
        traceback.print_exc(file=stderr_buffer)
        return {
            "run_index": run_index,
            "total_runs": total_runs,
            "params": run_params,
            "repeat_outputs": None,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    finally:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

    return {
        "run_index": run_index,
        "total_runs": total_runs,
        "params": run_params,
        "repeat_outputs": repeat_outputs,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "error": None,
    }


def _flush_completed_demand_design_result(run_root, result):
    stdout_text = result.get("stdout", "")
    stderr_text = result.get("stderr", "")
    if stdout_text:
        print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
    if stderr_text:
        print(stderr_text, end="" if stderr_text.endswith("\n") else "\n", file=sys.stderr)

    if result.get("error") is not None:
        error = result["error"]
        raise RuntimeError(
            "Demand-design parallel worker failed for "
            f"run_index={result['run_index']} (repeat={result['params'].get('repeat_id')}) "
            f"with {error['type']}: {error['message']}"
        )

    repeat_outputs = _normalize_repeat_outputs(result["params"], result["repeat_outputs"])
    _persist_demand_design_repeat_outputs(
        run_root,
        result["run_index"],
        result["total_runs"],
        result["params"],
        repeat_outputs["run_config_text"],
        repeat_outputs["ranges_text"],
        repeat_outputs["training_history"],
        final_results=repeat_outputs.get("final_results", {}),
        provenance=repeat_outputs.get("provenance"),
    )


def _make_parallel_executor_factory(executor_cls=None):
    executor_cls = ProcessPoolExecutor if executor_cls is None else executor_cls

    def factory(**kwargs):
        return executor_cls(**kwargs)

    return factory


def _run_demand_design_family_parallel(
    params,
    run_root,
    executor_factory=None,
    as_completed_fn=None,
):
    num_tasks = _resolve_num_tasks(params)
    runs = _materialize_demand_design_sweep_runs(params)
    if not runs:
        return

    use_gpu = bool(params.get("use_gpu", False))
    worker_slots = _resolve_parallel_gpu_slots(params)
    max_parallel = min(num_tasks, len(runs))
    worker_slots = worker_slots[:max_parallel]
    executor_factory = _make_parallel_executor_factory(executor_factory)
    as_completed_fn = as_completed if as_completed_fn is None else as_completed_fn

    print(
        f"Launching demand-design sweep with {len(runs)} concrete run(s) "
        f"across {max_parallel} parallel worker(s)."
    )

    executors = []
    future_to_run = {}
    spawn_context = multiprocessing.get_context("spawn")
    try:
        for worker_index in range(max_parallel):
            executors.append(
                executor_factory(
                    max_workers=1,
                    mp_context=spawn_context,
                    initializer=_initialize_parallel_demand_design_worker,
                    initargs=(use_gpu, worker_slots[worker_index]),
                )
            )

        for task_index, (run_index, total_runs, run_params) in enumerate(runs):
            executor = executors[task_index % max_parallel]
            future = executor.submit(
                _run_demand_design_parallel_worker,
                run_index,
                total_runs,
                run_params,
            )
            future_to_run[future] = (run_index, total_runs, run_params)

        pending_results = {}
        next_run_index = 1
        for future in as_completed_fn(list(future_to_run.keys())):
            result = future.result()
            pending_results[result["run_index"]] = result
            while next_run_index in pending_results:
                _flush_completed_demand_design_result(
                    run_root,
                    pending_results.pop(next_run_index),
                )
                next_run_index += 1
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def _run_demand_design_family(params, single_run_fn):
    """Run demand-design IV experiments aligned with the DFIV benchmark."""
    run_id = _build_demand_design_run_id(params)
    run_root = _demand_design_dumps_dir() / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _copy_demand_design_config_snapshot(params, run_root)

    active_path = _demand_design_active_window_path(params)
    controller = _DemandDesignActiveWindowController(active_path, params, run_root)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _ActiveWindowStream(original_stdout, controller)
    sys.stderr = _ActiveWindowStream(original_stderr, controller)
    controller.render(force=True)

    try:
        if _resolve_num_tasks(params) > 1:
            _run_demand_design_family_parallel(params, run_root)
        else:
            only_repeat = params.get("_only_repeat_id")
            for run_index, total_runs, run_params in _iter_demand_design_sweep_runs(params):
                if only_repeat is not None and int(run_params["repeat_id"]) != int(only_repeat):
                    continue
                _print_demand_design_sweep_banner(run_index, total_runs, run_params)
                repeat_outputs = _normalize_repeat_outputs(
                    run_params,
                    single_run_fn(run_params),
                )
                _persist_demand_design_repeat_outputs(
                    run_root,
                    run_index,
                    total_runs,
                    run_params,
                    repeat_outputs["run_config_text"],
                    repeat_outputs["ranges_text"],
                    repeat_outputs["training_history"],
                    final_results=repeat_outputs.get("final_results", {}),
                    provenance=repeat_outputs.get("provenance"),
                )
    except Exception:
        controller.set_status("local run failed")
        raise
    else:
        controller.set_status("local run completed successfully")
    finally:
        controller.render(force=True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def run_demand_design_iv(params):
    _run_demand_design_family(params, _run_single_demand_design_iv)


def run_demand_design_mnist_iv(params):
    _run_demand_design_family(params, _run_single_demand_design_mnist_iv)


def run_demand_design_vector_iv(params):
    _run_demand_design_family(params, _run_single_demand_design_vector_iv)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = args.config

    with open(config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["_config_source_path"] = config
    _apply_config_overrides(params, args.overrides)
    _apply_mcmc_inference_config(params)
    params["num_tasks"] = args.num_tasks
    if args.repeat_id is not None:
        if args.num_tasks != 1:
            raise ValueError("--repeat-id runs a single repeat; use -t 1.")
        params["_only_repeat_id"] = int(args.repeat_id)
    if args.mcmc_only is not None:
        mcmc_only_timestamp = str(args.mcmc_only).strip()
        if not mcmc_only_timestamp:
            parser.error("--mcmc-only requires a non-empty TIMESTAMP")
        if args.repeat_id is None:
            raise ValueError("--mcmc-only requires --repeat-id.")
        params[_MCMC_ONLY_TIMESTAMP_KEY] = mcmc_only_timestamp
    _apply_mcmc_inference_cli(params, args)
    _apply_demand_design_benchmark_defaults(params)
    _validate_map_only_structural_config(params)
    _validate_egm_multistart_run_shape(params)

    if _resolve_num_tasks(params) > 1 and not _supports_parallel_demand_design(params):
        raise ValueError(
            "`-t/--num_tasks` is currently supported only for the demand-design "
            "datasets (Sim_Demand_Design_IV / _Mnist_IV / _Vector_IV)."
        )

    if _uses_egm_multistart(params):
        # The parent prepares data before spawning the EGM candidates.  Pixel
        # data loading enters TensorFlow's eager context, after which physical
        # device options can no longer be changed.  Bind memory growth first;
        # the idempotent handoff call after candidate selection then only
        # verifies this configuration before restoring the winning state.
        _configure_tensorflow_devices(
            bool(params.get("use_gpu", False)),
            strict_memory_growth=bool(params.get("use_gpu", False)),
        )
        print(
            "TensorFlow parent device configuration initialized before data "
            "loading; candidate training remains deferred to spawned workers."
        )
    elif _is_parallel_demand_design_run(params):
        print(
            "TensorFlow device configuration deferred to spawned training workers."
        )
    else:
        _configure_tensorflow_devices(bool(params.get("use_gpu", False)))

    if params["dataset"] == "Sim_Demand_Design_IV":
        run_demand_design_iv(params)
    elif params["dataset"] == "Sim_Demand_Design_Mnist_IV":
        run_demand_design_mnist_iv(params)
    elif params["dataset"] == "Sim_Demand_Design_Vector_IV":
        run_demand_design_vector_iv(params)
    else:
        raise ValueError(
            "Unsupported dataset. This clean package supports only "
            "Sim_Demand_Design_IV, Sim_Demand_Design_Mnist_IV and "
            "Sim_Demand_Design_Vector_IV."
        )


if __name__ == "__main__":
    main()
