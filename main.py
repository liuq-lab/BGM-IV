import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import gc
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
from bgm_iv.features import export_image_representations
from bgm_iv.hashing import sha256_array, sha256_json, sha256_weights


def _load_mcmc():
    """Import the MCMC inference package on first use.

    ``bgm_iv.mcmc.sampler`` disables TensorFloat-32 and enables op determinism
    at import time; loading it lazily keeps training numerics unchanged and
    applies the sampler's settings only once MCMC starts.
    """
    from bgm_iv.mcmc import inference as inference_module
    from bgm_iv.mcmc import target as target_module

    global FAMILY_RECIPES, run_mcmc_grid, AffinePreprocessorSpec, FeaturePreprocessorSpec
    global PCAFeaturePreprocessorSpec, execution_environment
    FAMILY_RECIPES = inference_module.FAMILY_RECIPES
    run_mcmc_grid = inference_module.run_mcmc_grid
    AffinePreprocessorSpec = target_module.AffinePreprocessorSpec
    FeaturePreprocessorSpec = target_module.FeaturePreprocessorSpec
    PCAFeaturePreprocessorSpec = target_module.PCAFeaturePreprocessorSpec
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
FeaturePreprocessorSpec = _LazyName("FeaturePreprocessorSpec")
PCAFeaturePreprocessorSpec = _LazyName("PCAFeaturePreprocessorSpec")
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
    # MNIST representation model.
    "Sim_Demand_Design_Mnist_Feature_IV": {
        "title": "Sim_Demand_Design_Mnist_Feature_IV",
        "slug": "sim_demand_design_mnist_feature_iv",
        "config_name": "Sim_Demand_Design_Mnist_Feature_IV.yaml",
        "seed_key": "image_seed",
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
    "Sim_Demand_Design_Mnist_Feature_IV": {
        "image_seed": 42,
        "noise_seed": 42,
        "feature_dim": 64,
    },
}


_DATASET_OPTIONAL_BENCHMARK_DEFAULTS = {
    "Sim_Demand_Design_Mnist_IV": {
        "v_dim": 785,
    },
    "Sim_Demand_Design_Mnist_Feature_IV": {
        "pixel_v_dim": 785,
        "feature_map": "egm",
        "holdout_seed_offset": 1000,
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
    if dataset == "Sim_Demand_Design_Mnist_Feature_IV":
        pixel_v_dim = int(params["pixel_v_dim"])
        if pixel_v_dim < 785:
            raise ValueError(
                "`pixel_v_dim` must be >= 785 for Sim_Demand_Design_Mnist_Feature_IV; "
                f"got {pixel_v_dim!r}."
            )
        if str(params["feature_map"]) not in {"egm", "pca"}:
            raise ValueError(
                "`feature_map` must be 'egm' or 'pca' for Sim_Demand_Design_Mnist_Feature_IV; "
                f"got {params['feature_map']!r}."
            )
        # v~ = (time, phi[64], raw nuisance block[pixel_v_dim - 785])
        vector_dim = int(params["feature_dim"]) + max(0, pixel_v_dim - 785)
        for field_name, expected in (("vector_dim", vector_dim), ("v_dim", 1 + vector_dim)):
            if field_name in params and int(params[field_name]) != expected:
                raise ValueError(
                    f"`{field_name}` is derived ({expected}) from `pixel_v_dim` for "
                    f"Sim_Demand_Design_Mnist_Feature_IV; got {params[field_name]!r}."
                )
            params[field_name] = expected


def _demand_design_uses_rho(params):
    return bool(_get_demand_design_dataset_meta(params).get("uses_rho", True))


def _configure_tensorflow_threads(intra_op_threads=None, inter_op_threads=None):
    if intra_op_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_op_threads))
    if inter_op_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_op_threads))


def _configure_tensorflow_devices(use_gpu=False, gpu_slot=None, verbose=True):
    gpus = tf.config.list_physical_devices("GPU")

    if use_gpu:
        if not gpus:
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
            tf.config.set_visible_devices(selected_gpus, "GPU")
        for gpu in selected_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                pass
        if verbose:
            if gpu_slot is None:
                print(f"TensorFlow GPU enabled with {len(selected_gpus)} device(s).")
            else:
                print(f"TensorFlow GPU enabled for worker slot {gpu_slot}.")
        return

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
        "structural_methods",
        "mcmc_family",
        "holdout_seed_offset",
        "feature_map",
        "pixel_v_dim",
        "pixel_checkpoint_dir",
        "pixel_checkpoint_timestamp",
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
    "device_name",
    "hostname",
    "feature_map",
    "pixel_checkpoint_timestamp",
    "trunk_weights_sha256",
    "phi_train_sha256",
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
        "feature_map",
        "pixel_checkpoint_timestamp",
    ):
        if key in params:
            row[key] = _blank(params.get(key))
    if "outcome_to_particles_weight" not in params and "resolved_gamma" in provenance:
        row["outcome_to_particles_weight"] = provenance["resolved_gamma"]
    export = provenance.get("feature_export") or {}
    row["trunk_weights_sha256"] = _blank(export.get("trunk_weights_sha256"))
    row["phi_train_sha256"] = _blank((export.get("hashes") or {}).get("phi_train_sha256"))
    row["sigma_vector_softfloor_source"] = _blank(provenance.get("sigma_vector_softfloor_source"))
    if "sigma_vector_softfloor_value" in provenance:
        row["sigma_vector_softfloor"] = provenance["sigma_vector_softfloor_value"]
    pixel_stage = provenance.get("pixel_stage") or {}
    if pixel_stage.get("timestamp"):
        row["pixel_checkpoint_timestamp"] = pixel_stage["timestamp"]
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


def _standardize_demand_design_image_data(train, grid):
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
    # MNIST representation model.
    "Sim_Demand_Design_Mnist_Feature_IV": BGM_IV_Vector,
}

_MCMC_FAMILY_BY_DATASET = {
    "Sim_Demand_Design_IV": "demand",
    "Sim_Demand_Design_Mnist_IV": "mnist_pixel",
    "Sim_Demand_Design_Vector_IV": "vector",
    "Sim_Demand_Design_Mnist_Feature_IV": "mnist_feature",
}


def _model_class_for_dataset(dataset):
    try:
        return _MODEL_CLASS_BY_DATASET[dataset]
    except KeyError:
        raise ValueError(f"Unsupported demand-design dataset: {dataset}") from None


def _model_random_seed(params):
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    params.setdefault("mcmc_seed", run_seed + 10007)
    return run_seed if bool(params.get("deterministic_training", False)) else None


def _fit_demand_design_model(params, train, evaluation_callback=None):
    model_cls = _model_class_for_dataset(params["dataset"])
    random_seed = _model_random_seed(params)
    model = model_cls(params=params, random_seed=random_seed)
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
    """Restore a pinned checkpoint and verify it against its training manifest.

    Every variable of the model must be present in the checkpoint, and the
    manifest written at training time must agree with the current dataset,
    parameters, training data (when ``train`` is given), restored weights and
    ``manifest_extra``; any mismatch fails instead of evaluating a model that
    was not trained the way the run claims.
    """
    model_cls = _model_class_for_dataset(params["dataset"])
    random_seed = _model_random_seed(params)
    model = model_cls(params=params, timestamp=str(timestamp), random_seed=random_seed)
    if not model.ckpt_manager.latest_checkpoint:
        raise FileNotFoundError(
            f"no checkpoint to restore under {model.checkpoint_path}"
        )
    if str(getattr(model, "timestamp", "")) != str(timestamp):
        raise RuntimeError("restored model timestamp differs from the requested one")
    status = model.ckpt.restore(model.ckpt_manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    model.training_history = []
    _verify_training_manifest(model, params, train, manifest_extra)
    return model


def _fit_or_restore_demand_design_model(
    params, train, evaluation_callback=None, manifest_extra=None, manifest_notes=None
):
    timestamp = params.get("_mcmc_only_timestamp")
    if timestamp:
        print(f"Restoring checkpoint {timestamp} (mcmc-only; no training) ...")
        return _restore_demand_design_model(
            params, timestamp, train=train, manifest_extra=manifest_extra
        )
    model = _fit_demand_design_model(params, train, evaluation_callback=evaluation_callback)
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
        "pixel_checkpoint_dir",
        "pixel_checkpoint_timestamp",
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


def _build_training_manifest(model, params, train, extra=None, notes=None):
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
        "code_commit": _code_commit(),
        "execution_environment": execution_environment(),
    }
    payload["params_hash"] = sha256_json("training-manifest-params", payload["params"])
    return payload


def _write_training_manifest(model, params, train, extra=None, notes=None):
    """Write the manifest once; a second write for the same checkpoint must agree."""
    if not bool(params.get("save_model")):
        return None
    payload = _build_training_manifest(model, params, train, extra, notes)
    path = _training_manifest_path(model)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        comparable = {key: existing.get(key) for key in ("checkpoint_identity", "params_hash", "data", "extra")}
        if comparable != {key: payload[key] for key in comparable}:
            raise RuntimeError(f"training manifest already exists and differs: {path}")
        return existing
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
    "structural_methods",
    "mcmc_family",
    "z_dims",
    "v_dim",
    "vector_dim",
    "feature_map",
    "pixel_v_dim",
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
        "mcmc_only": bool(params.get("_mcmc_only_timestamp")),
    }
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
    )
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
    )
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


_PIXEL_STAGE_DROPPED_KEYS = (
    "sigma_time_softfloor",
    "sigma_vector",
    "sigma_vector_softfloor",
    "vector_blocks",
    "vector_dim",
    "sigma_noise_softfloor",
    "sigma_v",
    "sigma_v_softfloor",
    "mcmc_seed",
    "_mcmc_only_timestamp",
)


def _pixel_stage_params(params):
    """Build the image-encoder configuration."""
    pixel = {key: value for key, value in params.items() if key not in _PIXEL_STAGE_DROPPED_KEYS}
    pixel_v_dim = int(params["pixel_v_dim"])
    pixel.update(
        dataset="Sim_Demand_Design_Mnist_IV",
        v_dim=pixel_v_dim,
        z_dims=list(params.get("pixel_z_dims", [2, 1, 1, 2])),
        fit_epochs=int(params.get("pixel_fit_epochs", params.get("fit_epochs", 200))),
        fit_batch_size=int(params.get("pixel_fit_batch_size", params.get("fit_batch_size", 32))),
        fit_egm_n_iter=int(params.get("pixel_fit_egm_n_iter", params.get("fit_egm_n_iter", 50000))),
        outcome_to_particles_weight=0.0,
        sigma_time=0.1,
        sigma_y_softfloor=0.1,
        covariate_block_scale="sum",
        structural_methods=["map"],
        training_grid_monitor=False,
        save_model=True,
        output_dir=os.path.join(str(params.get("output_dir", ".")), "pixel_stage"),
    )
    pixel.pop("stop_outcome_to_particles", None)
    return pixel


def _resolve_stage2_floor(params, export):
    floor = params.get("sigma_vector_softfloor", "rule")
    if isinstance(floor, str):
        if floor.strip().lower() != "rule":
            raise ValueError(
                "`sigma_vector_softfloor` must be a number or 'rule'; got "
                f"{floor!r}."
            )
        return float(export.floor["sigma_vector_softfloor"]), "rule"
    return float(floor), "explicit"


def _run_single_demand_design_mnist_feature_iv(params):
    """Run the MNIST representation model."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    pixel_v_dim = int(params["pixel_v_dim"])
    feature_map = str(params.get("feature_map", "egm"))
    feature_dim = int(params.get("feature_dim", 64))
    n_samples = int(params.get("n_samples", 5000))
    rho = float(params.get("rho", 0.5))
    run_seed = int(params.get("run_seed", params.get("seed", 0)))
    train_px = simulate_demand_design_mnist_iv(
        n_samples=n_samples, rho=rho, seed=run_seed, v_dim=pixel_v_dim
    )
    grid_px = make_demand_design_mnist_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
        v_dim=pixel_v_dim,
        image_seed=int(params.get("image_seed", 42)),
        noise_seed=int(params.get("noise_seed", 42)),
    )
    holdout_seed, holdout_n = _resolve_holdout_settings(params)
    holdout_px = simulate_demand_design_mnist_iv(
        n_samples=holdout_n, rho=rho, seed=holdout_seed, v_dim=pixel_v_dim
    )
    ranges_text = _render_observed_ranges(train_px)
    print(ranges_text)
    methods = _resolve_structural_methods(params)

    # ---- image encoder ---------------------------------------------------------
    # Under mcmc-only the stage-2 manifest names the exact stage-1
    # checkpoint; a retrained trunk would silently change the feature space.
    stage2_manifest = None
    if params.get("_mcmc_only_timestamp"):
        stage2_manifest = _load_training_manifest(params, params["_mcmc_only_timestamp"])
    trunk = None
    pixel_identity = None
    if feature_map == "egm":
        pixel_params = _pixel_stage_params(params)
        train_px_std, _, _ = _standardize_demand_design_image_data(train_px, grid_px)
        pixel_timestamp = params.get("pixel_checkpoint_timestamp")
        pixel_dir = params.get("pixel_checkpoint_dir")
        if bool(pixel_timestamp) != bool(pixel_dir):
            raise ValueError(
                "`pixel_checkpoint_dir` and `pixel_checkpoint_timestamp` must be set together."
            )
        if stage2_manifest is not None:
            bound = (stage2_manifest.get("notes") or {}).get("pixel_stage") or {}
            if not bound.get("timestamp"):
                raise RuntimeError("stage-2 manifest does not name a stage-1 checkpoint")
            if pixel_timestamp and (
                str(pixel_timestamp) != str(bound["timestamp"])
                or str(pixel_dir) != str(bound["output_dir"])
            ):
                raise ValueError(
                    "mcmc-only: the yaml pixel checkpoint differs from the stage-1 "
                    "checkpoint recorded in the stage-2 training manifest"
                )
            pixel_timestamp, pixel_dir = str(bound["timestamp"]), str(bound["output_dir"])
        if pixel_timestamp:
            pixel_params["output_dir"] = str(pixel_dir)
            print(f"\nStage 1: restoring pixel checkpoint {pixel_timestamp} from {pixel_dir}")
            pixel_model = _restore_demand_design_model(
                pixel_params, pixel_timestamp, train=train_px_std
            )
        else:
            print("\nStage 1: training the pixel encoder (grid-blind)")
            pixel_model = _fit_demand_design_model(pixel_params, train_px_std)
            _write_training_manifest(pixel_model, pixel_params, train_px_std)
        trunk = pixel_model.e_net.feature_extractor
        pixel_identity = {
            "output_dir": str(pixel_params["output_dir"]),
            "timestamp": str(pixel_model.timestamp),
            "checkpoint_identity": _checkpoint_identity(pixel_model),
        }
        if stage2_manifest is not None:
            recorded = (stage2_manifest.get("extra") or {}).get("pixel_stage") or {}
            if recorded.get("checkpoint_identity") != pixel_identity["checkpoint_identity"]:
                raise RuntimeError(
                    "mcmc-only: restored stage-1 weights differ from those recorded "
                    "in the stage-2 training manifest"
                )
    elif feature_map != "pca":
        raise ValueError("`feature_map` must be 'egm' or 'pca'.")

    # ---- representation preprocessing -----------------------------------------
    export = export_image_representations(
        trunk=trunk,
        train=train_px,
        grid=grid_px,
        holdout=holdout_px,
        pixel_v_dim=pixel_v_dim,
        feature_map=feature_map,
        feature_dim=feature_dim,
        floor_factor=float(params.get("sigma_vector_softfloor_rule_factor", 0.5)),
        floor_minimum=float(params.get("sigma_vector_softfloor_min", 0.02)),
    )
    floor_value, floor_source = _resolve_stage2_floor(params, export)
    noise_dim = pixel_v_dim - 785
    stage2 = dict(params)
    for forbidden in ("fit_model_selection_metric", "fit_restore_best_weights"):
        stage2.pop(forbidden, None)
    if noise_dim > 0:
        stage2["vector_blocks"] = [feature_dim, noise_dim]
        stage2["sigma_vector_softfloor"] = [
            floor_value,
            float(params.get("sigma_noise_softfloor", 0.1)),
        ]
    else:
        stage2["sigma_vector_softfloor"] = floor_value
    print(
        f"\nStage 2: generative model on the image representation ({stage2['v_dim']}-d), "
        f"sigma_vector_softfloor={stage2['sigma_vector_softfloor']} ({floor_source}), "
        f"feature_map={feature_map}"
    )
    print("Feature export diagnostics:", json.dumps(export.floor))

    train_f = {
        "x": train_px["x"],
        "y": train_px["y"],
        "v": export.train_v,
        "w": train_px["w"],
        "y_struct": train_px["y_struct"],
    }
    grid_f = {"x": grid_px["x"], "v": export.grid_v, "y_struct": grid_px["y_struct"]}
    train_std, grid_std, stats = _standardize_demand_design_image_data(train_f, grid_f)
    holdout_model = {
        "v": export.holdout_v,
        "w": holdout_px["w"].astype(np.float32),
        "y": holdout_px["y"],
    }
    feature_extra = {
        "pixel_stage": None
        if pixel_identity is None
        else {
            "timestamp": pixel_identity["timestamp"],
            "checkpoint_identity": pixel_identity["checkpoint_identity"],
        },
        "feature": {
            "feature_map": feature_map,
            "feature_dim": int(feature_dim),
            "pixel_v_dim": int(pixel_v_dim),
            "trunk_weights_sha256": export.trunk_weights_sha256,
            "standardizer_mean_sha256": sha256_array(export.stats["mean"]),
            "standardizer_scale_sha256": sha256_array(export.stats["scale"]),
            **export.hashes,
            "sigma_vector_softfloor": stage2["sigma_vector_softfloor"],
            "sigma_vector_softfloor_source": floor_source,
        },
    }
    feature_notes = {
        "pixel_stage": None
        if pixel_identity is None
        else {"output_dir": pixel_identity["output_dir"], "timestamp": pixel_identity["timestamp"]}
    }
    model = _fit_or_restore_demand_design_model(
        stage2,
        train_std,
        evaluation_callback=_maybe_structural_monitor_callback(
            stage2, grid_std["x"], grid_std["v"], grid_f["y_struct"], y_stats=stats["y"]
        ),
        manifest_extra=feature_extra,
        manifest_notes=feature_notes,
    )
    noise_slice = slice(785, None) if noise_dim > 0 else None
    if feature_map == "egm":
        preprocessor = FeaturePreprocessorSpec(
            trunk=trunk,
            mean=export.stats["mean"],
            scale=export.stats["scale"],
            input_dimension=pixel_v_dim,
            feature_slice=slice(1, 1 + feature_dim),
            noise_slice=noise_slice,
            trunk_architecture=type(trunk).__name__,
            pixel_checkpoint=pixel_identity,
            name="image_representation_egm",
            provenance={"phi_train_sha256": export.hashes["phi_train_sha256"]},
        )
    else:
        preprocessor = PCAFeaturePreprocessorSpec(
            components=export.pca["components"],
            pca_mean=export.pca["mean"],
            mean=export.stats["mean"],
            scale=export.stats["scale"],
            input_dimension=pixel_v_dim,
            noise_slice=noise_slice,
            name="pca_feature_control",
        )
    mcmc_context = None
    if "mcmc" in methods:
        mcmc_context = _mcmc_context(
            stage2,
            grid_x_model=grid_std["x"],
            grid_v_raw=grid_px["v"],
            truth_rows=grid_px["y_struct"],
            truth_label="mnist_demand_design_grid_y_struct",
            preprocessor=preprocessor,
            x_stats=stats["x"],
            y_stats=stats["y"],
        )
    extra = {
        "feature_export": export.to_payload(),
        "feature_preprocessor": preprocessor.to_payload(),
        "pixel_stage": pixel_identity,
        "sigma_vector_softfloor_value": stage2["sigma_vector_softfloor"],
        "sigma_vector_softfloor_source": floor_source,
        "feature_map": feature_map,
    }
    return _finalize_demand_design_run(
        stage2,
        model,
        train_model=train_std,
        grid_x_model=grid_std["x"],
        grid_v_model=grid_std["v"],
        grid_truth=grid_f["y_struct"],
        y_stats=stats["y"],
        methods=methods,
        mcmc_context=mcmc_context,
        holdout=holdout_model,
        run_config_text=run_config_text,
        ranges_text=ranges_text,
        space_label="original outcome scale (image-representation model)",
        extra_provenance=extra,
    )


def _select_demand_design_single_run_fn(dataset):
    if dataset == "Sim_Demand_Design_IV":
        return _run_single_demand_design_iv
    if dataset == "Sim_Demand_Design_Mnist_IV":
        return _run_single_demand_design_mnist_iv
    if dataset == "Sim_Demand_Design_Vector_IV":
        return _run_single_demand_design_vector_iv
    if dataset == "Sim_Demand_Design_Mnist_Feature_IV":
        return _run_single_demand_design_mnist_feature_iv
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


def run_demand_design_mnist_feature_iv(params):
    _run_demand_design_family(params, _run_single_demand_design_mnist_feature_iv)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = args.config

    with open(config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["_config_source_path"] = config
    _apply_config_overrides(params, args.overrides)
    params["num_tasks"] = args.num_tasks
    if args.repeat_id is not None:
        if args.num_tasks != 1:
            raise ValueError("--repeat-id runs a single repeat; use -t 1.")
        params["_only_repeat_id"] = int(args.repeat_id)
    if args.mcmc_only:
        if args.repeat_id is None:
            raise ValueError("--mcmc-only requires --repeat-id.")
        params["_mcmc_only_timestamp"] = str(args.mcmc_only)
    _apply_demand_design_benchmark_defaults(params)
    _validate_map_only_structural_config(params)

    if _resolve_num_tasks(params) > 1 and not _supports_parallel_demand_design(params):
        raise ValueError(
            "`-t/--num_tasks` is currently supported only for the demand-design "
            "datasets (Sim_Demand_Design_IV / _Mnist_IV / _Vector_IV / _Mnist_Feature_IV)."
        )

    if _is_parallel_demand_design_run(params):
        print(
            "TensorFlow device configuration deferred to demand-design parallel workers."
        )
    else:
        _configure_tensorflow_devices(bool(params.get("use_gpu", False)))

    if params["dataset"] == "Sim_Demand_Design_IV":
        run_demand_design_iv(params)
    elif params["dataset"] == "Sim_Demand_Design_Mnist_IV":
        run_demand_design_mnist_iv(params)
    elif params["dataset"] == "Sim_Demand_Design_Vector_IV":
        run_demand_design_vector_iv(params)
    elif params["dataset"] == "Sim_Demand_Design_Mnist_Feature_IV":
        run_demand_design_mnist_feature_iv(params)
    else:
        raise ValueError(
            "Unsupported dataset. This clean package supports only "
            "Sim_Demand_Design_IV, Sim_Demand_Design_Mnist_IV, "
            "Sim_Demand_Design_Vector_IV and Sim_Demand_Design_Mnist_Feature_IV."
        )


if __name__ == "__main__":
    main()
