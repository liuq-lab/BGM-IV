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
import sys
import time
import traceback
from itertools import product
import yaml
from bayesgm.models import (
    CausalBGM_IV,
    CausalBGM_IV_Image,
    CausalBGM_IV_Vector,
)
from bayesgm.datasets import (
    simulate_demand_design_iv,
    make_demand_design_grid,
    simulate_demand_design_mnist_iv,
    make_demand_design_mnist_grid,
    simulate_demand_design_vector_iv,
    make_demand_design_vector_grid,
)
import tensorflow as tf


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
    "fit_model_selection_metric": "structural_mse",
    "fit_restore_best_weights": True,
    "fit_use_progress_bar": False,
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
        help="Path to one of the supported CausalBGM-IV config files.",
    )
    parser.add_argument(
        '-t',
        '--num_tasks',
        type=int,
        default=1,
        help='Number of parallel demand-design tasks to run (default: 1).',
    )
    return parser


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
        "latent_pzv_weight",
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


def _summarize_training_history(history, structural_keys=None):
    structural_keys = (
        _get_training_history_structural_keys(history)
        if structural_keys is None
        else structural_keys
    )
    summaries = []
    for key in structural_keys:
        method = key.removeprefix("structural_mse_")
        method_records = [record for record in history if key in record]
        if not method_records:
            continue
        summaries.append(
            {
                "method": method,
                "key": key,
                "best_record": min(method_records, key=lambda r: r[key]),
                "last_record": method_records[-1],
            }
        )
    return summaries


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

    for summary in _summarize_training_history(history, structural_keys):
        method = summary["method"]
        key = summary["key"]
        best_record = summary["best_record"]
        last_record = summary["last_record"]
        lines.append("")
        lines.append(
            "Best structural checkpoint "
            f"[{method}]: stage={best_record['stage']}, epoch={best_record['epoch']}, "
            f"structural_MSE={best_record[key]:.6f}"
        )
        lines.append(
            "Last logged structural checkpoint "
            f"[{method}]: stage={last_record['stage']}, epoch={last_record['epoch']}, "
            f"structural_MSE={last_record[key]:.6f}"
        )
    return "\n".join(lines)


def _get_training_history_columns(history):
    return (
        "stage",
        "epoch",
        "include_outcome",
        "mse_x",
        "mse_y",
        "mse_v",
        *_get_training_history_structural_keys(history),
    )


def _rectangularize_training_history(history):
    columns = _get_training_history_columns(history)
    rows = []
    for record in history:
        row = {}
        for column in columns:
            value = record.get(column)
            row[column] = "" if value is None else value
        rows.append(row)
    return columns, rows


def _format_rho_token(value):
    return _format_demand_design_sweep_value(value).replace("-", "m").replace(".", "p")


def _build_training_history_repeat_csv_name(params):
    slug = _get_demand_design_dataset_meta(params)["slug"]
    if not _demand_design_uses_rho(params):
        return (
            f"training_metric_history_{slug}_"
            f"n{int(params['n_samples'])}_"
            f"repeat{int(params.get('repeat_id', 0))}.csv"
        )
    return (
        f"training_metric_history_{slug}_"
        f"n{int(params['n_samples'])}_rho{_format_rho_token(params['rho'])}_"
        f"repeat{int(params.get('repeat_id', 0))}.csv"
    )


def _build_training_history_combo_md_name(params):
    slug = _get_demand_design_dataset_meta(params)["slug"]
    if not _demand_design_uses_rho(params):
        return f"training_metric_history_{slug}_n{int(params['n_samples'])}.md"
    return (
        f"training_metric_history_{slug}_"
        f"n{int(params['n_samples'])}_rho{_format_rho_token(params['rho'])}.md"
    )


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
    return (
        f"n_samples:{int(params['n_samples'])}"
        f"-rho:{_format_demand_design_sweep_value(params['rho'])}"
        f"-v_dim:{int(params['v_dim'])}"
    )


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
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_checkpoint_summary_rows(summaries, repeat_id, which):
    rows = []
    for summary in summaries:
        record = summary[f"{which}_record"]
        rows.append(
            {
                "repeat_id": int(repeat_id),
                "method": summary["method"],
                "stage": record["stage"],
                "epoch": record["epoch"],
                "include_outcome": record["include_outcome"],
                "mse_x": record["mse_x"],
                "mse_y": record["mse_y"],
                "mse_v": record["mse_v"],
                "structural_mse": record[summary["key"]],
            }
        )
    return rows


def _append_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def _render_repeat_history_section(
    run_index,
    total_runs,
    params,
    run_config_text,
    ranges_text,
    training_history_text,
):
    if _demand_design_uses_rho(params):
        header_detail = (
            f"n_samples={params['n_samples']}, rho={params['rho']}, "
            f"repeat={params['repeat_id']}"
        )
    else:
        header_detail = (
            f"n_samples={params['n_samples']}, repeat={params['repeat_id']}"
        )
    lines = [
        f"## Demand-design sweep run [{run_index}/{total_runs}]: "
        f"{header_detail}",
        "",
        "```text",
        run_config_text,
        ranges_text,
    ]
    if training_history_text:
        lines.extend(["", training_history_text])
    lines.extend(["```", ""])
    return "\n".join(lines)


def _persist_demand_design_repeat_outputs(
    run_root,
    run_index,
    total_runs,
    params,
    run_config_text,
    ranges_text,
    training_history,
):
    combo_dir = run_root / _build_demand_design_combo_dir_name(params)
    combo_dir.mkdir(parents=True, exist_ok=True)

    columns, rows = _rectangularize_training_history(training_history)
    repeat_csv_path = combo_dir / _build_training_history_repeat_csv_name(params)
    with repeat_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summaries = _summarize_training_history(training_history)
    summary_columns = (
        "repeat_id",
        "method",
        "stage",
        "epoch",
        "include_outcome",
        "mse_x",
        "mse_y",
        "mse_v",
        "structural_mse",
    )
    _csv_writer_append_rows(
        combo_dir / "best_structural_checkpoints.csv",
        summary_columns,
        _build_checkpoint_summary_rows(summaries, params.get("repeat_id", 0), "best"),
    )
    _csv_writer_append_rows(
        combo_dir / "last_structural_checkpoints.csv",
        summary_columns,
        _build_checkpoint_summary_rows(summaries, params.get("repeat_id", 0), "last"),
    )

    combo_md_path = combo_dir / _build_training_history_combo_md_name(params)
    if not combo_md_path.exists():
        combo_md_path.write_text("# Training Metric History\n\n", encoding="utf-8")
    _append_text(
        combo_md_path,
        _render_repeat_history_section(
            run_index,
            total_runs,
            params,
            run_config_text,
            ranges_text,
            _render_training_history(training_history),
        ),
    )


def _render_demand_design_active_window(
    active_path,
    status,
    params,
    run_root,
    command,
    driver_output,
):
    dataset_meta = _get_demand_design_dataset_meta(params)
    source_path = _resolve_source_config_path(params)
    source_link = str(source_path) if source_path is not None else str(
        params.get("_config_source_path", f"configs/{dataset_meta['config_name']}")
    )
    if source_path is not None:
        source_link = f"[{source_path.name}]({_relative_markdown_link(active_path, source_path)})"
    dumps_link = f"[{run_root.name}]({_relative_markdown_link(active_path, run_root)})"
    body = driver_output.rstrip() or "Waiting for run output..."
    return (
        f"# Active Window: {dataset_meta['title']}\n\n"
        f"- status: {status}\n"
        "- target: `bgm_iv`\n"
        f"- source config: {source_link}\n"
        f"- dumps root: {dumps_link}\n"
        f"- command: `{command}`\n\n"
        "## Driver Output\n\n"
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


def _make_demand_design_sweep_output_dir(base_output_dir, n_samples, repeat_id, rho=None):
    if rho is None:
        return f"{base_output_dir}/sweeps/n_samples={n_samples}__repeat={repeat_id}"
    rho_str = _format_demand_design_sweep_value(rho)
    return (
        f"{base_output_dir}/sweeps/"
        f"n_samples={n_samples}__rho={rho_str}__repeat={repeat_id}"
    )


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
    combinations = tuple(product(n_samples_values, rho_values))
    total_runs = len(combinations) * n_repeat

    for run_index, (n_samples, rho) in enumerate(combinations, start=1):
        for repeat_id in range(n_repeat):
            run_params = dict(params)
            run_params["n_samples"] = n_samples
            if rho is None:
                run_params.pop("rho", None)
            else:
                run_params["rho"] = rho
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


def _validate_map_only_structural_config(params):
    _require_map_only_method_list(params, "structural_methods")
    _require_map_only_method_list(params, "training_structural_methods")
    _require_map_only_method(params, "training_structural_monitor_method")
    _require_map_only_method(params, "structural_latent_method")


def _resolve_structural_methods(params):
    return _require_map_only_method_list(params, "structural_methods")


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


def _print_training_history(history):
    text = _render_training_history(history)
    if text:
        print(f"\n{text}")


def _fit_demand_design_model(params, train, evaluation_callback=None):
    if params["dataset"] == "Sim_Demand_Design_Mnist_IV":
        model_cls = CausalBGM_IV_Image
    elif params["dataset"] == "Sim_Demand_Design_Vector_IV":
        model_cls = CausalBGM_IV_Vector
    else:
        model_cls = CausalBGM_IV
    model = model_cls(params=params, random_seed=None)
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


def _evaluate_structural_methods(
    model,
    grid_x,
    grid_v,
    y_true,
    methods,
    y_stats=None,
):
    results = {}
    for method in methods:
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


def _run_single_demand_design_iv(params):
    """Run one demand-design IV experiment with concrete scalar settings."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    train = simulate_demand_design_iv(
        n_samples=int(params.get("n_samples", 5000)),
        rho=float(params.get("rho", 0.5)),
        seed=int(params.get("run_seed", params.get("seed", 0))),
    )
    grid = make_demand_design_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
    )
    ranges_text = _render_observed_ranges(train)
    print(ranges_text)

    methods = _resolve_structural_methods(params)
    normalize_before_training = bool(params.get("normalize_before_training", False))

    if normalize_before_training:
        print("\nNormalized-space experiment")
        train_std, grid_std, stats = _standardize_demand_design_data(train, grid)
        structural_monitor_method, training_methods = _resolve_training_monitor_methods(
            params,
            methods,
        )
        additional_monitor_methods = [
            method
            for method in training_methods
            if method != structural_monitor_method
        ]
        evaluation_callback = _make_structural_monitor_callback(
            grid_std["x"],
            grid_std["v"],
            grid["y_struct"],
            latent_method=structural_monitor_method,
            y_stats=stats["y"],
            additional_methods=additional_monitor_methods,
        )
        model = _fit_demand_design_model(
            params,
            train_std,
            evaluation_callback=evaluation_callback,
        )
        training_history = getattr(model, "training_history", [])
        training_history_text = _render_training_history(training_history)
        if training_history_text:
            print(f"\n{training_history_text}")
        causal_pre, mse_x, mse_y, mse_v = model.evaluate(
            data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
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
            grid_std["x"],
            grid_std["v"],
            grid["y_struct"],
            methods=methods,
            y_stats=stats["y"],
        )
        print("\nStructural MSE summary (DFIV-compatible original outcome space)")
        for method in methods:
            print(f"  normalized/{method}: {results[method]:.6f}")
        return {
            "training_history": training_history,
            "training_history_text": training_history_text,
            "run_config_text": run_config_text,
            "ranges_text": ranges_text,
        }

    print("\nOriginal-space experiment")
    structural_monitor_method, training_methods = _resolve_training_monitor_methods(
        params,
        methods,
    )
    additional_monitor_methods = [
        method for method in training_methods if method != structural_monitor_method
    ]
    evaluation_callback = _make_structural_monitor_callback(
        grid["x"],
        grid["v"],
        grid["y_struct"],
        latent_method=structural_monitor_method,
        additional_methods=additional_monitor_methods,
    )
    model = _fit_demand_design_model(
        params,
        train,
        evaluation_callback=evaluation_callback,
    )
    training_history = getattr(model, "training_history", [])
    training_history_text = _render_training_history(training_history)
    if training_history_text:
        print(f"\n{training_history_text}")
    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train["x"], train["y"], train["v"], train["w"]),
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
        grid["x"],
        grid["v"],
        grid["y_struct"],
        methods=methods,
    )
    print("\nStructural MSE summary")
    for method in methods:
        print(f"  original/{method}: {results[method]:.6f}")
    return {
        "training_history": training_history,
        "training_history_text": training_history_text,
        "run_config_text": run_config_text,
        "ranges_text": ranges_text,
    }


def _run_single_demand_design_mnist_iv(params):
    """Run one MNIST demand-design IV experiment with concrete scalar settings."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    v_dim = int(params.get("v_dim", 785))
    train = simulate_demand_design_mnist_iv(
        n_samples=int(params.get("n_samples", 5000)),
        rho=float(params.get("rho", 0.5)),
        seed=int(params.get("run_seed", params.get("seed", 0))),
        v_dim=v_dim,
    )
    grid = make_demand_design_mnist_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
        v_dim=v_dim,
        image_seed=int(params.get("image_seed", 42)),
        noise_seed=int(params.get("noise_seed", 42)),
    )
    ranges_text = _render_observed_ranges(train)
    print(ranges_text)

    methods = _resolve_structural_methods(params)
    print("\nDFIV-style normalized-space experiment")
    train_std, grid_std, stats = _standardize_demand_design_image_data(train, grid)
    structural_monitor_method, training_methods = _resolve_training_monitor_methods(
        params,
        methods,
    )
    additional_monitor_methods = [
        method for method in training_methods if method != structural_monitor_method
    ]
    evaluation_callback = _make_structural_monitor_callback(
        grid_std["x"],
        grid_std["v"],
        grid["y_struct"],
        latent_method=structural_monitor_method,
        y_stats=stats["y"],
        additional_methods=additional_monitor_methods,
    )
    model = _fit_demand_design_model(
        params,
        train_std,
        evaluation_callback=evaluation_callback,
    )
    training_history = getattr(model, "training_history", [])
    training_history_text = _render_training_history(training_history)
    if training_history_text:
        print(f"\n{training_history_text}")
    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
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
        grid_std["x"],
        grid_std["v"],
        grid["y_struct"],
        methods=methods,
        y_stats=stats["y"],
    )
    print("\nStructural MSE summary (DFIV-compatible original outcome space)")
    for method in methods:
        print(f"  normalized/{method}: {results[method]:.6f}")
    return {
        "training_history": training_history,
        "training_history_text": training_history_text,
        "run_config_text": run_config_text,
        "ranges_text": ranges_text,
    }


def _run_single_demand_design_vector_iv(params):
    """Run one vector-proxy demand-design IV experiment with concrete settings."""
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    v_dim = int(params.get("v_dim", 785))
    vector_dim = int(params.get("vector_dim", 784))
    train = simulate_demand_design_vector_iv(
        n_samples=int(params.get("n_samples", 5000)),
        rho=float(params.get("rho", 0.5)),
        seed=int(params.get("run_seed", params.get("seed", 0))),
        v_dim=v_dim,
        vector_dim=vector_dim,
        feature_seed=int(params.get("feature_seed", 42)),
        representation_sd=float(params.get("representation_sd", 0.5)),
    )
    grid = make_demand_design_vector_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
        v_dim=v_dim,
        vector_dim=vector_dim,
        feature_seed=int(params.get("feature_seed", 42)),
        test_vector_seed=int(params.get("test_vector_seed", 42)),
        representation_sd=float(params.get("representation_sd", 0.5)),
    )
    ranges_text = _render_observed_ranges(train)
    print(ranges_text)

    methods = _resolve_structural_methods(params)
    print("\nDFIV-style normalized-space experiment")
    train_std, grid_std, stats = _standardize_demand_design_image_data(train, grid)
    structural_monitor_method, training_methods = _resolve_training_monitor_methods(
        params,
        methods,
    )
    additional_monitor_methods = [
        method for method in training_methods if method != structural_monitor_method
    ]
    evaluation_callback = _make_structural_monitor_callback(
        grid_std["x"],
        grid_std["v"],
        grid["y_struct"],
        latent_method=structural_monitor_method,
        y_stats=stats["y"],
        additional_methods=additional_monitor_methods,
    )
    model = _fit_demand_design_model(
        params,
        train_std,
        evaluation_callback=evaluation_callback,
    )
    training_history = getattr(model, "training_history", [])
    training_history_text = _render_training_history(training_history)
    if training_history_text:
        print(f"\n{training_history_text}")
    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
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
        grid_std["x"],
        grid_std["v"],
        grid["y_struct"],
        methods=methods,
        y_stats=stats["y"],
    )
    print("\nStructural MSE summary (DFIV-compatible original outcome space)")
    for method in methods:
        print(f"  normalized/{method}: {results[method]:.6f}")
    return {
        "training_history": training_history,
        "training_history_text": training_history_text,
        "run_config_text": run_config_text,
        "ranges_text": ranges_text,
    }


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
        }
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
            for run_index, total_runs, run_params in _iter_demand_design_sweep_runs(params):
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
    params["num_tasks"] = args.num_tasks
    _apply_demand_design_benchmark_defaults(params)
    _validate_map_only_structural_config(params)

    if _resolve_num_tasks(params) > 1 and not _supports_parallel_demand_design(params):
        raise ValueError(
            "`-t/--num_tasks` is currently supported only for "
            "`Sim_Demand_Design_IV`, `Sim_Demand_Design_Mnist_IV`, and "
            "`Sim_Demand_Design_Vector_IV`."
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
    else:
        raise ValueError(
            "Unsupported dataset. This clean package supports only "
            "Sim_Demand_Design_IV, Sim_Demand_Design_Mnist_IV, and "
            "Sim_Demand_Design_Vector_IV."
        )


if __name__ == "__main__":
    main()
