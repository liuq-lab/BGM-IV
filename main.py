import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import gc
import multiprocessing
import os
import numpy as np
from pathlib import Path
import shutil
import time
import traceback
import json
import yaml
from bgm_iv.models import (
    BGM_IV,
    BGM_IV_Image,
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
from bgm_iv.proxy_transform import apply_pca, fit_pca, parse_pca_dim
from bgm_iv.egm_multistart import (
    EGM_SELECTION_CRITERION,
    EGM_SELECTOR_VERSION,
    derive_multistart_seeds,
    derive_seed,
    draw_model_seed,
    make_candidate_record,
    normalize_model_seed,
    select_candidate_by_criterion,
    validate_multistart_config,
)


def _mcmc_inference():
    from bgm_iv.mcmc import inference

    return inference


def _affine_preprocessor(mean, scale):
    _mcmc_inference()
    from bgm_iv.mcmc.target import AffinePreprocessorSpec

    return AffinePreprocessorSpec(mean=mean, scale=scale)


_DATASETS = {
    "Sim_Demand_Design_IV": ("_run_single_demand_design_iv", BGM_IV, "demand"),
    "Sim_Demand_Design_Mnist_IV": (
        "_run_single_demand_design_mnist_iv", BGM_IV_Image, "mnist_pixel"
    ),
    "Sim_Demand_Design_Vector_PCAOnly_IV": (
        "_run_single_demand_design_vector_pcaonly_iv", BGM_IV, "demand"
    ),
}


def _dataset(params):
    dataset = params.get("dataset", "Sim_Demand_Design_IV")
    if dataset not in _DATASETS:
        raise ValueError(f"Unsupported demand-design dataset: {dataset}")
    return dataset


def _single_run_fn(params):
    return globals()[_DATASETS[_dataset(params)][0]]


def _model_class(params):
    return _DATASETS[_dataset(params)][1]


_BENCHMARK_DIMENSIONS = {
    "Sim_Demand_Design_IV": {"w_dim": 1, "v_dim": 2},
    "Sim_Demand_Design_Mnist_IV": {"w_dim": 1, "v_dim": 785},
    "Sim_Demand_Design_Vector_PCAOnly_IV": {"w_dim": 1, "v_dim": 785},
}


_RETIRED_KEYS = (
    "sigma_x", "sigma_y", "sigma_v", "sigma_x_softfloor", "sigma_v_softfloor",
    "sigma_y_softfloor", "sigma_time",
    "seed", "price_points", "time_points", "image_seed", "feature_seed",
    "test_vector_seed", "vector_dim",
    "binary_treatment", "deterministic_training", "covariate_block_scale",
    "use_z_rec", "fit_use_progress_bar",
    "fit_first_stage_warmup_epochs", "first_stage_warmup_epochs",
    "fit_egm_batches_per_eval",
    "egm_outcome_loss", "egm_outcome_gh_nodes", "egm_outcome_sigma",
    "egm_outcome_sigma_cap", "egm_outcome_grad_path",
    "egm_selection_top_k",
)


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
        choices=[1],
        help="Number of cells per process (only 1 is supported).",
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
            "rerun the readouts on it (MCMC only with structural_methods=[map,mcmc])."
        ),
    )
    parser.add_argument(
        "--repeat-id",
        dest="repeat_id",
        type=int,
        default=None,
        help=(
            "The repeat to run, in 0..n_repeat-1 (required when n_repeat > 1); "
            "with --mcmc-only it is the repeat whose checkpoint is restored."
        ),
    )
    parser.add_argument(
        "--mcmc-artifact-root",
        dest="mcmc_artifact_root",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "With --mcmc-only: also save the retained HMC draws under DIR "
            "(requires --mcmc-arm-id)."
        ),
    )
    parser.add_argument(
        "--mcmc-arm-id",
        dest="mcmc_arm_id",
        type=str,
        default=None,
        metavar="ID",
        help="Name of the saved draw artifact under --mcmc-artifact-root.",
    )
    parser.add_argument(
        "--mcmc-readout-artifact",
        dest="mcmc_readout_artifact",
        type=str,
        default=None,
        metavar="MANIFEST",
        help=(
            "With --mcmc-only: recompute the interval readout from saved draws "
            "instead of sampling again."
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
            "--set rho=0.5; repeatable."
        ),
    )
    return parser


_MCMC_ARTIFACT_KEYS = ("mcmc_artifact_root", "mcmc_arm_id", "mcmc_readout_artifact")


def _apply_mcmc_artifact_options(params, args):
    for key in _MCMC_ARTIFACT_KEYS:
        value = getattr(args, key)
        if value is None:
            continue
        if not str(value).strip():
            raise ValueError(f"--{key.replace('_', '-')} must be non-empty")
        params[key] = value
    if _mcmc_only_timestamp(params) is None and any(
        params.get(key) is not None for key in _MCMC_ARTIFACT_KEYS
    ):
        raise ValueError("MCMC draw-artifact options require --mcmc-only TIMESTAMP")
    return params


def _apply_config_overrides(params, overrides):
    for item in overrides:
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}.")
        params[key] = yaml.safe_load(value)
    return params


def _apply_demand_design_benchmark_defaults(params):
    dataset = _dataset(params)
    for retired in _RETIRED_KEYS:
        if retired in params:
            raise ValueError(f"`{retired}` is no longer supported; remove it.")
    if "outcome_to_particles_weight" in params:
        gamma = params["outcome_to_particles_weight"]
        if isinstance(gamma, bool) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError(
                "`outcome_to_particles_weight` must be one number in [0, 1]; "
                f"got {gamma!r}."
            )
        params["outcome_to_particles_weight"] = float(gamma)

    for key, value in _BENCHMARK_DIMENSIONS[dataset].items():
        if params.get(key, value) != value:
            raise ValueError(
                f"`{key}` is fixed to {value!r} for this benchmark; got {params[key]!r}."
            )
        params[key] = value
    params.setdefault("save_model", True)
    if dataset == "Sim_Demand_Design_Vector_PCAOnly_IV":
        if "pca_dim" not in params:
            raise ValueError("the vector-proxy config must explicitly supply pca_dim")
        _require_scalar(params, "pca_dim")
        params["pca_dim"] = parse_pca_dim(params["pca_dim"])

    normalized = validate_multistart_config(params)
    params.clear()
    params.update(normalized)


def _configure_tensorflow_threads(intra_op_threads=None, inter_op_threads=None):
    if intra_op_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_op_threads))
    if inter_op_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_op_threads))


def _configure_tensorflow_devices(use_gpu=False):
    gpus = tf.config.list_physical_devices("GPU")
    if use_gpu:
        if not gpus:
            raise RuntimeError("TensorFlow GPU was required, but no GPU was detected")
        for gpu in gpus:
            if not tf.config.experimental.get_memory_growth(gpu):
                tf.config.experimental.set_memory_growth(gpu, True)
            if not tf.config.experimental.get_memory_growth(gpu):
                raise RuntimeError("TensorFlow GPU memory growth was not enabled")
        print(f"TensorFlow GPU enabled with {len(gpus)} device(s).")
        return
    if tf.config.get_visible_devices("GPU"):
        tf.config.set_visible_devices([], "GPU")
    print("TensorFlow GPU disabled. Using CPU only.")


def _fit_standardizer(data):
    mean = np.mean(data, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(data, axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def _transform(data, stats):
    return ((data - stats["mean"]) / stats["scale"]).astype(np.float32)


def _inverse_transform(data, stats):
    return (data * stats["scale"] + stats["mean"]).astype(np.float32)


def _render_demand_design_run_config(params):
    lines = ["Demand-design run config:"]
    keys = [
        "n_samples",
        "rho",
        "n_repeat",
        "repeat_id",
        "run_seed",
        "z_dims",
        "v_dim",
        "w_dim",
        "representation_sd",
        "pca_dim",
        "outcome_to_particles_weight",
        "egm_num_warm_starts",
        "structural_methods",
        "mcmc_num_chains",
        "mcmc_production_warmup_steps",
        "mcmc_production_draws",
    ]
    for key in keys:
        if key in params:
            lines.append(f"  {key}: {params.get(key)}")
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


def _build_demand_design_run_timestamp(now=None):
    now = datetime.now() if now is None else now
    return now.strftime("%Y-%m-%d_%H-%M-%S-%f")


def _build_demand_design_run_id(params, now=None):
    return f"{_dataset(params).lower()}_{_build_demand_design_run_timestamp(now)}"


def _demand_design_src_root():
    return Path(__file__).resolve().parent


def _demand_design_dumps_dir():
    return _demand_design_src_root() / "dumps"


def _build_demand_design_combo_dir_name(params):
    name = (
        f"n_samples:{int(params['n_samples'])}"
        f"-rho:{_format_demand_design_sweep_value(params['rho'])}"
        f"-v_dim:{int(params['v_dim'])}"
    )
    if params.get("dataset") == "Sim_Demand_Design_Vector_PCAOnly_IV":
        name += f"-pca_dim:{int(params['pca_dim'])}"
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
    destination_name = (
        source_path.name if source_path is not None else f"{_dataset(params)}.yaml"
    )
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


_FINAL_RESULT_COLUMNS = (
    "repeat_id",
    "run_seed",
    "model_seed",
    "structural_mse_map",
    "mcmc_cov90",
    "mcmc_cov95",
    "mcmc_cov99",
    "mcmc_width90",
    "mcmc_width95",
    "mcmc_width99",
    "mcmc_family",
    "mcmc_num_targets",
    "mcmc_num_queries",
    "mcmc_num_chains",
    "mcmc_draws_per_chain",
    "mcmc_seconds",
    "mcmc_uq_seconds",
    "checkpoint_timestamp",
    "checkpoint_path",
    "outcome_to_particles_weight",
    "egm_selected_candidate_id",
    "egm_selected_criterion",
    "egm_selected_train_iv_map",
    "egm_selected_train_mse_y",
    "device_name",
    "hostname",
    "pca_dim",
    "mcmc_only",
)


def _blank(value):
    return "" if value is None else value


def _build_final_results_row(params, final_results, provenance):
    row = {column: "" for column in _FINAL_RESULT_COLUMNS}
    row["repeat_id"] = int(params.get("repeat_id", 0))
    row["run_seed"] = _blank(params.get("run_seed"))
    row["structural_mse_map"] = final_results["map"]
    mcmc = final_results.get("_mcmc")
    if mcmc is not None:
        readout = mcmc["readout"]
        coverage = readout["coverage"]
        row.update(
            {
                "mcmc_cov90": coverage["0.9"],
                "mcmc_cov95": coverage["0.95"],
                "mcmc_cov99": coverage["0.99"],
                "mcmc_width90": readout["width90"],
                "mcmc_width95": readout["width95"],
                "mcmc_width99": readout["width99"],
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
        "model_seed",
        "checkpoint_timestamp",
        "checkpoint_path",
        "device_name",
    ):
        row[key] = _blank(provenance.get(key))
    row["hostname"] = _blank((provenance.get("execution_environment") or {}).get("hostname"))
    for key in (
        "outcome_to_particles_weight",
        "pca_dim",
    ):
        if key in params:
            row[key] = _blank(params.get(key))
    multistart = provenance.get("egm_multistart") or {}
    for key in (
        "egm_selected_candidate_id",
        "egm_selected_criterion",
        "egm_selected_train_iv_map",
        "egm_selected_train_mse_y",
    ):
        if key in multistart:
            row[key] = _blank(multistart.get(key))
    row["mcmc_only"] = _blank(provenance.get("mcmc_only"))
    return row


def _persist_demand_design_repeat_outputs(run_root, params, repeat_outputs):
    combo_dir = run_root / _build_demand_design_combo_dir_name(params)
    combo_dir.mkdir(parents=True, exist_ok=True)
    final_results = repeat_outputs["final_results"]
    provenance = repeat_outputs["provenance"]

    _csv_writer_append_rows(
        combo_dir / "results.csv",
        _FINAL_RESULT_COLUMNS,
        [_build_final_results_row(params, final_results, provenance)],
    )
    repeat_id = int(params.get("repeat_id", 0))
    timestamp = provenance.get("checkpoint_timestamp", "unknown")
    record_dir = combo_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "bgm-repeat-record",
        "repeat_id": repeat_id,
        "run_config_text": repeat_outputs["run_config_text"],
        "provenance": provenance,
        "final_results": {
            key: value for key, value in final_results.items() if not str(key).startswith("_")
        },
        "mcmc": final_results.get("_mcmc"),
    }
    with (record_dir / f"repeat{repeat_id}_{timestamp}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(record, handle, indent=1, default=_json_default)


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


def _make_demand_design_sweep_output_dir(base_output_dir, n_samples, repeat_id, rho):
    return (
        f"{base_output_dir}/sweeps/n_samples={n_samples}"
        f"__rho={_format_demand_design_sweep_value(rho)}__repeat={repeat_id}"
    )


def _require_scalar(params, field):
    if isinstance(params.get(field), (list, tuple)):
        raise ValueError(
            f"one process runs one cell; set a scalar {field} with --set {field}=..."
        )


def _resolve_demand_design_cell(params):
    for field in ("n_samples", "rho", "pca_dim"):
        _require_scalar(params, field)
    for field in ("n_samples", "rho"):
        if field not in params:
            raise ValueError(f"set {field} with --set {field}=...")
    n_repeat = _resolve_demand_design_repeat_count(params)
    repeat_id = params.get("_only_repeat_id")
    if repeat_id is None:
        if n_repeat > 1:
            raise ValueError(
                f"n_repeat={n_repeat}: pass --repeat-id in 0..{n_repeat - 1} "
                "(one process per repeat)"
            )
        repeat_id = 0
    repeat_id = int(repeat_id)
    if not 0 <= repeat_id < n_repeat:
        raise ValueError(
            f"--repeat-id must lie in 0..{n_repeat - 1} (n_repeat={n_repeat}); "
            f"got {repeat_id}."
        )
    run_params = dict(params)
    run_params["n_samples"] = int(params["n_samples"])
    run_params["rho"] = float(params["rho"])
    run_params["n_repeat"] = n_repeat
    run_params["repeat_id"] = repeat_id
    run_params["run_seed"] = repeat_id
    if n_repeat > 1 and run_params.get("save_model"):
        run_params["output_dir"] = _make_demand_design_sweep_output_dir(
            str(run_params.get("output_dir", ".")),
            run_params["n_samples"],
            repeat_id,
            run_params["rho"],
        )
    return run_params


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


_MCMC_BUDGET_KEYS = (
    "mcmc_num_chains", "mcmc_production_warmup_steps", "mcmc_production_draws"
)


def _check_structural_methods(params):
    methods = params.get("structural_methods", ["map"])
    if not isinstance(methods, (list, tuple)):
        methods = [methods]
    methods = [str(method).strip() for method in methods]
    if methods not in (["map"], ["map", "mcmc"]):
        raise ValueError(
            "`structural_methods` must be [map] or [map, mcmc]; "
            f"got {params.get('structural_methods')!r}."
        )
    params["structural_methods"] = methods
    missing = [key for key in _MCMC_BUDGET_KEYS if params.get(key) is None]
    if "mcmc" in methods and missing:
        raise ValueError(f"structural_methods=[map, mcmc] needs {', '.join(missing)}")


def _resolve_model_seed(params):
    if params.get("model_seed") is None:
        params["model_seed"] = draw_model_seed()
    params["model_seed"] = normalize_model_seed(params["model_seed"])
    print(f"model_seed: {params['model_seed']}")
    return params["model_seed"]


def _model_random_seed(params, stream, candidate_id=None):
    return derive_seed(params["model_seed"], stream, candidate_id)


def _mcmc_seeds(model_seed):
    return {
        "pilot": derive_seed(model_seed, "mcmc-pilot"),
        "production": derive_seed(model_seed, "mcmc-production"),
    }


_MCMC_ONLY_TIMESTAMP_KEY = "_mcmc_only_timestamp"


def _mcmc_only_timestamp(params):
    return params.get(_MCMC_ONLY_TIMESTAMP_KEY)


def _is_mcmc_only(params):
    return _mcmc_only_timestamp(params) is not None


def _uses_egm_multistart(params):
    return (
        not _is_mcmc_only(params)
        and int(params.get("egm_num_warm_starts", 1)) > 1
    )


def _initialize_egm_candidate_worker(use_gpu):
    _configure_tensorflow_threads(1, 1)
    _configure_tensorflow_devices(bool(use_gpu))
    if use_gpu and not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("EGM multistart requested GPU but the worker sees no GPU")


def _evaluate_training_iv_criterion(model, train, *, y_raw, y_stats):
    observed = np.asarray(y_raw, np.float64).reshape(-1)
    if observed.shape[0] != np.asarray(train["v"]).shape[0]:
        raise ValueError("criterion outcome rows must match the training rows")
    data_w = tf.constant(
        np.asarray(train["w"], np.float32).reshape(observed.shape[0], -1)
    )
    data_z = model.infer_latent_from_covariates(train["v"])
    prediction = model._integrated_outcome_mean(
        tf.constant(np.asarray(data_z, np.float32)), data_w
    ).numpy()
    if y_stats is not None:
        prediction = _inverse_transform(prediction, y_stats)
    score = float(
        np.mean((observed - np.asarray(prediction, np.float64).reshape(-1)) ** 2)
    )
    print(f"Training-set IV-moment residual [map] = {score:.6f}")
    return {"train_iv_map": score}


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
    candidate_root,
):
    candidate_root_path = Path(candidate_root)
    candidate_root_path.mkdir(parents=True, exist_ok=True)
    stdout_path = candidate_root_path / "candidate.stdout.log"
    stderr_path = candidate_root_path / "candidate.stderr.log"
    started_at = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
    try:
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_handle, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_handle, contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
            candidate_params = dict(params)
            candidate_params["output_dir"] = str(candidate_root)
            candidate_params["save_model"] = True
            model_cls = _model_class(candidate_params)
            egm_started = time.time()
            model = model_cls(
                params=candidate_params,
                timestamp=f"egm_candidate_{int(candidate_id):02d}",
                random_seed=int(init_seed),
                auto_restore_checkpoint=False,
            )
            # Shared schedule seed: candidates differ only in network initialization.
            tf.keras.utils.set_random_seed(int(schedule_seed))
            np.random.seed(int(schedule_seed))
            model.egm_init(
                data=(train["x"], train["y"], train["v"], train["w"]),
                egm_n_iter=int(candidate_params.get("fit_egm_n_iter", 10000)),
                batch_size=int(candidate_params.get("fit_batch_size", 32)),
                verbose=1,
            )
            checkpoint_path = model.save_model_state_checkpoint(
                candidate_root_path / "egm-transition" / "ckpt"
            )
            egm_seconds = time.time() - egm_started
            logical_gpus = [
                device.name for device in tf.config.list_logical_devices("GPU")
            ]
            device_names = logical_gpus or ["cpu"]
            del model
            tf.keras.backend.clear_session()
            gc.collect()

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
            tf.keras.utils.set_random_seed(int(post_egm_seed))
            np.random.seed(int(post_egm_seed))
            model.fit_bgm_from_egm(
                data=(train["x"], train["y"], train["v"], train["w"]),
                epochs=int(candidate_params.get("fit_epochs", 100)),
                epochs_per_eval=int(candidate_params.get("fit_epochs_per_eval", 10)),
                batch_size=int(candidate_params.get("fit_batch_size", 32)),
                verbose=1,
                initialize_latents_from_encoder=True,
            )
            bgm_seconds = time.time() - bgm_started

            tf.keras.utils.set_random_seed(int(criterion_seed))
            np.random.seed(int(criterion_seed))
            train_mse_x, train_mse_y, train_mse_v = model.evaluate(
                data=(train["x"], train["y"], train["v"], train["w"]),
                data_z=None,
            )
            criterion = _evaluate_training_iv_criterion(
                model,
                train,
                y_raw=criterion_y_raw,
                y_stats=criterion_y_stats,
            )
            print(
                f"Candidate {int(candidate_id)} post-BGM criterion: "
                f"train_iv_map={criterion['train_iv_map']:.6f}; "
                f"encoder-latent fit train_mse_y={float(train_mse_y):.6f}"
            )
            bgm_checkpoint_path = model.save_model_state_checkpoint(
                candidate_root_path / "bgm-final" / "ckpt"
            )
            diagnostics_finite = all(
                np.isfinite(float(value))
                for value in (train_mse_x, train_mse_y, train_mse_v)
            )
            if np.isfinite(float(criterion["train_iv_map"])):
                status, failure_reason = "completed", None
            else:
                status, failure_reason = (
                    "nonfinite_criterion",
                    "non-finite post-BGM training criterion",
                )
            finished_at = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
            record = make_candidate_record(
                candidate_id=int(candidate_id),
                init_seed=int(init_seed),
                criterion_seed=int(criterion_seed),
                status=status,
                failure_reason=failure_reason,
                bgm_checkpoint_path=str(bgm_checkpoint_path),
                train_iv_map=float(criterion["train_iv_map"]),
                train_mse_x=float(train_mse_x),
                train_mse_y=float(train_mse_y),
                train_mse_v=float(train_mse_v),
                started_at=started_at,
                finished_at=finished_at,
                egm_seconds=float(egm_seconds),
                bgm_seconds=float(bgm_seconds),
                device_names=device_names,
                diagnostics_finite=diagnostics_finite,
            )
            record_path = candidate_root_path / "candidate.json"
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        return {
            "candidate_id": int(candidate_id),
            "record": record,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": None,
        }
    except Exception as exc:
        with stderr_path.open("a", encoding="utf-8") as stderr_handle:
            traceback.print_exc(file=stderr_handle)
        return {
            "candidate_id": int(candidate_id),
            "record": None,
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


def _fit_demand_design_model_multistart(params, train, *, criterion_data):
    observed_train = {
        key: np.asarray(train[key], np.float32)
        for key in ("x", "y", "v", "w")
    }
    num_starts = int(params["egm_num_warm_starts"])
    criterion_y_raw = np.asarray(criterion_data["y_raw"], np.float64).reshape(-1)
    if criterion_y_raw.shape[0] != observed_train["y"].shape[0]:
        raise ValueError("criterion_data['y_raw'] must have one row per training row")
    criterion_y_stats = criterion_data.get("y_stats")

    n_samples = int(params["n_samples"])
    rho = float(params["rho"])
    repeat_id = int(params.get("repeat_id", 0))
    seeds = derive_multistart_seeds(params["model_seed"], num_starts)
    bundle_id = (
        f"n={n_samples}_rho={format(rho, '.17g')}_repeat={repeat_id}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
    )
    bundle_root = (
        Path(str(params.get("output_dir", ".")))
        / "egm_multistart"
        / bundle_id
    )
    bundle_root.mkdir(parents=True, exist_ok=False)

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
        initargs=(bool(params.get("use_gpu", False)),),
    ) as executor:
        futures = []
        for candidate_id, init_seed in enumerate(seeds["init_seeds"]):
            candidate_root = bundle_root / f"candidate_{candidate_id:02d}"
            futures.append(
                executor.submit(
                    _run_egm_candidate_worker,
                    candidate_id,
                    params,
                    observed_train,
                    init_seed=int(init_seed),
                    schedule_seed=int(seeds["schedule_seed"]),
                    post_egm_seed=int(seeds["post_egm_seed"]),
                    criterion_seed=int(seeds["criterion_seeds"][candidate_id]),
                    criterion_y_raw=criterion_y_raw,
                    criterion_y_stats=criterion_y_stats,
                    candidate_root=str(candidate_root),
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
        record = result["record"]
        if record.get("status") not in {"completed", "nonfinite_criterion"}:
            raise RuntimeError(f"EGM candidate {candidate_id} has invalid status")
        checkpoint_path = record.get("bgm_checkpoint_path")
        if not checkpoint_path or not Path(str(checkpoint_path) + ".index").is_file():
            raise RuntimeError(f"BGM candidate {candidate_id} checkpoint is missing")

    candidate_criteria = {
        result["candidate_id"]: result["record"].get(EGM_SELECTION_CRITERION)
        for result in results
    }
    selection = select_candidate_by_criterion(candidate_criteria)
    selection_path = bundle_root / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                **selection,
                "candidates": [result["record"] for result in results],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    selected_id = int(selection["selected_candidate_id"])
    selected = next(item for item in results if item["candidate_id"] == selected_id)
    selected_checkpoint = str(selected["record"]["bgm_checkpoint_path"])

    _configure_tensorflow_devices(bool(params.get("use_gpu", False)))
    if bool(params.get("use_gpu", False)) and not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("EGM multistart requested GPU but the parent sees no GPU")
    tf.keras.utils.set_random_seed(int(seeds["post_egm_seed"]))
    np.random.seed(int(seeds["post_egm_seed"]))
    model_cls = _model_class(params)
    model = model_cls(
        params=params,
        random_seed=int(seeds["post_egm_seed"]),
        auto_restore_checkpoint=False,
    )
    model.restore_model_state_checkpoint(selected_checkpoint)
    tf.keras.utils.set_random_seed(int(seeds["post_egm_seed"]))
    np.random.seed(int(seeds["post_egm_seed"]))
    model.egm_multistart_provenance = {
        "egm_num_warm_starts": num_starts,
        "egm_selector_version": EGM_SELECTOR_VERSION,
        "egm_selection_criterion": EGM_SELECTION_CRITERION,
        "egm_selected_candidate_id": selected_id,
        "egm_selected_criterion": float(selection["selected_criterion"]),
        "egm_selected_train_iv_map": float(selected["record"]["train_iv_map"]),
        "egm_selected_train_mse_y": (
            None
            if selected["record"]["train_mse_y"] is None
            else float(selected["record"]["train_mse_y"])
        ),
        "egm_candidate_criteria": selection["candidate_criteria"],
        "egm_selection_path": str(selection_path),
        "model_seed": int(seeds["model_seed"]),
        "init_seeds": [int(value) for value in seeds["init_seeds"]],
        "schedule_seed": int(seeds["schedule_seed"]),
        "post_egm_seed": int(seeds["post_egm_seed"]),
        "criterion_seeds": [int(value) for value in seeds["criterion_seeds"]],
    }
    print(
        "Multistart selected candidate "
        f"{selected_id} with {EGM_SELECTION_CRITERION}="
        f"{float(selection['selected_criterion']):.6f}."
    )
    return model


def _fit_demand_design_model(params, train, criterion_data=None):
    if _uses_egm_multistart(params):
        return _fit_demand_design_model_multistart(
            params, train, criterion_data=criterion_data
        )
    model_cls = _model_class(params)
    model = model_cls(
        params=params,
        random_seed=_model_random_seed(params, "egm-init", 0),
        auto_restore_checkpoint=False,
    )
    schedule_seed = _model_random_seed(params, "egm-schedule")
    tf.keras.utils.set_random_seed(int(schedule_seed))
    np.random.seed(int(schedule_seed))
    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=int(params.get("fit_epochs", 100)),
        epochs_per_eval=int(params.get("fit_epochs_per_eval", 10)),
        batch_size=int(params.get("fit_batch_size", 32)),
        use_egm_init=True,
        egm_n_iter=int(params.get("fit_egm_n_iter", 10000)),
        verbose=1,
    )
    return model


def _restore_demand_design_model(params, timestamp):
    config = _load_train_config(params, timestamp)
    saved_seed = normalize_model_seed(config.get("model_seed"))
    if params.get("model_seed") is not None and int(params["model_seed"]) != saved_seed:
        raise RuntimeError(
            f"model_seed={params['model_seed']} differs from the checkpoint's "
            f"model_seed={saved_seed}; drop the override for --mcmc-only"
        )
    params["model_seed"] = saved_seed
    model_cls = _model_class(params)
    model = model_cls(
        params=params,
        timestamp=timestamp,
        random_seed=_model_random_seed(params, "post-egm"),
        auto_restore_checkpoint=False,
    )
    if str(getattr(model, "timestamp", "")) != str(timestamp):
        raise RuntimeError("restored model timestamp differs from the requested one")
    recorded = config.get("params") or {}
    current = _training_params(model.params)
    differing = sorted(
        key for key in set(recorded) | set(current)
        if recorded.get(key) != current.get(key)
    )
    if config.get("dataset") != str(params["dataset"]) or differing:
        raise RuntimeError(
            f"checkpoint {model.checkpoint_path} was trained with a different "
            f"configuration; differing keys: {differing or ['dataset']}"
        )
    model.restore_model_state_checkpoint(_inference_state_prefix(model, config))
    multistart = (config.get("notes") or {}).get("egm_multistart")
    if multistart:
        model.egm_multistart_provenance = multistart
    return model


def _fit_or_restore_demand_design_model(params, train, criterion_data=None):
    timestamp = _mcmc_only_timestamp(params)
    if timestamp is not None:
        print(f"Restoring checkpoint {timestamp} (mcmc-only; no training) ...")
        return _restore_demand_design_model(params, timestamp)
    _resolve_model_seed(params)
    model = _fit_demand_design_model(
        params,
        train,
        criterion_data=criterion_data,
    )
    notes = {}
    multistart = getattr(model, "egm_multistart_provenance", None)
    if multistart:
        notes["egm_multistart"] = multistart
    _write_train_config(model, params, notes)
    return model


_TRAIN_CONFIG_NAME = "train_config.json"
_RUN_CONTROL_KEYS = frozenset(
    {
        "output_dir",
        "save_model",
        "use_gpu",
        "num_tasks",
        "n_repeat",
        "structural_methods",
        "model_seed",
        *_MCMC_BUDGET_KEYS,
        *_MCMC_ARTIFACT_KEYS,
    }
)


def _training_params(params):
    kept = {
        key: value
        for key, value in params.items()
        if not str(key).startswith("_") and key not in _RUN_CONTROL_KEYS
    }
    return json.loads(json.dumps(kept, sort_keys=True, default=str))


def _load_train_config(params, timestamp):
    path = (
        Path(str(params.get("output_dir", ".")))
        / "checkpoints"
        / str(params["dataset"])
        / str(timestamp)
        / _TRAIN_CONFIG_NAME
    )
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"no checkpoint directory {path.parent}; check TIMESTAMP, the --set "
            "n_samples/rho/pca_dim values and --repeat-id"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint {path.parent} has no {_TRAIN_CONFIG_NAME}; it cannot be evaluated"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_train_config(model, params, notes=None):
    if not bool(params.get("save_model")):
        return None
    checkpoint_root = Path(model.checkpoint_path)
    path = checkpoint_root / _TRAIN_CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RuntimeError(f"refusing to overwrite an existing {path}") from None
    handed_over = False
    try:
        checkpoint_prefix = Path(
            model.save_model_state_checkpoint(checkpoint_root / "inference-state" / "ckpt")
        )
        payload = {
            "schema_version": "bgm-train-config",
            "dataset": str(params["dataset"]),
            "checkpoint_timestamp": str(model.timestamp),
            "model_seed": int(params["model_seed"]),
            "params": _training_params(model.params),
            "inference_state_prefix": checkpoint_prefix.relative_to(
                checkpoint_root
            ).as_posix(),
            "notes": json.loads(json.dumps(notes or {}, sort_keys=True, default=str)),
            "execution_environment": _mcmc_inference().execution_environment(),
        }
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        handed_over = True
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except BaseException:
        if not handed_over:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return payload


def _inference_state_prefix(model, config):
    relative_prefix = Path(str(config.get("inference_state_prefix", "")))
    if (
        not relative_prefix.parts
        or relative_prefix.is_absolute()
        or ".." in relative_prefix.parts
    ):
        raise RuntimeError("inference-state checkpoint path is not relative and safe")
    checkpoint_prefix = Path(model.checkpoint_path).resolve() / relative_prefix
    if not Path(str(checkpoint_prefix) + ".index").is_file():
        raise FileNotFoundError(
            f"inference-state checkpoint is missing: {checkpoint_prefix}"
        )
    return checkpoint_prefix


def _training_device_name(params):
    if not bool(params.get("use_gpu", False)):
        return "cpu"
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        return "cpu"
    visible = tf.config.get_visible_devices("GPU")
    if visible:
        try:
            details = tf.config.experimental.get_device_details(visible[0])
        except Exception:
            details = {}
        if details.get("device_name"):
            return str(details["device_name"])
    return logical[0].name


def _run_provenance(params, model):
    environment = _mcmc_inference().execution_environment()
    provenance = {
        "schema_version": "bgm-run-provenance",
        "dataset": params.get("dataset"),
        "repeat_id": int(params.get("repeat_id", 0)),
        "run_seed": params.get("run_seed"),
        "model_seed": params.get("model_seed"),
        "checkpoint_timestamp": str(getattr(model, "timestamp", "")),
        "checkpoint_path": str(getattr(model, "checkpoint_path", "")),
        "resolved_gamma": float(model._outcome_to_particles_weight()),
        "device_name": _training_device_name(params),
        "execution_environment": environment,
        "mcmc_only": _is_mcmc_only(params),
    }
    multistart = getattr(model, "egm_multistart_provenance", None)
    if multistart:
        provenance["egm_multistart"] = json.loads(
            json.dumps(multistart, sort_keys=True, default=str)
        )
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
        "family": _DATASETS[_dataset(params)][2],
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
    grid_x_full = np.asarray(grid_x_model, np.float32).reshape(-1, 1)
    grid_v_full = np.asarray(grid_v_raw, np.float32)
    truth_full = np.asarray(truth_rows, np.float64).reshape(-1)
    y_shift, y_scale = _standardizer_scalars(y_stats)
    x_shift, x_scale = _standardizer_scalars(x_stats)
    return _mcmc_inference().run_mcmc_grid(
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
        seeds=_mcmc_seeds(params["model_seed"]),
        source={
            "dataset": str(params["dataset"]),
            "repeat_id": int(params.get("repeat_id", 0)),
            "checkpoint_timestamp": str(model.timestamp),
        },
        production_num_chains=params["mcmc_num_chains"],
        production_warmup_steps=params["mcmc_production_warmup_steps"],
        production_draws=params["mcmc_production_draws"],
        artifact_root=params.get("mcmc_artifact_root"),
        arm_id=params.get("mcmc_arm_id"),
        readout_artifact_manifest=params.get("mcmc_readout_artifact"),
    )


def _evaluate_structural_methods(
    model,
    grid_x,
    grid_v,
    y_true,
    y_stats=None,
    mcmc_context=None,
):
    results = {}
    data_y_pred = model.predict_structural(grid_x, grid_v)
    if y_stats is not None:
        data_y_pred = _inverse_transform(data_y_pred, y_stats)
    mse = float(np.mean((y_true - data_y_pred) ** 2))
    results["map"] = mse
    print(f"Structural MSE [map] = {mse:.6f}")
    if mcmc_context is not None:
        record = _run_structural_mcmc(model, **mcmc_context)
        coverage = record["readout"]["coverage"]
        results["_mcmc"] = record
        print(
            "MCMC predictive intervals: "
            f"coverage 90/95/99 = {coverage['0.9']:.4f}/"
            f"{coverage['0.95']:.4f}/{coverage['0.99']:.4f}, "
            f"width 90/95/99 = {record['readout']['width90']:.4f}/"
            f"{record['readout']['width95']:.4f}/"
            f"{record['readout']['width99']:.4f} "
            f"[{record['grid']['num_queries']} full-grid queries, "
            f"{record['readout']['num_components']} retained draws]"
        )
    return results


def _finalize_demand_design_run(
    params,
    model,
    *,
    grid_x_model,
    grid_v_model,
    grid_truth,
    y_stats,
    mcmc_context,
    run_config_text,
    space_label,
):
    results = _evaluate_structural_methods(
        model,
        grid_x_model,
        grid_v_model,
        grid_truth,
        y_stats=y_stats,
        mcmc_context=mcmc_context,
    )
    provenance = _run_provenance(params, model)
    print(f"\nStructural MSE summary ({space_label})")
    print(f"  map: {results['map']:.6f}")
    print(
        f"  checkpoint {provenance['checkpoint_timestamp']} "
        f"gamma={provenance['resolved_gamma']} device={provenance['device_name']}"
    )
    return {
        "run_config_text": run_config_text,
        "final_results": results,
        "provenance": provenance,
    }


def _run_single_demand_design_iv(params):
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    n_samples = int(params["n_samples"])
    rho = float(params["rho"])
    run_seed = int(params["run_seed"])
    train = simulate_demand_design_iv(n_samples=n_samples, rho=rho, seed=run_seed)
    grid = make_demand_design_grid()
    print(_render_observed_ranges(train))

    print("\nNormalized-space experiment")
    train_std, grid_std, stats = _standardize_demand_design_data(train, grid)
    preprocessor = _affine_preprocessor(
        mean=np.asarray(stats["v"]["mean"], np.float32).reshape(-1),
        scale=np.asarray(stats["v"]["scale"], np.float32).reshape(-1),
    )
    model = _fit_or_restore_demand_design_model(
        params,
        train_std,
        criterion_data={"y_raw": train["y"], "y_stats": stats["y"]},
    )
    mcmc_context = None
    if "mcmc" in params["structural_methods"]:
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
        grid_x_model=grid_std["x"],
        grid_v_model=grid_std["v"],
        grid_truth=grid["y_struct"],
        y_stats=stats["y"],
        mcmc_context=mcmc_context,
        run_config_text=run_config_text,
        space_label="original outcome space",
    )


def _run_single_demand_design_mnist_iv(params):
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    n_samples = int(params["n_samples"])
    rho = float(params["rho"])
    run_seed = int(params["run_seed"])
    train = simulate_demand_design_mnist_iv(
        n_samples=n_samples, rho=rho, seed=run_seed
    )
    grid = make_demand_design_mnist_grid()
    print(_render_observed_ranges(train))

    print("\nNormalized-space experiment")
    train_std, grid_std, stats = _standardize_demand_design_image_data(train, grid)
    model = _fit_or_restore_demand_design_model(
        params,
        train_std,
        criterion_data={"y_raw": train["y"], "y_stats": stats["y"]},
    )
    mcmc_context = None
    if "mcmc" in params["structural_methods"]:
        mcmc_context = _mcmc_context(
            params,
            grid_x_model=grid_std["x"],
            grid_v_raw=grid_std["v"],
            truth_rows=grid["y_struct"],
            truth_label="mnist_demand_design_grid_y_struct",
            preprocessor=_affine_preprocessor(
                mean=np.zeros(int(params["v_dim"]), np.float32),
                scale=np.ones(int(params["v_dim"]), np.float32),
            ),
            x_stats=stats["x"],
            y_stats=stats["y"],
        )
    return _finalize_demand_design_run(
        params,
        model,
        grid_x_model=grid_std["x"],
        grid_v_model=grid_std["v"],
        grid_truth=grid["y_struct"],
        y_stats=stats["y"],
        mcmc_context=mcmc_context,
        run_config_text=run_config_text,
        space_label="original outcome space",
    )


def _apply_proxy_arm(data, transform):
    out = dict(data)
    v = np.asarray(data["v"])
    out["v"] = np.concatenate(
        [v[:, :1].astype(np.float32), apply_pca(transform, v[:, 1:])], axis=1
    ).astype(np.float32)
    return out


def _standardize_demand_design_pcaonly_data(train, grid):
    stats = {key: _fit_standardizer(train[key]) for key in ("x", "y", "w")}
    v = np.asarray(train["v"], dtype=np.float32)
    mean = np.mean(v, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(v, axis=0, keepdims=True).astype(np.float32)
    block = np.sqrt(np.mean(np.square(scale[:, 1:]))).astype(np.float32)
    scale[:, 1:] = block
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    stats["v"] = {"mean": mean, "scale": scale}
    train_std = {key: _transform(train[key], stats[key]) for key in ("x", "y", "v", "w")}
    train_std["y_struct"] = train["y_struct"]
    grid_std = {"x": _transform(grid["x"], stats["x"]),
                "v": _transform(grid["v"], stats["v"]), "y_struct": grid["y_struct"]}
    return train_std, grid_std, stats


def _run_single_demand_design_vector_pcaonly_iv(params):
    params = dict(params)
    k = int(params["pca_dim"])
    n_samples = int(params["n_samples"])
    rho = float(params["rho"])
    run_seed = int(params["run_seed"])
    representation_sd = float(params.get("representation_sd", 0.5))
    train = simulate_demand_design_vector_iv(
        n_samples=n_samples, rho=rho, seed=run_seed, representation_sd=representation_sd)
    grid = make_demand_design_vector_grid(representation_sd=representation_sd)
    raw_proxy = np.asarray(train["v"])[:, 1:]
    transform = fit_pca(raw_proxy, k)
    train = _apply_proxy_arm(train, transform)
    grid = _apply_proxy_arm(grid, transform)
    params["v_dim"] = 1 + k
    print(f"Proxy PCA: {k} components; the model sees v_dim={1 + k}")
    run_config_text = _render_demand_design_run_config(params)
    print(run_config_text)
    print(_render_observed_ranges(train))
    train_std, grid_std, stats = _standardize_demand_design_pcaonly_data(train, grid)
    preprocessor = _affine_preprocessor(
        mean=np.asarray(stats["v"]["mean"], np.float32).reshape(-1),
        scale=np.asarray(stats["v"]["scale"], np.float32).reshape(-1))
    model = _fit_or_restore_demand_design_model(
        params, train_std,
        criterion_data={"y_raw": train["y"], "y_stats": stats["y"]})
    mcmc_context = None
    if "mcmc" in params["structural_methods"]:
        mcmc_context = _mcmc_context(
            params, grid_x_model=grid_std["x"], grid_v_raw=grid["v"],
            truth_rows=grid["y_struct"], truth_label="vector_pcaonly_demand_design_grid_y_struct",
            preprocessor=preprocessor, x_stats=stats["x"], y_stats=stats["y"])
    return _finalize_demand_design_run(
        params, model, grid_x_model=grid_std["x"],
        grid_v_model=grid_std["v"], grid_truth=grid["y_struct"], y_stats=stats["y"],
        mcmc_context=mcmc_context,
        run_config_text=run_config_text,
        space_label="original outcome space")


def _run_demand_design_family(run_params):
    run_root = _demand_design_dumps_dir() / _build_demand_design_run_id(run_params)
    run_root.mkdir(parents=True, exist_ok=True)
    _copy_demand_design_config_snapshot(run_params, run_root)
    print(
        f"\nDemand-design cell: n_samples={run_params['n_samples']}, "
        f"rho={run_params['rho']}, repeat={run_params['repeat_id']}"
    )
    repeat_outputs = _single_run_fn(run_params)(run_params)
    _persist_demand_design_repeat_outputs(run_root, run_params, repeat_outputs)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = args.config

    with open(config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["_config_source_path"] = config
    _apply_config_overrides(params, args.overrides)
    params.pop(_MCMC_ONLY_TIMESTAMP_KEY, None)
    if args.repeat_id is not None:
        params["_only_repeat_id"] = int(args.repeat_id)
    if args.mcmc_only is not None:
        mcmc_only_timestamp = str(args.mcmc_only).strip()
        if not mcmc_only_timestamp:
            parser.error("--mcmc-only requires a non-empty TIMESTAMP")
        if args.repeat_id is None:
            raise ValueError("--mcmc-only requires --repeat-id.")
        params[_MCMC_ONLY_TIMESTAMP_KEY] = mcmc_only_timestamp
    _apply_mcmc_artifact_options(params, args)
    _apply_demand_design_benchmark_defaults(params)
    _check_structural_methods(params)
    if "mcmc" not in params["structural_methods"] and any(
        params.get(key) is not None for key in _MCMC_ARTIFACT_KEYS
    ):
        raise ValueError(
            "--mcmc-artifact-root/--mcmc-arm-id/--mcmc-readout-artifact need "
            "--set structural_methods=[map,mcmc]"
        )
    run_params = _resolve_demand_design_cell(params)

    # Must run before data loading, which fixes TensorFlow's device options.
    _configure_tensorflow_devices(bool(run_params.get("use_gpu", False)))
    if _uses_egm_multistart(run_params):
        print(
            "TensorFlow parent device configuration initialized before data "
            "loading; candidate training remains deferred to spawned workers."
        )

    _run_demand_design_family(run_params)


if __name__ == "__main__":
    main()
