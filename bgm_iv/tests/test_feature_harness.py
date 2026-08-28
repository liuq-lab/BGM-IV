"""End-to-end harness smoke for the MNIST representation dataset (tiny nets).

Runs both grid-blind stages on fake MNIST (stage-1 pixel model trained in
process, stage-2 vector model on (time, phi)), checks the export provenance,
the held-out criterion, the results rows, and the mcmc-only restore path.
No mcmc here (connectivity of the sampler is covered in test_inference).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import main as main_module
import bgm_iv.datasets.simulator_image as simulator_image_module

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _fake_mnist_arrays():
    rng = np.random.default_rng(0)
    base = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28)
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for digit in range(10):
        for replica in range(6):
            train_images.append(((base + digit * 11 + replica) % 256).astype(np.uint8))
            train_labels.append(digit)
            test_images.append(((base + digit * 17 + replica + 5) % 256).astype(np.uint8))
            test_labels.append(digit)
    return {
        "train_images": np.stack(train_images),
        "train_labels": np.asarray(train_labels, np.int64),
        "test_images": np.stack(test_images),
        "test_labels": np.asarray(test_labels, np.int64),
    }


@pytest.fixture
def fake_mnist(monkeypatch):
    arrays = _fake_mnist_arrays()
    monkeypatch.setattr(simulator_image_module, "_load_mnist_arrays", lambda: arrays)
    return arrays


def _feature_params(tmp_path, **overrides):
    params = {
        "dataset": "Sim_Demand_Design_Mnist_Feature_IV",
        "output_dir": str(tmp_path),
        "save_res": False,
        "save_model": True,
        "use_gpu": False,
        "binary_treatment": False,
        "use_bnn": False,
        "n_samples": 40,
        "rho": 0.5,
        "n_repeat": 1,
        "seed": 0,
        "holdout_n_samples": 28,
        "pixel_fit_epochs": 1,
        "pixel_fit_batch_size": 8,
        "pixel_fit_egm_n_iter": 2,
        "pixel_z_dims": [1, 1, 1, 1],
        "fit_epochs": 1,
        "fit_epochs_per_eval": 1,
        "fit_batch_size": 8,
        "fit_egm_n_iter": 2,
        "fit_egm_batches_per_eval": 100,
        "fit_first_stage_warmup_epochs": 0,
        "z_dims": [1, 1, 1, 1],
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "e_units": [16, 16],
        "dz_units": [16, 8],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "structural_map_steps": 2,
        "structural_map_lr": 1e-4,
        "covariate_block_scale": "sum",
        "outcome_to_particles_weight": 0.25,
        "sigma_time_softfloor": 0.1,
        "sigma_y_softfloor": 0.1,
        "sigma_vector_softfloor": "rule",
        "deterministic_training": True,
        "training_grid_monitor": False,
        "structural_methods": ["map", "encoder"],
        "nb_intervals": 4,
    }
    params.update(overrides)
    return params


def _run(params):
    main_module._apply_demand_design_benchmark_defaults(params)
    main_module._validate_map_only_structural_config(params)
    runs = list(main_module._iter_demand_design_sweep_runs(params))
    assert len(runs) == 1
    _, _, run_params = runs[0]
    return run_params, main_module._run_single_demand_design_mnist_feature_iv(run_params)


def test_feature_runner_two_stage_grid_blind(fake_mnist, tmp_path):
    params = _feature_params(tmp_path)
    run_params, outputs = _run(params)
    results = outputs["final_results"]
    provenance = outputs["provenance"]
    assert np.isfinite(results["map"]) and np.isfinite(results["encoder"])
    assert np.isfinite(results["holdout_iv_mse_map"])
    assert provenance["params"]["outcome_to_particles_weight"] == 0.25
    assert provenance["resolved_gamma"] == 0.25
    export = provenance["feature_export"]
    assert export["feature_map"] == "egm" and export["feature_dim"] == 64 and export["noise_dim"] == 0
    assert len(export["trunk_weights_sha256"]) == 64
    assert provenance["sigma_vector_softfloor_source"] == "rule"
    assert provenance["sigma_vector_softfloor_value"] >= 0.02
    assert provenance["pixel_stage"]["timestamp"]
    assert provenance["feature_preprocessor"]["model_dimension"] == 65
    assert provenance["feature_preprocessor"]["trunk_weights_sha256"] == export["trunk_weights_sha256"]
    # no test-grid metric was recorded during training (grid-blind)
    assert all("structural_mse" not in record for record in outputs["training_history"])
    # persisted rows carry the provenance columns
    run_root = tmp_path / "dumps_root"
    run_root.mkdir()
    main_module._persist_demand_design_repeat_outputs(
        run_root, 1, 1, run_params, outputs["run_config_text"], outputs["ranges_text"],
        outputs["training_history"], final_results=results, provenance=provenance,
    )
    combo = next(run_root.iterdir())
    with (combo / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["trunk_weights_sha256"] == export["trunk_weights_sha256"]
    assert rows[0]["pixel_checkpoint_timestamp"] == provenance["pixel_stage"]["timestamp"]
    assert rows[0]["sigma_vector_softfloor_source"] == "rule"
    assert rows[0]["outcome_to_particles_weight"] == "0.25"
    assert rows[0]["training_grid_monitor"] == "False"
    assert rows[0]["structural_mse_mcmc"] == ""
    records = list((combo / "records").glob("repeat0_*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["provenance"]["feature_export"]["hashes"]["phi_train_sha256"]
    assert record["mcmc"] is None


def test_feature_runner_restores_pixel_checkpoint_and_mcmc_only(fake_mnist, tmp_path):
    params = _feature_params(tmp_path)
    run_params, outputs = _run(params)
    pixel = outputs["provenance"]["pixel_stage"]
    stage2_timestamp = outputs["provenance"]["checkpoint_timestamp"]
    reuse = _feature_params(
        tmp_path,
        pixel_checkpoint_dir=pixel["output_dir"],
        pixel_checkpoint_timestamp=pixel["timestamp"],
    )
    reuse_run_params, reuse_outputs = _run(reuse)
    assert reuse_outputs["provenance"]["feature_export"]["trunk_weights_sha256"] == (
        outputs["provenance"]["feature_export"]["trunk_weights_sha256"]
    )
    assert reuse_outputs["provenance"]["pixel_stage"]["timestamp"] == pixel["timestamp"]
    # mcmc-only: restore the stage-2 checkpoint, no training
    restore = _feature_params(
        tmp_path,
        pixel_checkpoint_dir=pixel["output_dir"],
        pixel_checkpoint_timestamp=pixel["timestamp"],
        _mcmc_only_timestamp=stage2_timestamp,
    )
    restore_run_params, restore_outputs = _run(restore)
    assert restore_outputs["provenance"]["mcmc_only"] is True
    assert restore_outputs["provenance"]["checkpoint_timestamp"] == stage2_timestamp
    assert restore_outputs["provenance"]["weights"] == outputs["provenance"]["weights"]
    assert restore_outputs["final_results"]["map"] == pytest.approx(
        outputs["final_results"]["map"], rel=1e-3
    )


def test_feature_runner_hd_three_block_floors(fake_mnist, tmp_path):
    params = _feature_params(tmp_path, pixel_v_dim=1000, sigma_noise_softfloor=0.1)
    run_params, outputs = _run(params)
    assert run_params["v_dim"] == 280 and run_params["vector_dim"] == 279
    floor = outputs["provenance"]["sigma_vector_softfloor_value"]
    assert isinstance(floor, list) and len(floor) == 2 and floor[1] == 0.1
    assert outputs["provenance"]["feature_export"]["noise_dim"] == 215
    assert outputs["provenance"]["feature_preprocessor"]["model_dimension"] == 280
    assert np.isfinite(outputs["final_results"]["map"])


def test_pca_control_arm_runs_without_pixel_stage(fake_mnist, tmp_path):
    # PCA-64 needs more than 64 training images
    params = _feature_params(tmp_path, feature_map="pca", n_samples=96)
    _, outputs = _run(params)
    assert outputs["provenance"]["pixel_stage"] is None
    assert outputs["provenance"]["feature_export"]["trunk_weights_sha256"] is None
    assert outputs["provenance"]["feature_preprocessor"]["kind"] == "pca-feature"
    assert np.isfinite(outputs["final_results"]["map"])
