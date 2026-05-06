import pytest
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import main as main_module

from bayesgm.datasets import (
    demand_design_h,
    demand_design_structural_function,
    make_demand_design_grid,
    simulate_demand_design_iv,
)
from bayesgm.models.causalbgm import CausalBGM_IV

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _make_params(output_dir):
    return {
        "dataset": "DemandDesignIVSmoke",
        "output_dir": str(output_dir),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 2,
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [16, 16],
        "e_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "dz_units": [16, 8],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "first_stage_warmup_epochs": 1,
        "structural_map_steps": 3,
    }


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _FakeExecutor:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.futures = []
        _FakeExecutor.instances.append(self)

    def submit(self, fn, *args, **kwargs):
        future = _FakeFuture(fn(*args, **kwargs))
        self.futures.append(future)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        return None


def test_demand_design_formula_matches_dfiv_reference():
    time = np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32)
    price = np.array([10.0, 12.5, 15.0, 17.5, 20.0], dtype=np.float32)
    group = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    psi_ref = 2.0 * (
        ((time - 5.0) ** 4) / 600.0
        + np.exp(-4.0 * (time - 5.0) ** 2)
        + time / 10.0
        - 2.0
    )
    structural_ref = 100.0 + (10.0 + price) * group * psi_ref - 2.0 * price

    np.testing.assert_allclose(demand_design_h(time), psi_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        demand_design_structural_function(price, time, group),
        structural_ref,
        rtol=1e-6,
        atol=1e-6,
    )


def test_demand_design_covariates_are_low_dimensional():
    train = simulate_demand_design_iv(n_samples=12, rho=0.5, seed=5)
    grid = make_demand_design_grid(price_points=3, time_points=2)

    assert train["v"].shape == (12, 2)
    assert grid["v"].shape == (3 * 2 * 7, 2)


def test_causalbgm_iv_smoke(tmp_path):
    train = simulate_demand_design_iv(n_samples=96, rho=0.5, seed=7)
    params = _make_params(tmp_path)

    model = CausalBGM_IV(params=params, random_seed=13)
    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=2,
        epochs_per_eval=1,
        batch_size=24,
        use_egm_init=False,
        verbose=0,
    )

    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train["x"], train["y"], train["v"], train["w"]),
        data_z=None,
        nb_intervals=6,
    )
    assert causal_pre.shape == (6,)
    assert np.isfinite(float(mse_x))
    assert np.isfinite(float(mse_y))
    assert np.isfinite(float(mse_v))

    grid = make_demand_design_grid(price_points=4, time_points=3)
    structural_pred = model.predict_structural(
        grid["x"], grid["v"], latent_method="map", map_steps=3
    )
    structural_mse = model.evaluate_structural_mse(
        grid["x"], grid["v"], grid["y_struct"], latent_method="map", map_steps=3
    )
    assert structural_pred.shape == grid["y_struct"].shape
    assert np.isfinite(structural_mse)

    encoder_pred = model.predict_structural(grid["x"], grid["v"], latent_method="encoder")
    assert encoder_pred.shape == grid["y_struct"].shape

    unsupported_method = "mc" + "mc"
    with pytest.raises(ValueError, match="`method`"):
        model.predict_structural(grid["x"][:6], grid["v"][:6], latent_method=unsupported_method)


def test_covariate_posterior_is_untempered(tmp_path):
    params = _make_params(tmp_path)
    model = CausalBGM_IV(params=params, random_seed=3)

    data_v = tf.constant([[0.2, -0.1], [0.4, 0.7]], dtype=tf.float32)
    data_z = tf.constant(
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.1, -0.3, 0.2]], dtype=tf.float32
    )

    g_output = model.g_net(data_z)
    mu_v = g_output[:, : params["v_dim"]]
    sigma_square_v = model._continuous_sigma(g_output, sigma_key="sigma_v")
    loss_pv_z = model._gaussian_nll(
        data_v, mu_v, sigma_square_v, event_dim=params["v_dim"]
    )
    loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
    expected = -(loss_pv_z + loss_prior_z)

    actual = model.get_log_covariate_posterior(data_v, data_z)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_causalbgm_iv_rejects_alpha_v_param(tmp_path):
    params = _make_params(tmp_path)
    params["alpha_v"] = 1.0

    with pytest.raises(ValueError, match="alpha_v"):
        CausalBGM_IV(params=params, random_seed=3)


def test_causalbgm_iv_records_structural_history(tmp_path):
    train = simulate_demand_design_iv(n_samples=64, rho=0.5, seed=11)
    grid = make_demand_design_grid(price_points=3, time_points=2)
    params = _make_params(tmp_path)
    params["use_bnn"] = False

    model = CausalBGM_IV(params=params, random_seed=13)

    def callback(model, stage, epoch, metrics):
        return {
            "structural_mse": model.evaluate_structural_mse(
                grid["x"],
                grid["v"],
                grid["y_struct"],
                latent_method="map",
            )
        }

    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=1,
        epochs_per_eval=1,
        batch_size=16,
        use_egm_init=True,
        egm_n_iter=0,
        egm_batches_per_eval=1,
        verbose=0,
        first_stage_warmup_epochs=0,
        evaluation_callback=callback,
    )

    assert len(model.training_history) >= 2
    assert model.training_history[0]["stage"] == "post_egm"
    assert np.isfinite(model.training_history[0]["structural_mse"])
    assert model.best_training_record["epoch"] == 0


def test_resolve_training_monitor_methods_defaults_to_structural_methods():
    params = {}

    structural_methods = main_module._resolve_structural_methods(params)
    monitor_method, training_methods = main_module._resolve_training_monitor_methods(
        params,
        structural_methods,
    )

    assert structural_methods == ("map",)
    assert training_methods == structural_methods
    assert monitor_method == "map"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("structural_methods", ["encoder"]),
        ("structural_methods", ["mc" + "mc"]),
        ("structural_methods", [""]),
        ("training_structural_methods", ["encoder"]),
        ("training_structural_monitor_method", "encoder"),
        ("structural_latent_method", "encoder"),
    ],
)
def test_map_only_config_rejects_non_map_structural_methods(field, value):
    params = {
        "structural_methods": ["map"],
        "training_structural_methods": ["map"],
        "training_structural_monitor_method": "map",
        "structural_latent_method": "map",
    }
    params[field] = value

    with pytest.raises(ValueError, match=field):
        main_module._validate_map_only_structural_config(params)


def test_benchmark_defaults_inject_fixed_values_and_map_methods():
    params = {"dataset": "Sim_Demand_Design_IV"}

    main_module._apply_demand_design_benchmark_defaults(params)
    main_module._validate_map_only_structural_config(params)

    assert params["seed"] == 0
    assert "noise_seed" not in params
    assert params["price_points"] == 20
    assert params["time_points"] == 20
    assert params["v_dim"] == 2
    assert params["w_dim"] == 1
    assert params["fit_model_selection_metric"] == "structural_mse"
    assert params["fit_restore_best_weights"] is True
    assert params["fit_use_progress_bar"] is False
    assert params["structural_methods"] == ["map"]
    assert params["training_structural_methods"] == ["map"]
    assert params["training_structural_monitor_method"] == "map"
    assert params["structural_latent_method"] == "map"


def test_benchmark_defaults_keep_mnist_hd_v_dim_visible():
    params = {"dataset": "Sim_Demand_Design_Mnist_IV", "v_dim": 1000}

    main_module._apply_demand_design_benchmark_defaults(params)

    assert params["v_dim"] == 1000
    assert params["w_dim"] == 1
    assert params["image_seed"] == 42
    assert params["noise_seed"] == 42


def test_benchmark_defaults_reject_invalid_mnist_v_dim():
    params = {"dataset": "Sim_Demand_Design_Mnist_IV", "v_dim": 200}

    with pytest.raises(ValueError, match="v_dim"):
        main_module._apply_demand_design_benchmark_defaults(params)


@pytest.mark.parametrize(
    ("dataset", "field", "value"),
    [
        ("Sim_Demand_Design_IV", "seed", 1),
        ("Sim_Demand_Design_IV", "v_dim", 3),
        ("Sim_Demand_Design_IV", "noise_seed", 42),
        ("Sim_Demand_Design_IV", "price_points", 21),
        ("Sim_Demand_Design_IV", "fit_model_selection_metric", "mse_y"),
        ("Sim_Demand_Design_IV", "fit_restore_best_weights", False),
        ("Sim_Demand_Design_Mnist_IV", "image_seed", 99),
        ("Sim_Demand_Design_Vector_IV", "vector_dim", 128),
        ("Sim_Demand_Design_Vector_IV", "test_vector_seed", 99),
    ],
)
def test_benchmark_defaults_reject_non_default_hidden_fields(dataset, field, value):
    params = {"dataset": dataset, field: value}

    with pytest.raises(ValueError, match=field):
        main_module._apply_demand_design_benchmark_defaults(params)


def test_benchmark_defaults_reject_alpha_v():
    params = {"dataset": "Sim_Demand_Design_IV", "alpha_v": 1.0}

    with pytest.raises(ValueError, match="alpha_v"):
        main_module._apply_demand_design_benchmark_defaults(params)


def test_benchmark_defaults_accept_backward_compatible_default_values():
    params = {
        "dataset": "Sim_Demand_Design_IV",
        "seed": 0,
        "v_dim": 2,
        "price_points": 20,
        "fit_restore_best_weights": True,
    }

    main_module._apply_demand_design_benchmark_defaults(params)

    assert params["seed"] == 0
    assert params["v_dim"] == 2
    assert params["fit_restore_best_weights"] is True


def test_print_demand_design_run_config(capsys):
    params = {
        "n_samples": 1000,
        "rho": 0.5,
        "n_repeat": 3,
        "repeat_id": 1,
        "seed": 0,
        "run_seed": 1,
        "z_dims": [2, 1, 1, 7],
        "v_dim": 2,
        "w_dim": 1,
    }

    main_module._print_demand_design_run_config(params)
    captured = capsys.readouterr().out

    assert "Demand-design run config:" in captured
    assert "n_samples: 1000" in captured
    assert "rho: 0.5" in captured
    assert "n_repeat: 3" in captured
    assert "repeat_id: 1" in captured
    assert "seed: 0" in captured
    assert "run_seed: 1" in captured
    assert "z_dims: [2, 1, 1, 7]" in captured
    assert "v_dim: 2" in captured
    assert "w_dim: 1" in captured


def test_build_arg_parser_accepts_num_tasks():
    parser = main_module._build_arg_parser()
    args = parser.parse_args(["-c", "configs/Sim_Demand_Design_IV.yaml", "-t", "5"])

    assert args.config == "configs/Sim_Demand_Design_IV.yaml"
    assert args.num_tasks == 5


def test_resolve_parallel_gpu_slots_rejects_oversubscription(monkeypatch):
    monkeypatch.setattr(
        main_module.tf.config,
        "list_physical_devices",
        lambda device_type: [object(), object()] if device_type == "GPU" else [],
    )

    with pytest.raises(ValueError, match="exceeds the number of visible GPU devices"):
        main_module._resolve_parallel_gpu_slots(
            {
                "dataset": "Sim_Demand_Design_IV",
                "use_gpu": True,
                "num_tasks": 3,
            }
        )


def test_run_demand_design_family_parallel_flushes_results_in_run_order(
    monkeypatch,
    tmp_path,
    capsys,
):
    _FakeExecutor.instances = []
    params = {
        "dataset": "Sim_Demand_Design_IV",
        "n_samples": [64],
        "rho": [0.25],
        "n_repeat": 3,
        "seed": 0,
        "z_dims": [2, 1, 1, 7],
        "v_dim": 2,
        "w_dim": 1,
        "num_tasks": 2,
        "use_gpu": False,
        "save_model": False,
        "save_res": False,
    }
    persisted = []

    def fake_worker(run_index, total_runs, run_params):
        return {
            "run_index": run_index,
            "total_runs": total_runs,
            "params": run_params,
            "repeat_outputs": {
                "training_history": [],
                "training_history_text": "",
                "run_config_text": f"run {run_index}",
                "ranges_text": "",
            },
            "stdout": f"worker {run_index}\n",
            "stderr": "",
            "error": None,
        }

    def fake_persist(
        run_root,
        run_index,
        total_runs,
        params,
        run_config_text,
        ranges_text,
        training_history,
    ):
        persisted.append((run_index, params["repeat_id"], run_config_text))

    def reversed_completion(futures):
        return list(reversed(list(futures)))

    monkeypatch.setattr(main_module, "_run_demand_design_parallel_worker", fake_worker)
    monkeypatch.setattr(main_module, "_persist_demand_design_repeat_outputs", fake_persist)

    main_module._run_demand_design_family_parallel(
        params,
        tmp_path / "run-root",
        executor_factory=_FakeExecutor,
        as_completed_fn=reversed_completion,
    )
    captured = capsys.readouterr().out

    assert [item[0] for item in persisted] == [1, 2, 3]
    assert captured.index("worker 1") < captured.index("worker 2") < captured.index("worker 3")
    assert len(_FakeExecutor.instances) == 2
    assert all(instance.kwargs["max_workers"] == 1 for instance in _FakeExecutor.instances)


def test_run_demand_design_family_parallel_assigns_one_gpu_slot_per_worker(
    monkeypatch,
    tmp_path,
):
    _FakeExecutor.instances = []
    monkeypatch.setattr(
        main_module,
        "_resolve_parallel_gpu_slots",
        lambda params: (0, 1),
    )
    monkeypatch.setattr(
        main_module,
        "_run_demand_design_parallel_worker",
        lambda run_index, total_runs, run_params: {
            "run_index": run_index,
            "total_runs": total_runs,
            "params": run_params,
            "repeat_outputs": {
                "training_history": [],
                "training_history_text": "",
                "run_config_text": "",
                "ranges_text": "",
            },
            "stdout": "",
            "stderr": "",
            "error": None,
        },
    )
    monkeypatch.setattr(
        main_module,
        "_persist_demand_design_repeat_outputs",
        lambda *args, **kwargs: None,
    )

    main_module._run_demand_design_family_parallel(
        {
            "dataset": "Sim_Demand_Design_IV",
            "n_samples": [64],
            "rho": [0.25],
            "n_repeat": 2,
            "seed": 0,
            "z_dims": [2, 1, 1, 7],
            "v_dim": 2,
            "w_dim": 1,
            "num_tasks": 2,
            "use_gpu": True,
            "save_model": False,
            "save_res": False,
        },
        tmp_path / "run-root",
        executor_factory=_FakeExecutor,
        as_completed_fn=lambda futures: list(futures),
    )

    assert [instance.kwargs["initargs"] for instance in _FakeExecutor.instances] == [
        (True, 0),
        (True, 1),
    ]


def test_iter_demand_design_sweep_runs_preserves_scalar_behavior(tmp_path):
    params = {
        "n_samples": 1000,
        "rho": 0.5,
        "seed": 7,
        "output_dir": str(tmp_path),
        "save_model": False,
        "save_res": False,
    }

    runs = list(main_module._iter_demand_design_sweep_runs(params))

    assert len(runs) == 1
    run_index, total_runs, run_params = runs[0]
    assert run_index == 1
    assert total_runs == 1
    assert run_params["n_samples"] == 1000
    assert run_params["rho"] == 0.5
    assert run_params["n_repeat"] == 1
    assert run_params["repeat_id"] == 0
    assert run_params["run_seed"] == 7
    assert run_params["output_dir"] == str(tmp_path)


def test_iter_demand_design_sweep_runs_expands_list_n_samples():
    params = {
        "n_samples": [1000, 5000],
        "rho": 0.1,
        "seed": 5,
        "save_model": False,
        "save_res": False,
    }

    runs = list(main_module._iter_demand_design_sweep_runs(params))

    assert [(run["n_samples"], run["rho"], run["repeat_id"], run["run_seed"]) for _, _, run in runs] == [
        (1000, 0.1, 0, 5),
        (5000, 0.1, 0, 5),
    ]


def test_iter_demand_design_sweep_runs_expands_list_rho():
    params = {
        "n_samples": 1000,
        "rho": [0.1, 0.5, 0.9],
        "seed": 5,
        "save_model": False,
        "save_res": False,
    }

    runs = list(main_module._iter_demand_design_sweep_runs(params))

    assert [(run["n_samples"], run["rho"], run["repeat_id"], run["run_seed"]) for _, _, run in runs] == [
        (1000, 0.1, 0, 5),
        (1000, 0.5, 0, 5),
        (1000, 0.9, 0, 5),
    ]


def test_iter_demand_design_sweep_runs_uses_cartesian_product_in_order():
    params = {
        "n_samples": [1000, 5000],
        "rho": [0.1, 0.5],
        "seed": 5,
        "save_model": False,
        "save_res": False,
    }

    runs = list(main_module._iter_demand_design_sweep_runs(params))

    assert [(run["n_samples"], run["rho"], run["repeat_id"], run["run_seed"]) for _, _, run in runs] == [
        (1000, 0.1, 0, 5),
        (1000, 0.5, 0, 5),
        (5000, 0.1, 0, 5),
        (5000, 0.5, 0, 5),
    ]


def test_iter_demand_design_sweep_runs_rejects_empty_lists():
    with pytest.raises(ValueError, match="n_samples.*empty list"):
        list(
            main_module._iter_demand_design_sweep_runs(
                {"n_samples": [], "rho": 0.5, "save_model": False, "save_res": False}
            )
        )

    with pytest.raises(ValueError, match="rho.*empty list"):
        list(
            main_module._iter_demand_design_sweep_runs(
                {"n_samples": 1000, "rho": [], "save_model": False, "save_res": False}
            )
        )


def test_iter_demand_design_sweep_runs_expands_repeat_ids_inside_each_combination():
    params = {
        "n_samples": [1000, 5000],
        "rho": [0.1, 0.5],
        "seed": 10,
        "n_repeat": 2,
        "save_model": False,
        "save_res": False,
    }

    runs = list(main_module._iter_demand_design_sweep_runs(params))

    assert [(run["n_samples"], run["rho"], run["repeat_id"], run["run_seed"]) for _, _, run in runs] == [
        (1000, 0.1, 0, 10),
        (1000, 0.1, 1, 11),
        (1000, 0.5, 0, 10),
        (1000, 0.5, 1, 11),
        (5000, 0.1, 0, 10),
        (5000, 0.1, 1, 11),
        (5000, 0.5, 0, 10),
        (5000, 0.5, 1, 11),
    ]


def test_iter_demand_design_sweep_runs_rejects_invalid_repeat_count():
    with pytest.raises(ValueError, match="`n_repeat` must be >= 1"):
        list(
            main_module._iter_demand_design_sweep_runs(
                {
                    "n_samples": 1000,
                    "rho": 0.5,
                    "n_repeat": 0,
                    "save_model": False,
                    "save_res": False,
                }
            )
        )


def test_iter_demand_design_sweep_runs_uses_unique_output_dirs_when_saving(tmp_path):
    params = {
        "n_samples": [1000, 5000],
        "rho": [0.1, 0.5],
        "n_repeat": 2,
        "output_dir": str(tmp_path),
        "save_model": True,
        "save_res": False,
    }

    runs = list(main_module._iter_demand_design_sweep_runs(params))
    output_dirs = [run["output_dir"] for _, _, run in runs]

    assert len(output_dirs) == 8
    assert len(set(output_dirs)) == 8
    assert all("/sweeps/" in output_dir for output_dir in output_dirs)
    assert all("__repeat=" in output_dir for output_dir in output_dirs)


def test_run_demand_design_iv_sweep_prints_banners_and_concrete_configs(monkeypatch, capsys):
    params = {
        "n_samples": [64, 96],
        "rho": [0.25],
        "n_repeat": 2,
        "seed": 0,
        "z_dims": [2, 1, 1, 7],
        "v_dim": 2,
        "w_dim": 1,
        "save_model": False,
        "save_res": False,
    }
    seen = []

    def fake_run(run_params):
        seen.append(
            (
                run_params["n_samples"],
                run_params["rho"],
                run_params["repeat_id"],
                run_params["run_seed"],
            )
        )
        main_module._print_demand_design_run_config(run_params)

    monkeypatch.setattr(main_module, "_run_single_demand_design_iv", fake_run)

    main_module.run_demand_design_iv(params)
    captured = capsys.readouterr().out

    assert seen == [
        (64, 0.25, 0, 0),
        (64, 0.25, 1, 1),
        (96, 0.25, 0, 0),
        (96, 0.25, 1, 1),
    ]
    assert "Demand-design sweep run [1/4]: n_samples=64, rho=0.25, repeat=0" in captured
    assert "Demand-design sweep run [4/4]: n_samples=96, rho=0.25, repeat=1" in captured
    assert "n_samples: 64" in captured
    assert "n_samples: 96" in captured
    assert "rho: 0.25" in captured
    assert "n_repeat: 2" in captured
    assert "repeat_id: 0" in captured
    assert "repeat_id: 1" in captured
    assert "run_seed: 0" in captured
    assert "run_seed: 1" in captured


def test_demand_design_repeat_seed_reproduces_identical_train_and_grid_data():
    params = {
        "seed": 10,
        "n_repeat": 2,
    }
    repeat_id = 1
    run_seed = main_module._resolve_demand_design_run_seed(params, repeat_id)

    train_a = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed)
    train_b = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed)
    grid_a = make_demand_design_grid(price_points=3, time_points=2)
    grid_b = make_demand_design_grid(price_points=3, time_points=2)

    np.testing.assert_allclose(train_a["x"], train_b["x"], atol=0.0)
    np.testing.assert_allclose(train_a["y"], train_b["y"], atol=0.0)
    np.testing.assert_allclose(train_a["v"], train_b["v"], atol=0.0)
    np.testing.assert_allclose(train_a["w"], train_b["w"], atol=0.0)
    np.testing.assert_allclose(grid_a["x"], grid_b["x"], atol=0.0)
    np.testing.assert_allclose(grid_a["v"], grid_b["v"], atol=0.0)
    np.testing.assert_allclose(grid_a["y_struct"], grid_b["y_struct"], atol=0.0)


def test_demand_design_repeat_id_changes_train_data_but_not_grid_data():
    params = {
        "seed": 10,
        "n_repeat": 2,
    }
    run_seed_0 = main_module._resolve_demand_design_run_seed(params, 0)
    run_seed_1 = main_module._resolve_demand_design_run_seed(params, 1)

    train_0 = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed_0)
    train_1 = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed_1)
    grid_0 = make_demand_design_grid(price_points=3, time_points=2)
    grid_1 = make_demand_design_grid(price_points=3, time_points=2)

    assert not np.allclose(train_0["x"], train_1["x"])
    assert not np.allclose(train_0["y"], train_1["y"])
    assert not np.allclose(train_0["v"], train_1["v"])
    assert not np.allclose(train_0["w"], train_1["w"])
    np.testing.assert_allclose(grid_0["x"], grid_1["x"], atol=0.0)
    np.testing.assert_allclose(grid_0["v"], grid_1["v"], atol=0.0)
    np.testing.assert_allclose(grid_0["y_struct"], grid_1["y_struct"], atol=0.0)


class _DummyStructuralModel:
    def __init__(self):
        self.called_methods = []

    def predict_structural(self, grid_x, grid_v, latent_method):
        self.called_methods.append(latent_method)
        return np.zeros((len(grid_x), 1), dtype=np.float32)


def test_training_structural_methods_limit_training_callback_but_not_final_summary():
    params = {
        "structural_methods": ["map"],
        "training_structural_methods": ["map"],
        "training_structural_monitor_method": "map",
    }
    structural_methods = main_module._resolve_structural_methods(params)
    monitor_method, training_methods = main_module._resolve_training_monitor_methods(
        params,
        structural_methods,
    )
    callback = main_module._make_structural_monitor_callback(
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        latent_method=monitor_method,
        additional_methods=[method for method in training_methods if method != monitor_method],
    )

    callback_model = _DummyStructuralModel()
    callback_metrics = callback(
        model=callback_model,
        stage="epoch_eval",
        epoch=0,
        metrics={},
    )

    assert callback_model.called_methods == ["map"]
    assert callback_metrics["structural_mse"] == callback_metrics["structural_mse_map"]

    final_model = _DummyStructuralModel()
    final_results = main_module._evaluate_structural_methods(
        final_model,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        methods=structural_methods,
    )

    assert final_model.called_methods == ["map"]
    assert tuple(final_results.keys()) == structural_methods


def test_training_structural_monitor_method_must_be_in_training_structural_methods():
    params = {
        "training_structural_methods": ["map"],
        "training_structural_monitor_method": "encoder",
    }

    with pytest.raises(
        ValueError,
        match="training_structural_monitor_method",
    ):
        main_module._resolve_training_monitor_methods(params, ("map",))


def test_build_demand_design_run_timestamp_format():
    stamp = main_module._build_demand_design_run_timestamp(
        datetime(2026, 5, 5, 14, 47, 28, 123456)
    )

    assert stamp == "2026-05-05_14-47-28-123456"


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        (
            "Sim_Demand_Design_IV",
            "sim_demand_design_iv_2026-05-05_14-47-28-123456",
        ),
        (
            "Sim_Demand_Design_Mnist_IV",
            "sim_demand_design_mnist_iv_2026-05-05_14-47-28-123456",
        ),
        (
            "Sim_Demand_Design_Vector_IV",
            "sim_demand_design_vector_iv_2026-05-05_14-47-28-123456",
        ),
    ],
)
def test_build_demand_design_run_id_prefixes_dataset_slug(dataset, expected):
    run_id = main_module._build_demand_design_run_id(
        {"dataset": dataset},
        datetime(2026, 5, 5, 14, 47, 28, 123456),
    )

    assert run_id == expected


def test_build_demand_design_combo_dir_name_uses_requested_format():
    params = {"n_samples": 1000, "rho": 0.25, "v_dim": 2}
    assert (
        main_module._build_demand_design_combo_dir_name(params)
        == "n_samples:1000-rho:0.25-v_dim:2"
    )


def test_rectangularize_training_history_expands_sorted_structural_columns():
    history = [
        {
            "stage": "egm_init",
            "epoch": None,
            "include_outcome": False,
            "mse_x": 0.3,
            "mse_y": 0.2,
            "mse_v": 0.1,
            "structural_mse_map": 10.0,
        },
        {
            "stage": "epoch_eval",
            "epoch": 10,
            "include_outcome": True,
            "mse_x": 0.2,
            "mse_y": 0.1,
            "mse_v": 0.05,
            "structural_mse_map": 8.0,
        },
    ]

    columns, rows = main_module._rectangularize_training_history(history)

    assert columns == (
        "stage",
        "epoch",
        "include_outcome",
        "mse_x",
        "mse_y",
        "mse_v",
        "structural_mse_map",
    )
    assert rows[0]["structural_mse_map"] == 10.0
    assert rows[1]["structural_mse_map"] == 8.0


def test_persist_demand_design_repeat_outputs_writes_expected_files(tmp_path):
    run_root = tmp_path / "sim_demand_design_iv_2026-05-05_14-47-28-123456"
    run_root.mkdir()
    params = {
        "n_samples": 1000,
        "rho": 0.25,
        "repeat_id": 1,
        "v_dim": 2,
    }
    history = [
        {
            "stage": "egm_init",
            "epoch": None,
            "include_outcome": False,
            "mse_x": 0.3,
            "mse_y": 0.2,
            "mse_v": 0.1,
            "structural_mse_map": 10.0,
        },
        {
            "stage": "epoch_eval",
            "epoch": 10,
            "include_outcome": True,
            "mse_x": 0.2,
            "mse_y": 0.1,
            "mse_v": 0.05,
            "structural_mse_map": 8.0,
        },
    ]

    main_module._persist_demand_design_repeat_outputs(
        run_root,
        2,
        8,
        params,
        "Demand-design run config:\n  n_samples: 1000",
        "Observed data ranges before normalization:\n  x: min=0.0000, max=1.0000, mean=0.5000, std=0.1000",
        history,
    )

    combo_dir = run_root / "n_samples:1000-rho:0.25-v_dim:2"
    assert combo_dir.exists()

    repeat_csv = combo_dir / "training_metric_history_sim_demand_design_iv_n1000_rho0p25_repeat1.csv"
    best_csv = combo_dir / "best_structural_checkpoints.csv"
    last_csv = combo_dir / "last_structural_checkpoints.csv"
    combo_md = combo_dir / "training_metric_history_sim_demand_design_iv_n1000_rho0p25.md"

    assert repeat_csv.exists()
    assert best_csv.exists()
    assert last_csv.exists()
    assert combo_md.exists()

    best_lines = best_csv.read_text(encoding="utf-8").splitlines()
    assert best_lines[0] == "repeat_id,method,stage,epoch,include_outcome,mse_x,mse_y,mse_v,structural_mse"
    assert any(",map," in line for line in best_lines[1:])

    combo_md_text = combo_md.read_text(encoding="utf-8")
    assert "## Demand-design sweep run [2/8]: n_samples=1000, rho=0.25, repeat=1" in combo_md_text
    assert "Demand-design run config:" in combo_md_text
    assert "Observed data ranges before normalization:" in combo_md_text
    assert "Training metric history" in combo_md_text
    assert "Best structural checkpoint [map]" in combo_md_text
    assert "Last logged structural checkpoint [map]" in combo_md_text


def test_render_demand_design_active_window_omits_updated_at_and_run_id(tmp_path):
    active_path = tmp_path / "logs" / "outputs_dev_sim_demand_design_iv_active.md"
    source_config = tmp_path / "configs" / "Sim_Demand_Design_IV.yaml"
    source_config.parent.mkdir(parents=True)
    source_config.write_text("dataset: Sim_Demand_Design_IV\n", encoding="utf-8")
    run_root = tmp_path / "dumps" / "sim_demand_design_iv_2026-05-05_14-47-28-123456"
    run_root.mkdir(parents=True)
    params = {"_config_source_path": str(source_config)}

    content = main_module._render_demand_design_active_window(
        active_path,
        "running",
        params,
        run_root,
        "python -u main.py -c configs/Sim_Demand_Design_IV.yaml",
        "hello\nworld",
    )

    assert "updated_at" not in content
    assert "run_id" not in content
    assert "- target: `bgm_iv`" in content
    assert "## Driver Output" in content
    assert "hello" in content


class _DummyExperimentalConfig:
    def __init__(self):
        self.memory_growth_calls = []

    def set_memory_growth(self, gpu, enabled):
        self.memory_growth_calls.append((gpu, enabled))


class _DummyTfConfig:
    def __init__(self, gpus):
        self.gpus = list(gpus)
        self.visible_device_calls = []
        self.experimental = _DummyExperimentalConfig()

    def list_physical_devices(self, device_type):
        assert device_type == "GPU"
        return list(self.gpus)

    def set_visible_devices(self, devices, device_type):
        self.visible_device_calls.append((devices, device_type))


def test_configure_tensorflow_devices_disables_gpu_when_requested(monkeypatch, capsys):
    dummy_config = _DummyTfConfig(gpus=["gpu0"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    main_module._configure_tensorflow_devices(use_gpu=False)

    assert dummy_config.visible_device_calls == [([], "GPU")]
    assert dummy_config.experimental.memory_growth_calls == []
    assert "TensorFlow GPU disabled. Using CPU only." in capsys.readouterr().out


def test_configure_tensorflow_devices_enables_gpu_when_available(monkeypatch, capsys):
    dummy_config = _DummyTfConfig(gpus=["gpu0", "gpu1"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    main_module._configure_tensorflow_devices(use_gpu=True)

    assert dummy_config.visible_device_calls == []
    assert dummy_config.experimental.memory_growth_calls == [
        ("gpu0", True),
        ("gpu1", True),
    ]
    assert "TensorFlow GPU enabled with 2 device(s)." in capsys.readouterr().out


def test_configure_tensorflow_devices_falls_back_to_cpu_when_gpu_missing(monkeypatch, capsys):
    dummy_config = _DummyTfConfig(gpus=[])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    main_module._configure_tensorflow_devices(use_gpu=True)

    assert dummy_config.visible_device_calls == []
    assert dummy_config.experimental.memory_growth_calls == []
    assert (
        "TensorFlow GPU requested but no GPU was detected. Falling back to CPU."
        in capsys.readouterr().out
    )


def test_use_gpu_config_defaults_to_false_when_omitted(monkeypatch, capsys):
    dummy_config = _DummyTfConfig(gpus=["gpu0"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    params = {}
    main_module._configure_tensorflow_devices(bool(params.get("use_gpu", False)))

    assert dummy_config.visible_device_calls == [([], "GPU")]
    assert "TensorFlow GPU disabled. Using CPU only." in capsys.readouterr().out
