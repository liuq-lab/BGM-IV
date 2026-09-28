import pytest
import numpy as np
import tensorflow as tf
import json
import yaml
from datetime import datetime
from pathlib import Path
import main as main_module

from bgm_iv.datasets import make_demand_design_grid, simulate_demand_design_iv
from bgm_iv.datasets.simulators import (
    demand_design_h,
    demand_design_structural_function,
)
from bgm_iv.models.bgm_iv import BGM_IV

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _make_params(output_dir):
    return {
        "dataset": "DemandDesignIVSmoke",
        "output_dir": str(output_dir),
        "save_model": False,
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
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "structural_map_steps": 3,
        "structural_map_lr": 5e-4,
    }


def test_demand_design_formula_matches_reference():
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
    grid = make_demand_design_grid()

    assert train["v"].shape == (12, 2)
    assert grid["v"].shape == (20 * 20 * 7, 2)
    assert grid["x"].shape == grid["y_struct"].shape == (20 * 20 * 7, 1)


def test_bgm_iv_smoke(tmp_path):
    train = simulate_demand_design_iv(n_samples=96, rho=0.5, seed=7)
    params = _make_params(tmp_path)

    model = BGM_IV(params=params, random_seed=13)
    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=2,
        epochs_per_eval=1,
        batch_size=24,
        use_egm_init=False,
        verbose=0,
    )

    mse_x, mse_y, mse_v = model.evaluate(
        data=(train["x"], train["y"], train["v"], train["w"]),
        data_z=None,
    )
    assert np.isfinite(float(mse_x))
    assert np.isfinite(float(mse_y))
    assert np.isfinite(float(mse_v))

    grid = {
        key: value[::200] for key, value in make_demand_design_grid().items()
    }
    structural_pred = model.predict_structural(grid["x"], grid["v"], map_steps=3)
    assert structural_pred.shape == grid["y_struct"].shape
    assert np.isfinite(float(np.mean((grid["y_struct"] - structural_pred) ** 2)))

    np.testing.assert_array_equal(
        model.infer_latent_from_covariates(grid["v"], map_steps=0),
        model.encoder_latent(grid["v"]),
    )


def test_covariate_posterior_is_untempered(tmp_path):
    params = _make_params(tmp_path)
    model = BGM_IV(params=params, random_seed=3)

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


_MCMC_BUDGET = {
    "mcmc_num_chains": 4,
    "mcmc_production_warmup_steps": 2000,
    "mcmc_production_draws": 5000,
}


def test_structural_methods_are_map_and_optional_mcmc():
    for given, expected in (
        (None, ["map"]),
        ("map", ["map"]),
        (["map"], ["map"]),
        ([" map ", "mcmc"], ["map", "mcmc"]),
    ):
        params = {} if given is None else {"structural_methods": given}
        params.update(_MCMC_BUDGET)
        main_module._check_structural_methods(params)
        assert params["structural_methods"] == expected


@pytest.mark.parametrize("missing", sorted(_MCMC_BUDGET))
def test_mcmc_readout_requires_the_chain_budget(missing):
    params = {"structural_methods": ["map", "mcmc"], **_MCMC_BUDGET}
    params.pop(missing)
    with pytest.raises(ValueError, match=missing):
        main_module._check_structural_methods(params)
    params["structural_methods"] = ["map"]
    main_module._check_structural_methods(params)


class _DummyStructuralModel:
    def __init__(self):
        self.calls = 0

    def predict_structural(self, grid_x, grid_v):
        self.calls += 1
        return np.zeros((len(grid_x), 1), dtype=np.float32)


def test_structural_methods_reject_other_forms():
    for methods in (
        ["map", "hmc"], ["map", "encoder"], [""], [], None, ["mcmc"],
        ["mcmc", "map"], ["map", "map"], ["map", "mcmc", "mcmc"],
    ):
        with pytest.raises(ValueError, match="structural_methods"):
            main_module._check_structural_methods({"structural_methods": methods})


def test_benchmark_defaults_add_only_the_fixed_dimensions():
    params = {"dataset": "Sim_Demand_Design_IV"}

    main_module._apply_demand_design_benchmark_defaults(params)
    main_module._check_structural_methods(params)

    assert params == {
        "dataset": "Sim_Demand_Design_IV",
        "w_dim": 1,
        "v_dim": 2,
        "save_model": True,
        "egm_num_warm_starts": 1,
        "structural_methods": ["map"],
    }


def test_model_seed_is_drawn_when_absent_and_kept_when_configured(monkeypatch):
    monkeypatch.setattr(main_module, "draw_model_seed", lambda: 424242)
    params = {}
    assert main_module._resolve_model_seed(params) == 424242
    assert params["model_seed"] == 424242
    params = {"model_seed": 5}
    assert main_module._resolve_model_seed(params) == 5
    for bad in (0, 2**31 - 1, 1.5):
        with pytest.raises(ValueError, match="model_seed"):
            main_module._resolve_model_seed({"model_seed": bad})


def test_benchmark_defaults_fix_the_mnist_covariate_dimension():
    params = {"dataset": "Sim_Demand_Design_Mnist_IV"}

    main_module._apply_demand_design_benchmark_defaults(params)

    assert params["v_dim"] == 785
    assert params["w_dim"] == 1


@pytest.mark.parametrize("v_dim", [200, 1000])
def test_benchmark_defaults_reject_other_mnist_v_dim(v_dim):
    params = {"dataset": "Sim_Demand_Design_Mnist_IV", "v_dim": v_dim}

    with pytest.raises(ValueError, match="v_dim"):
        main_module._apply_demand_design_benchmark_defaults(params)


@pytest.mark.parametrize(
    ("dataset", "field", "value"),
    [
        ("Sim_Demand_Design_IV", "v_dim", 3),
        ("Sim_Demand_Design_IV", "w_dim", 2),
        ("Sim_Demand_Design_Mnist_IV", "w_dim", 2),
        ("Sim_Demand_Design_Vector_PCAOnly_IV", "v_dim", 7),
    ],
)
def test_benchmark_defaults_reject_other_dimensions(dataset, field, value):
    params = {"dataset": dataset, field: value}

    with pytest.raises(ValueError, match=field):
        main_module._apply_demand_design_benchmark_defaults(params)


@pytest.mark.parametrize(
    "key",
    ["sigma_x", "sigma_y", "sigma_v_softfloor", "sigma_y_softfloor", "sigma_time",
     "seed", "use_z_rec", "egm_outcome_loss", "egm_selection_top_k",
     "fit_egm_batches_per_eval"],
)
def test_benchmark_defaults_reject_retired_keys(key):
    assert key in main_module._RETIRED_KEYS
    with pytest.raises(ValueError, match=f"`{key}` is no longer supported; remove it"):
        main_module._apply_demand_design_benchmark_defaults(
            {"dataset": "Sim_Demand_Design_IV", key: 0}
        )


def test_retired_key_given_with_set_is_refused_by_main(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "-c", str(Path(main_module.__file__).resolve().parent / "configs" / "Sim_Demand_Design_IV.yaml"),
         "--set", "use_z_rec=0", "--set", f"output_dir={tmp_path}"],
    )
    with pytest.raises(ValueError, match="`use_z_rec` is no longer supported"):
        main_module.main()


@pytest.mark.parametrize(
    "config",
    ["Sim_Demand_Design_IV.yaml", "Sim_Demand_Design_Mnist_IV.yaml",
     "Sim_Demand_Design_Vector_PCAOnly_IV.yaml"],
)
def test_configs_contain_no_retired_key(config):
    with (Path(main_module.__file__).resolve().parent / "configs" / config).open() as handle:
        params = yaml.safe_load(handle)
    assert not set(params) & set(main_module._RETIRED_KEYS)


def test_benchmark_defaults_accept_the_fixed_dimensions():
    params = {"dataset": "Sim_Demand_Design_IV", "v_dim": 2, "w_dim": 1}

    main_module._apply_demand_design_benchmark_defaults(params)

    assert (params["v_dim"], params["w_dim"]) == (2, 1)


def test_render_demand_design_run_config():
    params = {
        "n_samples": 1000,
        "rho": 0.5,
        "n_repeat": 3,
        "repeat_id": 1,
        "run_seed": 1,
        "z_dims": [2, 1, 1, 7],
        "v_dim": 2,
        "w_dim": 1,
    }

    captured = main_module._render_demand_design_run_config(params)

    assert "Demand-design run config:" in captured
    assert "n_samples: 1000" in captured
    assert "rho: 0.5" in captured
    assert "n_repeat: 3" in captured
    assert "repeat_id: 1" in captured
    assert "run_seed: 1" in captured
    assert "z_dims: [2, 1, 1, 7]" in captured
    assert "v_dim: 2" in captured
    assert "w_dim: 1" in captured


def test_set_overrides_parse_yaml_values_and_apply_in_order():
    parser = main_module._build_arg_parser()
    args = parser.parse_args(
        ["-c", "x.yaml", "--set", "n_samples=5000", "--set", "rho=0.5",
         "--set", "structural_methods=[map, mcmc]", "--set", "n_samples=1000"]
    )
    params = main_module._apply_config_overrides({"n_samples": [1, 2], "lr": 3}, args.overrides)
    assert params == {"n_samples": 1000, "rho": 0.5, "structural_methods": ["map", "mcmc"], "lr": 3}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        main_module._apply_config_overrides({}, ["n_samples"])


def test_mcmc_yaml_controls_are_inference_only_and_not_training_params():
    base = {"dataset": "Sim_Demand_Design_Mnist_IV", "fit_epochs": 200}
    params = {
        **base,
        "mcmc_num_chains": 4,
        "mcmc_production_warmup_steps": 2000,
        "mcmc_production_draws": 5000,
        "mcmc_artifact_root": "/tmp/a",
        "mcmc_arm_id": "a",
        "mcmc_readout_artifact": None,
    }
    assert main_module._training_params(params) == main_module._training_params(base)
    text = main_module._render_demand_design_run_config(params)
    assert text.endswith(
        "  mcmc_num_chains: 4\n"
        "  mcmc_production_warmup_steps: 2000\n"
        "  mcmc_production_draws: 5000"
    )


EXPECTED_CONFIGS = {
    "Sim_Demand_Design_IV.yaml": ([3, 2, 1, 2], 64, 2000, 5000),
    "Sim_Demand_Design_Vector_PCAOnly_IV.yaml": ([3, 2, 1, 2], 64, 2000, 5000),
    "Sim_Demand_Design_Mnist_IV.yaml": ([2, 1, 1, 2], 32, 1000, 1000),
}


def test_the_repository_ships_exactly_the_three_configs():
    root = Path(main_module.__file__).resolve().parent / "configs"
    shipped = sorted(
        path.name for path in root.iterdir() if not path.name.startswith(".")
    )
    assert shipped == sorted(EXPECTED_CONFIGS)


@pytest.mark.parametrize("name", sorted(EXPECTED_CONFIGS))
def test_configs_match_the_expected_settings(name):
    z_dims, batch_size, warmup, draws = EXPECTED_CONFIGS[name]
    root = Path(main_module.__file__).resolve().parent / "configs"
    with (root / name).open(encoding="utf-8") as handle:
        params = main_module.yaml.safe_load(handle)
    assert params["z_dims"] == z_dims
    assert params["fit_batch_size"] == batch_size
    assert params["outcome_to_particles_weight"] == 0.01
    assert "sigma_y_softfloor" not in params and "sigma_time" not in params
    assert params["egm_num_warm_starts"] == 10
    assert params["fit_egm_n_iter"] == 50000
    assert params["fit_epochs"] == 200
    assert params["iv_mc_samples"] == params["eval_mc_samples"] == 1000
    assert params["structural_map_steps"] == 1000
    assert params["structural_methods"] == ["map"]
    assert params["mcmc_num_chains"] == 4
    assert params["mcmc_production_warmup_steps"] == warmup
    assert params["mcmc_production_draws"] == draws
    assert params["n_repeat"] == 20
    assert params["use_gpu"] is False
    assert params["lr"] == 2e-4
    assert params["lr_theta"] == params["lr_z"] == params["structural_map_lr"] == 1e-4
    assert params["g_d_freq"] == 5
    assert params["e_units"] == params["g_units"] == [64] * 5
    assert params["f_units"] == params["h_units"] == params["dz_units"] == [64, 32, 8]
    assert "n_samples" not in params and "pca_dim" not in params
    if name == "Sim_Demand_Design_Vector_PCAOnly_IV.yaml":
        assert params["rho"] == 0.5
    else:
        assert "rho" not in params
    if name == "Sim_Demand_Design_Vector_PCAOnly_IV.yaml":
        assert params["representation_sd"] == 0.5
    params = dict(params, n_samples=1000, rho=0.5, _only_repeat_id=3)
    if name == "Sim_Demand_Design_Vector_PCAOnly_IV.yaml":
        params["pca_dim"] = 8
    main_module._apply_demand_design_benchmark_defaults(params)
    main_module._check_structural_methods(params)
    cell = main_module._resolve_demand_design_cell(params)
    assert (cell["repeat_id"], cell["run_seed"]) == (3, 3)
    assert cell["output_dir"] == "./sweeps/n_samples=1000__rho=0.5__repeat=3"


def test_models_use_the_expected_egm_and_optimizer_settings(tmp_path):
    params = _make_params(tmp_path)
    model = BGM_IV(params=params, random_seed=1)
    assert model.params == params
    nodes, weights = np.polynomial.hermite.hermgauss(8)
    np.testing.assert_array_equal(
        model._egm_gh_t.numpy().reshape(-1), nodes.astype(np.float32)
    )
    np.testing.assert_array_equal(
        model._egm_gh_w.numpy().reshape(-1), (weights / np.sqrt(np.pi)).astype(np.float32)
    )
    for optimizer in (
        model.g_optimizer,
        model.f_optimizer,
        model.h_optimizer,
        model.g_pre_optimizer,
        model.d_pre_optimizer,
        model.posterior_optimizer,
    ):
        assert float(optimizer.beta_1) == pytest.approx(0.9)
        assert float(optimizer.beta_2) == pytest.approx(0.99)


def test_test_time_map_uses_the_expected_adam_settings(monkeypatch, tmp_path):
    model = BGM_IV(params=_make_params(tmp_path), random_seed=1)
    made = []
    real_adam = tf.keras.optimizers.Adam

    def recording_adam(*args, **kwargs):
        made.append((args, kwargs))
        return real_adam(*args, **kwargs)

    monkeypatch.setattr(tf.keras.optimizers, "Adam", recording_adam)
    model.infer_latent_from_covariates(
        np.zeros((3, 2), np.float32), map_steps=1, map_lr=1e-4
    )
    assert made == [((1e-4,), {"beta_1": 0.9, "beta_2": 0.99})]


def test_build_arg_parser_accepts_only_one_task():
    parser = main_module._build_arg_parser()
    args = parser.parse_args(["-c", "configs/Sim_Demand_Design_IV.yaml", "-t", "1"])

    assert args.config == "configs/Sim_Demand_Design_IV.yaml"
    assert args.num_tasks == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["-c", "configs/Sim_Demand_Design_IV.yaml", "-t", "5"])


def test_build_arg_parser_uses_mcmc_only():
    parser = main_module._build_arg_parser()
    args = parser.parse_args(["-c", "x.yaml", "--mcmc-only", "stamp"])
    assert args.mcmc_only == "stamp"


def test_mcmc_artifact_cli_options_become_run_control_params():
    parser = main_module._build_arg_parser()
    args = parser.parse_args(
        [
            "-c",
            "x.yaml",
            "--mcmc-only",
            "stamp",
            "--mcmc-artifact-root",
            "/tmp/artifacts",
            "--mcmc-arm-id",
            "w2000_d5000",
        ]
    )
    params = {"_mcmc_only_timestamp": "stamp"}
    main_module._apply_mcmc_artifact_options(params, args)

    assert params == {
        "_mcmc_only_timestamp": "stamp",
        "mcmc_artifact_root": "/tmp/artifacts",
        "mcmc_arm_id": "w2000_d5000",
    }
    assert main_module._training_params(params) == {}


@pytest.mark.parametrize(
    "flag", ["--mcmc-artifact-root", "--mcmc-arm-id", "--mcmc-readout-artifact"]
)
@pytest.mark.parametrize("value", ["", "   "])
def test_mcmc_artifact_cli_rejects_empty_values(flag, value):
    parser = main_module._build_arg_parser()
    args = parser.parse_args(["-c", "x.yaml", "--mcmc-only", "stamp", f"{flag}={value}"])
    with pytest.raises(ValueError, match=f"{flag} must be non-empty"):
        main_module._apply_mcmc_artifact_options({"_mcmc_only_timestamp": "stamp"}, args)


def test_mcmc_artifact_cli_requires_restore_mode():
    parser = main_module._build_arg_parser()
    args = parser.parse_args(
        ["-c", "x.yaml", "--mcmc-readout-artifact", "/tmp/arm/manifest.json"]
    )
    with pytest.raises(ValueError, match="require --mcmc-only"):
        main_module._apply_mcmc_artifact_options({}, args)


def test_run_structural_mcmc_passes_budget_and_artifact_options(monkeypatch, tmp_path):
    captured = {}

    def fake_run_mcmc_grid(model, **kwargs):
        captured.update(kwargs)
        return {"readout": {"width95": 3.0}}

    monkeypatch.setattr("bgm_iv.mcmc.inference.run_mcmc_grid", fake_run_mcmc_grid)
    params = {
        "dataset": "Sim_Demand_Design_Mnist_IV",
        "repeat_id": 0,
        "model_seed": 31337,
        "mcmc_num_chains": 4,
        "mcmc_production_warmup_steps": 1000,
        "mcmc_production_draws": 1000,
        "mcmc_artifact_root": str(tmp_path),
        "mcmc_arm_id": "w1000_d1000",
    }
    model = type("Model", (), {"timestamp": "stamp"})()
    kwargs = dict(
        family="mnist_pixel",
        grid_x_model=np.zeros((2, 1), np.float32),
        grid_v_raw=np.zeros((2, 2), np.float32),
        truth_rows=np.zeros(2, np.float32),
        truth_label="truth",
        preprocessor=object(),
        x_stats={"mean": np.asarray([0.0]), "scale": np.asarray([1.0])},
        y_stats={"mean": np.asarray([0.0]), "scale": np.asarray([1.0])},
    )
    main_module._run_structural_mcmc(model, params, **kwargs)

    assert captured["production_warmup_steps"] == 1000
    assert captured["production_draws"] == 1000
    assert captured["production_num_chains"] == 4
    assert captured["artifact_root"] == str(tmp_path)
    assert captured["arm_id"] == "w1000_d1000"
    assert captured["readout_artifact_manifest"] is None
    assert captured["seeds"] == {
        "pilot": main_module.derive_seed(31337, "mcmc-pilot"),
        "production": main_module.derive_seed(31337, "mcmc-production"),
    }
    assert captured["source"] == {
        "dataset": "Sim_Demand_Design_Mnist_IV",
        "repeat_id": 0,
        "checkpoint_timestamp": "stamp",
    }

    for key in ("mcmc_artifact_root", "mcmc_arm_id"):
        params.pop(key)
    params["mcmc_readout_artifact"] = "/tmp/existing/manifest.json"
    captured.clear()
    main_module._run_structural_mcmc(model, params, **kwargs)
    assert captured["readout_artifact_manifest"] == "/tmp/existing/manifest.json"
    assert captured["artifact_root"] is None and captured["arm_id"] is None


def test_importing_main_does_not_load_the_mcmc_sampler():
    import subprocess
    import sys

    code = (
        "import sys, main; "
        "assert 'bgm_iv.mcmc.sampler' not in sys.modules; "
        "main._mcmc_inference(); "
        "assert 'bgm_iv.mcmc.sampler' in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(main_module.__file__).resolve().parent,
        check=True,
    )


def test_mcmc_only_marker_distinguishes_training_and_restore_modes():
    assert main_module._mcmc_only_timestamp({}) is None
    assert main_module._mcmc_only_timestamp(
        {"_mcmc_only_timestamp": "checkpoint_1"}
    ) == "checkpoint_1"


@pytest.mark.parametrize("timestamp", ["", "   "])
def test_main_rejects_empty_mcmc_only_before_creating_artifacts(
    monkeypatch, tmp_path, timestamp
):
    config = tmp_path / "config.yaml"
    config.write_text("dataset: Sim_Demand_Design_Vector_PCAOnly_IV\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "_demand_design_dumps_dir", lambda: tmp_path / "dumps")
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "-c",
            str(config),
            "--mcmc-only",
            timestamp,
            "--repeat-id",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        main_module.main()
    assert list(tmp_path.iterdir()) == [config]


def _repeat_run_seed(repeat_id):
    return main_module._resolve_demand_design_cell(
        {"n_samples": 32, "rho": 0.5, "n_repeat": 2, "_only_repeat_id": repeat_id}
    )["run_seed"]


def test_demand_design_repeat_seed_reproduces_identical_train_and_grid_data():
    run_seed = _repeat_run_seed(1)
    assert run_seed == 1

    train_a = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed)
    train_b = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed)
    grid_a = make_demand_design_grid()
    grid_b = make_demand_design_grid()

    np.testing.assert_allclose(train_a["x"], train_b["x"], atol=0.0)
    np.testing.assert_allclose(train_a["y"], train_b["y"], atol=0.0)
    np.testing.assert_allclose(train_a["v"], train_b["v"], atol=0.0)
    np.testing.assert_allclose(train_a["w"], train_b["w"], atol=0.0)
    np.testing.assert_allclose(grid_a["x"], grid_b["x"], atol=0.0)
    np.testing.assert_allclose(grid_a["v"], grid_b["v"], atol=0.0)
    np.testing.assert_allclose(grid_a["y_struct"], grid_b["y_struct"], atol=0.0)


def test_demand_design_repeat_id_changes_train_data_but_not_grid_data():
    run_seed_0 = _repeat_run_seed(0)
    run_seed_1 = _repeat_run_seed(1)

    train_0 = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed_0)
    train_1 = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=run_seed_1)
    grid_0 = make_demand_design_grid()
    grid_1 = make_demand_design_grid()

    assert not np.allclose(train_0["x"], train_1["x"])
    assert not np.allclose(train_0["y"], train_1["y"])
    assert not np.allclose(train_0["v"], train_1["v"])
    assert not np.allclose(train_0["w"], train_1["w"])
    np.testing.assert_allclose(grid_0["x"], grid_1["x"], atol=0.0)
    np.testing.assert_allclose(grid_0["v"], grid_1["v"], atol=0.0)
    np.testing.assert_allclose(grid_0["y_struct"], grid_1["y_struct"], atol=0.0)


def test_final_readout_calls_the_map_predictor_once():
    model = _DummyStructuralModel()
    results = main_module._evaluate_structural_methods(
        model,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
    )
    assert model.calls == 1
    assert results == {"map": 0.0}


def test_map_readout_is_pinned(tmp_path):
    model = BGM_IV(params=_make_params(tmp_path), random_seed=7)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(16, 1)).astype(np.float32)
    v = rng.normal(size=(16, 2)).astype(np.float32)

    def outcome_mean(z):
        z0, z1, _ = model._split_z(tf.convert_to_tensor(z, tf.float32))
        return model.f_net(tf.concat([z0, z1, x], axis=-1))[:, :1].numpy()

    steps, lr = 5, 0.1
    z_map = model.infer_latent_from_covariates(v, map_steps=steps, map_lr=lr)
    prediction = model.predict_structural(x, v, map_steps=steps, map_lr=lr)
    np.testing.assert_allclose(prediction, outcome_mean(z_map), rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(prediction - outcome_mean(model.encoder_latent(v)))) > 1e-4

    y_stats = {
        "mean": np.array([[3.0]], np.float32),
        "scale": np.array([[2.0]], np.float32),
    }
    y_true = rng.normal(size=(16, 1)).astype(np.float32)
    standardized = model.predict_structural(x, v)
    expected = float(np.mean((y_true - (standardized * 2.0 + 3.0)) ** 2))
    results = main_module._evaluate_structural_methods(model, x, v, y_true, y_stats=y_stats)
    assert results == {"map": pytest.approx(expected, rel=1e-6)}


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
            "Sim_Demand_Design_Vector_PCAOnly_IV",
            "sim_demand_design_vector_pcaonly_iv_2026-05-05_14-47-28-123456",
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


def test_persist_demand_design_repeat_outputs_writes_expected_files(tmp_path):
    run_root = tmp_path / "sim_demand_design_iv_2026-05-05_14-47-28-123456"
    run_root.mkdir()
    params = {
        "n_samples": 1000,
        "rho": 0.25,
        "repeat_id": 1,
        "v_dim": 2,
    }

    main_module._persist_demand_design_repeat_outputs(
        run_root,
        params,
        {
            "run_config_text": "Demand-design run config:\n  n_samples: 1000",
            "final_results": {"map": 8.0},
            "provenance": {"checkpoint_timestamp": "stamp"},
        },
    )

    combo_dir = run_root / "n_samples:1000-rho:0.25-v_dim:2"
    assert combo_dir.exists()

    assert sorted(path.name for path in combo_dir.iterdir()) == ["records", "results.csv"]

    import csv as _csv

    with (combo_dir / "results.csv").open(encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    assert list(rows[0].keys()) == list(main_module._FINAL_RESULT_COLUMNS)
    assert rows[0]["repeat_id"] == "1"
    assert rows[0]["structural_mse_map"] == "8.0"
    record = json.loads((combo_dir / "records" / "repeat1_stamp.json").read_text())
    assert set(record) == {
        "schema_version", "repeat_id", "run_config_text", "provenance",
        "final_results", "mcmc",
    }
    assert record["final_results"] == {"map": 8.0} and record["mcmc"] is None


def test_persist_mcmc_interval_headline_schema(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    params = {
        "n_samples": 1000,
        "rho": 0.5,
        "repeat_id": 0,
        "v_dim": 785,
    }
    mcmc = {
        "family": "demand",
        "grid": {"num_targets": 2, "num_queries": 3},
        "readout": {
            "coverage": {"0.9": 0.89, "0.95": 0.94, "0.99": 0.985},
            "width90": 7.25,
            "width95": 10.5,
            "width99": 14.75,
            "num_chains": 4,
            "draws_per_chain": 24000,
        },
        "timings": {"mcmc_seconds": 10.0, "uq_seconds": 2.0},
    }
    main_module._persist_demand_design_repeat_outputs(
        run_root,
        params,
        {
            "run_config_text": "config",
            "final_results": {"map": 11.0, "_mcmc": mcmc},
            "provenance": {"model_seed": 12345, "checkpoint_timestamp": "stamp"},
        },
    )
    combo = next(run_root.iterdir())
    import csv as _csv

    with (combo / "results.csv").open() as handle:
        row = next(_csv.DictReader(handle))
    assert row["structural_mse_map"] == "11.0"
    assert "structural_mse_mcmc" not in row
    assert (row["mcmc_cov90"], row["mcmc_cov95"], row["mcmc_cov99"]) == ("0.89", "0.94", "0.985")
    assert (row["mcmc_width90"], row["mcmc_width95"], row["mcmc_width99"]) == ("7.25", "10.5", "14.75")
    assert "wasserstein1" not in row
    assert not (combo / "certified_results.csv").exists()
    record = json.loads(next((combo / "records").glob("*.json")).read_text())
    assert record["mcmc"]["readout"]["coverage"] == mcmc["readout"]["coverage"]
    assert row["model_seed"] == "12345"
    assert record["provenance"]["model_seed"] == 12345


class _DummyExperimentalConfig:
    def __init__(self):
        self.memory_growth_calls = []
        self.memory_growth = {}

    def set_memory_growth(self, gpu, enabled):
        self.memory_growth_calls.append((gpu, enabled))
        self.memory_growth[gpu] = bool(enabled)

    def get_memory_growth(self, gpu):
        return self.memory_growth.get(gpu, False)


class _DummyTfConfig:
    def __init__(self, gpus):
        self.gpus = list(gpus)
        self.visible_gpus = list(gpus)
        self.visible_device_calls = []
        self.experimental = _DummyExperimentalConfig()

    def list_physical_devices(self, device_type):
        assert device_type == "GPU"
        return list(self.gpus)

    def set_visible_devices(self, devices, device_type):
        self.visible_device_calls.append((devices, device_type))
        self.visible_gpus = list(devices)

    def get_visible_devices(self, device_type):
        assert device_type == "GPU"
        return list(self.visible_gpus)


def test_configure_tensorflow_devices_disables_gpu_when_requested(monkeypatch, capsys):
    dummy_config = _DummyTfConfig(gpus=["gpu0"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    main_module._configure_tensorflow_devices(use_gpu=False)

    assert dummy_config.visible_device_calls == [([], "GPU")]
    assert dummy_config.experimental.memory_growth_calls == []
    assert "TensorFlow GPU disabled. Using CPU only." in capsys.readouterr().out


def test_configure_tensorflow_devices_disables_gpu_idempotently(monkeypatch):
    dummy_config = _DummyTfConfig(gpus=["gpu0"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    main_module._configure_tensorflow_devices(use_gpu=False)
    main_module._configure_tensorflow_devices(use_gpu=False)

    assert dummy_config.visible_device_calls == [([], "GPU")]


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


def test_configure_tensorflow_devices_is_idempotent(monkeypatch):
    dummy_config = _DummyTfConfig(gpus=["gpu0", "gpu1"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    main_module._configure_tensorflow_devices(use_gpu=True)
    main_module._configure_tensorflow_devices(use_gpu=True)

    assert dummy_config.experimental.memory_growth_calls == [
        ("gpu0", True),
        ("gpu1", True),
    ]


def test_configure_tensorflow_devices_requires_a_gpu_when_requested(monkeypatch):
    dummy_config = _DummyTfConfig(gpus=[])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    with pytest.raises(RuntimeError, match="no GPU was detected"):
        main_module._configure_tensorflow_devices(use_gpu=True)
    assert dummy_config.visible_device_calls == []
    assert dummy_config.experimental.memory_growth_calls == []


def test_initialize_egm_candidate_worker_configures_one_thread_and_the_devices(monkeypatch):
    seen = []
    monkeypatch.setattr(
        main_module, "_configure_tensorflow_threads", lambda *a: seen.append(("threads", a))
    )
    monkeypatch.setattr(
        main_module, "_configure_tensorflow_devices", lambda use_gpu: seen.append(("gpu", use_gpu))
    )
    monkeypatch.setattr(main_module.tf.config, "list_logical_devices", lambda kind: [])
    main_module._initialize_egm_candidate_worker(False)
    assert seen == [("threads", (1, 1)), ("gpu", False)]
    with pytest.raises(RuntimeError, match="worker sees no GPU"):
        main_module._initialize_egm_candidate_worker(True)


def test_use_gpu_config_defaults_to_false_when_omitted(monkeypatch, capsys):
    dummy_config = _DummyTfConfig(gpus=["gpu0"])
    monkeypatch.setattr(main_module.tf, "config", dummy_config)

    params = {}
    main_module._configure_tensorflow_devices(bool(params.get("use_gpu", False)))

    assert dummy_config.visible_device_calls == [([], "GPU")]
    assert "TensorFlow GPU disabled. Using CPU only." in capsys.readouterr().out


def test_main_configures_multistart_parent_before_dataset_runner(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: Sim_Demand_Design_Mnist_IV",
                "use_gpu: true",
                "n_samples: 8",
                "rho: 0.5",
                "n_repeat: 1",
                "egm_num_warm_starts: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    events = []

    monkeypatch.setattr("sys.argv", ["main.py", "-c", str(config)])
    monkeypatch.setattr(
        main_module, "_apply_demand_design_benchmark_defaults", lambda params: None
    )

    def configure(use_gpu, **kwargs):
        events.append(("configure", use_gpu, kwargs))

    def run(params):
        events.append(("run", dict(params)))

    monkeypatch.setattr(main_module, "_configure_tensorflow_devices", configure)
    monkeypatch.setattr(main_module, "_run_demand_design_family", run)

    main_module.main()

    assert events[0] == ("configure", True, {})
    assert events[1][0] == "run"
    assert events[1][1]["egm_num_warm_starts"] == 10

def _egm_params(tmp_path, **overrides):
    params = {
        "dataset": "unit", "output_dir": str(tmp_path),
        "save_model": False,
        "z_dims": [2, 2, 1, 2], "v_dim": 2, "w_dim": 1,
        "lr": 2e-4, "lr_theta": 1e-4, "lr_z": 1e-4,
        "g_units": [16, 16], "e_units": [16, 16], "f_units": [16, 8],
        "h_units": [16, 8], "dz_units": [16, 8],
        "g_d_freq": 1,
        "iv_mc_samples": 8, "eval_mc_samples": 8,
        "structural_map_steps": 100, "structural_map_lr": 1e-4,
    }
    params.update(overrides)
    return params


def _standardize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (arr - arr.mean(axis=0, keepdims=True)) / std


def test_egm_integral_gh8_finite_and_ema_calibrates(tmp_path):
    train = simulate_demand_design_iv(n_samples=96, rho=0.5, seed=11)
    data = tuple(_standardize(train[key]) for key in ("x", "y", "v", "w"))
    m = BGM_IV(_egm_params(tmp_path), random_seed=1)
    m.egm_init(data, egm_n_iter=60, batch_size=32, verbose=0)
    ema = float(m.egm_sigma2_x_ema.numpy())
    assert np.isfinite(ema)
    assert 0.0 < ema < 4.0


def _train_config_params(tmp_path):
    return {
        "dataset": "Sim_Demand_Design_IV",
        "output_dir": str(tmp_path),
        "save_model": True,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 2,
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [8, 8],
        "e_units": [8, 8],
        "f_units": [8, 4],
        "h_units": [8, 4],
        "dz_units": [8, 4],
        "lr": 5e-4,
        "g_d_freq": 1,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "structural_map_steps": 100,
        "structural_map_lr": 5e-4,
        "structural_methods": ["map"],
        "model_seed": 2024,
    }


def _weights(model):
    return [w.numpy().copy() for w in model.g_net.weights + model.f_net.weights]


def test_train_config_records_the_model_seed_and_training_params(tmp_path):
    params = _train_config_params(tmp_path)
    model = BGM_IV(
        params=params, random_seed=main_module._model_random_seed(params, "egm-init", 0)
    )
    config = main_module._write_train_config(model, params)
    path = Path(model.checkpoint_path) / "train_config.json"
    assert path.exists()
    assert json.loads(path.read_text()) == json.loads(json.dumps(config))
    assert config["model_seed"] == 2024
    assert config["dataset"] == "Sim_Demand_Design_IV"
    assert "output_dir" not in config["params"] and "lr" in config["params"]
    assert "model_seed" not in config["params"]
    assert "structural_methods" not in config["params"]
    state_prefix = Path(model.checkpoint_path) / config["inference_state_prefix"]
    state_variables = [name for name, _ in tf.train.list_variables(str(state_prefix))]
    assert not any("optimizer" in name.lower() for name in state_variables)
    assert not any("data_z" in name for name in state_variables)
    allowed_roots = (
        "g_net/",
        "e_net/",
        "f_net/",
        "h_net/",
        "dz_net/",
        "egm_sigma2_x_ema/",
        "save_counter/",
        "_CHECKPOINTABLE_OBJECT_GRAPH",
    )
    assert all(name.startswith(allowed_roots) for name in state_variables)
    with pytest.raises(RuntimeError, match="overwrite"):
        main_module._write_train_config(model, params)
    assert main_module._write_train_config(model, {**params, "save_model": False}) is None


def test_mcmc_only_restores_the_trained_state_and_its_model_seed(monkeypatch, tmp_path):
    params = _train_config_params(tmp_path)
    model = BGM_IV(params=params, random_seed=7)
    selection_provenance = {
        "egm_num_warm_starts": 10,
        "egm_selector_version": "train-iv-map-post-bgm",
        "egm_selection_criterion": "train_iv_map",
        "egm_selected_candidate_id": 4,
        "egm_selected_criterion": 161.6,
    }
    main_module._write_train_config(
        model, params, notes={"egm_multistart": selection_provenance}
    )
    expected = _weights(model)

    monkeypatch.setattr(
        main_module,
        "_fit_demand_design_model",
        lambda *args, **kwargs: pytest.fail("mcmc-only attempted training"),
    )
    restore_params = {**params, "_mcmc_only_timestamp": str(model.timestamp)}
    restore_params.pop("model_seed")
    restored = main_module._fit_or_restore_demand_design_model(restore_params, None)
    assert main_module._is_mcmc_only(restore_params)
    assert restore_params["model_seed"] == 2024
    assert restored.egm_multistart_provenance == selection_provenance
    for a, b in zip(expected, _weights(restored)):
        np.testing.assert_array_equal(a, b)
    main_module._restore_demand_design_model(
        {**params, "structural_methods": ["map", "mcmc"], "use_gpu": True},
        model.timestamp,
    )


def test_mcmc_only_lists_every_differing_training_parameter(tmp_path):
    params = _train_config_params(tmp_path)
    model = BGM_IV(params=params, random_seed=7)
    main_module._write_train_config(model, params)
    with pytest.raises(RuntimeError, match=r"differing keys: \['lr', 'n_samples'\]"):
        main_module._restore_demand_design_model(
            {**params, "lr": 1e-3, "n_samples": 5000}, model.timestamp
        )
    with pytest.raises(RuntimeError, match="differs from the checkpoint's model_seed"):
        main_module._restore_demand_design_model(
            {**params, "model_seed": 2025}, model.timestamp
        )
    with pytest.raises(FileNotFoundError, match="no checkpoint directory"):
        main_module._restore_demand_design_model(params, "no-such-timestamp")


def test_mcmc_only_requires_the_saved_inference_state(tmp_path):
    params = _train_config_params(tmp_path)
    model = BGM_IV(params=params, random_seed=7)
    config = main_module._write_train_config(model, params)
    prefix = Path(model.checkpoint_path) / config["inference_state_prefix"]
    Path(str(prefix) + ".index").unlink()
    with pytest.raises(FileNotFoundError, match="inference-state"):
        main_module._restore_demand_design_model(params, model.timestamp)
    path = Path(model.checkpoint_path) / "train_config.json"
    stored = json.loads(path.read_text())
    stored["inference_state_prefix"] = "../../elsewhere/ckpt-1"
    path.write_text(json.dumps(stored))
    with pytest.raises(RuntimeError, match="relative and safe"):
        main_module._restore_demand_design_model(params, model.timestamp)


def test_training_iv_criterion_uses_map_latents_observed_outcome_and_instrument(tmp_path):
    params = _train_config_params(tmp_path)
    params["save_model"] = False
    model = BGM_IV(params=params, random_seed=5)
    rng = np.random.default_rng(1)
    train = {
        "v": rng.normal(size=(5, 2)).astype(np.float32),
        "w": rng.normal(size=(5, 1)).astype(np.float32),
    }
    y_raw = rng.normal(size=(5, 1))
    stats = {"mean": np.array([[2.0]], np.float32), "scale": np.array([[3.0]], np.float32)}
    out = main_module._evaluate_training_iv_criterion(model, train, y_raw=y_raw, y_stats=stats)
    assert set(out) == {"train_iv_map"}
    assert np.isfinite(out["train_iv_map"]) and out["train_iv_map"] >= 0

    tf.keras.utils.set_random_seed(9)
    again = main_module._evaluate_training_iv_criterion(model, train, y_raw=y_raw, y_stats=stats)
    tf.keras.utils.set_random_seed(9)
    z = model.infer_latent_from_covariates(train["v"])
    mean = model._integrated_outcome_mean(
        tf.constant(z), tf.constant(train["w"])
    ).numpy()
    expected = np.mean((y_raw.reshape(-1) - (mean * 3.0 + 2.0).reshape(-1)) ** 2)
    assert again["train_iv_map"] == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="rows"):
        main_module._evaluate_training_iv_criterion(model, train, y_raw=y_raw[:3], y_stats=stats)


def _pretend_a_gpu_is_visible(monkeypatch):
    fake = type("D", (), {"name": "/device:GPU:0", "device_type": "GPU"})()
    monkeypatch.setattr(main_module.tf.config, "list_logical_devices", lambda kind: [fake])
    monkeypatch.setattr(main_module.tf.config, "get_visible_devices", lambda kind: [fake])
    monkeypatch.setattr(
        main_module.tf.config.experimental,
        "get_device_details",
        lambda device: {"device_name": "Test GPU"},
    )


def test_run_provenance_records_the_model_seed_and_the_training_device(
    monkeypatch, tmp_path
):
    _pretend_a_gpu_is_visible(monkeypatch)
    model = BGM_IV(params=_make_params(tmp_path), random_seed=1)
    params = {"dataset": "Sim_Demand_Design_IV", "repeat_id": 3,
              "run_seed": 3, "model_seed": 4242, "use_gpu": False}
    provenance = main_module._run_provenance(params, model)
    assert provenance["model_seed"] == 4242
    assert provenance["run_seed"] == 3
    assert provenance["device_name"] == "cpu"


def test_training_device_name_follows_use_gpu_and_visible_devices(monkeypatch):
    _pretend_a_gpu_is_visible(monkeypatch)
    assert main_module._training_device_name({"use_gpu": False}) == "cpu"
    assert main_module._training_device_name({"use_gpu": True}) == "Test GPU"
    monkeypatch.setattr(main_module.tf.config, "list_logical_devices", lambda kind: [])
    assert main_module._training_device_name({"use_gpu": True}) == "cpu"


@pytest.mark.parametrize("gamma", [1.5, -0.1, True, "2e0"])
def test_benchmark_defaults_reject_out_of_range_gamma(gamma):
    with pytest.raises(ValueError, match="outcome_to_particles_weight"):
        main_module._apply_demand_design_benchmark_defaults(
            {"dataset": "Sim_Demand_Design_IV", "outcome_to_particles_weight": gamma}
        )


@pytest.mark.parametrize("gamma", [[0.01, 0.1], "abc"])
def test_benchmark_defaults_reject_non_numeric_gamma(gamma):
    with pytest.raises((TypeError, ValueError)):
        main_module._apply_demand_design_benchmark_defaults(
            {"dataset": "Sim_Demand_Design_IV", "outcome_to_particles_weight": gamma}
        )


@pytest.mark.parametrize("value", ["1e-2", "0.01", 0.01, 0, 1])
def test_benchmark_defaults_accept_gamma_numbers_and_yaml_exponent_strings(value):
    params = {"dataset": "Sim_Demand_Design_IV", "outcome_to_particles_weight": value}
    main_module._apply_demand_design_benchmark_defaults(params)
    assert params["outcome_to_particles_weight"] == float(value)
    assert isinstance(params["outcome_to_particles_weight"], float)


def test_structural_map_steps_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="structural_map_steps"):
        BGM_IV(params={**_make_params(tmp_path), "structural_map_steps": 0}, random_seed=1)


def test_repeat_id_must_lie_inside_n_repeat_and_a_process_runs_one_cell():
    resolve = main_module._resolve_demand_design_cell
    cell = {"n_samples": 1000, "rho": 0.5}
    assert resolve({**cell, "n_repeat": 20, "_only_repeat_id": 19})["repeat_id"] == 19
    for bad in (20, -1):
        with pytest.raises(ValueError, match="--repeat-id"):
            resolve({**cell, "n_repeat": 20, "_only_repeat_id": bad})
    with pytest.raises(ValueError, match="pass --repeat-id"):
        resolve({**cell, "n_repeat": 20})
    for field in ("n_samples", "rho", "pca_dim"):
        with pytest.raises(ValueError, match=f"scalar {field}"):
            resolve({**cell, "n_repeat": 1, field: [1, 2]})
    for field in ("n_samples", "rho"):
        with pytest.raises(ValueError, match=f"set {field}"):
            resolve({key: value for key, value in cell.items() if key != field})


def test_cell_resolver_pins_the_seeds_types_and_cell_folder():
    resolve = main_module._resolve_demand_design_cell
    base = {"dataset": "Sim_Demand_Design_Vector_PCAOnly_IV", "n_samples": "5000",
            "rho": 1, "pca_dim": 8, "n_repeat": 20, "save_model": True,
            "output_dir": "out", "_only_repeat_id": 7}
    cell = resolve(base)
    assert list(cell) == list(base) + ["repeat_id", "run_seed"]
    assert (cell["n_samples"], cell["rho"], cell["n_repeat"]) == (5000, 1.0, 20)
    assert isinstance(cell["n_samples"], int) and isinstance(cell["rho"], float)
    assert (cell["repeat_id"], cell["run_seed"]) == (7, 7)
    assert cell["output_dir"] == "out/sweeps/n_samples=5000__rho=1__repeat=7"
    assert base["output_dir"] == "out"
    assert resolve({**base, "save_model": False})["output_dir"] == "out"
    single = resolve({**base, "n_repeat": 1, "_only_repeat_id": None})
    assert (single["repeat_id"], single["run_seed"], single["output_dir"]) == (0, 0, "out")
    assert resolve({**base, "rho": 0.25})["output_dir"].endswith("__rho=0.25__repeat=7")


def test_run_demand_design_family_runs_and_persists_one_cell(monkeypatch, capsys):
    seen = []

    def fake_run(run_params):
        seen.append(run_params)
        return {
            "run_config_text": main_module._render_demand_design_run_config(run_params),
            "final_results": {"map": 2.5},
            "provenance": {"model_seed": 77, "checkpoint_timestamp": "stamp"},
        }

    monkeypatch.setattr(main_module, "_run_single_demand_design_iv", fake_run)
    cell = main_module._resolve_demand_design_cell(
        {"dataset": "Sim_Demand_Design_IV", "n_samples": 64, "rho": 0.25, "v_dim": 2,
         "n_repeat": 2, "save_model": False, "_only_repeat_id": 1}
    )
    main_module._run_demand_design_family(cell)
    assert [(p["n_samples"], p["rho"], p["repeat_id"], p["run_seed"]) for p in seen] == [
        (64, 0.25, 1, 1)
    ]
    assert "Demand-design cell: n_samples=64, rho=0.25, repeat=1" in capsys.readouterr().out
    results = list(main_module._demand_design_dumps_dir().rglob("results.csv"))
    assert len(results) == 1
    import csv as _csv

    with results[0].open(encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    assert [(row["repeat_id"], row["run_seed"], row["model_seed"], row["structural_mse_map"])
            for row in rows] == [("1", "1", "77", "2.5")]


@pytest.mark.parametrize(
    "options",
    [["--mcmc-artifact-root", "ART", "--mcmc-arm-id", "a"],
     ["--mcmc-readout-artifact", "ART/a/manifest.json"]],
)
def test_main_rejects_artifact_options_without_mcmc_readout(monkeypatch, tmp_path, options):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset: Sim_Demand_Design_IV\nn_samples: 1000\nrho: 0.5\nn_repeat: 1\n",
        encoding="utf-8",
    )
    options = [item.replace("ART", str(tmp_path / "art")) for item in options]
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "-c", str(config), "--mcmc-only", "stamp", "--repeat-id", "0"] + options,
    )
    monkeypatch.setattr(
        main_module, "_run_demand_design_family", lambda params: pytest.fail("ran")
    )
    with pytest.raises(ValueError, match="structural_methods=\\[map,mcmc\\]"):
        main_module.main()


def test_single_start_training_streams_follow_the_model_seed(monkeypatch, tmp_path):
    seen = []
    fit_kwargs = []

    def fake_fit(self, **kwargs):
        seen.append((float(np.random.random()), float(tf.random.uniform([]).numpy())))
        fit_kwargs.append(kwargs)

    monkeypatch.setattr(BGM_IV, "fit", fake_fit)
    train = simulate_demand_design_iv(n_samples=32, rho=0.5, seed=0)
    budget = {"fit_egm_n_iter": 37, "fit_epochs": 3, "fit_epochs_per_eval": 2,
              "fit_batch_size": 16}
    for seed in (5, 5, 6):
        params = {**_make_params(tmp_path), "dataset": "Sim_Demand_Design_IV",
                  "model_seed": seed, **budget}
        main_module._fit_demand_design_model(params, train)
    for kwargs in fit_kwargs:
        assert kwargs["use_egm_init"] is True
        assert kwargs["egm_n_iter"] == budget["fit_egm_n_iter"]
        assert kwargs["epochs"] == budget["fit_epochs"]
        assert kwargs["epochs_per_eval"] == budget["fit_epochs_per_eval"]
        assert kwargs["batch_size"] == budget["fit_batch_size"]
    assert seen[0] == seen[1]
    assert seen[0][0] != seen[2][0] and seen[0][1] != seen[2][1]
    tf.keras.utils.set_random_seed(main_module.derive_seed(5, "egm-schedule"))
    np.random.seed(main_module.derive_seed(5, "egm-schedule"))
    assert seen[0] == (float(np.random.random()), float(tf.random.uniform([]).numpy()))


def test_run_control_keys_are_exactly_the_non_training_settings():
    assert main_module._RUN_CONTROL_KEYS == frozenset(
        {"output_dir", "save_model", "use_gpu", "num_tasks", "n_repeat",
         "structural_methods", "model_seed", "mcmc_num_chains",
         "mcmc_production_warmup_steps", "mcmc_production_draws",
         "mcmc_artifact_root", "mcmc_arm_id", "mcmc_readout_artifact"}
    )


@pytest.mark.parametrize(
    "key,value",
    [("lr", 1e-3), ("iv_mc_samples", 7), ("z_dims", [1, 1, 1, 2]), ("n_samples", 5000),
     ("rho", 0.9), ("run_seed", 3)],
)
def test_mcmc_only_refuses_each_changed_training_setting(tmp_path, key, value):
    params = _train_config_params(tmp_path)
    model = BGM_IV(params=params, random_seed=7)
    main_module._write_train_config(model, params)
    changed = {**params, key: value}
    with pytest.raises(RuntimeError, match=f"differing keys: \\['{key}'\\]"):
        main_module._restore_demand_design_model(changed, model.timestamp)


@pytest.mark.parametrize(
    "extra,match",
    [(["--repeat-id", "2"], "--repeat-id must lie"),
     (["--repeat-id", "0", "--mcmc-only", "stamp", "--set", "rho=[0.1,0.5]"], "scalar rho")],
)
def test_main_checks_repeat_id_and_restore_cell_before_running(
    monkeypatch, tmp_path, extra, match
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset: Sim_Demand_Design_IV\nn_samples: 1000\nrho: 0.5\nn_repeat: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["main.py", "-c", str(config)] + extra)
    monkeypatch.setattr(
        main_module, "_run_demand_design_family", lambda params: pytest.fail("ran")
    )
    with pytest.raises(ValueError, match=match):
        main_module.main()


def test_train_config_is_reserved_before_the_inference_state_is_written(tmp_path):
    params = _train_config_params(tmp_path)
    model = BGM_IV(params=params, random_seed=7)
    root = Path(model.checkpoint_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "train_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="overwrite"):
        main_module._write_train_config(model, params)
    assert not (root / "inference-state").exists()
    assert (root / "train_config.json").read_text(encoding="utf-8") == "{}"


def test_failed_train_config_write_leaves_no_reserved_file(monkeypatch, tmp_path):
    params = _train_config_params(tmp_path)
    model = BGM_IV(params=params, random_seed=7)

    def broken(prefix):
        raise OSError("disk full")

    monkeypatch.setattr(model, "save_model_state_checkpoint", broken)
    with pytest.raises(OSError, match="disk full"):
        main_module._write_train_config(model, params)
    assert not (Path(model.checkpoint_path) / "train_config.json").exists()


def test_mcmc_only_names_a_missing_checkpoint_directory(tmp_path):
    params = _train_config_params(tmp_path)
    with pytest.raises(FileNotFoundError, match="no checkpoint directory"):
        main_module._load_train_config(params, "no-such-timestamp")
    missing = tmp_path / "checkpoints" / params["dataset"] / "empty"
    missing.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="has no train_config.json"):
        main_module._load_train_config(params, "empty")
