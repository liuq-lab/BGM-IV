import types
from pathlib import Path

import numpy as np

import main as main_module
from bgm_iv.egm_multistart import make_candidate_manifest


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediateExecutor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, candidate_id, params, train, **kwargs):
        run_seed = int(params.get("run_seed", params.get("seed", 0)))
        del fn, params, train
        candidate_root = Path(kwargs["candidate_root"])
        candidate_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = candidate_root / "ckpt-0"
        (candidate_root / "ckpt-0.index").write_text("index", encoding="utf-8")
        stdout_path = candidate_root / "candidate.stdout.log"
        stderr_path = candidate_root / "candidate.stderr.log"
        stdout_path.write_text("candidate complete\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        scores = [0.04 + 0.001 * int(candidate_id)] * len(
            kwargs["evaluation_iterations"]
        )
        manifest = make_candidate_manifest(
            candidate_id=int(candidate_id),
            init_seed=int(kwargs["init_seed"]),
            schedule_seed=int(kwargs["schedule_seed"]),
            run_seed=run_seed,
            evaluation_iterations=kwargs["evaluation_iterations"],
            full_train_l2_loss_y=scores,
            status="completed",
            data_hash=kwargs["data_hash"],
            config_hash=kwargs["config_hash"],
            code_commit=kwargs["code_commit"],
            checkpoint_path=str(checkpoint_path),
            checkpoint_hash=main_module._checkpoint_files_hash(checkpoint_path),
            checkpoint_weight_hash="restored-weight-hash",
            worker_pid=1000 + int(candidate_id),
            device_names=["cpu"],
            device_hash="shared-device-hash",
        )
        return _ImmediateFuture(
            {
                "candidate_id": int(candidate_id),
                "manifest": manifest,
                "manifest_path": f"candidate-{candidate_id}/candidate_manifest.json",
                "training_history": [
                    {
                        "stage": "post_egm",
                        "epoch": None,
                        "include_outcome": True,
                        "mse_x": 0.1,
                        "mse_y": scores[-1],
                        "mse_v": 0.2,
                    }
                ],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "error": None,
            }
        )


class _RestoreStatus:
    def assert_existing_objects_matched(self):
        return self


class _Checkpoint:
    def __init__(self):
        self.restored = None

    def restore(self, path):
        self.restored = path
        return _RestoreStatus()


class _FakeWinnerModel:
    def __init__(self, params, random_seed, auto_restore_checkpoint=True):
        self.params = dict(params)
        self.random_seed = int(random_seed)
        self.auto_restore_checkpoint = bool(auto_restore_checkpoint)
        self.ckpt = _Checkpoint()
        self.training_history = []
        self.bgm_calls = 0

    def restore_model_state_checkpoint(self, path):
        return self.ckpt.restore(path)

    def fit_bgm_from_egm(self, **kwargs):
        self.bgm_calls += 1
        self.fit_kwargs = kwargs
        self.training_history.append(
            {
                "stage": "epoch_eval",
                "epoch": 2,
                "include_outcome": True,
                "mse_x": 0.08,
                "mse_y": 0.03,
                "mse_v": 0.1,
            }
        )
        return self.training_history


def _tiny_train(n=8):
    return {
        "x": np.zeros((n, 1), np.float32),
        "y": np.zeros((n, 1), np.float32),
        "v": np.zeros((n, 785), np.float32),
        "w": np.zeros((n, 1), np.float32),
        "y_struct": np.zeros((n, 1), np.float32),
    }


def _multistart_params(tmp_path):
    return {
        "dataset": "Sim_Demand_Design_Vector_IV",
        "output_dir": str(tmp_path),
        "n_samples": 8,
        "rho": 0.5,
        "repeat_id": 0,
        "num_tasks": 1,
        "egm_num_warm_starts": 10,
        "egm_selection_top_k": 3,
        "fit_egm_n_iter": 1_000,
        "fit_egm_batches_per_eval": 100,
        "fit_epochs": 2,
        "fit_epochs_per_eval": 1,
        "fit_batch_size": 4,
        "fit_first_stage_warmup_epochs": 0,
        "save_model": True,
        "save_res": False,
        "use_gpu": False,
        "deterministic_training": True,
        "training_grid_monitor": False,
    }


def test_multistart_bundle_waits_for_ten_candidates_and_runs_one_bgm(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(main_module, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(main_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(main_module, "_configure_tensorflow_devices", lambda *a, **k: None)
    monkeypatch.setattr(
        main_module, "_model_class_for_dataset", lambda dataset: _FakeWinnerModel
    )
    monkeypatch.setattr(
        main_module,
        "_model_weight_hashes",
        lambda model: {"g": "g", "e": "e", "f": "f", "h": "h"},
    )
    real_sha256_json = main_module.sha256_json
    monkeypatch.setattr(
        main_module,
        "sha256_json",
        lambda namespace, payload: (
            "restored-weight-hash"
            if namespace == "egm-candidate-network-weights"
            else real_sha256_json(namespace, payload)
        ),
    )

    model = main_module._fit_demand_design_model_multistart(
        _multistart_params(tmp_path), _tiny_train()
    )

    assert model.bgm_calls == 1
    assert len(model.egm_multistart_provenance["init_seeds"]) == 10
    assert model.egm_multistart_provenance["egm_num_warm_starts"] == 10
    assert model.egm_multistart_provenance["egm_selection_top_k"] == 3
    assert model.egm_multistart_provenance["egm_selected_rank"] in {1, 2, 3}
    assert model.ckpt.restored.endswith("ckpt-0")


def test_vector_multistart_constructs_grid_only_after_model_selection(monkeypatch):
    events = []
    train = _tiny_train()

    monkeypatch.setattr(
        main_module,
        "simulate_demand_design_vector_iv",
        lambda **kwargs: train,
    )

    def fit_model(params, train_std, **kwargs):
        del params, train_std, kwargs
        assert "grid" not in events
        events.append("fit")
        return types.SimpleNamespace()

    def make_grid(**kwargs):
        del kwargs
        events.append("grid")
        return _tiny_train(n=28)

    monkeypatch.setattr(main_module, "_fit_or_restore_demand_design_model", fit_model)
    monkeypatch.setattr(main_module, "make_demand_design_vector_grid", make_grid)
    monkeypatch.setattr(main_module, "_render_observed_ranges", lambda train: "ranges")
    monkeypatch.setattr(main_module, "_resolve_structural_methods", lambda params: ("map",))
    monkeypatch.setattr(
        main_module,
        "_finalize_demand_design_run",
        lambda *args, **kwargs: {"events": list(events)},
    )

    params = {
        "dataset": "Sim_Demand_Design_Vector_IV",
        "n_samples": 8,
        "rho": 0.5,
        "run_seed": 0,
        "v_dim": 785,
        "vector_dim": 784,
        "feature_seed": 42,
        "test_vector_seed": 42,
        "representation_sd": 0.5,
        "price_points": 2,
        "time_points": 2,
        "holdout_seed_offset": 1000,
        "egm_num_warm_starts": 10,
        "egm_selection_top_k": 3,
    }
    result = main_module._run_single_demand_design_vector_iv(params)
    assert result["events"] == ["fit", "grid"]
