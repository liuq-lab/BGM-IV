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
    """Stand-in for the candidate pool: emits one completed candidate per
    warm start, with EGM and BGM checkpoints and post-BGM criteria."""

    seen_kwargs = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, candidate_id, params, train, **kwargs):
        run_seed = int(params.get("run_seed", params.get("seed", 0)))
        del fn, params, train
        type(self).seen_kwargs.append(dict(kwargs))
        candidate_root = Path(kwargs["candidate_root"])
        candidate_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = candidate_root / "ckpt-0"
        (candidate_root / "ckpt-0.index").write_text("index", encoding="utf-8")
        bgm_checkpoint_path = candidate_root / "bgm-final" / "ckpt-1"
        bgm_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        (candidate_root / "bgm-final" / "ckpt-1.index").write_text(
            f"bgm-index-{candidate_id}", encoding="utf-8"
        )
        stdout_path = candidate_root / "candidate.stdout.log"
        stderr_path = candidate_root / "candidate.stderr.log"
        stdout_path.write_text("candidate complete\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        # EGM tail scores prefer candidate 0; the post-BGM criterion prefers
        # candidate 7, so the two orderings disagree on purpose.
        scores = [0.04 + 0.001 * int(candidate_id)] * len(
            kwargs["evaluation_iterations"]
        )
        train_iv_map = 200.0 - 5.0 * int(candidate_id) if int(candidate_id) <= 7 else 250.0
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
            bgm_checkpoint_path=str(bgm_checkpoint_path),
            bgm_checkpoint_hash=main_module._checkpoint_files_hash(bgm_checkpoint_path),
            bgm_checkpoint_weight_hash=f"bgm-weight-hash-{int(candidate_id)}",
            criterion_seed=int(kwargs["criterion_seed"]),
            train_iv_map=train_iv_map,
            train_iv_encoder=train_iv_map - 3.0,
            train_mse_x=0.08,
            train_mse_y=0.03,
            train_mse_v=0.1,
            bgm_seconds=12.5,
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
                    },
                    {
                        "stage": "epoch_eval",
                        "epoch": 2,
                        "include_outcome": True,
                        "mse_x": 0.08,
                        "mse_y": 0.03,
                        "mse_v": 0.1,
                    },
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


def _patch_parent(monkeypatch):
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

    def fake_sha256_json(namespace, payload):
        if namespace == "egm-candidate-network-weights":
            return "restored-weight-hash"
        if namespace == "bgm-candidate-network-weights":
            return "bgm-weight-hash-7"
        return real_sha256_json(namespace, payload)

    monkeypatch.setattr(main_module, "sha256_json", fake_sha256_json)


def test_multistart_bundle_selects_post_bgm_criterion_and_runs_no_parent_bgm(
    monkeypatch, tmp_path
):
    _patch_parent(monkeypatch)
    _ImmediateExecutor.seen_kwargs = []
    train = _tiny_train()
    y_raw = np.arange(8, dtype=np.float64).reshape(8, 1)

    model = main_module._fit_demand_design_model_multistart(
        _multistart_params(tmp_path),
        train,
        criterion_data={"y_raw": y_raw, "y_stats": {"mean": 0.0, "scale": 1.0}},
    )

    # Every candidate worker received the shared post-EGM seed, its own
    # criterion seed and the raw training outcome for the criterion.
    assert len(_ImmediateExecutor.seen_kwargs) == 10
    post_egm_seeds = {kw["post_egm_seed"] for kw in _ImmediateExecutor.seen_kwargs}
    assert len(post_egm_seeds) == 1
    criterion_seeds = [kw["criterion_seed"] for kw in _ImmediateExecutor.seen_kwargs]
    assert len(set(criterion_seeds)) == 10
    assert all(
        np.array_equal(kw["criterion_y_raw"], y_raw.reshape(-1))
        for kw in _ImmediateExecutor.seen_kwargs
    )

    # The parent restores the post-BGM state of the criterion argmin and does
    # not train again.
    assert model.bgm_calls == 0
    assert model.ckpt.restored.endswith("candidate_07/bgm-final/ckpt-1")
    provenance = model.egm_multistart_provenance
    assert provenance["egm_num_warm_starts"] == 10
    assert provenance["egm_selector_version"] == "train-iv-map-post-bgm"
    assert provenance["egm_selection_criterion"] == "train_iv_map"
    assert provenance["egm_selected_candidate_id"] == 7
    assert provenance["egm_selected_criterion"] == 165.0
    assert provenance["egm_selected_train_iv_map"] == 165.0
    assert provenance["egm_selected_train_iv_encoder"] == 162.0
    assert provenance["egm_selected_egm_tail_rank"] == 8
    assert len(provenance["init_seeds"]) == 10
    assert len(provenance["criterion_seeds"]) == 10
    assert "selector_seed" not in provenance
    assert "egm_selection_top_k" not in provenance
    assert provenance["uses_holdout"] is False
    assert provenance["uses_test_grid"] is False
    # The selected candidate's full (EGM + BGM) history is carried over.
    assert [row["stage"] for row in model.training_history] == ["post_egm", "epoch_eval"]
    assert (tmp_path / "egm_multistart").is_dir()
    selection_files = list((tmp_path / "egm_multistart").glob("*/selection_manifest.json"))
    assert len(selection_files) == 1


def test_multistart_requires_criterion_data(monkeypatch, tmp_path):
    _patch_parent(monkeypatch)
    try:
        main_module._fit_demand_design_model_multistart(
            _multistart_params(tmp_path), _tiny_train()
        )
    except ValueError as exc:
        assert "criterion_data" in str(exc)
    else:
        raise AssertionError("multistart accepted a run without criterion data")


def test_vector_multistart_constructs_grid_only_after_model_selection(monkeypatch):
    events = []
    train = _tiny_train()

    monkeypatch.setattr(
        main_module,
        "simulate_demand_design_vector_iv",
        lambda **kwargs: train,
    )

    def fit_model(params, train_std, **kwargs):
        del params, train_std
        assert "grid" not in events
        assert "criterion_data" in kwargs
        assert np.asarray(kwargs["criterion_data"]["y_raw"]).shape[0] == 8
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
    }
    result = main_module._run_single_demand_design_vector_iv(params)
    assert result["events"] == ["fit", "grid"]
