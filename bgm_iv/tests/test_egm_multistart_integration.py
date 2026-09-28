import json
from pathlib import Path

import numpy as np
import pytest

import main as main_module
from bgm_iv.egm_multistart import derive_multistart_seeds, make_candidate_record


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediateExecutor:
    seen_kwargs = []
    nan_train_mse_y_for = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, candidate_id, params, train, **kwargs):
        del fn, params, train
        type(self).seen_kwargs.append(dict(kwargs))
        candidate_root = Path(kwargs["candidate_root"])
        candidate_root.mkdir(parents=True, exist_ok=True)
        bgm_checkpoint_path = candidate_root / "bgm-final" / "ckpt-1"
        bgm_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        (candidate_root / "bgm-final" / "ckpt-1.index").write_text(
            f"bgm-index-{candidate_id}", encoding="utf-8"
        )
        stdout_path = candidate_root / "candidate.stdout.log"
        stderr_path = candidate_root / "candidate.stderr.log"
        stdout_path.write_text("candidate complete\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        train_iv_map = 200.0 - 5.0 * int(candidate_id) if int(candidate_id) <= 7 else 250.0
        record = make_candidate_record(
            candidate_id=int(candidate_id),
            init_seed=int(kwargs["init_seed"]),
            criterion_seed=int(kwargs["criterion_seed"]),
            status="completed",
            bgm_checkpoint_path=str(bgm_checkpoint_path),
            train_iv_map=train_iv_map,
            train_mse_x=0.08,
            train_mse_y=(
                float("nan")
                if type(self).nan_train_mse_y_for == int(candidate_id)
                else 0.03
            ),
            train_mse_v=0.1,
            bgm_seconds=12.5,
            device_names=["cpu"],
        )
        return _ImmediateFuture(
            {
                "candidate_id": int(candidate_id),
                "record": record,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "error": None,
            }
        )


class _RestoreStatus:
    def assert_consumed(self):
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
        self.bgm_calls = 0

    def restore_model_state_checkpoint(self, path):
        return self.ckpt.restore(path)

    def fit_bgm_from_egm(self, **kwargs):
        self.bgm_calls += 1
        self.fit_kwargs = kwargs


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
        "dataset": "Sim_Demand_Design_Vector_PCAOnly_IV",
        "output_dir": str(tmp_path),
        "n_samples": 8,
        "rho": 0.5,
        "repeat_id": 0,
        "egm_num_warm_starts": 10,
        "fit_egm_n_iter": 1_000,
        "fit_epochs": 2,
        "fit_epochs_per_eval": 1,
        "fit_batch_size": 4,
        "save_model": True,
        "use_gpu": False,
        "model_seed": 987654,
    }


def _patch_parent(monkeypatch):
    monkeypatch.setattr(main_module, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(main_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(main_module, "_configure_tensorflow_devices", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "_model_class", lambda params: _FakeWinnerModel)


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

    seeds = derive_multistart_seeds(987654, 10)
    assert [kw["init_seed"] for kw in _ImmediateExecutor.seen_kwargs] == seeds["init_seeds"]
    assert [kw["criterion_seed"] for kw in _ImmediateExecutor.seen_kwargs] == seeds["criterion_seeds"]
    assert {kw["schedule_seed"] for kw in _ImmediateExecutor.seen_kwargs} == {seeds["schedule_seed"]}
    assert len(_ImmediateExecutor.seen_kwargs) == 10
    post_egm_seeds = {kw["post_egm_seed"] for kw in _ImmediateExecutor.seen_kwargs}
    assert post_egm_seeds == {seeds["post_egm_seed"]}
    assert model.random_seed == seeds["post_egm_seed"]
    criterion_seeds = [kw["criterion_seed"] for kw in _ImmediateExecutor.seen_kwargs]
    assert len(set(criterion_seeds)) == 10
    assert all(
        np.array_equal(kw["criterion_y_raw"], y_raw.reshape(-1))
        for kw in _ImmediateExecutor.seen_kwargs
    )

    assert model.bgm_calls == 0
    assert model.ckpt.restored.endswith("candidate_07/bgm-final/ckpt-1")
    provenance = model.egm_multistart_provenance
    assert provenance["egm_num_warm_starts"] == 10
    assert provenance["egm_selector_version"] == "train-iv-map-post-bgm"
    assert provenance["egm_selection_criterion"] == "train_iv_map"
    assert provenance["egm_selected_candidate_id"] == 7
    assert provenance["egm_selected_criterion"] == 165.0
    assert provenance["egm_selected_train_iv_map"] == 165.0
    assert "egm_selected_train_iv_encoder" not in provenance
    assert provenance["model_seed"] == 987654
    assert len(provenance["init_seeds"]) == 10
    assert len(provenance["criterion_seeds"]) == 10
    assert "selector_seed" not in provenance
    assert (tmp_path / "egm_multistart").is_dir()
    selection_files = list((tmp_path / "egm_multistart").glob("*/selection.json"))
    assert len(selection_files) == 1
    selection = json.loads(selection_files[0].read_text())
    assert selection["selected_candidate_id"] == 7
    assert len(selection["candidates"]) == 10


def test_multistart_requires_criterion_data(monkeypatch, tmp_path):
    _patch_parent(monkeypatch)
    with pytest.raises(TypeError, match="criterion_data"):
        main_module._fit_demand_design_model_multistart(
            _multistart_params(tmp_path), _tiny_train()
        )


def test_selected_candidate_with_nonfinite_training_fit_is_still_reported(
    monkeypatch, tmp_path
):
    _patch_parent(monkeypatch)
    _ImmediateExecutor.seen_kwargs = []
    _ImmediateExecutor.nan_train_mse_y_for = 7
    try:
        model = main_module._fit_demand_design_model_multistart(
            _multistart_params(tmp_path),
            _tiny_train(),
            criterion_data={"y_raw": np.zeros((8, 1)), "y_stats": None},
        )
    finally:
        _ImmediateExecutor.nan_train_mse_y_for = None
    provenance = model.egm_multistart_provenance
    assert provenance["egm_selected_candidate_id"] == 7
    assert provenance["egm_selected_train_mse_y"] is None


def _worker_params(tmp_path):
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
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "structural_map_steps": 3,
        "structural_map_lr": 5e-4,
        "fit_egm_n_iter": 4,
        "fit_epochs": 0,
        "fit_epochs_per_eval": 1,
        "fit_batch_size": 16,
    }


def test_candidate_worker_streams_follow_their_seeds(tmp_path):
    import tensorflow as tf
    from bgm_iv.datasets import simulate_demand_design_iv

    data = simulate_demand_design_iv(n_samples=48, rho=0.5, seed=0)
    train = {key: np.asarray(data[key], np.float32) for key in ("x", "y", "v", "w")}
    params = _worker_params(tmp_path)

    def run(tag, **overrides):
        seeds = dict(init_seed=11, schedule_seed=12, post_egm_seed=13, criterion_seed=14)
        seeds.update(overrides)
        result = main_module._run_egm_candidate_worker(
            0,
            params,
            train,
            criterion_y_raw=train["y"],
            criterion_y_stats=None,
            candidate_root=str(tmp_path / tag),
            **seeds,
        )
        assert result["error"] is None, result["error"]
        reader = tf.train.load_checkpoint(result["record"]["bgm_checkpoint_path"])
        names = sorted(
            name for name in reader.get_variable_to_shape_map() if "save_counter" not in name
        )
        return result["record"], {name: reader.get_tensor(name) for name in names}

    def same_weights(a, b):
        return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)

    base_record, base_state = run("base")
    again_record, again_state = run("again")
    assert again_record["train_iv_map"] == base_record["train_iv_map"]
    assert same_weights(base_state, again_state)

    _, schedule_state = run("schedule", schedule_seed=99)
    assert not same_weights(base_state, schedule_state)

    _, post_state = run("post", post_egm_seed=99)
    assert not same_weights(base_state, post_state)

    criterion_record, criterion_state = run("criterion", criterion_seed=99)
    assert same_weights(base_state, criterion_state)
    assert criterion_record["train_iv_map"] != base_record["train_iv_map"]
