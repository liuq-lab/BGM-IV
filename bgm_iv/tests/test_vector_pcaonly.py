import csv
import json
from pathlib import Path

import numpy as np
import pytest
import main as m

from bgm_iv.proxy_transform import fit_pca, parse_pca_dim

NAME = "Sim_Demand_Design_Vector_PCAOnly_IV"


def params(k=6):
    p = m.yaml.safe_load((Path(m.__file__).parent / "configs" /
                          "Sim_Demand_Design_Vector_PCAOnly_IV.yaml").read_text())
    p.update(pca_dim=k, egm_num_warm_starts=1, n_samples=200, rho=0.5,
             n_repeat=1, repeat_id=0, run_seed=0, use_gpu=False)
    m._apply_demand_design_benchmark_defaults(p)
    return p


def test_pcaonly_registration_and_explicit_config():
    assert m._DATASETS[NAME] == ("_run_single_demand_design_vector_pcaonly_iv", m.BGM_IV, "demand")
    assert m._model_class({"dataset": NAME}) is m.BGM_IV
    assert m._single_run_fn({"dataset": NAME}) is m._run_single_demand_design_vector_pcaonly_iv
    p = params(6)
    assert p["pca_dim"] == 6
    assert (p["v_dim"], p["w_dim"]) == (785, 1)
    for bad, match in (({"dataset": NAME}, "pca_dim"),
                       ({"dataset": NAME, "pca_dim": "none"}, "1..784"),
                       ({"dataset": NAME, "pca_dim": [1, 2]}, "scalar pca_dim")):
        with pytest.raises(ValueError, match=match):
            m._apply_demand_design_benchmark_defaults(bad)


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 785, 6.0, 6.7, "6", "bad", [6], None, "none", ""]
)
def test_pcaonly_rejects_bad_dimension(value):
    with pytest.raises(ValueError, match="1..784"):
        parse_pca_dim(value)


@pytest.mark.parametrize("value", [1, 128, 784, np.int64(8)])
def test_pcaonly_accepts_integer_dimensions(value):
    assert parse_pca_dim(value) == int(value)
    assert type(parse_pca_dim(value)) is int


def test_pcaonly_standardizer_is_demand_at_one_component_and_preserves_other_data():
    d = m.simulate_demand_design_vector_iv(n_samples=64, rho=0.5, seed=0)
    t = fit_pca(d["v"][:, 1:], 1)
    changed = m._apply_proxy_arm(d, t)
    for key in ("x", "y", "w", "y_struct"):
        np.testing.assert_array_equal(changed[key], d[key])
    a = m._standardize_demand_design_data(changed, changed)
    b = m._standardize_demand_design_pcaonly_data(changed, changed)
    for key in ("x", "y", "v", "w"):
        np.testing.assert_array_equal(a[0][key], b[0][key])
    for key in ("x", "v"):
        np.testing.assert_array_equal(a[1][key], b[1][key])
    np.testing.assert_array_equal(a[2]["v"]["scale"], b[2]["v"]["scale"])


def test_proxy_block_has_one_shared_scale():
    d = m.simulate_demand_design_vector_iv(n_samples=64, rho=0.5, seed=3)
    d = m._apply_proxy_arm(d, fit_pca(d["v"][:, 1:], 6))
    stats = m._standardize_demand_design_pcaonly_data(d, d)[2]
    assert np.unique(stats["v"]["scale"][:, 1:]).size == 1
    expected = np.sqrt(np.mean(np.square(np.std(d["v"][:, 1:], axis=0))))
    np.testing.assert_allclose(stats["v"]["scale"][:, 1:], expected, rtol=1e-5)


@pytest.mark.parametrize("k", [1, 6])
def test_runner_narrows_locally_and_persists_pca_dim(monkeypatch, tmp_path, k):
    p = params(k)
    captured = {}

    def fit(model_params, data, **kwargs):
        captured.update(params=dict(model_params), data=data, criterion=kwargs["criterion_data"])
        return object()

    def finalize(model_params, model, **kwargs):
        return {"provenance": {"checkpoint_timestamp": "test"},
                "run_config_text": kwargs["run_config_text"],
                "final_results": {"map": 1.0}}

    monkeypatch.setattr(m, "_fit_or_restore_demand_design_model", fit)
    monkeypatch.setattr(m, "_finalize_demand_design_run", finalize)
    result = m._run_single_demand_design_vector_pcaonly_iv(p)
    width = k
    assert captured["data"]["v"].shape == (200, width+1)
    assert captured["params"]["v_dim"] == width+1
    assert p["v_dim"] == 785
    assert captured["criterion"]["y_raw"].shape == (200, 1)
    assert f"pca_dim: {k}" in result["run_config_text"]
    m._persist_demand_design_repeat_outputs(tmp_path, p, result)
    csv_path = next(tmp_path.rglob("results.csv"))
    row = list(csv.DictReader(csv_path.open()))[0]
    assert row["pca_dim"] == str(k)
    assert json.loads(next(tmp_path.rglob("repeat0_test.json")).read_text())
    trained = m._training_params(captured["params"])
    assert trained["pca_dim"] == k and trained["v_dim"] == width + 1
    changed = dict(captured["params"], pca_dim=k + 1)
    assert m._training_params(changed) != trained
