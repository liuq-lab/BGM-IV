"""PCA-only entrypoint, representation and real result-writer contracts."""
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
    assert m._MODEL_CLASS_BY_DATASET[NAME] is m.BGM_IV
    assert m._MCMC_FAMILY_BY_DATASET[NAME] == "demand"
    assert m._select_demand_design_single_run_fn(NAME) is m._run_single_demand_design_vector_pcaonly_iv
    p = params("none")
    assert p["pca_dim"] is None and p["proxy_transform"] == "none"
    assert (p["v_dim"], p["vector_dim"]) == (785, 784)
    for bad, match in (({"dataset": NAME}, "pca_dim"),
                       ({"dataset": NAME, "pca_dim": 6, "proxy_transform": "pca6"}, "use pca_dim")):
        with pytest.raises(ValueError, match=match):
            m._apply_demand_design_benchmark_defaults(bad)


@pytest.mark.parametrize("value", [True, False, 0, -1, 785, 6.0, "bad", [6]])
def test_pcaonly_rejects_bad_dimension(value):
    with pytest.raises(ValueError):
        parse_pca_dim(value)


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
    stats = m._standardize_demand_design_pcaonly_data(d)[2]
    assert np.unique(stats["v"]["scale"][:, 1:]).size == 1
    expected = np.sqrt(np.mean(np.square(np.std(d["v"][:, 1:], axis=0))))
    np.testing.assert_allclose(stats["v"]["scale"][:, 1:], expected, rtol=1e-5)


@pytest.mark.parametrize("k", [1, 6, None])
@pytest.mark.parametrize("writer", ["serial", "parallel"])
def test_runner_narrows_locally_and_persists_fitted_provenance(monkeypatch, tmp_path, k, writer):
    p = params(k)
    captured = {}

    def fit(model_params, data, **kwargs):
        captured.update(params=dict(model_params), data=data, criterion=kwargs["criterion_data"])
        return object()

    def finalize(model_params, model, **kwargs):
        prov = dict(kwargs["extra_provenance"], params=dict(model_params), checkpoint_timestamp="test")
        return {"provenance": prov, "training_history": [],
                "run_config_text": kwargs["run_config_text"], "ranges_text": kwargs["ranges_text"],
                "final_results": {"map": 1.0, "encoder": 2.0}}

    monkeypatch.setattr(m, "_fit_or_restore_demand_design_model", fit)
    monkeypatch.setattr(m, "_finalize_demand_design_run", finalize)
    result = m._run_single_demand_design_vector_pcaonly_iv(p)
    width = 784 if k is None else k
    assert captured["data"]["v"].shape == (200, width+1)
    assert captured["params"]["v_dim"] == width+1
    assert p["v_dim"] == 785 and p["vector_dim"] == 784
    assert captured["criterion"]["y_raw"].shape == (200, 1)
    assert result["provenance"]["model_v_dim"] == width+1
    if writer == "serial":
        m._persist_demand_design_repeat_outputs(
            tmp_path, 1, 1, p, result["run_config_text"], result["ranges_text"], [],
            final_results=result["final_results"], provenance=result["provenance"])
    else:
        m._flush_completed_demand_design_result(tmp_path, {
            "params": p, "run_index": 1, "total_runs": 1,
            "repeat_outputs": result, "error": None})
    csv_path = next(tmp_path.rglob("results.csv"))
    row = list(csv.DictReader(csv_path.open()))[0]
    for key in ("proxy_transform", "pca_fit_sha1", "train_proxy_sha1",
                "test_pca_sha1_r4", "vector_pca_version"):
        assert row[key] == str(result["provenance"][key]) and row[key]
    assert row["pca_dim"] == ("" if k is None else str(k))
    record = json.loads(next(tmp_path.rglob("repeat0_test.json")).read_text())
    assert record["provenance"]["model_vector_dim"] == width
    identity = m._manifest_params(captured["params"])
    changed = dict(captured["params"], vector_pca_version="different-version")
    assert m._manifest_params(changed) != identity


def test_pcaonly_rejects_raw_training():
    p = params(6)
    p["normalize_before_training"] = False
    with pytest.raises(ValueError, match="normalize_before_training"):
        m._run_single_demand_design_vector_pcaonly_iv(p)
