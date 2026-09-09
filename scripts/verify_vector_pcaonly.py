"""Check PCA-only contracts on the real Vector DGP without fitting a model."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("bgm_main", str(root / "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from bgm_iv.proxy_transform import fit_pca

d = m.simulate_demand_design_vector_iv(n_samples=400, rho=0.5, seed=0)
g = m.make_demand_design_vector_grid()
t = fit_pca(d["v"][:, 1:], 1)
d1, g1 = m._apply_proxy_arm(d, t), m._apply_proxy_arm(g, t)
for key in ("x", "y", "w", "y_struct"):
    np.testing.assert_array_equal(d1[key], d[key])
a = m._standardize_demand_design_data(d1, g1)
b = m._standardize_demand_design_pcaonly_data(d1, g1)
for key in ("x", "y", "v", "w"):
    np.testing.assert_array_equal(a[0][key], b[0][key])
    print("[verify] k=1 standardized %s: identical=True" % key)
for key in ("x", "v"):
    np.testing.assert_array_equal(a[1][key], b[1][key])
np.testing.assert_array_equal(a[2]["v"]["scale"], b[2]["v"]["scale"])
try:
    fit_pca(d["v"][:, 1:], 784)
except ValueError:
    print("[verify] n400/pca784 correctly rejected")
else:
    raise AssertionError("invalid full PCA accepted")
full = m.simulate_demand_design_vector_iv(n_samples=1000, rho=0.5, seed=0)
for k in (1, 2, 6, 8, 64, 128, 784, None):
    data = full if k == 784 else d
    t = fit_pca(data["v"][:, 1:], k)
    output = m._apply_proxy_arm(data, t)
    width = 784 if k is None else k
    assert output["v"].shape == (len(data["v"]), width+1)
    print("[verify] pca_dim=%s width=%d OK" % (k, width+1))
name = "Sim_Demand_Design_Vector_PCAOnly_IV"
assert m._MODEL_CLASS_BY_DATASET[name] is m.BGM_IV
assert m._MCMC_FAMILY_BY_DATASET[name] == "demand"
print("[verify] ALL CONTRACTS HOLD")
