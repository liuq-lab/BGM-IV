# BGM-IV

Bayesian generative modeling for nonlinear instrumental-variable regression.

BGM-IV is a latent Bayesian generative modeling approach for IV regression under
endogenous treatment assignment. The method learns a causally structured latent
space for observed covariates, treatment, and outcome, then uses an
IV-integrated pseudo-likelihood so outcome learning is driven by
instrument-induced treatment variation rather than the observed endogenous
treatment alone.

This repository is the minimal research package for the demand-design IV
experiments used in the BGM-IV paper. It keeps only the code needed for the
low-dimensional demand benchmark and the high-dimensional vector and MNIST
covariate benchmarks.

## Highlights

- Latent Bayesian generative modeling for nonlinear IV regression
- Structured latent components for shared confounding, outcome variation,
  treatment variation, and covariate-only nuisance information
- IV-integrated pseudo-likelihood for learning from instrument-induced
  treatment variation
- MAP structural prediction for paper experiments
- Reproducible demand-design benchmarks with fixed data-generation and
  evaluation-grid defaults
- Support for low-dimensional demand, vector-proxy demand, MNIST image
  covariates, and an MNIST-HD appendix variant

## Installation

### Create a conda environment

```bash
conda create -n bgmiv_env python=3.9 -y
conda activate bgmiv_env
```

### Install from source

```bash
git clone https://github.com/liuq-lab/BGM-IV.git
cd BGM-IV/src
pip install -e .
```

### Dependencies

This project is tested with:

- `python==3.9`
- `tensorflow==2.10.0`
- `tensorflow-probability==0.18.0`
- `numpy==1.24.2`
- `pyyaml`
- `tqdm`
- `python-dateutil`

For Linux GPU runs with TensorFlow 2.10, CUDA 11.2 and cuDNN 8.1 are the
expected compatible runtime libraries.

## Quickstart (Python API)

The public Python API remains under the `bayesgm` namespace. The primary model
classes are `BGM_IV`, `BGM_IV_Image`, and `BGM_IV_Vector`.

```python
import numpy as np

from bayesgm.datasets import make_demand_design_grid, simulate_demand_design_iv
from bayesgm.models import BGM_IV

train = simulate_demand_design_iv(n_samples=128, rho=0.5, seed=0)
grid = make_demand_design_grid(price_points=4, time_points=3)

params = {
    "dataset": "DemandDesignQuickstart",
    "output_dir": ".",
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
    "structural_map_steps": 5,
}

model = BGM_IV(params=params, random_seed=42)
model.fit(
    data=(train["x"], train["y"], train["v"], train["w"]),
    epochs=1,
    epochs_per_eval=1,
    batch_size=32,
    use_egm_init=False,
    verbose=0,
)

prediction = model.predict_structural(
    grid["x"],
    grid["v"],
    latent_method="map",
    map_steps=5,
)
mse = np.mean((prediction - grid["y_struct"]) ** 2)
print(f"Structural MSE: {mse:.4f}")
```

## Reproducing Experiments With `main.py`

Run all commands from `src/`:

```bash
python main.py -c configs/Sim_Demand_Design_IV.yaml -t 1
python main.py -c configs/Sim_Demand_Design_Mnist_IV.yaml -t 1
python main.py -c configs/Sim_Demand_Design_Vector_IV.yaml -t 1
```

Use `-t x` to set the number of parallel workers for demand-design sweeps.

### Provided configs

- `configs/Sim_Demand_Design_IV.yaml`: low-dimensional airline-demand IV
  benchmark with observed covariates `V=[time, customer_group]`.
- `configs/Sim_Demand_Design_Mnist_IV.yaml`: MNIST image-covariate IV
  benchmark with `V=[time, image_784]`; the current config uses `v_dim: 785`.
- `configs/Sim_Demand_Design_Vector_IV.yaml`: high-dimensional vector-proxy IV
  benchmark with a 784-dimensional proxy representation and
  `representation_sd: 0.5`.

The MNIST-HD appendix variant uses the MNIST config with `v_dim: 1000`, which
appends 215 iid Gaussian nuisance covariates after `[time, image_784]`.

The paper experiment entrypoint is MAP-only for structural evaluation. MAP is
the default when structural method fields are omitted, and other structural
method names are rejected before training starts.

### What `main.py` does

For each run, `main.py`:

- Loads a YAML config
- Injects fixed benchmark defaults for fair comparison
- Simulates the requested demand-design benchmark and structural evaluation grid
- Trains the appropriate BGM-IV model
- Computes MAP structural predictions on the original outcome scale
- Writes active logs and run summaries under ignored runtime directories

Structural performance is reported as structural MSE on the original outcome
scale using the same evaluation grid across methods.

## MNIST Cache

The repository does not include `src/data/mnist.npz`. The MNIST-IV simulator
uses `tf.keras.datasets.mnist.load_data(...)`, so the first MNIST-IV run will
download the dataset into `src/data/` if it is missing. If the runner has no
internet access, provide the cache file before running the MNIST config.

## Project Structure

```text
src/
  bayesgm/
    datasets/        # demand, MNIST, and vector-proxy simulators
    models/          # BGM-IV model implementations
    utils/           # data I/O helpers
    tests/           # focused BGM-IV tests
  configs/           # YAML configs for the three paper experiments
  main.py            # experiment entrypoint
  setup.py           # editable source install
```

## Outputs

Runtime files are intentionally ignored by Git:

- `src/logs/`
- `src/dumps/`
- `src/sweeps/`
- `src/data/`
- checkpoints and result folders

The active markdown logs are recreated automatically at runtime.

## Citation

If you use BGM-IV in your research, please cite the paper:

```bibtex
@misc{luo2026bgmiv,
  title  = {BGM-IV: an AI-powered Bayesian generative modeling approach for instrumental variable analysis},
  author = {Luo, Guyue and Liu, Qiao},
  year   = {2026},
}
```
