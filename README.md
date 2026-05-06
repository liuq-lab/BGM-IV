# BGM-IV

Bayesian generative modeling for nonlinear instrumental-variable regression.

BGM-IV is a Bayesian generative modeling package for nonlinear
instrumental-variable regression with high-dimensional covariates. It jointly
models covariates, treatment, outcome, and instrument-induced treatment
variation through a structured latent generative model, enabling structural
prediction under endogenous treatment assignment.

## Highlights

- Nonlinear instrumental-variable regression under endogeneity
- Latent Bayesian generative modeling for structured covariate representations
- IV-integrated pseudo-likelihood for endogeneity correction
- MAP structural prediction for fast point estimates

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

Below is a minimal example that mirrors the main workflow: load a configuration,
simulate demand-design IV data, train BGM-IV, and evaluate MAP structural
predictions. Run it from `src/`. The example uses the real
`Sim_Demand_Design_IV` configuration and only overrides runtime settings so the
demo finishes quickly.

```python
import numpy as np
import yaml

from bayesgm.datasets import make_demand_design_grid, simulate_demand_design_iv
from bayesgm.models import BGM_IV

# Load the low-dimensional demand-design configuration
params = yaml.safe_load(open("configs/Sim_Demand_Design_IV.yaml", "r"))

# Small demo overrides so the example runs quickly
params.update(
    {
        "output_dir": ".",
        "save_res": False,
        "save_model": False,
        "use_bnn": False,
        "n_samples": 128,
        "rho": 0.5,
        "n_repeat": 1,
        "fit_epochs": 1,
        "fit_egm_n_iter": 0,
        "fit_batch_size": 32,
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "structural_map_steps": 5,
        "v_dim": 2,
        "w_dim": 1,
        "seed": 0,
    }
)

train = simulate_demand_design_iv(
    n_samples=params["n_samples"],
    rho=params["rho"],
    seed=params["seed"],
)
grid = make_demand_design_grid(price_points=4, time_points=3)

model = BGM_IV(params=params, random_seed=42)
model.fit(
    data=(train["x"], train["y"], train["v"], train["w"]),
    epochs=params["fit_epochs"],
    batch_size=params["fit_batch_size"],
    use_egm_init=False,
    verbose=0,
)

prediction = model.predict_structural(
    grid["x"],
    grid["v"],
    latent_method="map",
    map_steps=params["structural_map_steps"],
)
mse = np.mean((prediction - grid["y_struct"]) ** 2)
print(f"Structural MSE: {mse:.4f}")
```

## Reproducing Experiments With `main.py`

`main.py` is the primary experiment entrypoint. It reads a YAML configuration
with `-c` and runs the requested demand-design IV benchmark.

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

For MNIST-HD, set `v_dim: 1000` in the MNIST config or in a copied config. This
appends 215 iid Gaussian nuisance covariates after `[time, image_784]`.

### What `main.py` does

For each run:

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
  configs/           # YAML configs for experiments
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

Running `main.py` creates `src/logs/` and `src/dumps/` automatically. Each run
gets a dump root named with the benchmark slug and run start time:

```text
src/dumps/<dataset_slug>_<YYYY-MM-DD_HH-MM-SS-microseconds>/
```

For the provided configs, `<dataset_slug>` is one of
`sim_demand_design_iv`, `sim_demand_design_mnist_iv`, or
`sim_demand_design_vector_iv`. The dump root stores a config snapshot and
per-setting training summaries, with subfolders such as:

```text
src/dumps/sim_demand_design_iv_2026-05-06_14-30-10-123456/
  Sim_Demand_Design_IV.yaml
  n_samples:<n>-rho:<rho>-v_dim:<v_dim>/
    training_metric_history_*.csv
    training_metric_history_*.md
    best_structural_checkpoints.csv
    last_structural_checkpoints.csv
```

The active markdown log is also recreated automatically:

```text
src/logs/outputs_dev_<dataset_slug>_active.md
```

## Citation

If you use BGM-IV in your research, please cite the paper:

```bibtex
@misc{luo2026bgmiv,
  title  = {BGM-IV: an AI-powered Bayesian generative modeling approach for instrumental variable analysis},
  author = {Luo, Guyue and Liu, Qiao},
  year   = {2026},
}
```
