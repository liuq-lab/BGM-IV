# BGM-IV

Latent Bayesian generative modeling for nonlinear instrumental-variable analysis
with high-dimensional covariates.

BGM-IV estimates structural treatment-response functions from observational data
when treatment assignment is endogenous and valid instruments are available. It
learns a causally structured latent representation of covariates and replaces
the confounded outcome likelihood with an IV-integrated pseudo-likelihood
so that outcome learning is driven by instrument-induced treatment variation.
It also supports MCMC integration over the latent covariate posterior for
uncertainty-aware structural prediction.

## Highlights

- Nonlinear instrumental-variable regression under endogeneity
- Causally structured latent space `Z = (Z0, Z1, Z2, Z3)`: shared, outcome-only,
  treatment-only and covariate-only blocks
- IV-integrated pseudo-likelihood for endogeneity correction
- Structural prediction from the covariate-only MAP latent state
- Predictive intervals from multi-chain HMC on the latent posterior `p(z | v)`

## Model overview

BGM-IV models `Z ~ N(0, I)`, `V ~ p_theta(v | z)`, `X ~ p_phi(x | w, z0, z2)`
and `Y ~ p_omega(y | x, z0, z1)`. The outcome network is trained on the
IV-integrated pseudo-likelihood

```math
p_{\mathrm{IV}}(y\mid w,z)=\int p_\omega(y\mid x,z_0,z_1)\,p_\phi(x\mid w,z_0,z_2)\,dx,
```

and the latent states are updated on a tempered quasi-posterior. The structural
function is `g(x, v) = mu_omega(x, z0, z1)`, evaluated at the MAP of the
covariate-only posterior `p(z | v)`.

## Installation

### Create a conda environment

```bash
conda create -n bgmiv_env python=3.9 -y
conda activate bgmiv_env
```

### Install from source

```bash
git clone https://github.com/liuq-lab/BGM-IV.git
cd BGM-IV
pip install -e .
```

### Dependencies

This project is tested with:

- `python==3.9`
- `tensorflow==2.10.0`
- `tensorflow-probability==0.18.0`
- `numpy==1.24.2`
- `scipy==1.13.1`
- `pyyaml`
- `threadpoolctl`
- `python-dateutil`

For Linux GPU runs with TensorFlow 2.10, CUDA 11.2 and cuDNN 8.1 are the
expected compatible runtime libraries.

## Reproducing experiments with `main.py`

Each run trains one cell of a configuration: choose the sample size, the
confounding level (and, for the vector proxy, the number of principal
components) with `--set` and the repetition with `--repeat-id`.

### 1) Demand design

```bash
python main.py -c configs/Sim_Demand_Design_IV.yaml \
  --set n_samples=5000 --set rho=0.5 --repeat-id 0
```

### 2) High-dimensional vector proxy

```bash
python main.py -c configs/Sim_Demand_Design_Vector_PCAOnly_IV.yaml \
  --set n_samples=5000 --set pca_dim=8 --repeat-id 0
```

### 3) MNIST image covariates

```bash
python main.py -c configs/Sim_Demand_Design_Mnist_IV.yaml \
  --set n_samples=5000 --set rho=0.5 --repeat-id 0
```

The first MNIST run downloads the MNIST data into `data/`.

### Predictive intervals

Add `mcmc` to the readouts:

```bash
python main.py -c configs/Sim_Demand_Design_Vector_PCAOnly_IV.yaml \
  --set n_samples=5000 --set pca_dim=8 --set "structural_methods=[map,mcmc]" \
  --repeat-id 0
```

To sample from a trained model without retraining, repeat the same command with
`--mcmc-only TIMESTAMP`, where `TIMESTAMP` is the `checkpoint_timestamp` in the
checkpoint's `train_config.json`.

### What `main.py` does

- Simulates the training data of repetition `r` with seed `r` and builds the
  evaluation grid.
- Trains K (10 by default) warm-start candidates in parallel worker processes
  and keeps the one with the smallest training-set IV-moment residual; the model
  seed is drawn at random for every run and recorded with the results.
- Reports the structural MSE of the MAP readout on the evaluation grid and, with
  `mcmc`, the coverage and width of the 90/95/99% predictive intervals.
- Writes a `results.csv` row and a JSON record under `dumps/`, and the
  checkpoint with its `train_config.json` under `output_dir`.

## Project structure

```text
bgm_iv/
  datasets/          # data simulators
  models/            # BGM-IV models and networks
  mcmc/              # HMC on p(z | v) and the interval readout
  egm_multistart.py  # model seeds and warm-start selection
  proxy_transform.py # PCA of the vector proxy
  tests/
configs/             # experiment configurations
main.py              # experiment entrypoint
```

## Citation

If you use BGM-IV in your research, please cite the paper:

```bibtex
@misc{luo2026bgmivaipoweredbayesiangenerative,
      title={BGM-IV: an AI-powered Bayesian generative modeling approach for instrumental variable analysis}, 
      author={Guyue Luo and Qiao Liu},
      year={2026},
      eprint={2605.07029},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.07029}, 
}
```
