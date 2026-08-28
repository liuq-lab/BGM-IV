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

This design is intended for nonlinear IV settings where useful causal
information may be embedded in high-dimensional or noisy covariates, while
remaining applicable to low-dimensional covariates.

## Highlights

- Nonlinear instrumental-variable regression under endogeneity
- Latent Bayesian generative modeling for structured covariate representations
- IV-integrated pseudo-likelihood for endogeneity correction
- Structural prediction with MAP, encoder, and posterior-integration readouts
- Uncertainty quantification through multi-chain latent-posterior MCMC
- Reproducible benchmarks for low-dimensional, high-dimensional, and image
  covariates

## Uncertainty Quantification

In addition to point prediction, BGM-IV can average the structural outcome
network over draws from the fitted latent posterior,

```math
\widehat g(x,v)
=
\frac{1}{M}\sum_{m=1}^{M}
\mu_{\widehat\omega}\!\left(x,z_Y^{(m)}\right),
\qquad
z^{(m)}\sim p_{\widehat\theta}(z\mid v).
```

This propagates latent-state uncertainty into posterior-integrated structural
estimates and outcome predictive intervals. Four overdispersed HMC chains are
run over the complete target catalog. Structural MSE, 50/80/95% coverage and
the corresponding 50/80/95% interval lengths use every post-warmup draw and
every evaluation-grid query.
These summaries are conditional on the fitted networks; they quantify latent
and predictive uncertainty rather than a full posterior over network weights.

## Installation

### Create a conda environment

```bash
conda create -n bgmiv_env python=3.9 -y
conda activate bgmiv_env
```

### Install from PyPI

```bash
pip install bgm-iv
```

This installs the importable Python package:

```python
from bgm_iv.models import BGM_IV
```

### Install from source

```bash
git clone https://github.com/liuq-lab/BGM-IV.git
cd BGM-IV
pip install -e .
```

Use the source install when you want to run the provided benchmark
configurations with `main.py`.

### Dependencies

This project is tested with:

- `python==3.9`
- `tensorflow==2.10.0`
- `tensorflow-probability==0.18.0`
- `numpy==1.24.2`
- `scipy==1.13.1`
- `pyyaml`
- `tqdm`
- `python-dateutil`

For Linux GPU runs with TensorFlow 2.10, CUDA 11.2 and cuDNN 8.1 are the
expected compatible runtime libraries.

## Running Experiments

`main.py` is the source-repository experiment entrypoint. Run it from the
repository root with one of the YAML configurations under `configs/`:

```bash
python main.py -c configs/Sim_Demand_Design_IV.yaml -t 1
```

A configuration may define a Cartesian-product sweep. The vector configuration
contains 3 sample sizes, 5 values of `rho`, and 20 repeats. Scalar `--set`
overrides collapse only the named axes; unspecified axes remain active. Using
`configs/Sim_Demand_Design_Vector_IV.yaml`, the common cases are:

| Additional arguments | Concrete runs |
| --- | ---: |
| `-t 1` | 300, sequentially |
| `--set n_samples=5000 -t 2` | 100; all 5 values of `rho` and 20 repeats, at most 2 concurrently |
| `--set n_samples=5000 --set rho=0.5 -t 1` | 20 repeats, sequentially |
| `--set n_samples=5000 --repeat-id 0 -t 1` | 5; repeat 0 at every value of `rho` |
| `--set n_samples=5000 --set rho=0.5 --repeat-id 0 -t 1` | 1 |

For example, the second case is:

```bash
python main.py -c configs/Sim_Demand_Design_Vector_IV.yaml \
  --set n_samples=5000 -t 2
```

`-t N` changes only the maximum concurrency, not the number of runs. When
`use_gpu: true`, each worker requires a distinct visible GPU. `--repeat-id`
accepts one integer in `0, ..., n_repeat - 1`, requires `-t 1`, and applies to
every sweep-axis combination that remains active.

To restore a saved training checkpoint and run only structural evaluation and
MCMC inference:

```bash
python main.py -c configs/Sim_Demand_Design_Mnist_IV.yaml \
  --set n_samples=5000 --set rho=0.5 --repeat-id 0 \
  --mcmc-only TIMESTAMP -t 1
```

## MNIST Cache

The repository does not include `data/mnist.npz`. The MNIST-IV simulator
uses `tf.keras.datasets.mnist.load_data(...)`, so the first MNIST-IV run will
download the dataset into `data/` if it is missing. If the runner has no
internet access, provide the cache file before running the MNIST config.

## Project Structure

```text
bgm_iv/
  datasets/        # demand, MNIST, and vector-proxy simulators
  features/        # image-representation export and preprocessing
  hashing.py       # SHA-256 digests shared by features/ and mcmc/
  models/          # BGM-IV model implementations
  mcmc/            # full-grid inference (target, sampler, readout, inference)
  utils/           # data I/O helpers
  tests/           # focused BGM-IV tests
configs/           # YAML configs for experiments
main.py            # experiment entrypoint
setup.py           # editable source install
```

## Outputs

Runtime files are intentionally ignored by Git:

- `logs/`
- `dumps/`
- `sweeps/`
- `data/`
- checkpoints and result folders

`main.py` writes run directories under `dumps/`, including a configuration
snapshot and result files for each benchmark setting. Headline MSE, coverage
and interval length are stored in one `results.csv`; per-repeat JSON records
contain sampler/target provenance and the finite-chain bias sensitivity.
Runtime logs include the HMC acceptance rate and are stored under `logs/`.

MCMC calibration columns are `mcmc_cov50`, `mcmc_cov80`, `mcmc_cov95`,
`mcmc_width50`, `mcmc_width80`, and `mcmc_width95`.

The all-draw Gaussian-mixture readout is expected to add roughly 20--40 minutes
per Pixel/Vector/MNIST-Feature repeat and 10--25 minutes per Demand repeat.
These are planning estimates; cluster wall time and peak memory must be
measured on the user's Yale jobs.

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
