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
estimates and outcome predictive intervals. The MCMC pipeline uses multiple
overdispersed HMC chains and records convergence, effective-sample-size,
divergence, and Monte Carlo error diagnostics before reporting a certified
result. These uncertainty summaries are conditional on the fitted networks;
they quantify latent and predictive uncertainty rather than a full posterior
over all network parameters.

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

The YAML files cover the included low-dimensional, vector, and image
benchmarks and are the authoritative source for experiment-specific settings.

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
  mcmc/            # certified posterior integration (target, sampler,
                   # diagnostics, readout, certify)
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
snapshot and result files for each benchmark setting. Summary metrics are
stored in `results.csv`; MCMC runs also write `certified_results.csv` and
per-repeat JSON records. Runtime logs are stored under `logs/`.

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
