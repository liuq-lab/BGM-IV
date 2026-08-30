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
contains 3 sample sizes, 5 values of `rho`, and 20 repeats. Its canonical
multistart recipe requires one concrete outer cell per process so that the
winner BGM/MCMC CUDA context exits before another ten-candidate bundle starts.
Use scalar `n_samples`, scalar `rho`, one `--repeat-id`, and `-t 1`:

| Additional arguments | Concrete runs |
| --- | ---: |
| `--set n_samples=5000 --set rho=0.5 --repeat-id 0 -t 1` | 1 |

For example:

```bash
python main.py -c configs/Sim_Demand_Design_Vector_IV.yaml \
  --set n_samples=5000 --set rho=0.5 --repeat-id 0 -t 1
```

For a legacy single-start Cartesian sweep, override both multistart fields to
`1`; then `-t N` changes only outer-run concurrency and each worker still
requires a distinct visible GPU. `--repeat-id` accepts one integer in
`0, ..., n_repeat - 1` and requires `-t 1`.

### EGM multistart initialization

The Vector benchmark enables a training-only EGM multistart procedure with:

```yaml
egm_num_warm_starts: 10
egm_selection_top_k: 3
```

All starts use the same complete training sample and optimization schedule but
different, content-derived initialization seeds.  During the final ten EGM
evaluation points, each start is scored by the same `l2_loss_y` term evaluated
deterministically over every training row with the model's fixed eight-node
Gauss--Hermite treatment integral.  No validation split, simulated holdout,
evaluation grid, or structural truth is available to the selector.

The three lowest-scoring starts receive probabilities from a fixed
relative-loss softmax with temperature `0.05`; a manifest-derived random draw
selects exactly one terminal EGM checkpoint.  Only that checkpoint continues
through BGM, MAP, encoder, and MCMC evaluation.  Omitting the fields defaults
to the legacy single-start path (`1` start and top `1`).  The extra EGM starts
are part of the estimator's compute budget and are not independent repeats.

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
