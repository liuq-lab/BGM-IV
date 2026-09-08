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

### Warm-start multistart and model selection

Every benchmark configuration enables a training-only multistart procedure
with:

```yaml
egm_num_warm_starts: 10
```

All starts use the same complete training sample and optimization schedule but
different initialization seeds derived from the cell's `run_seed`.  The same
master seed also deterministically derives the shared EGM schedule, the shared
post-EGM/BGM stream, and one criterion stream per start.  Each start is trained
end to end in its own worker process: the EGM stage, then the BGM stage from
the persisted EGM state (a fresh model under the shared post-EGM seed, exactly
as a single continuation would be), then the selection criterion.

The selection criterion is the training-set IV-moment residual under the MAP
latent readout, measured after BGM on the rows the start was fitted on: the
mean squared gap between the observed training outcome and
`E[f(X, z) | w, z]` with `z` inferred from `v` alone and `X` drawn from the
model's own first stage.  The start with the smallest residual is selected by
a deterministic argmin (candidate-id tie break), and MAP, encoder, and MCMC
readouts are all reported from that one model.  No validation split, simulated
holdout, evaluation grid, or structural truth is available to the selector.
The EGM tail-window score (`l2_loss_y` over every training row with the fixed
eight-node Gauss--Hermite treatment integral, averaged over the last ten EGM
evaluation points) is still recorded per start as a diagnostic, but it does not
enter the selection.  Omitting the field defaults to the legacy single-start
path; `egm_selection_top_k` is no longer accepted.  The extra starts are part
of the estimator's compute budget and are not independent repeats.

Candidate workers persist an optimizer-free `egm-transition` checkpoint (the
trained networks and EGM variance EMA that initialize the BGM stage; EGM
optimizer slots are deliberately not resumable) and, after BGM, an
optimizer-free `bgm-final` checkpoint that the parent restores for the selected
start.  The training manifest then binds a separate optimizer-free
`inference-state` checkpoint used by structural evaluation and `--mcmc-only`.

To restore a saved training checkpoint and run only structural evaluation and
MCMC inference:

```bash
python main.py -c configs/Sim_Demand_Design_Mnist_IV.yaml \
  --set n_samples=5000 --set rho=0.5 --repeat-id 0 \
  --mcmc-only TIMESTAMP -t 1
```

`--mcmc-only` restores the manifest-bound inference state, skips EGM,
selection, and BGM, then reruns structural evaluation and full-grid MCMC from
the beginning.  It does not resume a partially completed MCMC chain.  Empty or
whitespace-only timestamps are rejected before training can start.  Pilot and
production seeds remain deterministically derived from `run_seed`, MCMC
family, checkpoint identity, and stage; there is no public `mcmc_seed` option.

Every demand-design YAML exposes the production sampling budget explicitly:

```yaml
# Draws are retained draws per chain.
mcmc_num_chains: 4
mcmc_production_warmup_steps: 2000
mcmc_production_draws: 5000
```

The three values must be supplied together.  Production uses at least four
chains; fewer chains fail before sampling.  These inference-only controls do
not alter the training-manifest or checkpoint identity.  The family recipe is
the backward-compatible fallback for older configs, while the YAML values
override that fallback.  Private ablation CLI flags, when present, override
the YAML warmup/draw counts without changing the YAML chain count.

## MNIST Cache

The repository does not include `data/mnist.npz`. The MNIST-IV simulator
uses `tf.keras.datasets.mnist.load_data(...)`, so the first MNIST-IV run will
download the dataset into `data/` if it is missing. If the runner has no
internet access, provide the cache file before running the MNIST config.

## Project Structure

```text
bgm_iv/
  datasets/        # demand, MNIST, and vector-proxy simulators
  hashing.py       # SHA-256 digests shared by main.py and mcmc/
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
per Pixel/Vector repeat and 10--25 minutes per Demand repeat.
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
