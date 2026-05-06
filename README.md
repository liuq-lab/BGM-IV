# CausalBGM-IV

This repository is the minimal CausalBGM-IV package for the demand-design
instrumental-variable experiments.

It keeps only the code needed to run:

- `Sim_Demand_Design_IV`
- `Sim_Demand_Design_Mnist_IV` 
- `Sim_Demand_Design_Vector_IV`

The Python namespace remains `bayesgm`, but the public model API is limited to
`CausalBGM_IV`, `CausalBGM_IV_Image`, and `CausalBGM_IV_Vector`.

The paper experiment entrypoint is MAP-only for structural evaluation. MAP is
the default when structural method fields are omitted, and other structural
method names are rejected before training starts.

Benchmark constants such as random seeds, instrument dimension, evaluation-grid
size, and fixed image/vector seeds are set in code defaults so the three configs
use the same inputs and grids for fair comparison. The MNIST config keeps
`v_dim` visible because `v_dim: 1000` selects the MNIST-HD setting with 215
Gaussian nuisance covariates appended after `[time, image_784]`.

## Setup

From the package source directory:

```bash
cd src
pip install -e .
```

The expected runtime stack is TensorFlow 2.10, TensorFlow Probability 0.18, and
NumPy 1.24.

## Run Experiments

Run all commands from `src/`:

```bash
python main.py -c configs/Sim_Demand_Design_IV.yaml -t 1
python main.py -c configs/Sim_Demand_Design_Mnist_IV.yaml -t 1
python main.py -c configs/Sim_Demand_Design_Vector_IV.yaml -t 1
```

Use `-t x` to set the number of parallel workers for demand-design sweeps.

## MNIST Cache

The repository does not include `src/data/mnist.npz`. The MNIST-IV simulator
uses `tf.keras.datasets.mnist.load_data(...)`, so the first MNIST-IV run will
download the dataset into `src/data/` if it is missing. If the runner has no
internet access, provide the cache file before running the MNIST-IV config.

## Outputs

Runtime files are intentionally ignored by Git:

- `src/logs/`
- `src/dumps/`
- `src/data/`
- checkpoints and result folders

The active markdown logs are recreated automatically at runtime.
