import numpy as np

from .simulators import (
    make_demand_design_grid,
    replace_customer_group,
    simulate_demand_design_iv,
)


def attach_demand_design_vectors(labels, vector_seed, representation_sd=0.5, feature_seed=42):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    prototypes = (
        np.random.default_rng(feature_seed).normal(0.0, 1.0, size=(7, 784)).astype(np.float32)
    )
    perturbation = np.random.default_rng(vector_seed).normal(
        0.0, float(representation_sd), size=(labels.shape[0], 784)
    ).astype(np.float32)
    return prototypes[labels - 1] + perturbation


def simulate_demand_design_vector_iv(n_samples=5000, rho=0.5, seed=0, representation_sd=0.5):
    data = simulate_demand_design_iv(n_samples=n_samples, rho=rho, seed=seed)
    proxies = attach_demand_design_vectors(data["customer_group"], seed, representation_sd)
    return replace_customer_group(data, proxies)


def make_demand_design_vector_grid(representation_sd=0.5, test_vector_seed=42):
    grid = make_demand_design_grid()
    proxies = attach_demand_design_vectors(
        grid["customer_group"], test_vector_seed, representation_sd
    )
    return replace_customer_group(grid, proxies)
