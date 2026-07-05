from pathlib import Path

import numpy as np
import tensorflow as tf

from .simulators import (
    demand_design_h,
    demand_design_structural_function,
)


def _resolve_mnist_extra_covariate_dim(v_dim):
    v_dim = int(v_dim)
    if v_dim < 785:
        raise ValueError(
            "`Sim_Demand_Design_Mnist_IV` requires `v_dim >= 785` because the "
            "base observed covariates are `(time, flattened_mnist_image)`."
        )
    return v_dim - 785


def _append_gaussian_covariates(covariates, extra_covariate_dim, rng):
    extra_covariate_dim = int(extra_covariate_dim)
    if extra_covariate_dim <= 0:
        return covariates.astype(np.float32)

    nuisance_covariates = rng.normal(
        0.0,
        1.0,
        size=(covariates.shape[0], extra_covariate_dim),
    ).astype(np.float32)
    return np.concatenate([covariates.astype(np.float32), nuisance_covariates], axis=1)


def _mnist_cache_dir():
    return Path(__file__).resolve().parents[2] / "data"


def _load_mnist_arrays():
    cache_dir = _mnist_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tf.device("/CPU:0"):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(
            path=str(cache_dir / "mnist.npz")
        )

    return {
        "train_images": np.asarray(train_images, dtype=np.uint8),
        "train_labels": np.asarray(train_labels, dtype=np.int64),
        "test_images": np.asarray(test_images, dtype=np.uint8),
        "test_labels": np.asarray(test_labels, dtype=np.int64),
    }


def _attach_mnist_images(labels, split="train", seed=42):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any((labels < 1) | (labels > 7)):
        raise ValueError("MNIST demand-design labels must lie in {1, ..., 7}.")

    mnist = _load_mnist_arrays()
    if split == "train":
        images = mnist["train_images"]
        image_labels = mnist["train_labels"]
    elif split == "test":
        images = mnist["test_images"]
        image_labels = mnist["test_labels"]
    else:
        raise ValueError("`split` must be one of {'train', 'test'}.")

    rng = np.random.default_rng(seed)
    attached = []
    for label in labels:
        candidates = images[image_labels == label]
        if len(candidates) == 0:
            raise ValueError(f"No MNIST images available for digit {label}.")
        attached.append(candidates[rng.choice(len(candidates))].reshape(1, -1))
    return np.concatenate(attached, axis=0).astype(np.float32)


def simulate_demand_design_mnist_iv(
    n_samples=5000,
    rho=0.5,
    seed=0,
    v_dim=785,
):
    """Simulate the MNIST demand-design IV benchmark with image covariates."""
    extra_covariate_dim = _resolve_mnist_extra_covariate_dim(v_dim)

    rng = np.random.default_rng(seed)
    customer_group = rng.integers(1, 8, size=n_samples)
    time = rng.uniform(0.0, 10.0, size=n_samples)
    instrument = rng.normal(0.0, 1.0, size=n_samples)
    latent_confounder = rng.normal(0.0, 1.0, size=n_samples)

    noise_scale = np.sqrt(max(1.0 - float(rho) ** 2, 1e-6))
    epsilon = rho * latent_confounder + rng.normal(0.0, noise_scale, size=n_samples)

    price = 25.0 + (instrument + 3.0) * demand_design_h(time) + latent_confounder
    structural_mean = demand_design_structural_function(price, time, customer_group)
    demand = structural_mean + epsilon

    image_features = _attach_mnist_images(customer_group, split="train", seed=seed)
    base_covariates = np.concatenate(
        [time.reshape(-1, 1).astype(np.float32), image_features],
        axis=1,
    )
    covariates = _append_gaussian_covariates(
        base_covariates,
        extra_covariate_dim,
        rng,
    )

    return {
        "x": price.reshape(-1, 1).astype(np.float32),
        "y": demand.reshape(-1, 1).astype(np.float32),
        "v": covariates.astype(np.float32),
        "w": instrument.reshape(-1, 1).astype(np.float32),
        "y_struct": structural_mean.reshape(-1, 1).astype(np.float32),
        "time": time.reshape(-1, 1).astype(np.float32),
        "customer_group": customer_group.reshape(-1, 1).astype(np.float32),
    }


def make_demand_design_mnist_grid(
    price_points=20,
    time_points=20,
    v_dim=785,
    image_seed=42,
    noise_seed=42,
):
    """Construct the MNIST demand-design evaluation grid."""
    extra_covariate_dim = _resolve_mnist_extra_covariate_dim(v_dim)

    prices = np.linspace(10.0, 25.0, num=price_points, dtype=np.float64)
    times = np.linspace(0.0, 10.0, num=time_points, dtype=np.float64)
    customer_groups = np.arange(1.0, 8.0, dtype=np.float64)

    price_grid, time_grid, customer_grid = np.meshgrid(
        prices, times, customer_groups, indexing="ij"
    )
    flat_groups = customer_grid.reshape(-1).astype(np.int64)
    image_features = _attach_mnist_images(flat_groups, split="test", seed=image_seed)

    x = price_grid.reshape(-1, 1).astype(np.float32)
    base_v = np.concatenate(
        [time_grid.reshape(-1, 1).astype(np.float32), image_features],
        axis=1,
    )
    v = _append_gaussian_covariates(
        base_v,
        extra_covariate_dim,
        np.random.default_rng(noise_seed),
    )
    y_struct = demand_design_structural_function(
        price_grid.reshape(-1),
        time_grid.reshape(-1),
        customer_grid.reshape(-1),
    ).reshape(-1, 1)

    return {
        "x": x,
        "v": v.astype(np.float32),
        "y_struct": y_struct.astype(np.float32),
        "time": time_grid.reshape(-1, 1).astype(np.float32),
        "customer_group": customer_grid.reshape(-1, 1).astype(np.float32),
    }
