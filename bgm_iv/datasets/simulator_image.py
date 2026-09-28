from pathlib import Path

import numpy as np
import tensorflow as tf

from .simulators import make_demand_design_grid, replace_customer_group, simulate_demand_design_iv


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


def simulate_demand_design_mnist_iv(n_samples=5000, rho=0.5, seed=0):
    data = simulate_demand_design_iv(n_samples=n_samples, rho=rho, seed=seed)
    images = _attach_mnist_images(data["customer_group"], split="train", seed=seed)
    return replace_customer_group(data, images)


def make_demand_design_mnist_grid(image_seed=42):
    grid = make_demand_design_grid()
    images = _attach_mnist_images(grid["customer_group"], split="test", seed=image_seed)
    return replace_customer_group(grid, images)
