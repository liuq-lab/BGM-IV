import numpy as np

from .simulators import (
    demand_design_h,
    demand_design_structural_function,
)


def _validate_vector_dimensions(v_dim, vector_dim):
    v_dim = int(v_dim)
    vector_dim = int(vector_dim)
    expected_v_dim = 1 + vector_dim
    if v_dim != expected_v_dim:
        raise ValueError(
            "`Sim_Demand_Design_Vector_IV` requires `v_dim == 1 + vector_dim`; "
            f"got v_dim={v_dim}, vector_dim={vector_dim}."
        )
    return v_dim, vector_dim


def make_demand_design_vector_prototypes(vector_dim=784, feature_seed=42):
    """Create fixed type-specific vector prototypes for S in {1, ..., 7}."""
    rng = np.random.default_rng(feature_seed)
    return rng.normal(0.0, 1.0, size=(7, int(vector_dim))).astype(np.float32)


def attach_demand_design_vectors(
    labels,
    vector_dim=784,
    feature_seed=42,
    vector_seed=0,
    representation_sd=0.5,
):
    """Attach a high-dimensional vector proxy for each latent customer type."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any((labels < 1) | (labels > 7)):
        raise ValueError("Demand-design vector labels must lie in {1, ..., 7}.")

    prototypes = make_demand_design_vector_prototypes(
        vector_dim=vector_dim,
        feature_seed=feature_seed,
    )
    rng = np.random.default_rng(vector_seed)
    perturbation = rng.normal(
        0.0,
        float(representation_sd),
        size=(labels.shape[0], int(vector_dim)),
    ).astype(np.float32)
    return prototypes[labels - 1] + perturbation


def simulate_demand_design_vector_iv(
    n_samples=5000,
    rho=0.5,
    seed=0,
    v_dim=785,
    vector_dim=784,
    feature_seed=42,
    representation_sd=0.5,
):
    """Simulate demand-design IV data with a vector proxy for customer type S."""
    _validate_vector_dimensions(v_dim, vector_dim)

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

    vector_features = attach_demand_design_vectors(
        customer_group,
        vector_dim=vector_dim,
        feature_seed=feature_seed,
        vector_seed=seed,
        representation_sd=representation_sd,
    )
    covariates = np.concatenate(
        [time.reshape(-1, 1).astype(np.float32), vector_features],
        axis=1,
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


def make_demand_design_vector_grid(
    price_points=20,
    time_points=20,
    v_dim=785,
    vector_dim=784,
    feature_seed=42,
    test_vector_seed=42,
    representation_sd=0.5,
):
    """Construct the vector-proxy demand-design evaluation grid."""
    _validate_vector_dimensions(v_dim, vector_dim)

    prices = np.linspace(10.0, 25.0, num=price_points, dtype=np.float64)
    times = np.linspace(0.0, 10.0, num=time_points, dtype=np.float64)
    customer_groups = np.arange(1.0, 8.0, dtype=np.float64)

    price_grid, time_grid, customer_grid = np.meshgrid(
        prices, times, customer_groups, indexing="ij"
    )
    flat_groups = customer_grid.reshape(-1).astype(np.int64)
    vector_features = attach_demand_design_vectors(
        flat_groups,
        vector_dim=vector_dim,
        feature_seed=feature_seed,
        vector_seed=test_vector_seed,
        representation_sd=representation_sd,
    )

    x = price_grid.reshape(-1, 1).astype(np.float32)
    v = np.concatenate(
        [time_grid.reshape(-1, 1).astype(np.float32), vector_features],
        axis=1,
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
