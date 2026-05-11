import numpy as np


def demand_design_h(time):
    """Demand-design seasonality function used in DFIV/DeepIV benchmarks."""
    time = np.asarray(time, dtype=np.float64)
    return 2.0 * (
        ((time - 5.0) ** 4) / 600.0
        + np.exp(-4.0 * (time - 5.0) ** 2)
        + time / 10.0
        - 2.0
    )


def demand_design_structural_function(price, time, customer_group):
    """Ground-truth structural function for the demand-design IV benchmark."""
    price = np.asarray(price, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    customer_group = np.asarray(customer_group, dtype=np.float64)
    return 100.0 + (10.0 + price) * customer_group * demand_design_h(time) - 2.0 * price


def simulate_demand_design_iv(
    n_samples=5000,
    rho=0.5,
    seed=0,
):
    """Simulate the demand-design IV benchmark from DFIV."""
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

    covariates = np.column_stack([time, customer_group]).astype(np.float32)
    return {
        "x": price.reshape(-1, 1).astype(np.float32),
        "y": demand.reshape(-1, 1).astype(np.float32),
        "v": covariates,
        "w": instrument.reshape(-1, 1).astype(np.float32),
        "y_struct": structural_mean.reshape(-1, 1).astype(np.float32),
        "time": time.reshape(-1, 1).astype(np.float32),
        "customer_group": customer_group.reshape(-1, 1).astype(np.float32),
    }


def make_demand_design_grid(
    price_points=20,
    time_points=20,
):
    """Construct the low-dimensional demand-design evaluation grid."""
    prices = np.linspace(10.0, 25.0, num=price_points, dtype=np.float64)
    times = np.linspace(0.0, 10.0, num=time_points, dtype=np.float64)
    customer_groups = np.arange(1.0, 8.0, dtype=np.float64)

    price_grid, time_grid, customer_grid = np.meshgrid(
        prices, times, customer_groups, indexing="ij"
    )
    x = price_grid.reshape(-1, 1).astype(np.float32)
    v = np.column_stack([time_grid.reshape(-1), customer_grid.reshape(-1)]).astype(
        np.float32
    )
    y_struct = demand_design_structural_function(
        price_grid.reshape(-1),
        time_grid.reshape(-1),
        customer_grid.reshape(-1),
    ).reshape(-1, 1)

    return {
        "x": x,
        "v": v,
        "y_struct": y_struct.astype(np.float32),
        "time": time_grid.reshape(-1, 1).astype(np.float32),
        "customer_group": customer_grid.reshape(-1, 1).astype(np.float32),
    }
