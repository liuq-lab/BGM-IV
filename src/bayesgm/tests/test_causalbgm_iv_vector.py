import numpy as np
import pytest
import tensorflow as tf

from bayesgm.datasets import (
    make_demand_design_grid,
    make_demand_design_vector_grid,
    simulate_demand_design_iv,
    simulate_demand_design_vector_iv,
)
from bayesgm.models.causalbgm import CausalBGM_IV_Vector
from bayesgm.models.networks import DemandVectorFeatureExtractor


def _make_vector_params(output_dir):
    return {
        "dataset": "Sim_Demand_Design_Vector_IV",
        "output_dir": str(output_dir),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "vector_dim": 784,
        "v_dim": 785,
        "w_dim": 1,
        "feature_seed": 42,
        "test_vector_seed": 42,
        "representation_sd": 0.5,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [16, 16],
        "e_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "dz_units": [16, 8],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 2,
        "eval_mc_samples": 2,
        "first_stage_warmup_epochs": 0,
        "structural_map_steps": 2,
        "latent_pzv_weight": 0.5,
    }


def test_simulate_demand_design_vector_iv_preserves_dgp_and_replaces_s():
    seed = 7
    train = simulate_demand_design_vector_iv(
        n_samples=12,
        rho=0.5,
        seed=seed,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        representation_sd=0.5,
    )
    base = simulate_demand_design_iv(n_samples=12, rho=0.5, seed=seed)

    np.testing.assert_allclose(train["x"], base["x"], atol=1e-6)
    np.testing.assert_allclose(train["y"], base["y"], atol=1e-6)
    np.testing.assert_allclose(train["w"], base["w"], atol=1e-6)
    np.testing.assert_allclose(train["y_struct"], base["y_struct"], atol=1e-6)
    np.testing.assert_allclose(train["v"][:, :1], base["time"], atol=1e-6)
    assert train["v"].shape == (12, 785)
    assert not np.allclose(train["v"][:, 1], base["customer_group"].reshape(-1))


def test_make_demand_design_vector_grid_matches_base_grid_layout():
    grid = make_demand_design_vector_grid(
        price_points=3,
        time_points=2,
        v_dim=785,
        vector_dim=784,
        feature_seed=42,
        test_vector_seed=42,
        representation_sd=0.5,
    )
    base_grid = make_demand_design_grid(price_points=3, time_points=2)

    np.testing.assert_allclose(grid["x"], base_grid["x"], atol=1e-6)
    np.testing.assert_allclose(grid["y_struct"], base_grid["y_struct"], atol=1e-6)
    np.testing.assert_allclose(grid["v"][:, :1], base_grid["time"], atol=1e-6)
    assert grid["v"].shape == (42, 785)


def test_vector_feature_extractor_output_shape():
    extractor = DemandVectorFeatureExtractor(v_dim=785, vector_dim=784)
    features = extractor(tf.zeros((2, 785), dtype=tf.float32), training=False)

    assert features.shape == (2, 65)


def test_causalbgm_iv_vector_accepts_v_dim(tmp_path):
    model = CausalBGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)

    data_v = tf.zeros((2, 785), dtype=tf.float32)
    data_z = model.e_net(data_v, training=False)
    decoded = model.g_net(data_z, training=False)

    assert data_z.shape == (2, 4)
    assert decoded["public_v"].shape == (2, 785)
    assert decoded["vector_mean"].shape == (2, 784)


def test_vector_model_rejects_alpha_v(tmp_path):
    params = _make_vector_params(tmp_path)
    params["alpha_v"] = 1.0

    with pytest.raises(ValueError, match="alpha_v"):
        CausalBGM_IV_Vector(params=params, random_seed=13)


def test_vector_covariate_posterior_is_untempered(tmp_path):
    model = CausalBGM_IV_Vector(params=_make_vector_params(tmp_path), random_seed=13)
    data_v = tf.zeros((2, 785), dtype=tf.float32)
    data_z = tf.constant(
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.1, -0.3, 0.2]],
        dtype=tf.float32,
    )

    loss_pv_z, _, _ = model._covariate_loss_terms(data_v, data_z, training=False)
    loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
    expected = -(loss_pv_z + loss_prior_z)
    actual = model.get_log_covariate_posterior(data_v, data_z)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
