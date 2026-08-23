"""Chain diagnostics, the structural metric, and the fail-closed gate."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from bgm_iv.mcmc.diagnostics import (
    Action,
    BatchIdentity,
    GateError,
    OutcomeTransform,
    PrecisionPolicy,
    SamplerEvidence,
    assess_sampler,
    chain_diagnostics,
    choose_block_len,
    functional_iact,
    score_batch,
    structural_metric,
)
from bgm_iv.mcmc.sampler import (
    FrozenBatch,
    FrozenVectorizedHMC,
    GaussianContextEvaluator,
    ProductionConfig,
    TrajectoryPolicy,
    WarmupConfig,
)
from bgm_iv.mcmc.target import sha256_array


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


# --- chain diagnostics -------------------------------------------------------


def test_chain_diagnostics_separate_mixed_shifted_constant_and_invalid():
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((400, 4, 5))
    draws[:, 1, 1] += 3.0                      # one chain shifted: R-hat > 1.01
    draws[:, :, 2] = 0.7                       # constant scalar
    draws[5, 0, 3] = np.nan                    # invalid scalar
    draws[:, :, 4] = np.cumsum(rng.standard_normal((400, 4)), axis=0)  # strongly autocorrelated
    diag = chain_diagnostics(draws)
    assert diag["rhat_rank"][0] < 1.01 and diag["rhat_folded"][0] < 1.01
    assert diag["ess_bulk"][0] > 800 and diag["ess_tail"][0] > 500
    assert diag["rhat_rank"][1] > 1.05
    assert diag["constant_mask"][2] and not diag["constant_mask"][0]
    assert diag["invalid_mask"][3] and not diag["invalid_mask"][0]
    assert diag["ess_bulk"][4] < diag["ess_bulk"][0] / 10


def test_functional_iact_and_block_length_rule():
    rng = np.random.default_rng(0)
    diag = chain_diagnostics(rng.standard_normal((240, 4, 5)))
    iact = functional_iact(diag, 240, 4)
    assert iact == pytest.approx(max(1.0, 960.0 / np.min(diag["ess_bulk"])))
    constant = chain_diagnostics(np.ones((240, 4, 2)))
    assert functional_iact(constant, 240, 4) == 0.0

    for t, max_iact in ((2400, 7.5), (12000, 3.0), (24000, 40.0)):
        b = choose_block_len(t, max_iact)
        assert b >= math.ceil(math.sqrt(t)) and b >= math.ceil(5.0 * max_iact)
        assert t % (2 * b) == 0 and t // (2 * b) >= 5
        assert all(
            not (t % (2 * c) == 0 and t // (2 * c) >= 5)
            for c in range(max(math.ceil(math.sqrt(t)), math.ceil(5.0 * max_iact)), b)
        )
    with pytest.raises(GateError, match="EXTEND_PRECISION"):
        choose_block_len(120, 40.0)


def test_structural_metric_identity_and_jackknife():
    rng = np.random.default_rng(1)
    functional = 5.0 + 0.3 * rng.standard_normal((240, 4, 11))
    truth = rng.standard_normal(11) + 5.0
    out = structural_metric(functional, truth, 20)
    plugin = np.mean((functional.mean(axis=(0, 1)) - truth) ** 2)
    assert out["mse_plugin"] == pytest.approx(plugin, rel=1e-12)
    assert out["mse_plugin"] - out["mse_u"] == pytest.approx(out["penalty_identity"], abs=1e-12)
    assert out["metric_valid"] is True and out["deterministic_functional"] is False
    assert out["halfwidth95"] >= 1.96 * max(out["mcse_jack_2b"], out["mcse_chain_jack"]) - 1e-12
    assert 0.5 <= out["batch_stability_ratio"] <= 2.0
    assert out["mcse_estimand"] == "mse_plugin_first_order"


# --- gate --------------------------------------------------------------------


def _evidence(**overrides):
    base = dict(
        num_chains=4, draws_per_chain=2000, rank_rhat_max=1.004, folded_rhat_max=1.003,
        bulk_ess_min=900.0, tail_ess_min=800.0, nonfinite_count=0, latent_constant_count=0,
        independent_initialization_provenance=True, fixed_production_kernel=True,
        target_gradient_finite=True, draws_hash=digest("draws"),
    )
    base.update(overrides)
    return SamplerEvidence(**base)


def test_assess_sampler_returns_the_only_legal_next_action():
    assert assess_sampler(_evidence()).action is Action.REPORTABLE
    assert assess_sampler(_evidence(num_chains=3)).action is Action.CONFIG_INVALID
    assert assess_sampler(_evidence(fixed_production_kernel=False)).action is Action.CONFIG_INVALID
    assert assess_sampler(_evidence(nonfinite_count=1)).action is Action.NUMERICAL_RESTART
    assert assess_sampler(_evidence(target_gradient_finite=False)).action is Action.NUMERICAL_RESTART
    assert assess_sampler(_evidence(geometry_action="MASS_UPGRADE")).action is Action.MASS_UPGRADE_REQUIRED
    assert assess_sampler(_evidence(latent_constant_count=1)).action is Action.STATIONARITY_RESTART
    assert assess_sampler(_evidence(rank_rhat_max=1.011)).action is Action.STATIONARITY_RESTART
    assert assess_sampler(_evidence(tail_ess_min=399.0)).action is Action.EXTEND_PRECISION
    assert assess_sampler(_evidence(bulk_ess_min=1e9)).action is Action.CONFIG_INVALID


def _identity(latent, x, inverse):
    return BatchIdentity(
        target_hash=digest("target"),
        decoder_hash=digest("decoder"),
        preprocessor_hash=digest("preprocessor"),
        evaluator_identity=digest("evaluator"),
        kernel_hash=digest("kernel"),
        outcome_hash=digest("outcome"),
        draws_hash=sha256_array(latent, kind="posterior-draws"),
        ordered_target_values=np.array([[0.0, 1.0], [1.0, 2.0]]),
        ordered_query_values=np.column_stack([x, inverse]),
        query_inverse=inverse,
    )


def problem(*, t=2000, deterministic_functional=False):
    rng = np.random.default_rng(20260811)
    c, u, d, q = 4, 2, 2, 4
    loc = np.array([[0.4, -0.2], [-1.0, 1.3]])
    scale = np.array([[0.7, 1.2], [1.4, 0.6]])
    latent = rng.normal(size=(t, c, u, d)) * scale[None, None] + loc[None, None]
    inverse = np.array([0, 0, 1, 1], np.int64)
    x = np.array([-1.0, 0.5, -0.25, 1.25])
    if deterministic_functional:
        functional = np.broadcast_to(np.array([1.0, 1.5, -0.8, 0.2]), (t, c, q)).copy()
    else:
        functional = np.empty((t, c, q))
        for j in range(q):
            z = latent[:, :, inverse[j]]
            functional[:, :, j] = 0.6 * x[j] + z[..., 0] + 0.15 * z[..., 1]
    transform = OutcomeTransform(shift=100.0, scale=3.5)
    truth = transform.to_original(functional).mean(axis=(0, 1)) + np.array([0.8, -0.5, 1.0, -0.7])
    log_ratio = np.minimum(0.0, rng.normal(-0.2, 0.3, size=(t, c, u)))
    return {
        "latent_draws": latent,
        "functional_draws_model_units": functional,
        "truth_original_units": truth,
        "accepted": rng.random((t, c, u)) < 0.8,
        "log_accept_ratio": log_ratio,
        "energy_error": -log_ratio,
        "has_nonfinite": np.zeros((t, c, u), bool),
        "divergence": np.zeros((t, c, u), bool),
        "numerical_anomaly": np.zeros((t, c, u), bool),
        "trajectory_length": rng.choice(np.array([3, 5, 7], np.int32), size=(t, c, u)),
        "identity": _identity(latent, x, inverse),
        "outcome_transform": transform,
        "precision_policy": PrecisionPolicy(10.0, 0.0, 1.0),
    }


def _with_latent(inputs, latent):
    inputs["latent_draws"] = latent
    identity = inputs["identity"]
    inputs["identity"] = BatchIdentity(
        **{**identity.__dict__, "draws_hash": sha256_array(latent, kind="posterior-draws")}
    )
    return inputs


def test_reportable_batch_is_scored_in_original_units():
    inputs = problem()
    record = score_batch(**inputs)
    functional_original = inputs["outcome_transform"].to_original(inputs["functional_draws_model_units"])
    direct = np.mean((functional_original.mean(axis=(0, 1)) - inputs["truth_original_units"]) ** 2)
    np.testing.assert_allclose(record["metric"]["plugin_mse"], direct, rtol=0, atol=1e-12)
    assert record["metric"]["original_outcome_units"] is True
    assert record["reportability"] == "REPORTABLE"
    assert record["trace"]["divergence_count"] == 0
    assert 0.7 < record["trace"]["accept_rate_min"] < 0.9
    assert record["identity"]["draws_hash"] == inputs["identity"].draws_hash
    assert record["block_len"] >= math.ceil(math.sqrt(2000))


def test_draws_hash_must_match_the_scored_draws():
    inputs = problem()
    inputs["latent_draws"] = inputs["latent_draws"] + 1e-6
    with pytest.raises(GateError, match="draws_hash"):
        score_batch(**inputs)


def test_any_divergence_or_anomaly_prevents_scoring():
    inputs = problem()
    inputs["divergence"][7, 1, 0] = True
    inputs["numerical_anomaly"][7, 1, 0] = True
    inputs["log_accept_ratio"][7, 1, 0] = -np.inf
    inputs["energy_error"][7, 1, 0] = np.inf
    with pytest.raises(GateError, match="NUMERICAL_RESTART"):
        score_batch(**inputs)

    inputs = problem()
    inputs["log_accept_ratio"][8, 2, 1] = -1500.0
    inputs["energy_error"][8, 2, 1] = 1500.0
    inputs["divergence"][8, 2, 1] = True
    inputs["numerical_anomaly"][8, 2, 1] = True
    with pytest.raises(GateError, match="NUMERICAL_RESTART"):
        score_batch(**inputs)

    inputs = problem()
    inputs["log_accept_ratio"][9, 3, 0] = 2000.0
    inputs["energy_error"][9, 3, 0] = -2000.0
    inputs["numerical_anomaly"][9, 3, 0] = True
    with pytest.raises(GateError, match="symmetric energy"):
        score_batch(**inputs)


def test_malformed_traces_are_rejected():
    inputs = problem()
    inputs["log_accept_ratio"][3, 0, 1] = -np.inf
    inputs["energy_error"][3, 0, 1] = np.inf
    with pytest.raises(GateError, match="requires matching divergence"):
        score_batch(**inputs)

    inputs = problem()
    inputs["has_nonfinite"][11, 2, 1] = True
    with pytest.raises(GateError, match="requires matching divergence"):
        score_batch(**inputs)
    inputs["divergence"][11, 2, 1] = True
    inputs["numerical_anomaly"][11, 2, 1] = True
    with pytest.raises(GateError, match="non-finite proposals"):
        score_batch(**inputs)

    inputs = problem()
    inputs["energy_error"][0, 0, 0] += 0.1
    with pytest.raises(GateError, match="must equal"):
        score_batch(**inputs)


def test_stuck_separated_chains_cannot_report_despite_high_acceptance():
    inputs = problem()
    t, c, u, d = inputs["latent_draws"].shape
    offsets = np.array([-4.0, -1.5, 1.5, 4.0])[None, :, None, None]
    _with_latent(inputs, offsets + 0.03 * np.random.default_rng(88).normal(size=(t, c, u, d)))
    inputs["accepted"][:] = True
    with pytest.raises(ValueError, match="STATIONARITY_RESTART"):
        score_batch(**inputs)


def test_deterministic_functional_is_legal_but_constant_latent_is_not():
    record = score_batch(**problem(deterministic_functional=True))
    assert record["metric"]["deterministic_functional"] is True
    assert record["metric"]["plugin_mcse"] == 0.0
    assert record["metric"]["max_iact"] == 0.0

    inputs = problem(deterministic_functional=True)
    latent = inputs["latent_draws"].copy()
    latent[..., 0] = 0.0
    _with_latent(inputs, latent)
    with pytest.raises(GateError, match="STATIONARITY_RESTART"):
        score_batch(**inputs)


def test_precision_policy_blocks_imprecise_batches():
    inputs = problem(t=240)
    inputs["precision_policy"] = PrecisionPolicy(0.0, 1e-6, 1.0)
    with pytest.raises(ValueError, match="EXTEND_PRECISION"):
        score_batch(**inputs)


@pytest.mark.slow
def test_frozen_hmc_on_a_gaussian_target_reaches_reportable():
    loc = np.array([[0.4, -0.3], [-1.1, 0.8]], np.float32)
    scale = np.array([[0.7, 1.2], [1.3, 0.6]], np.float32)
    context = np.concatenate([loc, np.log(scale)], axis=1).astype(np.float32)
    runner = FrozenVectorizedHMC(
        evaluator=GaussianContextEvaluator(latent_dim=2),
        num_chains=4,
        max_batch_size=4,
        warmup_config=WarmupConfig(warmup_steps=300, adaptation_fraction=0.8, initial_step_size=0.1, num_leapfrog_steps=5, target_accept_prob=0.8),
        production_config=ProductionConfig(segment_size=2000, trajectory_policy=TrajectoryPolicy((3, 5, 7)), max_energy_diff=1000.0),
    )
    epoch = runner.bind_epoch(FrozenBatch((17, 83), context, batch_index=0), run_seed=20260811, run_key=digest("gaussian-gate"), production_epoch=0)
    offsets = np.array([-3.0, -1.0, 1.0, 3.0], np.float32)[:, None, None]
    variance = np.square(scale).astype(np.float32)
    warm = runner.warmup(epoch, initial_state=(loc[None] + offsets * scale[None]).astype(np.float32), state_variance=variance)
    segment = runner.run_segment(epoch, segment_index=0, pre_state=warm.final_state, step_size=warm.step_size, state_variance=variance)
    assert not segment.has_numerical_anomaly.any()
    np.testing.assert_allclose(segment.draws.mean(axis=(0, 1)), loc, atol=0.06, rtol=0)

    query_inverse = np.array([0, 0, 1, 1], np.int64)
    treatment = np.array([-1.0, 0.75, -0.5, 1.25], np.float32)
    functional = np.empty((segment.draws.shape[0], 4, 4), np.float32)
    for j, position in enumerate(query_inverse):
        z = segment.draws[:, :, position]
        functional[:, :, j] = 0.4 * treatment[j] + z[..., 0] + 0.15 * z[..., 1]
    transform = OutcomeTransform(shift=100.0, scale=3.5)
    analytic = np.array([0.4 * treatment[j] + loc[query_inverse[j], 0] + 0.15 * loc[query_inverse[j], 1] for j in range(4)])
    truth = transform.to_original(analytic) + np.array([0.8, -0.5, 1.0, -0.7])
    identity = BatchIdentity(
        target_hash=digest("gaussian-target"), decoder_hash=digest("gaussian-evaluator"),
        preprocessor_hash=digest("identity-preprocessor"), evaluator_identity=runner.evaluator.evaluator_identity,
        kernel_hash=runner.kernel_identity, outcome_hash=digest("linear-outcome"),
        draws_hash=sha256_array(segment.draws, kind="posterior-draws"),
        ordered_target_values=context, ordered_query_values=np.column_stack([treatment, query_inverse]),
        query_inverse=query_inverse,
    )
    record = score_batch(
        latent_draws=segment.draws, functional_draws_model_units=functional, truth_original_units=truth,
        accepted=segment.is_accepted, log_accept_ratio=segment.log_accept_ratio, energy_error=segment.energy_error,
        has_nonfinite=segment.has_nonfinite, divergence=segment.has_divergence,
        numerical_anomaly=segment.has_numerical_anomaly, trajectory_length=segment.trajectory_length,
        identity=identity, outcome_transform=transform,
        precision_policy=PrecisionPolicy(absolute_halfwidth=0.2, relative_halfwidth=0.0, reference_scale=1.0),
    )
    assert record["reportability"] == "REPORTABLE"
    assert record["sampler"]["num_chains"] == 4
    assert record["sampler"]["bulk_ess_min"] >= 400 and record["sampler"]["tail_ess_min"] >= 400
