"""Whether a batch of draws may be reported, and with which structural metric.

Chain diagnostics (rank and folded split R-hat, bulk and tail ESS) and the
structural Monte Carlo metric with its block-jackknife MCSE feed a
fail-closed gate: a batch is scored only when every transition is finite and
non-divergent, every latent coordinate and the structural functional pass
R-hat <= 1.01 and ESS >= 100 per chain, and the MCSE meets the precision
policy.  Any other outcome is a named state, never a number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy.stats import norm, rankdata, t as student_t

from .target import require_digest, sha256_array


# --- chain diagnostics -------------------------------------------------------


def _split_chains(draws: np.ndarray) -> np.ndarray:
    """Split [T,C,...] into [floor(T/2),2C,...], dropping the middle draw."""

    if draws.ndim < 2 or draws.shape[0] < 4 or draws.shape[1] < 2:
        raise ValueError("draws must have shape [T>=4,C>=2,...]")
    n = draws.shape[0] // 2
    return np.concatenate([draws[:n], draws[-n:]], axis=1)


def _rank_normalize(draws: np.ndarray) -> np.ndarray:
    """Blom rank-normalize each trailing scalar independently."""

    n, m = draws.shape[:2]
    event_shape = draws.shape[2:]
    flat = draws.reshape(n * m, -1)
    ranks = rankdata(flat, axis=0, method="average", nan_policy="propagate")
    z = norm.ppf((ranks - 3.0 / 8.0) / (n * m + 1.0 / 4.0))
    return z.reshape((n, m) + event_shape)


def _basic_rhat(draws: np.ndarray, atol: float = 1e-15) -> np.ndarray:
    """Conventional square-root R-hat for already transformed [N,M,...] draws."""

    n, m = draws.shape[:2]
    if n < 2 or m < 2:
        raise ValueError("R-hat requires N>=2 and M>=2")
    chain_mean = np.mean(draws, axis=0)
    chain_var = np.var(draws, axis=0, ddof=1)
    within = np.mean(chain_var, axis=0)
    between = n * np.var(chain_mean, axis=0, ddof=1)
    scale = np.maximum(1.0, np.max(np.abs(draws), axis=(0, 1)))
    zero_within = within <= atol * scale**2
    zero_between = between <= atol * scale**2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt((between / within + n - 1.0) / n)
    out = np.where(zero_within & ~zero_between, np.inf, out)
    out = np.where(zero_within & zero_between, np.nan, out)
    return out


def _autocovariance_biased(x: np.ndarray) -> np.ndarray:
    """Biased FFT autocovariance for x[N,M], matching posterior/Stan ESS."""

    n = x.shape[0]
    centered = x - np.mean(x, axis=0, keepdims=True)
    transformed = np.fft.rfft(centered, n=2 * n, axis=0)
    return np.fft.irfft(
        transformed * np.conjugate(transformed), n=2 * n, axis=0
    )[:n].real / n


def _ess_scalar(draws: np.ndarray) -> float:
    """Geyer IPS/IMS multi-chain ESS for a scalar matrix [N,M]."""

    x = np.asarray(draws, dtype=np.float64)
    n, m = x.shape
    if n < 3 or m < 2 or not np.all(np.isfinite(x)):
        return float("nan")
    if np.max(x) - np.min(x) <= 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(x))):
        return float("nan")

    acov = _autocovariance_biased(x)
    acov_mean = np.mean(acov, axis=1)
    mean_var = acov_mean[0] * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n + np.var(np.mean(x, axis=0), ddof=1)
    if not np.isfinite(var_plus) or var_plus <= 0:
        return float("nan")

    rho = np.zeros(n, dtype=np.float64)
    t = 0
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_var - acov_mean[1]) / var_plus
    rho[1] = rho_odd

    # Geyer's initial positive sequence.
    while t < n - 5 and np.isfinite(rho_even + rho_odd) and rho_even + rho_odd > 0:
        t += 2
        rho_even = 1.0 - (mean_var - acov_mean[t]) / var_plus
        rho_odd = 1.0 - (mean_var - acov_mean[t + 1]) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t] = rho_even
            rho[t + 1] = rho_odd
    max_t = t
    if rho_even > 0:
        rho[max_t] = rho_even

    # Geyer's initial monotone sequence.
    t = 0
    while t <= max_t - 4:
        t += 2
        previous_pair = rho[t - 2] + rho[t - 1]
        current_pair = rho[t] + rho[t + 1]
        if current_pair > previous_pair:
            rho[t] = previous_pair / 2.0
            rho[t + 1] = previous_pair / 2.0

    total = m * n
    # Match posterior:::.ess exactly at the short-chain boundary.  The R
    # implementation evaluates ``rho_hat_t[1:0]`` when max_t == 0; R drops
    # index zero but retains index one, so the lag-zero term is included once
    # in the prefix and once as the endpoint.  A direct Python ``rho[:0]``
    # would instead be empty and spuriously hit the antithetic ESS cap.
    rho_prefix = rho[0] if max_t == 0 else np.sum(rho[:max_t])
    tau_hat = -1.0 + 2.0 * rho_prefix + rho[max_t]
    tau_hat = max(tau_hat, 1.0 / np.log10(total))
    return float(total / tau_hat)


def _ess_event(draws: np.ndarray) -> np.ndarray:
    event_shape = draws.shape[2:]
    flat = draws.reshape(draws.shape[0], draws.shape[1], -1)
    values = np.array([_ess_scalar(flat[:, :, j]) for j in range(flat.shape[2])])
    return values.reshape(event_shape)


def chain_diagnostics(draws: np.ndarray) -> Dict[str, np.ndarray]:
    """Rank/folded split-Rhat plus bulk/tail ESS for [T,C,...] draws."""

    x = np.asarray(draws, dtype=np.float64)
    if x.ndim < 3:
        raise ValueError("draws must have shape [T,C,...]")
    if x.shape[0] < 12:
        raise ValueError("split diagnostics require at least 12 draws per chain")
    event_shape = x.shape[2:]
    finite = np.all(np.isfinite(x), axis=(0, 1))
    span = np.max(x, axis=(0, 1)) - np.min(x, axis=(0, 1))
    magnitude = np.maximum(1.0, np.max(np.abs(x), axis=(0, 1)))
    constant = finite & (span <= 64 * np.finfo(float).eps * magnitude)
    invalid = ~finite

    split = _split_chains(x)
    rank_split = _rank_normalize(split)
    rhat_rank = _basic_rhat(rank_split)

    median = np.median(x, axis=(0, 1), keepdims=True)
    folded_split = _split_chains(np.abs(x - median))
    folded_span = np.max(folded_split, axis=(0, 1)) - np.min(
        folded_split, axis=(0, 1)
    )
    folded_degenerate = finite & ~constant & (
        folded_span <= 64 * np.finfo(float).eps * magnitude
    )
    folded_rank = _rank_normalize(folded_split)
    rhat_folded = _basic_rhat(folded_rank)
    rhat_folded = np.where(folded_degenerate, np.nan, rhat_folded)
    rhat = np.fmax(rhat_rank, rhat_folded)

    ess_bulk = _ess_event(rank_split)
    q05 = np.quantile(x, 0.05, axis=(0, 1), keepdims=True)
    q95 = np.quantile(x, 0.95, axis=(0, 1), keepdims=True)
    low = _split_chains((x <= q05).astype(np.float64))
    high = _split_chains((x <= q95).astype(np.float64))
    ess_low = _ess_event(low)
    ess_high = _ess_event(high)
    ess_tail = np.fmin(ess_low, ess_high)
    tail_degenerate = ~np.isfinite(ess_tail)

    for array in (rhat_rank, rhat_folded, rhat, ess_bulk, ess_tail):
        array[invalid | constant] = np.nan

    return {
        "rhat_rank": rhat_rank.reshape(event_shape),
        "rhat_folded": rhat_folded.reshape(event_shape),
        "rhat": rhat.reshape(event_shape),
        "ess_bulk": ess_bulk.reshape(event_shape),
        "ess_tail": ess_tail.reshape(event_shape),
        "constant_mask": constant.reshape(event_shape),
        "invalid_mask": invalid.reshape(event_shape),
        "folded_degenerate_mask": folded_degenerate.reshape(event_shape),
        "tail_degenerate_mask": tail_degenerate.reshape(event_shape),
    }


def structural_metric(
    mu_draws: np.ndarray,
    truth: np.ndarray,
    batch_len: int,
) -> Dict[str, Any]:
    """Finite/U structural MSE and joint nonlinear block-jackknife MCSE.

    The caller must additionally verify that ``batch_len`` exceeds the sampled
    functional's dependence range (a conservative rule is at least five times
    its estimated IACT).  The purely algebraic checks here cannot establish
    that condition from one chosen batching alone.
    """

    mu = np.asarray(mu_draws, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    if mu.ndim != 3:
        raise ValueError("mu_draws must have shape [T,C,Q]")
    t, c, q = mu.shape
    if y.shape != (q,):
        raise ValueError("truth must have shape [Q]")
    if c < 4:
        raise ValueError("formal metrics require at least four chains")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite metric inputs")
    b = int(batch_len)
    if b <= 0 or t % (2 * b):
        raise ValueError("require T divisible by 2 * batch_len")
    if b < int(np.ceil(np.sqrt(t))):
        raise ValueError("batch_len must be at least ceil(sqrt(T))")
    if t // (2 * b) < 5:
        raise ValueError("require at least five 2b blocks per chain")

    chain_mean = np.mean(mu, axis=0)  # [C,Q]
    # Work in residual space.  Expanding y**2 - 2*y*mu + mu**2 loses all
    # useful digits when the outcome has a large affine offset.
    chain_residual = np.mean(mu - y[None, None, :], axis=0)
    mean_residual = np.mean(chain_residual, axis=0)
    mean = y + mean_residual
    mse_plugin = np.mean(mean_residual**2)
    # Algebraically identical form based on the sample variance of chain means.
    penalty_identity = np.mean(np.var(chain_residual, axis=0, ddof=1) / c)
    # This is exactly the cross-chain U correction in real arithmetic.  Keep
    # the correction itself as the authoritative MC-bias estimate: recovering
    # it later as mse_plugin - mse_u catastrophically cancels when the structural
    # residual is huge and the integration variance is small.
    mc_bias_hat = penalty_identity
    mse_u = mse_plugin - penalty_identity

    def block_estimates(block):
        n_batch_per_chain = t // block
        blocks = (
            (mu - y[None, None, :]).reshape(n_batch_per_chain, block, c, q)
            .mean(axis=1)
            .transpose(1, 0, 2)
            .reshape(c * n_batch_per_chain, q)
        )
        k = blocks.shape[0]
        grand_residual = np.mean(blocks, axis=0)
        gradient = 2 * grand_residual / q
        projected = (blocks - grand_residual[None, :]) @ gradient
        se_delta = np.sqrt(np.var(projected, ddof=1) / k)

        leave_residual = (k * grand_residual[None, :] - blocks) / (k - 1)
        leave_metric = np.mean(leave_residual**2, axis=1)
        se_jack = np.sqrt(
            (k - 1)
            / k
            * np.sum((leave_metric - np.mean(leave_metric)) ** 2)
        )
        return float(se_delta), float(se_jack), int(k)

    delta_b, jack_b, k_b = block_estimates(b)
    delta_2b, jack_2b, k_2b = block_estimates(2 * b)
    leave_chain_residual = (
        c * mean_residual[None, :] - chain_residual
    ) / (c - 1)
    leave_chain_metric = np.mean(leave_chain_residual**2, axis=1)
    se_chain = np.sqrt(
        (c - 1)
        / c
        * np.sum((leave_chain_metric - np.mean(leave_chain_metric)) ** 2)
    )
    halfwidth_batch = 1.96 * jack_2b
    halfwidth_chain = student_t.ppf(0.975, c - 1) * se_chain
    mu_span = np.max(mu, axis=(0, 1)) - np.min(mu, axis=(0, 1))
    mu_magnitude = np.maximum(1.0, np.max(np.abs(mu), axis=(0, 1)))
    deterministic_functional = bool(
        np.all(mu_span <= 64 * np.finfo(float).eps * mu_magnitude)
    )
    if deterministic_functional:
        delta_b = jack_b = delta_2b = jack_2b = se_chain = 0.0
        halfwidth_batch = halfwidth_chain = 0.0
        stability_ratio = 1.0
    else:
        stability_ratio = jack_b / jack_2b if jack_2b > 0 else float("nan")
    metric_reasons = []
    if not np.isfinite(stability_ratio) or not 0.5 <= stability_ratio <= 2.0:
        metric_reasons.append("block-jackknife MCSE is unstable between b and 2b")

    return {
        "mean": mean,
        "chain_means": chain_mean,
        "mse_plugin": float(mse_plugin),
        "mse_u": float(mse_u),
        "mc_bias_hat": float(mc_bias_hat),
        "penalty_identity": float(penalty_identity),
        "mcse_delta_b": delta_b,
        "mcse_jack_b": jack_b,
        "mcse_delta_2b": delta_2b,
        "mcse_jack_2b": jack_2b,
        "mcse_chain_jack": float(se_chain),
        "halfwidth95": float(max(halfwidth_batch, halfwidth_chain)),
        "batch_stability_ratio": float(stability_ratio),
        "num_batches_b": k_b,
        "num_batches_2b": k_2b,
        "metric_valid": not metric_reasons,
        "metric_reasons": tuple(metric_reasons),
        "mcse_estimand": "mse_plugin_first_order",
        "deterministic_functional": deterministic_functional,
        "block_len_requires_external_iact_check": True,
    }



# --- reportability gate ------------------------------------------------------


class GateError(ValueError):
    """A malformed or non-reportable production snapshot."""


class Action(str, Enum):
    CONFIG_INVALID = "CONFIG_INVALID"
    NUMERICAL_RESTART = "NUMERICAL_RESTART"
    MASS_UPGRADE_REQUIRED = "MASS_UPGRADE_REQUIRED"
    STATIONARITY_RESTART = "STATIONARITY_RESTART"
    EXTEND_PRECISION = "EXTEND_PRECISION"
    METRIC_INVALID = "METRIC_INVALID"
    REPORTABLE = "REPORTABLE"


def _require_real(name: str, value: Any, *, finite: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    result = float(value)
    if finite and not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SamplerEvidence:
    """Chain-level evidence the gate decides on (one production segment)."""

    num_chains: int
    draws_per_chain: int
    rank_rhat_max: float
    folded_rhat_max: float
    bulk_ess_min: float
    tail_ess_min: float
    nonfinite_count: int
    latent_constant_count: int
    independent_initialization_provenance: bool
    fixed_production_kernel: bool
    target_gradient_finite: bool
    draws_hash: str
    geometry_action: str = "KEEP_IDENTITY"
    diagnostics_version: str = "rank-folded-bulk-tail"

    def validate_types(self) -> None:
        for name in ("num_chains", "draws_per_chain", "nonfinite_count", "latent_constant_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.num_chains <= 0 or self.draws_per_chain <= 0:
            raise ValueError("chain and draw counts must be positive")
        if self.draws_per_chain < 12:
            raise ValueError("split diagnostics require T >= 12")
        if self.nonfinite_count < 0 or self.latent_constant_count < 0:
            raise ValueError("failure counts cannot be negative")
        for name in ("rank_rhat_max", "folded_rhat_max", "bulk_ess_min", "tail_ess_min"):
            _require_real(name, getattr(self, name))
        for name in (
            "independent_initialization_provenance",
            "fixed_production_kernel",
            "target_gradient_finite",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        require_digest("draws_hash", self.draws_hash)
        split_n = self.draws_per_chain // 2
        rhat_floor = math.sqrt((split_n - 1.0) / split_n)
        if min(self.rank_rhat_max, self.folded_rhat_max) < rhat_floor - 1e-12:
            raise ValueError("Rhat is below its split-chain algebraic lower bound")
        split_total = 2 * self.num_chains * split_n
        ess_cap = split_total * math.log10(split_total)
        if max(self.bulk_ess_min, self.tail_ess_min) > ess_cap * (1.0 + 1e-9):
            raise ValueError("ESS exceeds the finite-draw antithetic cap")
        if self.geometry_action not in {"KEEP_IDENTITY", "MASS_UPGRADE", "RESTART"}:
            raise ValueError("unknown geometry_action")
        if self.diagnostics_version != "rank-folded-bulk-tail":
            raise ValueError("unsupported diagnostics implementation")


@dataclass(frozen=True)
class Assessment:
    action: Action
    reasons: Tuple[str, ...]


def assess_sampler(evidence: SamplerEvidence) -> Assessment:
    """Return the only legal next action for a production segment.

    Gates, in order: at least four independently initialized chains under a
    fixed kernel; no non-finite state / target / gradient; geometry verdict;
    no constant latent coordinate; rank and folded split R-hat <= 1.01;
    bulk and tail ESS >= 100 per chain.
    """

    try:
        evidence.validate_types()
    except (TypeError, ValueError) as exc:
        return Assessment(Action.CONFIG_INVALID, (str(exc),))
    if evidence.num_chains < 4:
        return Assessment(Action.CONFIG_INVALID, ("at least four chains are required",))
    if not evidence.independent_initialization_provenance:
        return Assessment(
            Action.CONFIG_INVALID,
            ("fresh independent production initialization is not established",),
        )
    if not evidence.fixed_production_kernel:
        return Assessment(
            Action.CONFIG_INVALID,
            ("adaptation wrapper or mutable kernel remained in production",),
        )
    if evidence.nonfinite_count or not evidence.target_gradient_finite:
        return Assessment(
            Action.NUMERICAL_RESTART,
            ("non-finite state/target/gradient/trace invalidates the segment",),
        )
    if evidence.geometry_action == "MASS_UPGRADE":
        return Assessment(
            Action.MASS_UPGRADE_REQUIRED,
            ("within-mode anisotropy requests a fresh mass estimate",),
        )
    if evidence.geometry_action == "RESTART":
        return Assessment(
            Action.STATIONARITY_RESTART,
            ("pilot classified the segment as restart/reparameterize",),
        )
    if evidence.latent_constant_count:
        return Assessment(
            Action.STATIONARITY_RESTART,
            ("constant latent coordinates are not deterministic functionals",),
        )
    if max(evidence.rank_rhat_max, evidence.folded_rhat_max) > 1.01:
        return Assessment(
            Action.STATIONARITY_RESTART,
            ("rank/folded Rhat exceeds 1.01; extension cannot cure stationarity",),
        )
    required_ess = 100.0 * evidence.num_chains
    if min(evidence.bulk_ess_min, evidence.tail_ess_min) < required_ess:
        return Assessment(
            Action.EXTEND_PRECISION,
            (f"bulk/tail ESS is below {required_ess:g}; extend the same fixed kernel",),
        )
    return Assessment(Action.REPORTABLE, ())


@dataclass(frozen=True)
class StructuralMetric:
    """Structural MSE evidence with its Monte Carlo precision contract."""

    plugin_mse: float
    u_corrected_mse: float
    integration_penalty: float
    plugin_mcse: float
    plugin_halfwidth95: float
    time_draws: int
    block_len: int
    max_iact: float
    block_stability_ratio: float
    absolute_halfwidth_tolerance: float
    relative_halfwidth_tolerance: float
    precision_reference_scale: float
    original_outcome_units: bool
    joint_query_mcse: bool
    deterministic_functional: bool
    iact_num_chains: int
    metric_version: str = "joint-block-jackknife-u"

    def validate(self, *, identity_tolerance: float = 1e-9) -> None:
        for name in ("time_draws", "block_len", "iact_num_chains"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TypeError(f"{name} must be a positive integer")
        for name in (
            "plugin_mse",
            "u_corrected_mse",
            "integration_penalty",
            "plugin_mcse",
            "plugin_halfwidth95",
            "max_iact",
            "block_stability_ratio",
            "absolute_halfwidth_tolerance",
            "relative_halfwidth_tolerance",
            "precision_reference_scale",
        ):
            _require_real(name, getattr(self, name))
        if self.plugin_mse < 0 or self.integration_penalty < 0:
            raise ValueError("plugin MSE and integration penalty must be nonnegative")
        if self.plugin_mcse < 0 or self.plugin_halfwidth95 < 0:
            raise ValueError("MC uncertainty summaries must be nonnegative")
        if self.max_iact < 0:
            raise ValueError("max_iact cannot be negative")
        if self.absolute_halfwidth_tolerance < 0 or self.relative_halfwidth_tolerance < 0:
            raise ValueError("precision tolerances cannot be negative")
        if self.precision_reference_scale <= 0:
            raise ValueError("precision_reference_scale must be positive")
        for name in ("original_outcome_units", "joint_query_mcse", "deterministic_functional"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        # U may legitimately be negative; the stable identity is the contract.
        scale = max(1.0, abs(self.plugin_mse), abs(self.u_corrected_mse))
        mismatch = abs(self.plugin_mse - self.u_corrected_mse - self.integration_penalty)
        if mismatch > identity_tolerance * scale:
            raise ValueError("plugin - U must equal the stable integration penalty")
        if self.plugin_halfwidth95 + identity_tolerance < 1.96 * self.plugin_mcse:
            raise ValueError("95% halfwidth is inconsistent with plugin MCSE")
        if not self.original_outcome_units:
            raise ValueError("scientific metrics must be in original outcome units")
        if not self.joint_query_mcse:
            raise ValueError("shared-target queries require joint MCSE propagation")
        if self.deterministic_functional:
            if any(value != 0.0 for value in (self.plugin_mcse, self.plugin_halfwidth95, self.max_iact)):
                raise ValueError("deterministic functional must have zero MC uncertainty")
            if abs(self.block_stability_ratio - 1.0) > identity_tolerance:
                raise ValueError("deterministic functional must use stability ratio one")
        elif self.max_iact <= 0.0:
            raise ValueError("non-deterministic functional requires a positive max_iact")
        if self.metric_version != "joint-block-jackknife-u":
            raise ValueError("unsupported metric implementation")

    def precision_reasons(self) -> Tuple[str, ...]:
        """Finite-MC failures that a fixed-kernel extension may cure."""

        reasons = []
        if self.time_draws < 12:
            reasons.append("metric_time_draws_below_12")
        if self.time_draws % (2 * self.block_len):
            reasons.append("time_not_divisible_by_2b")
        elif self.time_draws // (2 * self.block_len) < 5:
            reasons.append("fewer_than_five_2b_blocks_per_chain")
        if self.block_len < math.ceil(math.sqrt(self.time_draws)):
            reasons.append("block_shorter_than_sqrt_T")
        if not self.deterministic_functional and self.block_len < math.ceil(5.0 * self.max_iact):
            reasons.append("block_shorter_than_five_IACT")
        if not self.deterministic_functional and not 0.5 <= self.block_stability_ratio <= 2.0:
            reasons.append("b_vs_2b_mcse_unstable")
        allowed = self.absolute_halfwidth_tolerance + (
            self.relative_halfwidth_tolerance * self.precision_reference_scale
        )
        if self.plugin_halfwidth95 > allowed:
            reasons.append("joint_structural_metric_halfwidth_too_large")
        return tuple(reasons)


def structural_metric_evidence(
    output: Dict[str, Any],
    *,
    time_draws: int,
    block_len: int,
    max_iact: float,
    absolute_halfwidth_tolerance: float,
    relative_halfwidth_tolerance: float,
    precision_reference_scale: float,
    original_outcome_units: bool,
    iact_num_chains: int,
) -> StructuralMetric:
    """Validated metric evidence from a ``structural_metric`` output."""

    required = {
        "mse_plugin",
        "mse_u",
        "penalty_identity",
        "mcse_jack_2b",
        "mcse_chain_jack",
        "halfwidth95",
        "metric_valid",
        "batch_stability_ratio",
        "deterministic_functional",
        "mcse_estimand",
        "block_len_requires_external_iact_check",
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"metric output is missing {sorted(missing)}")
    if output["mcse_estimand"] != "mse_plugin_first_order":
        raise ValueError("MCSE is not for the plug-in structural metric")
    if output["block_len_requires_external_iact_check"] is not True:
        raise ValueError("metric output did not declare its external IACT obligation")
    if output["metric_valid"] is not True:
        reasons = output.get("metric_reasons", ())
        raise ValueError(f"metric failed its internal gate: {reasons}")
    deterministic = output["deterministic_functional"]
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic_functional flag must be boolean")
    jack_2b = _require_real("mcse_jack_2b", output["mcse_jack_2b"])
    chain_jack = _require_real("mcse_chain_jack", output["mcse_chain_jack"])
    metric = StructuralMetric(
        plugin_mse=_require_real("mse_plugin", output["mse_plugin"]),
        u_corrected_mse=_require_real("mse_u", output["mse_u"]),
        integration_penalty=_require_real("penalty_identity", output["penalty_identity"]),
        plugin_mcse=max(jack_2b, chain_jack),
        plugin_halfwidth95=_require_real("halfwidth95", output["halfwidth95"]),
        time_draws=time_draws,
        block_len=block_len,
        max_iact=_require_real("max_iact", max_iact),
        block_stability_ratio=_require_real("batch_stability_ratio", output["batch_stability_ratio"]),
        absolute_halfwidth_tolerance=_require_real("absolute_halfwidth_tolerance", absolute_halfwidth_tolerance),
        relative_halfwidth_tolerance=_require_real("relative_halfwidth_tolerance", relative_halfwidth_tolerance),
        precision_reference_scale=_require_real("precision_reference_scale", precision_reference_scale),
        original_outcome_units=original_outcome_units,
        joint_query_mcse=True,
        deterministic_functional=deterministic,
        iact_num_chains=iact_num_chains,
    )
    metric.validate()
    return metric


def build_batch_record(sampler: SamplerEvidence, metric: StructuralMetric) -> Dict[str, Any]:
    """Assemble a reportable record only after every gate passes.

    Raises ``ValueError("scientific score forbidden: <STATE>: ...")`` so the
    caller can classify the failure.
    """

    assessment = assess_sampler(sampler)
    if assessment.action is not Action.REPORTABLE:
        raise ValueError(
            f"scientific score forbidden: {assessment.action.value}: "
            + "; ".join(assessment.reasons)
        )
    try:
        metric.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scientific score forbidden: {Action.METRIC_INVALID.value}: {exc}")
    if metric.time_draws != sampler.draws_per_chain:
        raise ValueError("scientific score forbidden: metric T differs from sampler T")
    if metric.iact_num_chains != sampler.num_chains:
        raise ValueError("scientific score forbidden: IACT chain count differs from sampler")
    precision_reasons = metric.precision_reasons()
    if precision_reasons:
        raise ValueError(
            f"scientific score forbidden: {Action.EXTEND_PRECISION.value}: "
            + "; ".join(precision_reasons)
        )
    return {
        "sampler": asdict(sampler),
        "metric": asdict(metric),
        "reportability": Action.REPORTABLE.value,
    }


# --- one batch, one verdict --------------------------------------------------


@dataclass(frozen=True)
class BatchIdentity:
    """What a batch score is conditional on: target, kernel, readout, draws."""

    target_hash: str
    decoder_hash: str
    preprocessor_hash: str
    evaluator_identity: str
    kernel_hash: str
    outcome_hash: str
    draws_hash: str
    ordered_target_values: np.ndarray
    ordered_query_values: np.ndarray
    query_inverse: np.ndarray

    def validate(self, *, num_targets: int, num_queries: int) -> None:
        for name in (
            "target_hash",
            "decoder_hash",
            "preprocessor_hash",
            "evaluator_identity",
            "kernel_hash",
            "outcome_hash",
            "draws_hash",
        ):
            require_digest(name, getattr(self, name))
        targets = np.asarray(self.ordered_target_values)
        queries = np.asarray(self.ordered_query_values)
        inverse = np.asarray(self.query_inverse)
        if targets.ndim != 2 or targets.shape[0] != num_targets:
            raise GateError("ordered_target_values must be [U,V]")
        if queries.ndim != 2 or queries.shape[0] != num_queries:
            raise GateError("ordered_query_values must be [Q,K]")
        if inverse.dtype.kind not in "iu" or inverse.shape != (num_queries,):
            raise GateError("query_inverse must be integer [Q]")
        if np.any(inverse < 0) or np.any(inverse >= num_targets):
            raise GateError("query_inverse points outside ordered targets")
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(queries)):
            raise GateError("target/query payloads must be finite")

    def payload(self) -> Dict[str, Any]:
        return {
            "target_hash": self.target_hash,
            "decoder_hash": self.decoder_hash,
            "preprocessor_hash": self.preprocessor_hash,
            "evaluator_identity": self.evaluator_identity,
            "kernel_hash": self.kernel_hash,
            "outcome_hash": self.outcome_hash,
            "draws_hash": self.draws_hash,
            "ordered_target_hash": sha256_array(self.ordered_target_values, kind="ordered-targets"),
            "ordered_query_hash": sha256_array(self.ordered_query_values, kind="ordered-queries"),
            "query_inverse_hash": sha256_array(np.asarray(self.query_inverse, np.int64), kind="query-inverse"),
        }


@dataclass(frozen=True)
class OutcomeTransform:
    """Model outcome units -> original units: ``shift + scale * value``."""

    shift: float
    scale: float

    def validate(self) -> None:
        if isinstance(self.shift, bool) or not np.isfinite(self.shift):
            raise GateError("outcome shift must be finite")
        if isinstance(self.scale, bool) or not np.isfinite(self.scale) or self.scale <= 0:
            raise GateError("outcome scale must be finite and positive")

    def to_original(self, values: Any) -> np.ndarray:
        self.validate()
        return float(self.shift) + float(self.scale) * np.asarray(values, np.float64)


@dataclass(frozen=True)
class PrecisionPolicy:
    """Allowed 95% halfwidth: ``absolute + relative * reference_scale``."""

    absolute_halfwidth: float
    relative_halfwidth: float
    reference_scale: float

    def validate(self) -> None:
        for name in ("absolute_halfwidth", "relative_halfwidth", "reference_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value):
                raise GateError(f"{name} must be finite")
        if self.absolute_halfwidth < 0 or self.relative_halfwidth < 0:
            raise GateError("precision tolerances cannot be negative")
        if self.reference_scale <= 0:
            raise GateError("reference_scale must be positive")


def _active_diagnostic_summary(
    latent: Mapping[str, np.ndarray],
    functional: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    latent_constant = np.asarray(latent["constant_mask"], bool)
    latent_invalid = np.asarray(latent["invalid_mask"], bool)
    functional_constant = np.asarray(functional["constant_mask"], bool)
    functional_invalid = np.asarray(functional["invalid_mask"], bool)

    if np.any(latent_invalid) or np.any(functional_invalid):
        return {
            "valid": False,
            "latent_constant_count": int(np.sum(latent_constant)),
            "functional_constant": bool(np.all(functional_constant)),
        }

    # Constant derived functionals are legal only after the latent variables
    # pass; they are omitted from the extrema rather than given an invented ESS.
    arrays: Dict[str, list] = {
        "rank": [np.asarray(latent["rhat_rank"], np.float64).ravel()],
        "folded": [np.asarray(latent["rhat_folded"], np.float64).ravel()],
        "bulk": [np.asarray(latent["ess_bulk"], np.float64).ravel()],
        "tail": [np.asarray(latent["ess_tail"], np.float64).ravel()],
    }
    active_f = ~functional_constant
    if np.any(active_f):
        arrays["rank"].append(np.asarray(functional["rhat_rank"])[active_f].ravel())
        arrays["folded"].append(np.asarray(functional["rhat_folded"])[active_f].ravel())
        arrays["bulk"].append(np.asarray(functional["ess_bulk"])[active_f].ravel())
        arrays["tail"].append(np.asarray(functional["ess_tail"])[active_f].ravel())

    merged = {name: np.concatenate(parts) for name, parts in arrays.items()}
    valid = not np.any(latent_constant) and all(
        values.size and np.all(np.isfinite(values)) for values in merged.values()
    )
    return {
        "valid": bool(valid),
        "latent_constant_count": int(np.sum(latent_constant)),
        "functional_constant": bool(np.all(functional_constant)),
        "rank_rhat_max": float(np.max(merged["rank"])) if valid else float("nan"),
        "folded_rhat_max": float(np.max(merged["folded"])) if valid else float("nan"),
        "bulk_ess_min": float(np.min(merged["bulk"])) if valid else float("nan"),
        "tail_ess_min": float(np.min(merged["tail"])) if valid else float("nan"),
    }


def functional_iact(functional_diag: Mapping[str, np.ndarray], t: int, c: int) -> float:
    """Largest integrated autocorrelation time implied by the bulk ESS."""

    constant = np.asarray(functional_diag["constant_mask"], bool)
    if np.all(constant):
        return 0.0
    bulk = np.asarray(functional_diag["ess_bulk"], np.float64)[~constant]
    if bulk.size == 0 or not np.all(np.isfinite(bulk)) or np.any(bulk <= 0):
        raise GateError("functional IACT cannot be estimated from invalid bulk ESS")
    return float(max(1.0, (t * c) / np.min(bulk)))


def choose_block_len(t: int, max_iact: float) -> int:
    """Smallest legal block length: >= sqrt(T) and 5 IACT, dividing T/2,
    leaving at least five 2b blocks per chain."""

    lower = max(int(math.ceil(math.sqrt(t))), int(math.ceil(5.0 * max_iact)))
    candidates = [
        b
        for b in range(lower, t // 10 + 1)
        if t % (2 * b) == 0 and t // (2 * b) >= 5
    ]
    if not candidates:
        raise GateError(
            "EXTEND_PRECISION: no block length satisfies sqrt(T), 5*IACT, "
            "divisibility and five-2b-block requirements"
        )
    return candidates[0]


def _validate_trace(
    accepted: Any,
    log_accept_ratio: Any,
    energy_error: Any,
    has_nonfinite: Any,
    divergence: Any,
    numerical_anomaly: Any,
    trajectory_length: Any,
    *,
    shape: Tuple[int, int, int],
) -> Dict[str, Any]:
    accepted_array = np.asarray(accepted)
    log_ratio = np.asarray(log_accept_ratio)
    energy = np.asarray(energy_error)
    nonfinite = np.asarray(has_nonfinite)
    diverged = np.asarray(divergence)
    anomaly = np.asarray(numerical_anomaly)
    trajectory = np.asarray(trajectory_length)
    if accepted_array.dtype != np.dtype(bool) or accepted_array.shape != shape:
        raise GateError("accepted must be bool [T,C,U]")
    if diverged.dtype != np.dtype(bool) or diverged.shape != shape:
        raise GateError("divergence must be bool [T,C,U]")
    if nonfinite.dtype != np.dtype(bool) or nonfinite.shape != shape:
        raise GateError("has_nonfinite must be bool [T,C,U]")
    if anomaly.dtype != np.dtype(bool) or anomaly.shape != shape:
        raise GateError("numerical_anomaly must be bool [T,C,U]")
    if log_ratio.shape != shape or log_ratio.dtype.kind not in "fc":
        raise GateError("log_accept_ratio must be floating [T,C,U]")
    if energy.shape != shape or energy.dtype.kind not in "fc":
        raise GateError("energy_error must be floating [T,C,U]")
    if trajectory.shape != shape or trajectory.dtype.kind not in "iu":
        raise GateError("trajectory_length must be integer [T,C,U]")
    if np.any(trajectory <= 0):
        raise GateError("trajectory_length must be positive")
    if np.any(np.isnan(log_ratio)) or np.any(np.isposinf(log_ratio)):
        raise GateError("NaN/+Inf log_accept_ratio is never legal")
    if np.any(np.isnan(energy)) or np.any(np.isneginf(energy)):
        raise GateError("NaN/-Inf energy_error is never legal")
    if np.any(np.isneginf(log_ratio) & ~diverged):
        raise GateError("-Inf log_accept_ratio requires matching divergence")
    if np.any(np.isposinf(energy) & ~diverged):
        raise GateError("+Inf energy_error requires matching divergence")
    if np.any(nonfinite & ~diverged):
        raise GateError("non-finite proposal trace requires matching divergence")
    if np.any(diverged & ~anomaly):
        raise GateError("a divergence must also be a numerical anomaly")
    finite_pair = np.isfinite(log_ratio) & np.isfinite(energy)
    if not np.array_equal(energy[finite_pair], -log_ratio[finite_pair]):
        raise GateError("energy_error must equal -log_accept_ratio")
    return {
        "nonfinite_count": int(np.sum(nonfinite)),
        "divergence_count": int(np.sum(diverged)),
        "numerical_anomaly_count": int(np.sum(anomaly)),
        "accept_rate_min": float(np.min(np.mean(accepted_array, axis=0))),
        "accept_rate_max": float(np.max(np.mean(accepted_array, axis=0))),
    }


def score_batch(
    *,
    latent_draws: Any,
    functional_draws_model_units: Any,
    truth_original_units: Any,
    accepted: Any,
    log_accept_ratio: Any,
    energy_error: Any,
    has_nonfinite: Any,
    divergence: Any,
    numerical_anomaly: Any,
    trajectory_length: Any,
    identity: BatchIdentity,
    outcome_transform: OutcomeTransform,
    precision_policy: PrecisionPolicy,
    independent_initialization_provenance: bool = True,
    fixed_production_kernel: bool = True,
    target_gradient_finite: bool = True,
    geometry_action: str = "KEEP_IDENTITY",
) -> Dict[str, Any]:
    """One reportable record for one batch, or the exact unmet contract.

    Any non-finite or divergent transition is ``NUMERICAL_RESTART``; chain
    diagnostics on the latent draws AND on the structural functional must
    pass the stationarity and ESS gates; the block-jackknife MCSE must meet
    the precision policy.
    """

    latent = np.asarray(latent_draws)
    functional_model = np.asarray(functional_draws_model_units)
    truth = np.asarray(truth_original_units, np.float64).reshape(-1)
    if latent.ndim != 4:
        raise GateError("latent_draws must be [T,C,U,D]")
    t, c, u, _ = latent.shape
    if c < 4:
        raise GateError("scoring requires at least four chains")
    if functional_model.ndim != 3 or functional_model.shape[:2] != (t, c):
        raise GateError("functional_draws_model_units must be [T,C,Q]")
    q = functional_model.shape[2]
    if truth.shape != (q,):
        raise GateError("truth_original_units must be [Q]")
    if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(functional_model)):
        raise GateError("draw arrays contain non-finite values")
    identity.validate(num_targets=u, num_queries=q)
    outcome_transform.validate()
    precision_policy.validate()
    if identity.draws_hash != sha256_array(latent, kind="posterior-draws"):
        raise GateError("draws_hash does not match the latent draws being scored")

    trace = _validate_trace(
        accepted,
        log_accept_ratio,
        energy_error,
        has_nonfinite,
        divergence,
        numerical_anomaly,
        trajectory_length,
        shape=(t, c, u),
    )
    if trace["nonfinite_count"]:
        raise GateError(
            f"NUMERICAL_RESTART: production contains {trace['nonfinite_count']} "
            "non-finite proposals"
        )
    if trace["divergence_count"]:
        raise GateError(
            f"NUMERICAL_RESTART: production contains {trace['divergence_count']} divergences"
        )
    if trace["numerical_anomaly_count"]:
        raise GateError(
            "NUMERICAL_RESTART: production contains "
            f"{trace['numerical_anomaly_count']} symmetric energy/non-finite "
            "anomalies"
        )

    functional_original = outcome_transform.to_original(functional_model)
    latent_diag = chain_diagnostics(latent)
    functional_diag = chain_diagnostics(functional_original)
    summary = _active_diagnostic_summary(latent_diag, functional_diag)
    if not summary["valid"]:
        raise GateError("STATIONARITY_RESTART: invalid/constant diagnostic scalar")

    max_iact = functional_iact(functional_diag, t, c)
    block_len = choose_block_len(t, max_iact)
    metric_output = structural_metric(functional_original, truth, block_len)

    sampler = SamplerEvidence(
        num_chains=c,
        draws_per_chain=t,
        rank_rhat_max=float(summary["rank_rhat_max"]),
        folded_rhat_max=float(summary["folded_rhat_max"]),
        bulk_ess_min=float(summary["bulk_ess_min"]),
        tail_ess_min=float(summary["tail_ess_min"]),
        nonfinite_count=0,
        latent_constant_count=int(summary["latent_constant_count"]),
        independent_initialization_provenance=independent_initialization_provenance,
        fixed_production_kernel=fixed_production_kernel,
        target_gradient_finite=target_gradient_finite,
        draws_hash=identity.draws_hash,
        geometry_action=geometry_action,
    )
    metric = structural_metric_evidence(
        metric_output,
        time_draws=t,
        block_len=block_len,
        max_iact=max_iact,
        absolute_halfwidth_tolerance=precision_policy.absolute_halfwidth,
        relative_halfwidth_tolerance=precision_policy.relative_halfwidth,
        precision_reference_scale=precision_policy.reference_scale,
        original_outcome_units=True,
        iact_num_chains=c,
    )
    record = build_batch_record(sampler, metric)
    record.update(
        {
            "identity": identity.payload(),
            "trace": trace,
            "block_len": block_len,
            "max_iact": max_iact,
            "outcome_transform": {
                "shift": float(outcome_transform.shift),
                "scale": float(outcome_transform.scale),
            },
        }
    )
    return record


__all__ = [
    "Action",
    "Assessment",
    "BatchIdentity",
    "GateError",
    "OutcomeTransform",
    "PrecisionPolicy",
    "SamplerEvidence",
    "StructuralMetric",
    "assess_sampler",
    "build_batch_record",
    "chain_diagnostics",
    "choose_block_len",
    "functional_iact",
    "score_batch",
    "structural_metric",
    "structural_metric_evidence",
]
