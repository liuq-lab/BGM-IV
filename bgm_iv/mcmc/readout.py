"""Full-grid structural MCMC readout and predictive calibration.

The sampler draws one latent state for every unique covariate target.  This
module reuses those targets over every treatment/query row, streams the frozen
outcome network over all retained draws, and reports the ordinary plug-in
structural MSE plus exact Gaussian-mixture interval calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr
import tensorflow as tf

from .target import sha256_array, sha256_json, sha256_weights


class ReadoutError(RuntimeError):
    """A structural-readout contract violation."""


@dataclass(frozen=True)
class StructuralQueryTable:
    """Stable unique-target catalog and the complete query table."""

    unique_v: np.ndarray       # [U,V] model scale
    query_x: np.ndarray        # [Q,1] model scale
    query_inverse: np.ndarray  # [Q], query -> target

    def __post_init__(self) -> None:
        unique_v = np.asarray(self.unique_v, np.float32)
        query_x = np.asarray(self.query_x, np.float32)
        inverse = np.asarray(self.query_inverse, np.int64)
        if unique_v.ndim != 2 or not len(unique_v):
            raise ReadoutError("unique_v must be a non-empty [U,V] matrix")
        if query_x.ndim != 2 or query_x.shape[1] != 1 or not len(query_x):
            raise ReadoutError("query_x must be a non-empty [Q,1] column")
        if inverse.shape != (query_x.shape[0],):
            raise ReadoutError("query_inverse must align with query_x")
        if inverse.min(initial=0) < 0 or inverse.max(initial=0) >= len(unique_v):
            raise ReadoutError("query_inverse points outside the target catalog")
        if not np.all(np.isfinite(unique_v)) or not np.all(np.isfinite(query_x)):
            raise ReadoutError("query table payloads must be finite")
        object.__setattr__(self, "unique_v", np.ascontiguousarray(unique_v))
        object.__setattr__(self, "query_x", np.ascontiguousarray(query_x))
        object.__setattr__(self, "query_inverse", np.ascontiguousarray(inverse))

    @property
    def num_targets(self) -> int:
        return int(self.unique_v.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.query_x.shape[0])

    @property
    def catalog_hash(self) -> str:
        return sha256_array(self.unique_v)

    @property
    def query_hash(self) -> str:
        return sha256_json(
            "structural-query-table",
            {
                "catalog_hash": self.catalog_hash,
                "query_x_hash": sha256_array(self.query_x),
                "query_inverse_hash": sha256_array(self.query_inverse),
            },
        )

    @property
    def representative_rows(self) -> np.ndarray:
        rep = np.full(self.num_targets, -1, np.int64)
        for row, target in enumerate(self.query_inverse):
            if rep[target] < 0:
                rep[target] = int(row)
        if np.any(rep < 0):
            raise ReadoutError("every target must own at least one query")
        return rep

    def row_length_array(self, values: Any, name: str = "values") -> np.ndarray:
        array = np.asarray(values)
        if array.shape[0] != self.num_queries:
            raise ReadoutError(
                f"{name} must have one entry per query row "
                f"({array.shape[0]} != {self.num_queries})"
            )
        return array


def build_query_table(grid_x: Any, grid_v: Any) -> StructuralQueryTable:
    """Deduplicate covariate rows in stable first-occurrence order."""

    grid_x = np.asarray(grid_x, np.float32).reshape(-1, 1)
    grid_v = np.asarray(grid_v, np.float32)
    if grid_v.ndim != 2 or grid_v.shape[0] != grid_x.shape[0]:
        raise ReadoutError("grid_x and grid_v must have matching row counts")
    order: dict[bytes, int] = {}
    inverse = np.empty(grid_v.shape[0], np.int64)
    rows = []
    for index, row in enumerate(grid_v):
        key = row.tobytes()
        position = order.get(key)
        if position is None:
            position = len(rows)
            order[key] = position
            rows.append(row)
        inverse[index] = position
    return StructuralQueryTable(
        unique_v=np.stack(rows),
        query_x=grid_x,
        query_inverse=inverse,
    )


def gaussian_mixture_quantiles(
    means: np.ndarray,
    sds: np.ndarray,
    probabilities: np.ndarray,
    *,
    iterations: int = 60,
) -> np.ndarray:
    """Exact quantiles of row-wise equal-weight Gaussian mixtures."""

    means = np.asarray(means, np.float64)
    sds = np.asarray(sds, np.float64)
    probabilities = np.asarray(probabilities, np.float64).reshape(-1)
    if means.ndim != 2 or means.shape != sds.shape:
        raise ValueError("means and sds must be [R,M] with equal shapes")
    if np.any(sds <= 0.0) or not np.all(np.isfinite(means)) or not np.all(
        np.isfinite(sds)
    ):
        raise ValueError("mixture components must be finite with positive sd")
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("probabilities must lie strictly inside (0,1)")
    if int(iterations) < 20:
        raise ValueError("bisection iterations must be at least 20")
    lo = np.min(means - 10.0 * sds, axis=1)[:, None]
    hi = np.max(means + 10.0 * sds, axis=1)[:, None]
    lo = np.repeat(lo, probabilities.size, axis=1)
    hi = np.repeat(hi, probabilities.size, axis=1)
    target = probabilities[None, :]
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        cdf = ndtr(
            (mid[:, :, None] - means[:, None, :]) / sds[:, None, :]
        ).mean(axis=2)
        below = cdf < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ReadoutConfig:
    """Memory-bounded, all-draw full-grid readout settings."""

    levels: Sequence[float] = (0.5, 0.8, 0.95)
    truth_noise_sd: float = 1.0
    query_chunk: int = 16
    draw_chunk: int = 16384
    bisection_iterations: int = 60

    def validate(self) -> "ReadoutConfig":
        levels = tuple(float(level) for level in self.levels)
        if levels != (0.5, 0.8, 0.95):
            raise ValueError("levels are fixed to (0.5,0.8,0.95)")
        if not np.isfinite(self.truth_noise_sd) or self.truth_noise_sd <= 0:
            raise ValueError("truth_noise_sd must be positive")
        if int(self.query_chunk) < 1 or int(self.draw_chunk) < 1:
            raise ValueError("readout chunks must be positive")
        if int(self.bisection_iterations) < 20:
            raise ValueError("bisection_iterations must be at least 20")
        return self

    def to_payload(self) -> dict[str, Any]:
        return {
            "levels": [float(level) for level in self.levels],
            "truth_noise_sd": float(self.truth_noise_sd),
            "query_chunk": int(self.query_chunk),
            "draw_chunk": int(self.draw_chunk),
            "bisection_iterations": int(self.bisection_iterations),
            "scoring_unit": "query",
            "draw_usage": "all_post_warmup_draws",
            "quantile_method": "exact_gaussian_mixture_cdf_bisection",
        }


class FullGridReadout:
    """Compute MSE and predictive intervals over every query and draw."""

    def __init__(
        self,
        model: Any,
        table: StructuralQueryTable,
        truth_rows_original: Any,
        *,
        outcome_shift: float,
        outcome_scale: float,
        config: ReadoutConfig = ReadoutConfig(),
    ):
        if bool(model.params.get("use_bnn", False)):
            raise ReadoutError("stochastic BNN outcome head is not a fixed readout")
        self.model = model
        self.table = table
        self.truth = np.asarray(
            table.row_length_array(truth_rows_original, "truth"), np.float64
        ).reshape(-1)
        self.outcome_shift = float(outcome_shift)
        self.outcome_scale = float(outcome_scale)
        if not np.isfinite(self.outcome_shift):
            raise ReadoutError("outcome_shift must be finite")
        if not np.isfinite(self.outcome_scale) or self.outcome_scale <= 0:
            raise ReadoutError("outcome_scale must be positive")
        self.config = config.validate()
        self.latent_dim = int(sum(int(v) for v in model.params["z_dims"]))
        self._expected_f_hash = sha256_weights(model.f_net)

    @property
    def outcome_hash(self) -> str:
        return self._expected_f_hash

    def assert_runtime_identity(self) -> None:
        if sha256_weights(self.model.f_net) != self._expected_f_hash:
            raise ReadoutError("outcome network state changed during readout")

    @property
    def _probabilities(self) -> np.ndarray:
        bounds = []
        for level in self.config.levels:
            alpha = (1.0 - float(level)) / 2.0
            bounds.extend([alpha, 1.0 - alpha])
        return np.asarray(bounds, np.float64)

    def __call__(self, latent_draws: Any) -> dict[str, Any]:
        latent = np.asarray(latent_draws, np.float32)
        if latent.ndim != 4 or latent.shape[2] != self.table.num_targets:
            raise ReadoutError("latent draws must be [T,C,U,D] over all targets")
        if latent.shape[3] != self.latent_dim:
            raise ReadoutError("latent dimension does not match the model")
        if not np.all(np.isfinite(latent)):
            raise ReadoutError("latent draws must be finite")
        t_size, c_size, u_size, d_size = latent.shape
        if c_size < 2:
            raise ReadoutError("chain sensitivity requires at least two chains")
        flat = latent.reshape(t_size * c_size, u_size, d_size)
        num_components = int(flat.shape[0])
        plugin_sse = 0.0
        penalty_sum = 0.0
        coverage_sum = {float(level): 0.0 for level in self.config.levels}
        width_sum = {float(level): 0.0 for level in self.config.levels}
        component_seconds = 0.0
        quantile_seconds = 0.0
        probabilities = self._probabilities
        self.assert_runtime_identity()

        for q_start in range(0, self.table.num_queries, self.config.query_chunk):
            q_stop = min(q_start + self.config.query_chunk, self.table.num_queries)
            inverse = self.table.query_inverse[q_start:q_stop]
            query_x = self.table.query_x[q_start:q_stop, 0]
            truth = self.truth[q_start:q_stop]
            width = q_stop - q_start
            means = np.empty((width, num_components), np.float64)
            sds = np.empty((width, num_components), np.float64)
            chain_sums = np.zeros((c_size, width), np.float64)
            local_counts = np.zeros(c_size, np.int64)

            started = time.perf_counter()
            for d_start in range(0, num_components, self.config.draw_chunk):
                d_stop = min(d_start + self.config.draw_chunk, num_components)
                block = flat[d_start:d_stop]
                rows = block[:, inverse, :]
                block_size = d_stop - d_start
                x = np.broadcast_to(
                    query_x[None, :, None], (block_size, width, 1)
                )
                output = self.model._outcome_output(
                    tf.constant(rows.reshape(block_size * width, d_size)),
                    tf.constant(np.ascontiguousarray(x.reshape(-1, 1)), tf.float32),
                )
                mean_block = (
                    np.asarray(output[:, :1]).reshape(block_size, width).astype(
                        np.float64
                    )
                    * self.outcome_scale
                    + self.outcome_shift
                )
                sd_block = (
                    np.sqrt(
                        np.asarray(
                            self.model._continuous_sigma(
                                output, sigma_key="sigma_y"
                            )
                        ).reshape(block_size, width)
                    ).astype(np.float64)
                    * self.outcome_scale
                )
                if not np.all(np.isfinite(mean_block)) or not np.all(
                    np.isfinite(sd_block)
                ) or np.any(sd_block <= 0.0):
                    raise ReadoutError("predictive mixture components are invalid")
                means[:, d_start:d_stop] = mean_block.T
                sds[:, d_start:d_stop] = sd_block.T
                chain_index = np.arange(d_start, d_stop, dtype=np.int64) % c_size
                for chain in range(c_size):
                    mask = chain_index == chain
                    chain_sums[chain] += mean_block[mask].sum(axis=0)
                    local_counts[chain] += int(np.sum(mask))
            component_seconds += time.perf_counter() - started
            if np.any(local_counts != t_size):
                raise ReadoutError("pooled draw order did not preserve chain counts")
            chain_means = chain_sums / local_counts[:, None]
            pooled_mean = chain_means.mean(axis=0)
            residual = pooled_mean - truth
            plugin_sse += float(np.sum(residual**2))
            penalty_sum += float(
                np.sum(np.var(chain_means, axis=0, ddof=1) / c_size)
            )

            started = time.perf_counter()
            quantiles = gaussian_mixture_quantiles(
                means,
                sds,
                probabilities,
                iterations=self.config.bisection_iterations,
            )
            quantile_seconds += time.perf_counter() - started
            for index, level in enumerate(self.config.levels):
                lo = quantiles[:, 2 * index]
                hi = quantiles[:, 2 * index + 1]
                coverage = ndtr(
                    (hi - truth) / float(self.config.truth_noise_sd)
                ) - ndtr((lo - truth) / float(self.config.truth_noise_sd))
                coverage_sum[float(level)] += float(np.sum(coverage))
                width_sum[float(level)] += float(np.sum(hi - lo))

        self.assert_runtime_identity()
        num_queries = float(self.table.num_queries)
        plugin_mse = float(plugin_sse / num_queries)
        penalty = float(penalty_sum / num_queries)
        return {
            "schema_version": "bgm-mcmc-full-grid-readout",
            "config": self.config.to_payload(),
            "num_targets": int(self.table.num_targets),
            "num_queries": int(self.table.num_queries),
            "num_chains": int(c_size),
            "draws_per_chain": int(t_size),
            "num_components": int(num_components),
            "structural_mse_plugin": plugin_mse,
            "sensitivity": {
                "chain_mean_variance_penalty": penalty,
                "penalty_fraction_of_plugin": (
                    None if plugin_mse == 0.0 else float(penalty / plugin_mse)
                ),
            },
            "coverage": {
                str(float(level)): float(coverage_sum[float(level)] / num_queries)
                for level in self.config.levels
            },
            "width50": float(width_sum[0.5] / num_queries),
            "width80": float(width_sum[0.8] / num_queries),
            "width95": float(width_sum[0.95] / num_queries),
            "timings": {
                "component_seconds": float(component_seconds),
                "quantile_seconds": float(quantile_seconds),
                "total_seconds": float(component_seconds + quantile_seconds),
            },
            "outcome_hash": self.outcome_hash,
        }


__all__ = [
    "FullGridReadout",
    "ReadoutConfig",
    "ReadoutError",
    "StructuralQueryTable",
    "build_query_table",
    "gaussian_mixture_quantiles",
]
