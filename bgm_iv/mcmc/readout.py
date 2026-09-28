from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr
import tensorflow as tf


class ReadoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuralQueryTable:
    unique_v: np.ndarray
    query_x: np.ndarray
    query_inverse: np.ndarray

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


def build_query_table(grid_x: Any, grid_v: Any) -> StructuralQueryTable:
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


# The chunk sizes fix the summation order, and hence the floating-point result.
LEVELS = (0.9, 0.95, 0.99)
TRUTH_NOISE_SD = 1.0
QUERY_CHUNK = 16
DRAW_CHUNK = 16384
BISECTION_ITERATIONS = 60


class FullGridReadout:
    def __init__(
        self,
        model: Any,
        table: StructuralQueryTable,
        truth_rows_original: Any,
        *,
        outcome_shift: float,
        outcome_scale: float,
    ):
        self.model = model
        self.table = table
        truth = np.asarray(truth_rows_original, np.float64)
        if truth.shape[0] != table.num_queries:
            raise ReadoutError(
                "truth must have one entry per query row "
                f"({truth.shape[0]} != {table.num_queries})"
            )
        self.truth = truth.reshape(-1)
        self.outcome_shift = float(outcome_shift)
        self.outcome_scale = float(outcome_scale)
        if not np.isfinite(self.outcome_shift):
            raise ReadoutError("outcome_shift must be finite")
        if not np.isfinite(self.outcome_scale) or self.outcome_scale <= 0:
            raise ReadoutError("outcome_scale must be positive")
        self.latent_dim = int(sum(int(v) for v in model.params["z_dims"]))

    @staticmethod
    def _probabilities() -> np.ndarray:
        bounds = []
        for level in LEVELS:
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
        query_chunk = int(QUERY_CHUNK)
        draw_chunk = int(DRAW_CHUNK)
        t_size, c_size, u_size, d_size = latent.shape
        flat = latent.reshape(t_size * c_size, u_size, d_size)
        num_components = int(flat.shape[0])
        coverage_sum = {float(level): 0.0 for level in LEVELS}
        width_sum = {float(level): 0.0 for level in LEVELS}
        probabilities = self._probabilities()

        for q_start in range(0, self.table.num_queries, query_chunk):
            q_stop = min(q_start + query_chunk, self.table.num_queries)
            inverse = self.table.query_inverse[q_start:q_stop]
            query_x = self.table.query_x[q_start:q_stop, 0]
            truth = self.truth[q_start:q_stop]
            width = q_stop - q_start
            means = np.empty((width, num_components), np.float64)
            sds = np.empty((width, num_components), np.float64)

            for d_start in range(0, num_components, draw_chunk):
                d_stop = min(d_start + draw_chunk, num_components)
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

            quantiles = gaussian_mixture_quantiles(
                means,
                sds,
                probabilities,
                iterations=int(BISECTION_ITERATIONS),
            )
            for index, level in enumerate(LEVELS):
                lo = quantiles[:, 2 * index]
                hi = quantiles[:, 2 * index + 1]
                coverage = ndtr(
                    (hi - truth) / float(TRUTH_NOISE_SD)
                ) - ndtr((lo - truth) / float(TRUTH_NOISE_SD))
                coverage_sum[float(level)] += float(np.sum(coverage))
                width_sum[float(level)] += float(np.sum(hi - lo))

        num_queries = float(self.table.num_queries)
        return {
            "schema_version": "bgm-mcmc-full-grid-readout",
            "num_targets": int(self.table.num_targets),
            "num_queries": int(self.table.num_queries),
            "num_chains": int(c_size),
            "draws_per_chain": int(t_size),
            "num_components": int(num_components),
            "coverage": {
                str(float(level)): float(coverage_sum[float(level)] / num_queries)
                for level in LEVELS
            },
            "width90": float(width_sum[0.9] / num_queries),
            "width95": float(width_sum[0.95] / num_queries),
            "width99": float(width_sum[0.99] / num_queries),
        }


__all__ = [
    "FullGridReadout",
    "ReadoutError",
    "StructuralQueryTable",
    "build_query_table",
    "gaussian_mixture_quantiles",
]
