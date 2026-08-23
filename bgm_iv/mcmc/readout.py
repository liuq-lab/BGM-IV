"""From latent draws to structural numbers.

:class:`StructuralQueryTable` orders the evaluation grid into a first-
occurrence catalog of unique covariate targets; :class:`FunctionalReadout`
maps latent draws ``[T, C, U, D]`` through the outcome network to structural
functional draws ``[T, C, Q]`` in model outcome units while binding the
network state by hash; :class:`PredictiveCalibrationAccumulator` scores the
predictive distribution of the outcome with exact Gaussian-mixture quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import numpy as np
from scipy.special import ndtr, ndtri
import tensorflow as tf

from .target import sha256_array, sha256_json, sha256_weights


# --- query table and readout -------------------------------------------------


class ReadoutError(RuntimeError):
    """A structural-readout contract violation."""


@dataclass(frozen=True)
class StructuralQueryTable:
    """Stable ordered unique-target catalog plus the full query table.

    ``unique_v`` rows keep FIRST-OCCURRENCE order of the evaluation grid —
    never an implicit ``np.unique`` sort — so the catalog order itself is part
    of the run identity.  ``query_inverse[q]`` maps query ``q`` to its row in
    ``unique_v``; ``query_x`` is the treatment value in MODEL scale.
    """

    unique_v: np.ndarray      # [U, V] float32, model scale
    query_x: np.ndarray       # [Q, 1] float32, model scale
    query_inverse: np.ndarray  # [Q] int64

    def __post_init__(self) -> None:
        unique_v = np.asarray(self.unique_v, np.float32)
        query_x = np.asarray(self.query_x, np.float32)
        inverse = np.asarray(self.query_inverse, np.int64)
        if unique_v.ndim != 2 or not len(unique_v):
            raise ReadoutError("unique_v must be a non-empty [U, V] matrix")
        if query_x.ndim != 2 or query_x.shape[1] != 1 or not len(query_x):
            raise ReadoutError("query_x must be a non-empty [Q, 1] column")
        if inverse.shape != (query_x.shape[0],):
            raise ReadoutError("query_inverse must align with query_x")
        if inverse.min(initial=0) < 0 or inverse.max(initial=0) >= len(unique_v):
            raise ReadoutError("query_inverse points outside the catalog")
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
    def ordered_query_values(self) -> np.ndarray:
        """The ``[Q, 2]`` payload bound into the batch identity: (x, inverse)."""
        return np.column_stack(
            [self.query_x[:, 0], self.query_inverse.astype(np.float32)]
        ).astype(np.float32)

    @property
    def representative_rows(self) -> np.ndarray:
        """First grid row of every catalog target, ``[U]`` int64.

        Invariant: ``query_inverse[representative_rows] == arange(num_targets)``.
        Any per-target scoring that needs a row-length array (truth, treatment)
        must index it through this map — NEVER by the target index itself
        (a grid with duplicate covariate rows has ``num_targets < num_queries``
        and target-indexed truth is silently misaligned).
        """
        rep = np.full(self.num_targets, -1, np.int64)
        for row, target in enumerate(self.query_inverse):
            if rep[target] < 0:
                rep[target] = int(row)
        if np.any(rep < 0):
            raise ReadoutError("every catalog target must own at least one query")
        return rep

    def query_rows_of_targets(self, global_ids) -> np.ndarray:
        """All query rows (ascending) whose target is in ``global_ids``."""
        ids = np.asarray(global_ids, np.int64)
        return np.flatnonzero(np.isin(self.query_inverse, ids)).astype(np.int64)

    def row_length_array(self, values, name: str = "values") -> np.ndarray:
        """Validate a per-query array; refuse target-length arrays."""
        array = np.asarray(values)
        if array.shape[0] != self.num_queries:
            hint = ""
            if array.shape[0] == self.num_targets and self.num_targets != self.num_queries:
                hint = (
                    " (array is target-length but the grid has duplicate covariate "
                    "rows; index through representative_rows instead)"
                )
            raise ReadoutError(
                f"{name} must have one entry per query row "
                f"({array.shape[0]} != {self.num_queries}){hint}"
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
    for index in range(grid_v.shape[0]):
        key = grid_v[index].tobytes()
        position = order.get(key)
        if position is None:
            position = len(rows)
            order[key] = position
            rows.append(grid_v[index])
        inverse[index] = position
    return StructuralQueryTable(
        unique_v=np.stack(rows, axis=0),
        query_x=grid_x,
        query_inverse=inverse,
    )


class FunctionalReadout:
    """Callable ``latent [T, C, U, D] -> mu_f draws [T, C, Q]`` (model units).

    Evaluates the frozen outcome mean head ``f_net(concat([z0, z1, x]))[:, :1]``
    over posterior draws, chunked along time and queries to bound memory.
    """

    def __init__(
        self,
        model: Any,
        query_table: StructuralQueryTable,
        *,
        time_chunk: int = 64,
        query_chunk: int = 512,
    ):
        if bool(model.params.get("use_bnn", False)):
            raise ReadoutError("stochastic BNN outcome head is not a fixed readout")
        self.model = model
        self.table = query_table
        self.time_chunk = int(time_chunk)
        self.query_chunk = int(query_chunk)
        if self.time_chunk < 1 or self.query_chunk < 1:
            raise ReadoutError("chunk sizes must be positive")
        z_dims = [int(value) for value in model.params["z_dims"]]
        self.latent_dim = sum(z_dims)
        self.zy_dim = z_dims[0] + z_dims[1]
        self._expected_f_hash = sha256_weights(model.f_net)

    @property
    def outcome_hash(self) -> str:
        return self._expected_f_hash

    def assert_runtime_identity(self) -> None:
        if sha256_weights(self.model.f_net) != self._expected_f_hash:
            raise ReadoutError("outcome network state changed during readout")

    def __call__(self, latent_draws: Any) -> np.ndarray:
        latent = np.asarray(latent_draws, np.float32)
        if latent.ndim != 4 or latent.shape[2] != self.table.num_targets:
            raise ReadoutError("latent draws must be [T, C, U, D] over the catalog")
        if latent.shape[3] != self.latent_dim:
            raise ReadoutError("latent dimension does not match the model")
        if not np.all(np.isfinite(latent)):
            raise ReadoutError("latent draws must be finite")
        self.assert_runtime_identity()
        t_size, c_size, _, _ = latent.shape
        q_size = self.table.num_queries
        z_y = latent[:, :, :, : self.zy_dim]
        functional = np.empty((t_size, c_size, q_size), np.float32)
        inverse = self.table.query_inverse
        query_x = self.table.query_x[:, 0]
        for q_start in range(0, q_size, self.query_chunk):
            q_stop = min(q_start + self.query_chunk, q_size)
            chunk_inverse = inverse[q_start:q_stop]
            chunk_x = query_x[q_start:q_stop]
            width = q_stop - q_start
            for t_start in range(0, t_size, self.time_chunk):
                t_stop = min(t_start + self.time_chunk, t_size)
                depth = t_stop - t_start
                gathered = z_y[t_start:t_stop][:, :, chunk_inverse, :]
                flat_z = gathered.reshape(depth * c_size * width, self.zy_dim)
                flat_x = np.broadcast_to(
                    chunk_x[None, None, :], (depth, c_size, width)
                ).reshape(-1, 1)
                output = self.model.f_net(
                    tf.concat(
                        [
                            tf.constant(flat_z, tf.float32),
                            tf.constant(np.ascontiguousarray(flat_x), tf.float32),
                        ],
                        axis=-1,
                    )
                )
                functional[t_start:t_stop, :, q_start:q_stop] = (
                    np.asarray(output.numpy()[:, 0], np.float32).reshape(
                        depth, c_size, width
                    )
                )
        self.assert_runtime_identity()
        if not np.all(np.isfinite(functional)):
            raise ReadoutError("structural functional draws are non-finite")
        return functional


def batch_query_view(
    table: StructuralQueryTable, global_ids: Tuple[int, ...]
) -> StructuralQueryTable:
    """Restrict a query table to one frozen batch of catalog rows.

    ``global_ids`` are catalog row indices in batch order.  Queries whose
    target is outside the batch are dropped; the inverse map is re-indexed to
    batch-local positions so the view aligns with that batch's ``[T, C, B, D]``
    draws.
    """

    ids = np.asarray(global_ids, np.int64)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ReadoutError("global_ids must be non-empty unique catalog rows")
    if ids.min() < 0 or ids.max() >= table.num_targets:
        raise ReadoutError("global_ids point outside the catalog")
    local_of_global = {int(gid): local for local, gid in enumerate(ids)}
    keep = np.isin(table.query_inverse, ids)
    if not np.any(keep):
        raise ReadoutError("batch contains no queries")
    local_inverse = np.asarray(
        [local_of_global[int(gid)] for gid in table.query_inverse[keep]], np.int64
    )
    return StructuralQueryTable(
        unique_v=table.unique_v[ids],
        query_x=table.query_x[keep],
        query_inverse=local_inverse,
    )


# --- predictive calibration --------------------------------------------------

# p(y | x, v) = mean_m N(y; mu_f(x, z_m), sigma_y(x, z_m)^2) over retained draws
# z_m ~ p(z | v), in original outcome units.  The benchmark truth is
# y ~ N(g0(x, v), s^2) with known noise scale s, so interval coverage is exact
# and predictive quantiles are solved from the mixture CDF by bisection.
# Scoring unit "target": one record per catalog target at its representative
# grid row; "query": one record per grid row through ``query_inverse``.
# Two channels are always produced: ``certified`` (REPORTABLE batches only,
# the only channel that is a result) and ``diagnostic_unscored`` (every batch).

SCORING_UNITS = ("target", "query")


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration settings; every field enters the result payload."""

    levels: Sequence[float] = (0.5, 0.8, 0.95)
    num_draws: int = 2000
    quantile_grid_size: int = 199
    truth_noise_sd: float = 1.0
    scoring_unit: str = "target"
    bisection_iterations: int = 60

    def validate(self) -> "CalibrationConfig":
        levels = tuple(float(v) for v in self.levels)
        if not levels or any(not 0.0 < v < 1.0 for v in levels):
            raise ValueError("levels must lie in (0, 1)")
        if int(self.num_draws) < 2:
            raise ValueError("num_draws must be at least 2")
        if int(self.quantile_grid_size) < 3:
            raise ValueError("quantile_grid_size must be at least 3")
        if not np.isfinite(self.truth_noise_sd) or self.truth_noise_sd <= 0:
            raise ValueError("truth_noise_sd must be positive")
        if self.scoring_unit not in SCORING_UNITS:
            raise ValueError(f"scoring_unit must be one of {SCORING_UNITS}")
        if int(self.bisection_iterations) < 20:
            raise ValueError("bisection_iterations must be at least 20")
        return self

    @property
    def quantile_grid(self) -> np.ndarray:
        return np.linspace(0.005, 0.995, int(self.quantile_grid_size))

    def to_payload(self) -> dict[str, Any]:
        return {
            "levels": [float(v) for v in self.levels],
            "num_draws": int(self.num_draws),
            "quantile_grid_size": int(self.quantile_grid_size),
            "quantile_grid": "linspace(0.005, 0.995, quantile_grid_size)",
            "truth_noise_sd": float(self.truth_noise_sd),
            "scoring_unit": str(self.scoring_unit),
            "quantile_method": "exact_gaussian_mixture_cdf_bisection",
            "bisection_iterations": int(self.bisection_iterations),
        }


def gaussian_mixture_quantiles(
    means: np.ndarray,
    sds: np.ndarray,
    probabilities: np.ndarray,
    *,
    iterations: int = 60,
) -> np.ndarray:
    """Exact quantiles of row-wise equal-weight Gaussian mixtures.

    ``means``/``sds`` are ``[R, M]`` (M components per row); returns ``[R, P]``
    quantiles at ``probabilities`` ``[P]`` via monotone bisection on the exact
    mixture CDF ``F(y) = mean_m Phi((y - mu_m) / sd_m)``.
    """

    means = np.asarray(means, np.float64)
    sds = np.asarray(sds, np.float64)
    probabilities = np.asarray(probabilities, np.float64).reshape(-1)
    if means.ndim != 2 or means.shape != sds.shape:
        raise ValueError("means and sds must be [R, M] with equal shapes")
    if np.any(sds <= 0.0) or not np.all(np.isfinite(means)) or not np.all(np.isfinite(sds)):
        raise ValueError("mixture components must be finite with positive sd")
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("probabilities must lie strictly inside (0, 1)")
    lo = np.min(means - 10.0 * sds, axis=1)[:, None].repeat(probabilities.size, axis=1)
    hi = np.max(means + 10.0 * sds, axis=1)[:, None].repeat(probabilities.size, axis=1)
    target = probabilities[None, :]
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        cdf = ndtr((mid[:, :, None] - means[:, None, :]) / sds[:, None, :]).mean(axis=2)
        below = cdf < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


@dataclass
class _Channel:
    coverage: dict = field(default_factory=dict)
    width80: list = field(default_factory=list)
    wasserstein1: list = field(default_factory=list)
    units: int = 0

    def summary(self, levels: Sequence[float]) -> dict[str, Any]:
        out: dict[str, Any] = {"units": int(self.units)}
        out["coverage_mean"] = {
            str(float(lv)): (float(np.mean(self.coverage[lv])) if self.coverage.get(lv) else None)
            for lv in levels
        }
        out["coverage_gap_mean"] = {
            str(float(lv)): (
                float(np.mean(np.abs(np.asarray(self.coverage[lv]) - float(lv))))
                if self.coverage.get(lv)
                else None
            )
            for lv in levels
        }
        out["wasserstein1_mean"] = (
            float(np.mean(self.wasserstein1)) if self.wasserstein1 else None
        )
        out["width80_mean"] = float(np.mean(self.width80)) if self.width80 else None
        return out


class PredictiveCalibrationAccumulator:
    """Streaming per-batch predictive calibration over a certification run.

    Construct once per grid, then call :meth:`add_batch` with every
    ``BatchOutcome`` and its ``[T, C, B, D]`` draws (``run_mcmc``'s
    ``batch_callback``); :meth:`summary` returns both channels plus the
    per-batch explicitly-unscored plugin MSE.
    """

    def __init__(
        self,
        model: Any,
        table: StructuralQueryTable,
        truth_rows_original: Any,
        *,
        outcome_shift: float,
        outcome_scale: float,
        config: CalibrationConfig = CalibrationConfig(),
    ):
        self.model = model
        self.table = table
        self.config = config.validate()
        self.truth_rows = np.asarray(
            table.row_length_array(truth_rows_original, "truth"), np.float64
        ).reshape(-1)
        self.x_rows = np.asarray(table.query_x, np.float32).reshape(-1)
        self.rep_rows = table.representative_rows
        self.outcome_shift = float(outcome_shift)
        self.outcome_scale = float(outcome_scale)
        if not np.isfinite(self.outcome_scale) or self.outcome_scale <= 0:
            raise ValueError("outcome_scale must be positive")
        self.levels = tuple(float(v) for v in self.config.levels)
        self.qgrid = self.config.quantile_grid
        self.channels = {"certified": _Channel(), "diagnostic": _Channel()}
        self.unscored_plugin_mse_per_batch: dict[int, float] = {}
        self.units_per_batch: dict[int, int] = {}
        self.batch_states: dict[int, str] = {}
        self._seen: set[int] = set()

    # -- per-batch evaluation -------------------------------------------------

    def _rows_for_batch(self, global_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(global_ids, np.int64)
        if self.config.scoring_unit == "target":
            rows = self.rep_rows[ids]
            local = np.arange(ids.size, dtype=np.int64)
        else:
            rows = self.table.query_rows_of_targets(ids)
            local_of_global = {int(g): i for i, g in enumerate(ids)}
            local = np.asarray(
                [local_of_global[int(t)] for t in self.table.query_inverse[rows]], np.int64
            )
        return rows, local

    def add_batch(self, outcome: Any, draws: Any) -> dict[str, Any]:
        """Score one batch; returns the per-batch record that was accumulated."""

        batch_index = int(outcome.batch_index)
        if batch_index in self._seen:
            raise ReadoutError(f"batch {batch_index} was already calibrated")
        draws = np.asarray(draws, np.float32)
        if draws.ndim != 4 or draws.shape[2] != len(outcome.global_ids):
            raise ReadoutError("draws must be [T, C, B, D] over the batch targets")
        t_size, c_size, b_size, d_size = draws.shape
        flat = draws.reshape(t_size * c_size, b_size, d_size)
        stride = max(1, (t_size * c_size) // int(self.config.num_draws))
        sub = flat[::stride][: int(self.config.num_draws)]
        m_size = sub.shape[0]
        rows, local = self._rows_for_batch(outcome.global_ids)
        z_rows = sub[:, local, :]                                  # [M, R, D]
        r_size = rows.size
        x_flat = np.repeat(self.x_rows[rows][None, :], m_size, axis=0).reshape(-1, 1)
        output = self.model._outcome_output(
            tf.constant(z_rows.reshape(m_size * r_size, d_size)),
            tf.constant(x_flat.astype(np.float32)),
        )
        mu = (
            np.asarray(output[:, :1]).reshape(m_size, r_size).astype(np.float64)
            * self.outcome_scale
            + self.outcome_shift
        )
        sd = (
            np.sqrt(
                np.asarray(
                    self.model._continuous_sigma(output, sigma_key="sigma_y")
                ).reshape(m_size, r_size)
            ).astype(np.float64)
            * self.outcome_scale
        )
        sd = np.maximum(sd, 1e-8)
        truth = self.truth_rows[rows]
        # unscored plugin MSE over ALL retained draws of this batch (diagnostic)
        plugin = float(np.mean((truth - mu.mean(axis=0)) ** 2))
        self.unscored_plugin_mse_per_batch[batch_index] = plugin
        self.units_per_batch[batch_index] = int(r_size)
        self.batch_states[batch_index] = str(outcome.state)

        bounds = []
        for level in self.levels:
            alpha = (1.0 - level) / 2.0
            bounds.extend([alpha, 1.0 - alpha])
        probabilities = np.concatenate([np.asarray(bounds), self.qgrid])
        quantiles = gaussian_mixture_quantiles(
            mu.T, sd.T, probabilities, iterations=self.config.bisection_iterations
        )                                                            # [R, P]
        noise_sd = float(self.config.truth_noise_sd)
        record = {"batch_index": batch_index, "units": int(r_size), "state": str(outcome.state)}
        reportable = str(outcome.state) == "REPORTABLE"
        targets = ("certified", "diagnostic") if reportable else ("diagnostic",)
        for k, level in enumerate(self.levels):
            lo = quantiles[:, 2 * k]
            hi = quantiles[:, 2 * k + 1]
            coverage = ndtr((hi - truth) / noise_sd) - ndtr((lo - truth) / noise_sd)
            record[f"coverage_{level}"] = float(np.mean(coverage))
            for name in targets:
                self.channels[name].coverage.setdefault(level, []).extend(coverage.tolist())
                if abs(level - 0.8) < 1e-12:
                    self.channels[name].width80.extend((hi - lo).tolist())
        grid_q = quantiles[:, 2 * len(self.levels):]
        truth_q = truth[:, None] + noise_sd * ndtri(self.qgrid)[None, :]
        w1 = np.mean(np.abs(grid_q - truth_q), axis=1)
        record["wasserstein1"] = float(np.mean(w1))
        for name in targets:
            self.channels[name].wasserstein1.extend(w1.tolist())
            self.channels[name].units += int(r_size)
        self._seen.add(batch_index)
        return record

    # -- aggregation ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        certified = self.channels["certified"].summary(self.levels)
        diagnostic = self.channels["diagnostic"].summary(self.levels)
        weights = self.units_per_batch
        unscored_all = None
        if weights:
            unscored_all = float(
                sum(self.unscored_plugin_mse_per_batch[b] * weights[b] for b in weights)
                / sum(weights.values())
            )
        return {
            "schema_version": "bgm-predictive-calibration",
            "config": self.config.to_payload(),
            "units_calibrated": int(certified["units"]),
            **{k: v for k, v in certified.items() if k != "units"},
            "diagnostic_unscored": diagnostic,
            "unscored_plugin_mse_per_batch": {
                str(b): float(v) for b, v in sorted(self.unscored_plugin_mse_per_batch.items())
            },
            "unscored_plugin_mse_all_batches": unscored_all,
            "batch_states": {str(b): s for b, s in sorted(self.batch_states.items())},
        }


__all__ = [
    "SCORING_UNITS",
    "CalibrationConfig",
    "FunctionalReadout",
    "PredictiveCalibrationAccumulator",
    "ReadoutError",
    "StructuralQueryTable",
    "batch_query_view",
    "build_query_table",
    "gaussian_mixture_quantiles",
]
