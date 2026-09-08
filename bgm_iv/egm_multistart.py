"""Pure configuration, scoring, and selection utilities for EGM multistart.

This module deliberately has no TensorFlow dependency.  Candidate training is
owned by the caller; the functions here define the reproducible contract used
to select one fully trained candidate.

Protocol (one rule for every benchmark): every EGM warm start is continued
through the BGM stage, then the candidate with the smallest *training-set*
IV-moment residual under the MAP latent readout is selected, and every
structural readout (MAP, encoder, MCMC) is reported from that one model.  The
EGM tail-window score is still recorded per candidate as a diagnostic, but it
no longer enters the selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Optional

from .hashing import sha256_json


DEFAULT_EGM_NUM_WARM_STARTS = 1
EGM_SCORE_WINDOW_SIZE = 10
EGM_SELECTOR_VERSION = "train-iv-map-post-bgm"
EGM_SELECTION_CRITERION = "train_iv_map"
EGM_CANDIDATE_MANIFEST_VERSION = "egm-candidate-manifest"
EGM_SELECTION_MANIFEST_VERSION = "egm-selection-manifest"

EGM_INIT_SEED_NAMESPACE = "egm-init"
EGM_SCHEDULE_SEED_NAMESPACE = "egm-schedule"
POST_EGM_SEED_NAMESPACE = "post-egm"
EGM_CRITERION_SEED_NAMESPACE = "egm-criterion"

# Namespaces whose derived seed is specific to one candidate.
_CANDIDATE_SEED_NAMESPACES = frozenset(
    {EGM_INIT_SEED_NAMESPACE, EGM_CRITERION_SEED_NAMESPACE}
)
_SEED_NAMESPACES = frozenset(
    {
        EGM_INIT_SEED_NAMESPACE,
        EGM_SCHEDULE_SEED_NAMESPACE,
        POST_EGM_SEED_NAMESPACE,
        EGM_CRITERION_SEED_NAMESPACE,
    }
)
_SEED_MODULUS = 2**31 - 1
_REMOVED_SELECTION_KEYS = ("egm_selection_top_k",)


class MultistartConfigurationError(ValueError):
    """Raised when an EGM multistart configuration violates the contract."""


class CandidateSelectionError(RuntimeError):
    """Raised when there are not enough scientifically usable candidates."""


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MultistartConfigurationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise MultistartConfigurationError(f"{name} must be >= {minimum}")
    return result


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise MultistartConfigurationError(f"{name} must be a boolean")
    return value


def validate_multistart_config(
    params: Mapping[str, Any], *, mcmc_only: bool = False
) -> dict[str, Any]:
    """Return a normalized copy of ``params`` or fail on an invalid setup.

    Omitting both public fields preserves the historical single-start path.
    The additional safety invariants apply only when more than one warm start
    is requested.
    """

    if not isinstance(params, Mapping):
        raise MultistartConfigurationError("params must be a mapping")
    normalized = dict(params)
    for removed in _REMOVED_SELECTION_KEYS:
        if removed in normalized:
            raise MultistartConfigurationError(
                f"{removed} is no longer supported: every warm start is continued "
                "through BGM and the candidate with the smallest training-set "
                "IV-moment residual (MAP readout) is selected; remove the key"
            )
    num_starts = _require_int(
        "egm_num_warm_starts",
        normalized.get(
            "egm_num_warm_starts", DEFAULT_EGM_NUM_WARM_STARTS
        ),
        minimum=1,
    )

    if num_starts > 1:
        if not _require_bool("save_model", normalized.get("save_model", False)):
            raise MultistartConfigurationError(
                "EGM multistart requires save_model=true"
            )
        if not _require_bool(
            "deterministic_training",
            normalized.get("deterministic_training", False),
        ):
            raise MultistartConfigurationError(
                "EGM multistart requires deterministic_training=true"
            )
        if _require_bool(
            "training_grid_monitor",
            normalized.get("training_grid_monitor", False),
        ):
            raise MultistartConfigurationError(
                "EGM multistart requires training_grid_monitor=false"
            )
        # ``--mcmc-only`` restores an already-trained multistart checkpoint and
        # does not launch candidate training.  Keep the public warm-start count
        # in the manifest so restore can verify the estimator that produced the
        # checkpoint.  The caller is responsible for dispatching restore
        # instead of training whenever the flag is present.
        del mcmc_only

    normalized["egm_num_warm_starts"] = num_starts
    return normalized


def score_evaluation_iterations(
    fit_egm_n_iter: int,
    fit_egm_batches_per_eval: int,
    *,
    window_size: int = EGM_SCORE_WINDOW_SIZE,
) -> tuple[int, ...]:
    """Return the fixed tail window ``T-(W-1)E, ..., T``.

    Iteration zero is valid because the existing EGM loop has inclusive
    iteration semantics.  A negative first point is rejected rather than
    silently shortening the window.
    """

    terminal = _require_int("fit_egm_n_iter", fit_egm_n_iter, minimum=0)
    interval = _require_int(
        "fit_egm_batches_per_eval", fit_egm_batches_per_eval, minimum=1
    )
    count = _require_int("window_size", window_size, minimum=1)
    first = terminal - (count - 1) * interval
    if first < 0:
        raise MultistartConfigurationError(
            "fit_egm_n_iter is too small for the requested score window"
        )
    return tuple(first + index * interval for index in range(count))


def derive_multistart_seed(
    dataset: str,
    n_samples: int,
    rho: Real,
    repeat_id: int,
    namespace: str,
    *,
    run_seed: int,
    candidate_id: Optional[int] = None,
) -> int:
    """Derive a stable positive 31-bit seed from the experiment identity.

    ``run_seed`` is the cell-level master seed.  It is required explicitly so
    changing the configured base seed changes every derived multistart stream
    instead of only changing the simulated data.
    """

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    run_seed_value = _require_int("run_seed", run_seed, minimum=0)
    n_value = _require_int("n_samples", n_samples, minimum=1)
    repeat_value = _require_int("repeat_id", repeat_id, minimum=0)
    if isinstance(rho, bool) or not isinstance(rho, Real):
        raise ValueError("rho must be a finite real number")
    rho_value = float(rho)
    if not math.isfinite(rho_value):
        raise ValueError("rho must be a finite real number")
    if namespace not in _SEED_NAMESPACES:
        raise ValueError(f"unknown multistart seed namespace: {namespace!r}")

    if namespace in _CANDIDATE_SEED_NAMESPACES:
        if candidate_id is None:
            raise ValueError(
                f"candidate_id is required for the {namespace!r} seed namespace"
            )
        candidate_value: Optional[int] = _require_int(
            "candidate_id", candidate_id, minimum=0
        )
    else:
        if candidate_id is not None:
            raise ValueError(
                "candidate_id is only valid for the per-candidate seed namespaces"
            )
        candidate_value = None

    identity = {
        "run_seed": run_seed_value,
        "dataset": dataset,
        "n_samples": n_value,
        "rho": format(rho_value, ".17g"),
        "repeat_id": repeat_value,
        "namespace": namespace,
        "candidate_id": candidate_value,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(b"bgm-iv-egm-multistart-seed\0" + encoded).digest()
    # Avoid zero while staying in the positive signed 31-bit range accepted by
    # NumPy and TensorFlow seed APIs.
    return int.from_bytes(digest[:8], "big") % (_SEED_MODULUS - 1) + 1


def derive_multistart_seeds(
    dataset: str,
    n_samples: int,
    rho: Real,
    repeat_id: int,
    num_warm_starts: int,
    *,
    run_seed: int,
) -> dict[str, Any]:
    """Derive all independent random streams for one outer data repeat.

    ``init_seeds`` differ per candidate (network initialization only);
    ``schedule_seed`` and ``post_egm_seed`` are shared by every candidate so the
    EGM minibatch schedule and the BGM continuation are identical across
    candidates; ``criterion_seeds`` fix the Monte Carlo draws of the per-candidate
    selection criterion.
    """

    count = _require_int(
        "egm_num_warm_starts", num_warm_starts, minimum=1
    )
    common = (dataset, n_samples, rho, repeat_id)
    return {
        "init_seeds": [
            derive_multistart_seed(
                *common,
                EGM_INIT_SEED_NAMESPACE,
                run_seed=run_seed,
                candidate_id=candidate_id,
            )
            for candidate_id in range(count)
        ],
        "schedule_seed": derive_multistart_seed(
            *common, EGM_SCHEDULE_SEED_NAMESPACE, run_seed=run_seed
        ),
        "post_egm_seed": derive_multistart_seed(
            *common, POST_EGM_SEED_NAMESPACE, run_seed=run_seed
        ),
        "criterion_seeds": [
            derive_multistart_seed(
                *common,
                EGM_CRITERION_SEED_NAMESPACE,
                run_seed=run_seed,
                candidate_id=candidate_id,
            )
            for candidate_id in range(count)
        ],
    }


def _candidate_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("candidate IDs must be non-negative integers")
    result = int(value)
    if result < 0:
        raise ValueError("candidate IDs must be non-negative integers")
    return result


def _finite_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    if result < 0.0:
        raise ValueError("candidate losses must be non-negative")
    return result


def rank_finite_candidates(
    candidate_scores: Mapping[int, Any], *, top_k: int
) -> tuple[dict[str, Any], ...]:
    """Rank finite candidates by ``(score, candidate_id)``.

    NaN, infinity, and ``None`` are scientific failures and are excluded.  A
    stable candidate-ID tie break makes the ranking independent of mapping
    insertion order.
    """

    if not isinstance(candidate_scores, Mapping):
        raise ValueError("candidate_scores must be a mapping")
    required = _require_int("top_k", top_k, minimum=1)
    finite: list[tuple[float, int]] = []
    seen: set[int] = set()
    for raw_id, raw_score in candidate_scores.items():
        candidate_id = _candidate_id(raw_id)
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)
        score = _finite_score(raw_score)
        if score is not None:
            finite.append((score, candidate_id))
    finite.sort(key=lambda item: (item[0], item[1]))
    if len(finite) < required:
        raise CandidateSelectionError(
            f"need at least {required} finite candidates; found {len(finite)}"
        )
    return tuple(
        {
            "rank": rank,
            "candidate_id": candidate_id,
            "score": score,
        }
        for rank, (score, candidate_id) in enumerate(finite, start=1)
    )


def _json_payload_hash(namespace: str, payload: Mapping[str, Any]) -> str:
    # sha256_json also proves that the payload is JSON serializable and contains
    # no NaN or infinity values.
    return sha256_json(namespace, payload)


def _serialize_scores(candidate_scores: Mapping[int, Any]) -> list[dict[str, Any]]:
    serialized = []
    seen: set[int] = set()
    for raw_id, raw_score in candidate_scores.items():
        candidate_id = _candidate_id(raw_id)
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)
        finite_score = _finite_score(raw_score)
        serialized.append(
            {
                "candidate_id": candidate_id,
                "score": finite_score,
                "finite": finite_score is not None,
            }
        )
    serialized.sort(key=lambda record: record["candidate_id"])
    return serialized


def select_candidate_by_criterion(
    candidate_criteria: Mapping[int, Any],
    *,
    egm_tail_scores: Optional[Mapping[int, Any]] = None,
) -> dict[str, Any]:
    """Select the candidate with the smallest post-BGM training criterion.

    ``candidate_criteria`` maps candidate id -> training-set IV-moment residual
    under the MAP latent readout, measured after the BGM stage on the same
    training rows the candidate was fitted on.  The rule is a deterministic
    argmin with a candidate-id tie break; NaN/inf/None values are scientific
    failures and are excluded, and at least one finite candidate is required.
    ``egm_tail_scores`` (the EGM tail-window score of each candidate) is only
    recorded as a diagnostic so the two orderings can be compared post hoc.
    """

    if not isinstance(candidate_criteria, Mapping):
        raise ValueError("candidate_criteria must be a mapping")
    ranking = rank_finite_candidates(candidate_criteria, top_k=1)
    selected = ranking[0]
    serialized_criteria = _serialize_scores(candidate_criteria)

    egm_ranking: list[dict[str, Any]] = []
    selected_egm_rank: Optional[int] = None
    if egm_tail_scores is not None:
        if not isinstance(egm_tail_scores, Mapping):
            raise ValueError("egm_tail_scores must be a mapping")
        if set(_candidate_id(key) for key in egm_tail_scores) != set(
            record["candidate_id"] for record in serialized_criteria
        ):
            raise ValueError(
                "egm_tail_scores must cover exactly the candidates being selected"
            )
        try:
            egm_ranking = list(rank_finite_candidates(egm_tail_scores, top_k=1))
        except CandidateSelectionError:
            egm_ranking = []
        for record in egm_ranking:
            if record["candidate_id"] == selected["candidate_id"]:
                selected_egm_rank = int(record["rank"])

    payload: dict[str, Any] = {
        "manifest_version": EGM_SELECTION_MANIFEST_VERSION,
        "selector_version": EGM_SELECTOR_VERSION,
        "selection_criterion": EGM_SELECTION_CRITERION,
        "candidate_criteria": serialized_criteria,
        "finite_ranking": list(ranking),
        "selected_rank": int(selected["rank"]),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_criterion": float(selected["score"]),
        "egm_tail_scores": (
            _serialize_scores(egm_tail_scores) if egm_tail_scores is not None else []
        ),
        "egm_tail_ranking": egm_ranking,
        "selected_egm_tail_rank": selected_egm_rank,
        "uses_validation": False,
        "uses_holdout": False,
        "uses_test_grid": False,
    }
    payload["selection_manifest_hash"] = _json_payload_hash(
        "egm-selection-manifest", payload
    )
    return payload


def make_candidate_manifest(
    *,
    candidate_id: int,
    init_seed: int,
    schedule_seed: int,
    run_seed: int,
    evaluation_iterations: Sequence[int],
    full_train_l2_loss_y: Sequence[Any],
    status: str,
    data_hash: str,
    config_hash: str,
    code_commit: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_hash: Optional[str] = None,
    checkpoint_weight_hash: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    failure_reason: Optional[str] = None,
    worker_pid: Optional[int] = None,
    device_names: Optional[Sequence[str]] = None,
    device_hash: Optional[str] = None,
    bgm_checkpoint_path: Optional[str] = None,
    bgm_checkpoint_hash: Optional[str] = None,
    bgm_checkpoint_weight_hash: Optional[str] = None,
    criterion_seed: Optional[int] = None,
    train_iv_map: Optional[Any] = None,
    train_iv_encoder: Optional[Any] = None,
    train_mse_x: Optional[Any] = None,
    train_mse_y: Optional[Any] = None,
    train_mse_v: Optional[Any] = None,
    bgm_seconds: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a canonical, JSON-safe candidate artifact with its own digest.

    ``full_train_l2_loss_y`` / ``tail_mean_score`` describe the EGM stage
    (diagnostic only).  The ``bgm_*`` and ``train_*`` fields describe the same
    candidate after its BGM continuation; ``train_iv_map`` is the selection
    criterion.  ``status`` is ``"completed"`` only when both stages finished
    with finite values.
    """

    candidate_value = _candidate_id(candidate_id)
    init_value = _require_int("init_seed", init_seed, minimum=0)
    schedule_value = _require_int("schedule_seed", schedule_seed, minimum=0)
    run_seed_value = _require_int("run_seed", run_seed, minimum=0)
    iterations = [
        _require_int("evaluation iteration", value, minimum=0)
        for value in evaluation_iterations
    ]
    if len(iterations) != EGM_SCORE_WINDOW_SIZE:
        raise ValueError(
            f"candidate score history must contain {EGM_SCORE_WINDOW_SIZE} evaluations"
        )
    if any(right <= left for left, right in zip(iterations, iterations[1:])):
        raise ValueError("evaluation iterations must be strictly increasing")
    if len(full_train_l2_loss_y) != len(iterations):
        raise ValueError("score values must match evaluation iterations")

    scores = [_finite_score(value) for value in full_train_l2_loss_y]
    tail_score = math.fsum(scores) / len(scores) if all(
        score is not None for score in scores
    ) else None
    if not isinstance(status, str) or not status:
        raise ValueError("status must be a non-empty string")
    for name, value in (
        ("data_hash", data_hash),
        ("config_hash", config_hash),
        ("code_commit", code_commit),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    payload: dict[str, Any] = {
        "manifest_version": EGM_CANDIDATE_MANIFEST_VERSION,
        "candidate_id": candidate_value,
        "init_seed": init_value,
        "schedule_seed": schedule_value,
        "run_seed": run_seed_value,
        "evaluation_iterations": iterations,
        "full_train_l2_loss_y": scores,
        "tail_mean_score": tail_score,
        "status": status,
        "failure_reason": failure_reason,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "code_commit": code_commit,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_weight_hash": checkpoint_weight_hash,
        "started_at": started_at,
        "finished_at": finished_at,
        "worker_pid": (
            None if worker_pid is None else _require_int("worker_pid", worker_pid, minimum=1)
        ),
        "device_names": [str(value) for value in (device_names or ())],
        "device_hash": device_hash,
        "bgm_checkpoint_path": bgm_checkpoint_path,
        "bgm_checkpoint_hash": bgm_checkpoint_hash,
        "bgm_checkpoint_weight_hash": bgm_checkpoint_weight_hash,
        "criterion_seed": (
            None
            if criterion_seed is None
            else _require_int("criterion_seed", criterion_seed, minimum=0)
        ),
        "train_iv_map": _finite_score(train_iv_map),
        "train_iv_encoder": _finite_score(train_iv_encoder),
        "train_mse_x": _finite_score(train_mse_x),
        "train_mse_y": _finite_score(train_mse_y),
        "train_mse_v": _finite_score(train_mse_v),
        "bgm_seconds": (
            None if bgm_seconds is None else float(bgm_seconds)
        ),
    }
    # Validate optional fields before hashing so Path objects and NaN timestamps
    # cannot leak into a manifest that only appears JSON serializable.
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["candidate_manifest_hash"] = _json_payload_hash(
        "egm-candidate-manifest", payload
    )
    return payload


def verify_manifest_hash(
    payload: Mapping[str, Any], *, hash_field: str, namespace: str
) -> bool:
    """Verify a self-hashed JSON manifest without mutating the input."""

    if not isinstance(payload, Mapping):
        return False
    recorded = payload.get(hash_field)
    if not isinstance(recorded, str):
        return False
    unhashed = dict(payload)
    unhashed.pop(hash_field, None)
    try:
        expected = _json_payload_hash(namespace, unhashed)
    except (TypeError, ValueError):
        return False
    return recorded == expected


__all__ = [
    "CandidateSelectionError",
    "DEFAULT_EGM_NUM_WARM_STARTS",
    "EGM_CANDIDATE_MANIFEST_VERSION",
    "EGM_CRITERION_SEED_NAMESPACE",
    "EGM_INIT_SEED_NAMESPACE",
    "EGM_SCHEDULE_SEED_NAMESPACE",
    "EGM_SCORE_WINDOW_SIZE",
    "EGM_SELECTION_CRITERION",
    "EGM_SELECTION_MANIFEST_VERSION",
    "EGM_SELECTOR_VERSION",
    "MultistartConfigurationError",
    "POST_EGM_SEED_NAMESPACE",
    "derive_multistart_seed",
    "derive_multistart_seeds",
    "make_candidate_manifest",
    "rank_finite_candidates",
    "score_evaluation_iterations",
    "select_candidate_by_criterion",
    "validate_multistart_config",
    "verify_manifest_hash",
]
