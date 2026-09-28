from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Optional

import numpy as np


DEFAULT_EGM_NUM_WARM_STARTS = 1
EGM_SELECTOR_VERSION = "train-iv-map-post-bgm"
EGM_SELECTION_CRITERION = "train_iv_map"

_STREAMS = {
    "egm-init": 0,
    "egm-schedule": 1,
    "post-egm": 2,
    "egm-criterion": 3,
    "mcmc-pilot": 4,
    "mcmc-production": 5,
}
_PER_CANDIDATE_STREAMS = frozenset({"egm-init", "egm-criterion"})
_SEED_MODULUS = 2**31 - 1


class MultistartConfigurationError(ValueError):
    pass


class CandidateSelectionError(RuntimeError):
    pass


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


def validate_multistart_config(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise MultistartConfigurationError("params must be a mapping")
    normalized = dict(params)
    num_starts = _require_int(
        "egm_num_warm_starts",
        normalized.get("egm_num_warm_starts", DEFAULT_EGM_NUM_WARM_STARTS),
        minimum=1,
    )
    if num_starts > 1:
        if not _require_bool("save_model", normalized.get("save_model", False)):
            raise MultistartConfigurationError(
                "EGM multistart requires save_model=true"
            )
    normalized["egm_num_warm_starts"] = num_starts
    return normalized


def normalize_model_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MultistartConfigurationError("model_seed must be an integer")
    seed = int(value)
    if not 1 <= seed < _SEED_MODULUS:
        raise MultistartConfigurationError(
            f"model_seed must lie in 1..{_SEED_MODULUS - 1}"
        )
    return seed


def draw_model_seed() -> int:
    return secrets.randbelow(_SEED_MODULUS - 1) + 1


def derive_seed(
    model_seed: int, stream: str, candidate_id: Optional[int] = None
) -> int:
    seed = normalize_model_seed(model_seed)
    if stream not in _STREAMS:
        raise ValueError(f"unknown seed stream: {stream!r}")
    if stream in _PER_CANDIDATE_STREAMS:
        if candidate_id is None:
            raise ValueError(f"candidate_id is required for the {stream!r} stream")
        spawn_key = (_STREAMS[stream], _require_int("candidate_id", candidate_id, minimum=0))
    else:
        if candidate_id is not None:
            raise ValueError("candidate_id is only valid for per-candidate streams")
        spawn_key = (_STREAMS[stream],)
    state = np.random.SeedSequence(seed, spawn_key=spawn_key).generate_state(
        1, dtype=np.uint64
    )[0]
    return int(state % np.uint64(_SEED_MODULUS - 1)) + 1


def derive_multistart_seeds(model_seed: int, num_warm_starts: int) -> dict[str, Any]:
    count = _require_int("egm_num_warm_starts", num_warm_starts, minimum=1)
    return {
        "model_seed": normalize_model_seed(model_seed),
        "init_seeds": [
            derive_seed(model_seed, "egm-init", candidate_id)
            for candidate_id in range(count)
        ],
        "schedule_seed": derive_seed(model_seed, "egm-schedule"),
        "post_egm_seed": derive_seed(model_seed, "post-egm"),
        "criterion_seeds": [
            derive_seed(model_seed, "egm-criterion", candidate_id)
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
    candidate_scores: Mapping[int, Any],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(candidate_scores, Mapping):
        raise ValueError("candidate_scores must be a mapping")
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
    if not finite:
        raise CandidateSelectionError("need at least 1 finite candidate; found 0")
    return tuple(
        {
            "rank": rank,
            "candidate_id": candidate_id,
            "score": score,
        }
        for rank, (score, candidate_id) in enumerate(finite, start=1)
    )


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
) -> dict[str, Any]:
    if not isinstance(candidate_criteria, Mapping):
        raise ValueError("candidate_criteria must be a mapping")
    ranking = rank_finite_candidates(candidate_criteria)
    selected = ranking[0]
    return {
        "selector_version": EGM_SELECTOR_VERSION,
        "selection_criterion": EGM_SELECTION_CRITERION,
        "candidate_criteria": _serialize_scores(candidate_criteria),
        "finite_ranking": list(ranking),
        "selected_rank": int(selected["rank"]),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_criterion": float(selected["score"]),
    }


def make_candidate_record(
    *,
    candidate_id: int,
    init_seed: int,
    criterion_seed: int,
    status: str,
    bgm_checkpoint_path: Optional[str] = None,
    train_iv_map: Optional[Any] = None,
    train_mse_x: Optional[Any] = None,
    train_mse_y: Optional[Any] = None,
    train_mse_v: Optional[Any] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    egm_seconds: Optional[Any] = None,
    bgm_seconds: Optional[Any] = None,
    failure_reason: Optional[str] = None,
    device_names: Optional[list] = None,
    diagnostics_finite: bool = True,
) -> dict[str, Any]:
    if not isinstance(status, str) or not status:
        raise ValueError("status must be a non-empty string")
    return {
        "candidate_id": _candidate_id(candidate_id),
        "init_seed": _require_int("init_seed", init_seed, minimum=0),
        "criterion_seed": _require_int("criterion_seed", criterion_seed, minimum=0),
        "status": status,
        "failure_reason": failure_reason,
        "bgm_checkpoint_path": bgm_checkpoint_path,
        "train_iv_map": _finite_score(train_iv_map),
        "train_mse_x": _finite_score(train_mse_x),
        "train_mse_y": _finite_score(train_mse_y),
        "train_mse_v": _finite_score(train_mse_v),
        "started_at": started_at,
        "finished_at": finished_at,
        "egm_seconds": None if egm_seconds is None else float(egm_seconds),
        "bgm_seconds": None if bgm_seconds is None else float(bgm_seconds),
        "device_names": [str(value) for value in (device_names or ())],
        "diagnostics_finite": bool(diagnostics_finite),
    }


__all__ = [
    "CandidateSelectionError",
    "DEFAULT_EGM_NUM_WARM_STARTS",
    "EGM_SELECTION_CRITERION",
    "EGM_SELECTOR_VERSION",
    "MultistartConfigurationError",
    "derive_multistart_seeds",
    "derive_seed",
    "draw_model_seed",
    "make_candidate_record",
    "normalize_model_seed",
    "rank_finite_candidates",
    "select_candidate_by_criterion",
    "validate_multistart_config",
]
