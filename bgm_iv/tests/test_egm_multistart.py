import json
import math

import pytest

from bgm_iv.egm_multistart import (
    CandidateSelectionError,
    EGM_SELECTION_CRITERION,
    EGM_SELECTOR_VERSION,
    MultistartConfigurationError,
    derive_multistart_seeds,
    derive_seed,
    draw_model_seed,
    make_candidate_record,
    normalize_model_seed,
    rank_finite_candidates,
    select_candidate_by_criterion,
    validate_multistart_config,
)


def _multistart_params(**overrides):
    params = {
        "egm_num_warm_starts": 10,
        "save_model": True,
    }
    params.update(overrides)
    return params


def test_config_defaults_preserve_single_start():
    normalized = validate_multistart_config({})
    assert normalized["egm_num_warm_starts"] == 1


def test_config_accepts_ten_starts_without_mutating_input():
    params = _multistart_params()
    normalized = validate_multistart_config(params)
    assert normalized == params
    assert normalized is not params


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"egm_num_warm_starts": 0}, "egm_num_warm_starts"),
        ({"save_model": False}, "save_model"),
    ],
)
def test_config_rejects_invalid_multistart(overrides, message):
    with pytest.raises(MultistartConfigurationError, match=message):
        validate_multistart_config(_multistart_params(**overrides))


def test_model_seed_is_validated():
    assert normalize_model_seed(1) == 1
    assert normalize_model_seed(2**31 - 2) == 2**31 - 2
    for bad in (0, -3, 2**31 - 1, True, 1.5, "7"):
        with pytest.raises(MultistartConfigurationError, match="model_seed"):
            normalize_model_seed(bad)


def test_drawn_model_seeds_are_valid_and_vary():
    draws = {draw_model_seed() for _ in range(32)}
    assert all(normalize_model_seed(seed) == seed for seed in draws)
    assert len(draws) > 1


def test_seed_derivation_is_a_pure_function_of_the_model_seed():
    seeds = derive_multistart_seeds(123456, 10)
    assert seeds == derive_multistart_seeds(123456, 10)
    assert seeds["model_seed"] == 123456
    assert len(seeds["init_seeds"]) == 10
    assert len(seeds["criterion_seeds"]) == 10
    every = [
        *seeds["init_seeds"],
        *seeds["criterion_seeds"],
        seeds["schedule_seed"],
        seeds["post_egm_seed"],
    ]
    assert all(0 < seed < 2**31 - 1 for seed in every)
    assert len(set(every)) == 22


def test_seed_derivation_changes_every_stream_with_the_model_seed():
    first = derive_multistart_seeds(123456, 10)
    second = derive_multistart_seeds(123457, 10)
    assert first["init_seeds"] != second["init_seeds"]
    assert first["schedule_seed"] != second["schedule_seed"]
    assert first["post_egm_seed"] != second["post_egm_seed"]
    assert first["criterion_seeds"] != second["criterion_seeds"]


def test_seed_derivation_matches_pinned_values():
    assert {
        stream: derive_seed(
            123456, stream, 0 if stream in ("egm-init", "egm-criterion") else None
        )
        for stream in (
            "egm-init",
            "egm-schedule",
            "post-egm",
            "egm-criterion",
            "mcmc-pilot",
            "mcmc-production",
        )
    } == {
        "egm-init": 995929169,
        "egm-schedule": 805669775,
        "post-egm": 325600991,
        "egm-criterion": 1722722885,
        "mcmc-pilot": 255668837,
        "mcmc-production": 133642074,
    }
    assert derive_multistart_seeds(123456, 2) == {
        "model_seed": 123456,
        "init_seeds": [995929169, 1144069070],
        "schedule_seed": 805669775,
        "post_egm_seed": 325600991,
        "criterion_seeds": [1722722885, 1515719907],
    }


def test_candidate_streams_are_a_prefix_of_a_larger_multistart():
    small = derive_multistart_seeds(99, 3)
    large = derive_multistart_seeds(99, 10)
    assert large["init_seeds"][:3] == small["init_seeds"]
    assert large["criterion_seeds"][:3] == small["criterion_seeds"]
    assert large["schedule_seed"] == small["schedule_seed"]


def test_named_streams_match_the_multistart_seeds_and_differ_from_mcmc():
    seeds = derive_multistart_seeds(42, 2)
    assert seeds["init_seeds"] == [derive_seed(42, "egm-init", i) for i in range(2)]
    assert seeds["schedule_seed"] == derive_seed(42, "egm-schedule")
    assert seeds["post_egm_seed"] == derive_seed(42, "post-egm")
    assert seeds["criterion_seeds"] == [
        derive_seed(42, "egm-criterion", i) for i in range(2)
    ]
    mcmc = {derive_seed(42, "mcmc-pilot"), derive_seed(42, "mcmc-production")}
    assert len(mcmc) == 2
    assert not mcmc & {seeds["schedule_seed"], seeds["post_egm_seed"]}


def test_per_candidate_seeds_require_candidate_and_shared_streams_forbid_it():
    for stream in ("egm-init", "egm-criterion"):
        with pytest.raises(ValueError, match="candidate_id is required"):
            derive_seed(5, stream)
    for stream in ("egm-schedule", "post-egm", "mcmc-pilot", "mcmc-production"):
        with pytest.raises(ValueError, match="only valid"):
            derive_seed(5, stream, 0)
    with pytest.raises(ValueError, match="unknown seed stream"):
        derive_seed(5, "holdout")


def test_ranking_is_stable_and_excludes_nonfinite_scores():
    ranking = rank_finite_candidates(
        {8: math.nan, 4: 0.04, 2: 0.04, 7: math.inf, 1: 0.05}
    )
    assert [record["candidate_id"] for record in ranking] == [2, 4, 1]
    assert [record["rank"] for record in ranking] == [1, 2, 3]


def test_ranking_fails_closed_without_a_finite_candidate():
    with pytest.raises(CandidateSelectionError, match="at least 1"):
        rank_finite_candidates({0: math.nan, 1: None, 2: math.inf})


def test_selection_is_argmin_of_post_bgm_training_criterion():
    criteria = {
        0: 190.0,
        1: 161.6,
        2: 175.0,
        3: math.nan,
        4: 161.6,
    }
    first = select_candidate_by_criterion(criteria)
    assert first == select_candidate_by_criterion(criteria)
    assert first["selected_candidate_id"] == 1
    assert first["selected_rank"] == 1
    assert first["selected_criterion"] == pytest.approx(161.6)
    assert first["selection_criterion"] == EGM_SELECTION_CRITERION
    assert first["selector_version"] == EGM_SELECTOR_VERSION
    assert [record["candidate_id"] for record in first["finite_ranking"]] == [
        1, 4, 2, 0
    ]
    assert [record["finite"] for record in first["candidate_criteria"]] == [
        True, True, True, False, True
    ]
    json.dumps(first, allow_nan=False)


def test_selection_fails_closed_without_a_finite_candidate():
    assert select_candidate_by_criterion({0: 0.5, 1: 0.25})["selected_candidate_id"] == 1
    with pytest.raises(CandidateSelectionError, match="at least 1"):
        select_candidate_by_criterion({0: math.nan, 1: None, 2: math.inf})


def test_candidate_record_is_json_safe():
    payload = make_candidate_record(
        candidate_id=2,
        init_seed=11,
        criterion_seed=77,
        status="completed",
        bgm_checkpoint_path="candidate_02/bgm-final/ckpt-1",
        train_iv_map=161.6,
        train_mse_x=0.05,
        train_mse_y=0.02,
        train_mse_v=0.01,
        started_at="2026-08-30T00:00:00Z",
        finished_at="2026-08-30T00:01:00Z",
        egm_seconds=100.0,
        bgm_seconds=811.5,
        device_names=["/device:GPU:0"],
    )
    assert payload["criterion_seed"] == 77
    assert payload["train_iv_map"] == pytest.approx(161.6)
    assert payload["device_names"] == ["/device:GPU:0"]
    assert payload["bgm_seconds"] == pytest.approx(811.5)
    json.dumps(payload, allow_nan=False)


def test_candidate_record_serializes_nonfinite_values_as_null():
    payload = make_candidate_record(
        candidate_id=2,
        init_seed=11,
        criterion_seed=12,
        status="nonfinite_criterion",
        failure_reason="non-finite post-BGM training criterion",
        train_iv_map=math.inf,
        train_mse_y=math.nan,
    )
    assert payload["train_iv_map"] is None
    assert payload["train_mse_y"] is None
    assert payload["bgm_checkpoint_path"] is None
    json.dumps(payload, allow_nan=False)


def test_candidate_record_reports_nonfinite_diagnostics_separately():
    payload = make_candidate_record(
        candidate_id=0,
        init_seed=1,
        criterion_seed=2,
        status="completed",
        train_iv_map=10.0,
        train_mse_y=math.nan,
        diagnostics_finite=False,
    )
    assert payload["status"] == "completed"
    assert payload["train_iv_map"] == 10.0
    assert payload["train_mse_y"] is None
    assert payload["diagnostics_finite"] is False
