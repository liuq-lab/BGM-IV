import json
import math

import pytest

from bgm_iv.egm_multistart import (
    CandidateSelectionError,
    EGM_CANDIDATE_MANIFEST_VERSION,
    EGM_INIT_SEED_NAMESPACE,
    EGM_SCHEDULE_SEED_NAMESPACE,
    EGM_SCORE_WINDOW_SIZE,
    EGM_SELECTION_MANIFEST_VERSION,
    EGM_SELECTOR_DRAW_NAMESPACE,
    EGM_SELECTOR_TEMPERATURE,
    EGM_SELECTOR_VERSION,
    MultistartConfigurationError,
    derive_multistart_seed,
    derive_multistart_seeds,
    make_candidate_manifest,
    rank_finite_candidates,
    relative_loss_softmax,
    score_evaluation_iterations,
    select_egm_candidate,
    selector_uniform_draw,
    validate_multistart_config,
    verify_manifest_hash,
)


def _multistart_params(**overrides):
    params = {
        "egm_num_warm_starts": 10,
        "egm_selection_top_k": 3,
        "save_model": True,
        "deterministic_training": True,
        "training_grid_monitor": False,
    }
    params.update(overrides)
    return params


def test_config_defaults_preserve_single_start():
    normalized = validate_multistart_config({})
    assert normalized["egm_num_warm_starts"] == 1
    assert normalized["egm_selection_top_k"] == 1


def test_config_accepts_ms10_top3_without_mutating_input():
    params = _multistart_params()
    normalized = validate_multistart_config(params)
    assert normalized == params
    assert normalized is not params


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"egm_num_warm_starts": 0}, "egm_num_warm_starts"),
        ({"egm_selection_top_k": 11}, "egm_selection_top_k"),
        ({"save_model": False}, "save_model"),
        ({"deterministic_training": False}, "deterministic_training"),
        ({"training_grid_monitor": True}, "training_grid_monitor"),
    ],
)
def test_config_rejects_invalid_multistart(overrides, message):
    with pytest.raises(MultistartConfigurationError, match=message):
        validate_multistart_config(_multistart_params(**overrides))


def test_config_keeps_multistart_identity_for_mcmc_only_restore():
    normalized = validate_multistart_config(
        _multistart_params(), mcmc_only=True
    )
    assert normalized["egm_num_warm_starts"] == 10
    assert normalized["egm_selection_top_k"] == 3


def test_score_window_is_last_ten_fixed_evaluation_events():
    assert score_evaluation_iterations(50_000, 500) == tuple(
        range(45_500, 50_001, 500)
    )
    assert score_evaluation_iterations(1_000, 100) == tuple(
        range(100, 1_001, 100)
    )
    assert len(score_evaluation_iterations(1_000, 100)) == EGM_SCORE_WINDOW_SIZE


def test_score_window_fails_instead_of_silently_shortening():
    with pytest.raises(MultistartConfigurationError, match="too small"):
        score_evaluation_iterations(800, 100)


def test_seed_derivation_is_reproducible_and_namespaced():
    seeds = derive_multistart_seeds(
        "vector", 5_000, 0.5, 7, 10, run_seed=107
    )
    assert seeds == derive_multistart_seeds(
        "vector", 5_000, 0.5, 7, 10, run_seed=107
    )
    assert len(seeds["init_seeds"]) == 10
    assert len(set(seeds["init_seeds"])) == 10
    assert all(0 < seed < 2**31 - 1 for seed in seeds["init_seeds"])
    assert len(
        {
            *seeds["init_seeds"],
            seeds["schedule_seed"],
            seeds["selector_seed"],
            seeds["post_egm_seed"],
        }
    ) == 13
    assert seeds != derive_multistart_seeds(
        "vector", 5_000, 0.5, 8, 10, run_seed=108
    )


def test_seed_derivation_changes_every_stream_with_master_run_seed():
    first = derive_multistart_seeds(
        "vector", 5_000, 0.5, 7, 10, run_seed=107
    )
    second = derive_multistart_seeds(
        "vector", 5_000, 0.5, 7, 10, run_seed=108
    )
    assert first["init_seeds"] != second["init_seeds"]
    assert first["schedule_seed"] != second["schedule_seed"]
    assert first["selector_seed"] != second["selector_seed"]
    assert first["post_egm_seed"] != second["post_egm_seed"]


def test_multistart_contract_names_are_versionless():
    assert EGM_SELECTOR_VERSION == "relative-loss-softmax"
    assert EGM_CANDIDATE_MANIFEST_VERSION == "egm-candidate-manifest"
    assert EGM_SELECTION_MANIFEST_VERSION == "egm-selection-manifest"
    assert EGM_INIT_SEED_NAMESPACE == "egm-init"
    assert EGM_SCHEDULE_SEED_NAMESPACE == "egm-schedule"
    assert EGM_SELECTOR_DRAW_NAMESPACE == "selector-draw"


def test_init_seed_requires_candidate_and_other_streams_forbid_it():
    with pytest.raises(ValueError, match="candidate_id is required"):
        derive_multistart_seed(
            "vector",
            5_000,
            0.5,
            0,
            EGM_INIT_SEED_NAMESPACE,
            run_seed=0,
        )
    with pytest.raises(ValueError, match="only valid"):
        derive_multistart_seed(
            "vector",
            5_000,
            0.5,
            0,
            EGM_SCHEDULE_SEED_NAMESPACE,
            run_seed=0,
            candidate_id=0,
        )


def test_ranking_is_stable_and_excludes_nonfinite_scores():
    ranking = rank_finite_candidates(
        {8: math.nan, 4: 0.04, 2: 0.04, 7: math.inf, 1: 0.05},
        top_k=3,
    )
    assert [record["candidate_id"] for record in ranking] == [2, 4, 1]
    assert [record["rank"] for record in ranking] == [1, 2, 3]


def test_ranking_fails_closed_when_top_k_finite_candidates_are_unavailable():
    with pytest.raises(CandidateSelectionError, match="at least 3"):
        rank_finite_candidates(
            {0: 0.04, 1: math.nan, 2: math.inf}, top_k=3
        )


def test_relative_loss_softmax_matches_documented_example():
    probabilities = relative_loss_softmax([0.040, 0.041, 0.044])
    assert EGM_SELECTOR_TEMPERATURE == 0.05
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities == pytest.approx(
        (0.57409699, 0.34820743, 0.07769558), rel=1e-6
    )


def test_relative_loss_softmax_is_nearly_uniform_for_near_ties():
    probabilities = relative_loss_softmax([0.0400, 0.0402, 0.0404])
    assert probabilities == pytest.approx(
        (0.3671654, 0.3322250, 0.3006096), rel=1e-5
    )


def test_selector_is_reproducible_and_records_inverse_cdf_draw():
    scores = {
        0: 0.040,
        1: 0.041,
        2: 0.044,
        3: 0.20,
        4: math.nan,
    }
    first = select_egm_candidate(scores, top_k=3, selector_seed=1234)
    second = select_egm_candidate(scores, top_k=3, selector_seed=1234)
    assert first == second
    assert first["uniform_draw"] == selector_uniform_draw(1234)
    assert first["selected_candidate_id"] in {0, 1, 2}
    selected = next(
        record
        for record in first["top_k_candidates"]
        if record["candidate_id"] == first["selected_candidate_id"]
    )
    lower = 0.0 if selected["rank"] == 1 else first["top_k_candidates"][
        selected["rank"] - 2
    ]["cdf_upper"]
    assert lower <= first["uniform_draw"] < selected["cdf_upper"]
    assert first["selected_probability"] == selected["probability"]
    assert not first["uses_validation"]
    assert not first["uses_holdout"]
    assert not first["uses_test_grid"]
    assert verify_manifest_hash(
        first,
        hash_field="selection_manifest_hash",
        namespace="egm-selection-manifest",
    )
    json.dumps(first, allow_nan=False)


def test_selector_hash_detects_tampering():
    payload = select_egm_candidate(
        {0: 0.04, 1: 0.041, 2: 0.044},
        top_k=3,
        selector_seed=99,
    )
    payload["selected_candidate_id"] = 9
    assert not verify_manifest_hash(
        payload,
        hash_field="selection_manifest_hash",
        namespace="egm-selection-manifest",
    )


def test_candidate_manifest_is_json_safe_self_hashed_and_uses_tail_mean():
    iterations = score_evaluation_iterations(1_000, 100)
    scores = [0.05 - index * 0.001 for index in range(10)]
    payload = make_candidate_manifest(
        candidate_id=2,
        init_seed=11,
        schedule_seed=12,
        run_seed=10,
        evaluation_iterations=iterations,
        full_train_l2_loss_y=scores,
        status="completed",
        data_hash="d" * 64,
        config_hash="c" * 64,
        code_commit="a" * 40,
        checkpoint_path="candidate_02/egm_terminal",
        checkpoint_hash="b" * 64,
        checkpoint_weight_hash="e" * 64,
        started_at="2026-08-30T00:00:00Z",
        finished_at="2026-08-30T00:01:00Z",
        worker_pid=1234,
        device_names=["/device:GPU:0"],
        device_hash="f" * 64,
    )
    assert payload["tail_mean_score"] == pytest.approx(sum(scores) / 10)
    assert payload["run_seed"] == 10
    assert payload["worker_pid"] == 1234
    assert payload["device_names"] == ["/device:GPU:0"]
    assert payload["device_hash"] == "f" * 64
    assert payload["checkpoint_weight_hash"] == "e" * 64
    assert verify_manifest_hash(
        payload,
        hash_field="candidate_manifest_hash",
        namespace="egm-candidate-manifest",
    )
    json.dumps(payload, allow_nan=False)


def test_candidate_manifest_serializes_nonfinite_loss_as_null():
    scores = [0.04] * 9 + [math.nan]
    payload = make_candidate_manifest(
        candidate_id=2,
        init_seed=11,
        schedule_seed=12,
        run_seed=10,
        evaluation_iterations=score_evaluation_iterations(1_000, 100),
        full_train_l2_loss_y=scores,
        status="nonfinite_score",
        failure_reason="terminal score is NaN",
        data_hash="d" * 64,
        config_hash="c" * 64,
        code_commit="a" * 40,
    )
    assert payload["full_train_l2_loss_y"][-1] is None
    assert payload["tail_mean_score"] is None
    json.dumps(payload, allow_nan=False)
