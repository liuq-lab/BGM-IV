import json
import math

import pytest

from bgm_iv.egm_multistart import (
    CandidateSelectionError,
    EGM_CANDIDATE_MANIFEST_VERSION,
    EGM_CRITERION_SEED_NAMESPACE,
    EGM_INIT_SEED_NAMESPACE,
    EGM_SCHEDULE_SEED_NAMESPACE,
    EGM_SCORE_WINDOW_SIZE,
    EGM_SELECTION_CRITERION,
    EGM_SELECTION_MANIFEST_VERSION,
    EGM_SELECTOR_VERSION,
    MultistartConfigurationError,
    POST_EGM_SEED_NAMESPACE,
    derive_multistart_seed,
    derive_multistart_seeds,
    make_candidate_manifest,
    rank_finite_candidates,
    score_evaluation_iterations,
    select_candidate_by_criterion,
    validate_multistart_config,
    verify_manifest_hash,
)


def _multistart_params(**overrides):
    params = {
        "egm_num_warm_starts": 10,
        "save_model": True,
        "deterministic_training": True,
        "training_grid_monitor": False,
    }
    params.update(overrides)
    return params


def test_config_defaults_preserve_single_start():
    normalized = validate_multistart_config({})
    assert normalized["egm_num_warm_starts"] == 1
    assert "egm_selection_top_k" not in normalized


def test_config_accepts_ten_starts_without_mutating_input():
    params = _multistart_params()
    normalized = validate_multistart_config(params)
    assert normalized == params
    assert normalized is not params


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"egm_num_warm_starts": 0}, "egm_num_warm_starts"),
        ({"egm_selection_top_k": 3}, "egm_selection_top_k"),
        ({"egm_selection_top_k": 1}, "no longer supported"),
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
    assert len(seeds["criterion_seeds"]) == 10
    assert all(0 < seed < 2**31 - 1 for seed in seeds["init_seeds"])
    assert all(0 < seed < 2**31 - 1 for seed in seeds["criterion_seeds"])
    assert len(
        {
            *seeds["init_seeds"],
            *seeds["criterion_seeds"],
            seeds["schedule_seed"],
            seeds["post_egm_seed"],
        }
    ) == 22
    assert seeds != derive_multistart_seeds(
        "vector", 5_000, 0.5, 8, 10, run_seed=108
    )


def test_seed_derivation_keeps_historical_streams_and_adds_criterion_seeds():
    # The init / schedule / post-EGM streams are the same functions of the cell
    # identity as before the selection rule changed, so candidate trajectories
    # remain comparable with earlier campaigns; only the criterion stream is new.
    common = ("vector", 5_000, 0.5, 7)
    seeds = derive_multistart_seeds(*common, 3, run_seed=107)
    assert seeds["init_seeds"] == [
        derive_multistart_seed(
            *common, EGM_INIT_SEED_NAMESPACE, run_seed=107, candidate_id=index
        )
        for index in range(3)
    ]
    assert seeds["schedule_seed"] == derive_multistart_seed(
        *common, EGM_SCHEDULE_SEED_NAMESPACE, run_seed=107
    )
    assert seeds["post_egm_seed"] == derive_multistart_seed(
        *common, POST_EGM_SEED_NAMESPACE, run_seed=107
    )
    assert seeds["criterion_seeds"] == [
        derive_multistart_seed(
            *common, EGM_CRITERION_SEED_NAMESPACE, run_seed=107, candidate_id=index
        )
        for index in range(3)
    ]
    assert "selector_seed" not in seeds


def test_seed_derivation_changes_every_stream_with_master_run_seed():
    first = derive_multistart_seeds(
        "vector", 5_000, 0.5, 7, 10, run_seed=107
    )
    second = derive_multistart_seeds(
        "vector", 5_000, 0.5, 7, 10, run_seed=108
    )
    assert first["init_seeds"] != second["init_seeds"]
    assert first["schedule_seed"] != second["schedule_seed"]
    assert first["post_egm_seed"] != second["post_egm_seed"]
    assert first["criterion_seeds"] != second["criterion_seeds"]


def test_multistart_contract_names_are_versionless():
    assert EGM_SELECTOR_VERSION == "train-iv-map-post-bgm"
    assert EGM_SELECTION_CRITERION == "train_iv_map"
    assert EGM_CANDIDATE_MANIFEST_VERSION == "egm-candidate-manifest"
    assert EGM_SELECTION_MANIFEST_VERSION == "egm-selection-manifest"
    assert EGM_INIT_SEED_NAMESPACE == "egm-init"
    assert EGM_SCHEDULE_SEED_NAMESPACE == "egm-schedule"
    assert POST_EGM_SEED_NAMESPACE == "post-egm"
    assert EGM_CRITERION_SEED_NAMESPACE == "egm-criterion"


def test_per_candidate_seeds_require_candidate_and_shared_streams_forbid_it():
    for namespace in (EGM_INIT_SEED_NAMESPACE, EGM_CRITERION_SEED_NAMESPACE):
        with pytest.raises(ValueError, match="candidate_id is required"):
            derive_multistart_seed(
                "vector",
                5_000,
                0.5,
                0,
                namespace,
                run_seed=0,
            )
    for namespace in (EGM_SCHEDULE_SEED_NAMESPACE, POST_EGM_SEED_NAMESPACE):
        with pytest.raises(ValueError, match="only valid"):
            derive_multistart_seed(
                "vector",
                5_000,
                0.5,
                0,
                namespace,
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


def test_selection_is_argmin_of_post_bgm_training_criterion():
    criteria = {
        0: 190.0,
        1: 161.6,
        2: 175.0,
        3: math.nan,
        4: 161.6,
    }
    egm_scores = {0: 0.040, 1: 0.044, 2: 0.041, 3: 0.039, 4: 0.045}
    first = select_candidate_by_criterion(criteria, egm_tail_scores=egm_scores)
    second = select_candidate_by_criterion(criteria, egm_tail_scores=egm_scores)
    assert first == second
    # Deterministic argmin with a candidate-id tie break; the EGM score does
    # not enter (candidate 3 has the best EGM score but a NaN criterion).
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
    assert first["selected_egm_tail_rank"] == 4
    assert [record["candidate_id"] for record in first["egm_tail_ranking"]] == [
        3, 0, 2, 1, 4
    ]
    assert not first["uses_validation"]
    assert not first["uses_holdout"]
    assert not first["uses_test_grid"]
    assert verify_manifest_hash(
        first,
        hash_field="selection_manifest_hash",
        namespace="egm-selection-manifest",
    )
    json.dumps(first, allow_nan=False)


def test_selection_works_without_egm_diagnostics_and_fails_closed():
    payload = select_candidate_by_criterion({0: 0.5, 1: 0.25})
    assert payload["selected_candidate_id"] == 1
    assert payload["egm_tail_scores"] == []
    assert payload["egm_tail_ranking"] == []
    assert payload["selected_egm_tail_rank"] is None
    with pytest.raises(CandidateSelectionError, match="at least 1"):
        select_candidate_by_criterion({0: math.nan, 1: None, 2: math.inf})
    with pytest.raises(ValueError, match="exactly the candidates"):
        select_candidate_by_criterion({0: 0.5, 1: 0.25}, egm_tail_scores={0: 0.1})


def test_selection_hash_detects_tampering():
    payload = select_candidate_by_criterion({0: 0.04, 1: 0.041, 2: 0.044})
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
        bgm_checkpoint_path="candidate_02/bgm-final/ckpt-1",
        bgm_checkpoint_hash="1" * 64,
        bgm_checkpoint_weight_hash="2" * 64,
        criterion_seed=77,
        train_iv_map=161.6,
        train_iv_encoder=154.4,
        train_mse_x=0.05,
        train_mse_y=0.02,
        train_mse_v=0.01,
        bgm_seconds=811.5,
    )
    assert payload["tail_mean_score"] == pytest.approx(sum(scores) / 10)
    assert payload["run_seed"] == 10
    assert payload["worker_pid"] == 1234
    assert payload["device_names"] == ["/device:GPU:0"]
    assert payload["device_hash"] == "f" * 64
    assert payload["checkpoint_weight_hash"] == "e" * 64
    assert payload["bgm_checkpoint_weight_hash"] == "2" * 64
    assert payload["criterion_seed"] == 77
    assert payload["train_iv_map"] == pytest.approx(161.6)
    assert payload["train_iv_encoder"] == pytest.approx(154.4)
    assert payload["bgm_seconds"] == pytest.approx(811.5)
    assert verify_manifest_hash(
        payload,
        hash_field="candidate_manifest_hash",
        namespace="egm-candidate-manifest",
    )
    json.dumps(payload, allow_nan=False)


def test_candidate_manifest_serializes_nonfinite_values_as_null():
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
        train_iv_map=math.inf,
    )
    assert payload["full_train_l2_loss_y"][-1] is None
    assert payload["tail_mean_score"] is None
    assert payload["train_iv_map"] is None
    assert payload["bgm_checkpoint_path"] is None
    json.dumps(payload, allow_nan=False)
