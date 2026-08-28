"""Resolution semantics of the outcome->particle coupling gamma.

The particle objective is loss_pv_z + loss_prior_z + loss_px_z +
gamma * loss_py_z; gamma=0 is the severed C-form, gamma=1 the joint
objective.  Explicit `outcome_to_particles_weight` wins over the
boolean `stop_outcome_to_particles` alias.
"""

import pytest

from bgm_iv.models.bgm_iv.instrument import BGM_IV


def _resolve(extra):
    params = dict(extra)
    return BGM_IV._outcome_to_particles_weight(
        type("P", (), {"params": params})()
    )


def test_default_is_joint_objective():
    assert _resolve({}) == 1.0


def test_boolean_alias_maps_to_endpoints():
    assert _resolve({"stop_outcome_to_particles": True}) == 0.0
    assert _resolve({"stop_outcome_to_particles": False}) == 1.0


def test_explicit_gamma_wins_over_boolean():
    assert _resolve(
        {"stop_outcome_to_particles": True, "outcome_to_particles_weight": 0.5}
    ) == 0.5
    assert _resolve(
        {"stop_outcome_to_particles": False, "outcome_to_particles_weight": 0.0}
    ) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_gamma_outside_unit_interval_raises(bad):
    with pytest.raises(ValueError, match="0, 1"):
        _resolve({"outcome_to_particles_weight": bad})
