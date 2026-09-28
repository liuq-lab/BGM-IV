import pytest

from bgm_iv.models.bgm_iv.instrument import BGM_IV


def _resolve(extra):
    params = dict(extra)
    return BGM_IV._outcome_to_particles_weight(
        type("P", (), {"params": params})()
    )


def test_default_value():
    assert _resolve({}) == 0.01


def test_explicit_gamma_is_used():
    assert _resolve({"outcome_to_particles_weight": 0.5}) == 0.5
    assert _resolve({"outcome_to_particles_weight": 0.0}) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_gamma_outside_unit_interval_raises(bad):
    with pytest.raises(ValueError, match="0, 1"):
        _resolve({"outcome_to_particles_weight": bad})
