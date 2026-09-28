from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from bgm_iv.mcmc.inference import run_mcmc_grid
from bgm_iv.tests.mcmc.test_inference import (
    SEEDS,
    SMOKE_COUNTS,
    _identity,
    _model_and_grid,
    smoke,
)

_LAST_DRAW_TARGET0 = [
    [-0.18184131383895874, 0.019841007888317108, 0.4470210373401642, -0.5692297220230103],
    [0.5999626517295837, -1.2519192695617676, 0.7183522582054138, -1.4696780443191528],
    [-0.9564603567123413, 0.8781145811080933, -0.1165241077542305, 0.06633785367012024],
    [-0.5118927955627441, -0.13726051151752472, -0.181794673204422, -0.6853287816047668],
]
_FIRST_DRAW_CHAIN0 = [
    [-0.8206112384796143, -0.39387500286102295, 0.5168888568878174, 1.0809125900268555],
    [-0.7850319743156433, -0.047045860439538956, -0.4903753101825714, -0.5389809608459473],
]


@pytest.mark.slow
def test_tiny_demand_run_reproduces_pinned_draws(tmp_path, smoke):
    model, grid_x, grid_v, truth = _model_and_grid("demand")
    result = run_mcmc_grid(
        model,
        family="demand",
        grid_x_model=grid_x,
        grid_v_raw=grid_v,
        preprocessor=_identity(grid_v.shape[1]),
        truth_original_units=truth,
        truth_label="pinned",
        outcome_shift=0.0,
        outcome_scale=1.0,
        treatment_transform={"shift": 0.0, "scale": 1.0},
        seeds=SEEDS,
        **SMOKE_COUNTS,
        artifact_root=tmp_path,
        arm_id="pinned",
        progress=None,
    )
    draws = np.load(result["artifact"]["draws_path"], allow_pickle=False)
    assert draws.shape == (6, 4, 2, 4)
    np.testing.assert_allclose(draws[-1, :, 0, :], _LAST_DRAW_TARGET0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(draws[0, 0], _FIRST_DRAW_CHAIN0, rtol=1e-6, atol=1e-6)
    assert result["pilot"]["raw_variance_median"] == pytest.approx(
        0.8013190031051636, rel=1e-5
    )
    assert result["pilot"]["regularized_variance_median"] == pytest.approx(
        0.8051695823669434, rel=1e-5
    )
    assert result["sampler"]["acceptance"]["per_chain"] == pytest.approx(
        [1.0, 1.0, 1.0, 0.9166666666666667]
    )
    assert result["readout"]["coverage"] == pytest.approx(
        {"0.9": 0.5341126768403482, "0.95": 0.6812557304819301, "0.99": 0.9144406837274519},
        rel=1e-5,
    )
    assert [result["readout"][key] for key in ("width90", "width95", "width99")] == (
        pytest.approx([3.5011927359725328, 4.238650522700785, 5.809887811232481], rel=1e-5)
    )
