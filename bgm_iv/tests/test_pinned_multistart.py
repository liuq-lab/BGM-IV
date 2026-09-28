import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SETTINGS = [
    "n_samples=300",
    "rho=0.5",
    "model_seed=12345",
    "structural_methods=[map]",
    "fit_egm_n_iter=90",
    "fit_epochs=2",
    "fit_epochs_per_eval=1",
    "egm_num_warm_starts=2",
    "iv_mc_samples=50",
    "eval_mc_samples=50",
    "structural_map_steps=20",
    "use_gpu=false",
]

PINNED_TRAIN_IV_MAP = {0: 12129.801800659297, 1: 11452.703676857362}
PINNED_SELECTED_CANDIDATE = 1
PINNED_STRUCTURAL_MSE_MAP = 14972.8681640625


@pytest.mark.slow
def test_two_start_demand_cell_is_pinned(tmp_path):
    argv = ["main.py", "-c", "configs/Sim_Demand_Design_IV.yaml", "--repeat-id", "0"]
    for item in SETTINGS + [f"output_dir={tmp_path / 'out'}"]:
        argv += ["--set", item]
    dumps = tmp_path / "dumps"
    code = (
        "import sys; from pathlib import Path; import main; "
        f"main._demand_design_dumps_dir = lambda: Path({str(dumps)!r}); "
        f"sys.argv = {argv!r}; main.main()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=dict(os.environ, PYTHONHASHSEED="0", TF_CPP_MIN_LOG_LEVEL="3"),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    records = sorted(dumps.rglob("records/*.json"))
    assert len(records) == 1, records
    record = json.loads(records[0].read_text())

    multistart = record["provenance"]["egm_multistart"]
    criteria = {
        entry["candidate_id"]: entry["score"] for entry in multistart["egm_candidate_criteria"]
    }
    assert criteria == PINNED_TRAIN_IV_MAP
    assert multistart["egm_selected_candidate_id"] == PINNED_SELECTED_CANDIDATE
    assert record["final_results"]["map"] == PINNED_STRUCTURAL_MSE_MAP
