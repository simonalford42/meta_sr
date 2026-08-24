import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpo_pysr import load_reselection_source


def _write_run(tmp_path, trial_numbers):
    run_dir = tmp_path / "source_hpo"
    run_dir.mkdir()
    data = {
        "config": {"seed": 42, "n_runs": 3},
        "baseline": {"avg_score": 0.25, "score_vector": [0.25]},
        "trials": [
            {"trial_number": number, "params": {"x": number}, "avg_score": number / 10}
            for number in trial_numbers
        ],
        "final_candidates": [{"trial_number": 4}],
    }
    (run_dir / "run_data.json").write_text(json.dumps(data))
    return run_dir


def test_load_reselection_source_intervenes_at_trial_cutoff(tmp_path):
    run_dir = _write_run(tmp_path, range(5))

    trials, baseline, config = load_reselection_source(str(run_dir), 3)

    assert [trial["trial_number"] for trial in trials] == [0, 1, 2]
    assert baseline["avg_score"] == 0.25
    assert config == {"seed": 42, "n_runs": 3}


def test_load_reselection_source_rejects_missing_early_trial(tmp_path):
    run_dir = _write_run(tmp_path, [0, 2, 3])

    with pytest.raises(ValueError, match=r"missing trial\(s\) 1"):
        load_reselection_source(str(run_dir), 3)
