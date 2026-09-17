"""Check hard timeouts and resumable snapshot shards without submitting jobs."""
import json
import signal
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import compare_srbench_snapshots as scorer


def _stuck_checker(*args):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)


def test_bounded_check_matches():
    assert scorer.bounded_check("x0*x1", "x0*x1", ["x0", "x1"])["match"]
    assert not scorer.bounded_check("x0", "x0*x1", ["x0", "x1"])["match"]


def test_hard_timeout_kills_stuck_checker():
    start = time.monotonic()
    with patch.object(scorer, "_check_child", _stuck_checker):
        result = scorer.bounded_check("x0", "x0", ["x0"], wall_seconds=.2)
    assert result == {"match": False, "error": "hard wall timeout"}
    assert time.monotonic() - start < 2


def test_shards_resume_and_keep_noise(tmp_path):
    out = tmp_path / "datasets"
    out.mkdir()
    rows = [dict(method="baseline", dataset="feynman_I_12_1", seed=10000+i, noise=.01,
                 trace=[dict(status="ok", scheduled_seconds=20, elapsed_seconds=20.1,
                             equations=[{"equation": "x0*x1"}])]) for i in range(4)]
    item = ("feynman_I_12_1", rows, {"x0*x1": {"match": True, "error": None}}, str(out))
    first = scorer.score_dataset(item, 0, 2)
    second = scorer.score_dataset(item, 1, 2)
    assert [r["seed"] for r in first["records"]] == [10000, 10002]
    assert [r["seed"] for r in second["records"]] == [10001, 10003]
    assert all(r["first_scheduled"] == 20 and r["noise"] == .01 for r in first["records"])
    result_path = tmp_path / "shards/feynman_I_12_1.000.json"
    result_path.rename(tmp_path / "saved_output.json")
    progress_path = tmp_path / "progress/feynman_I_12_1.000.json"
    saved = json.loads(progress_path.read_text())
    saved["records"] = saved["records"][:1]
    progress_path.write_text(json.dumps(saved))
    assert scorer.score_dataset(item, 0, 2)["records"] == first["records"]


if __name__ == "__main__":
    test_bounded_check_matches()
    test_hard_timeout_kills_stuck_checker()
    with tempfile.TemporaryDirectory(prefix="srbench_snapshot_test_") as directory:
        test_shards_resume_and_keep_noise(Path(directory))
    print("All three snapshot scorer checks passed")
