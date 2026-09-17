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


def test_binary_search():
    for threshold in range(10):
        checked = []
        def check(index):
            checked.append(index)
            return "match" if index >= threshold else None
        result = scorer.first_matching_snapshot(list(range(9)), check, True)
        assert result == ((threshold, "match") if threshold < 9 else None)
        assert len(checked) <= 5
    # The accepted approximation intentionally misses transient early matches.
    assert scorer.first_matching_snapshot(list(range(9)), lambda i: "match" if i == 2 else None, True) is None


def test_binary_final_gate(tmp_path):
    out = tmp_path / "datasets"
    out.mkdir()
    row = dict(method="baseline", dataset="feynman_I_12_1", seed=10000, noise=0,
               final_solved=False, trace=[dict(status="ok", scheduled_seconds=20, elapsed_seconds=20.1,
                                             equations=[{"equation": "x0*x1"}])])
    payload = scorer.score_dataset((row['dataset'], [row], {}, str(out)), binary_search=True)
    result = payload["records"][0]
    assert result["checked_snapshots"] == [] and result["first_scheduled"] is None
    assert result["scoring_method"] == "binary_search_final_gate"


if __name__ == "__main__":
    test_bounded_check_matches()
    test_hard_timeout_kills_stuck_checker()
    test_binary_search()
    with tempfile.TemporaryDirectory(prefix="srbench_snapshot_test_") as directory:
        test_shards_resume_and_keep_noise(Path(directory))
    with tempfile.TemporaryDirectory(prefix="srbench_snapshot_test_") as directory:
        test_binary_final_gate(Path(directory))
    print("All five snapshot scorer checks passed")
