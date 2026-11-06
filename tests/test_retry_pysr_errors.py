import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.retry_pysr_errors import archive_stale_results, failed_task_indices


def test_finds_only_missing_error_and_empty_frontier_tasks(tmp_path):
    batch = tmp_path / "eval_0000"
    results = batch / "results"
    results.mkdir(parents=True)
    (batch / "tasks.json").write_text(json.dumps([{}, {}, {}, {}]))
    (results / "task_000000.json").write_text(json.dumps({
        "error": None, "pareto_frontier": [{"equation": "x0"}],
    }))
    (results / "task_000001.json").write_text(json.dumps({
        "error": "failed", "pareto_frontier": None,
    }))
    (results / "task_000002.json").write_text(json.dumps({
        "error": None, "pareto_frontier": [],
    }))

    assert failed_task_indices(batch) == [1, 2, 3]


def test_archives_stale_results_instead_of_deleting_them(tmp_path):
    batch = tmp_path / "eval_0000"
    results = batch / "results"
    results.mkdir(parents=True)
    stale = results / "task_000001.json"
    stale.write_text('{"error": "failed"}')

    archive = archive_stale_results(batch, [1, 2])

    assert not stale.exists()
    assert (archive / stale.name).read_text() == '{"error": "failed"}'
