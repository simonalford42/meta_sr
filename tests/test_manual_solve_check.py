import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import manual_solve_check as scorer


def _make_run(tmp_path: Path, n_seeds: int = 2) -> Path:
    run_dir = tmp_path / "run"
    batch_dir = run_dir / "batch_000"
    results_dir = batch_dir / "results"
    results_dir.mkdir(parents=True)
    datasets = ["first_principles_hubble", "first_principles_kepler"]
    tasks = []
    for dataset in datasets:
        for run_index in range(n_seeds):
            tasks.append({
                "dataset_name": dataset,
                "seed": 10000,
                "run_index": run_index,
                "target_noise": 0,
            })
    (run_dir / "manifest.json").write_text(json.dumps({
        "datasets": datasets,
        "batches": [{"batch_dir": "batch_000"}],
    }))
    (batch_dir / "tasks.json").write_text(json.dumps(tasks))
    for index, task in enumerate(tasks):
        equation = "2.0*x0" if task["dataset_name"].endswith("hubble") else "x0*sqrt(x0)"
        (results_dir / f"task_{index:06d}.json").write_text(json.dumps({
            "pareto_frontier": [{"complexity": 3, "equation": equation}],
        }))
    return run_dir


def test_one_request_per_frontier_and_explicit_cache(tmp_path):
    items = scorer.load_review_items(_make_run(tmp_path))
    assert len(items) == 4
    assert len({item["custom_id"] for item in items}) == 4
    assert {item["seed"] for item in items} == {10000, 10001}

    request = scorer.build_request(items[0], scorer.DEFAULT_MODEL, "medium", 1000)
    body = request["body"]
    assert request["url"] == "/v1/responses"
    assert body["model"] == "gpt-5.6-terra"
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert body["text"]["format"]["strict"] is True


def test_cost_guard_is_conservative_and_below_default_for_fixture(tmp_path):
    items = scorer.load_review_items(_make_run(tmp_path))
    requests = [scorer.build_request(item, scorer.DEFAULT_MODEL, "medium", 1000) for item in items]
    estimate = scorer.estimate_cost_upper(requests, scorer.DEFAULT_MODEL, 1000)
    assert estimate["estimated_input_tokens"] > 0
    assert estimate["maximum_output_tokens"] == 4000
    assert 0 < estimate["maximum_cost_usd"] < 5


def test_batch_output_writes_per_frontier_and_aggregate(tmp_path):
    run_dir = _make_run(tmp_path, n_seeds=1)
    items = scorer.load_review_items(run_dir)
    lines = []
    for item in items:
        review = {
            "classification": "exact",
            "best_frontier_indices": [0],
            "matching_equation": item["frontier"][0]["equation"],
            "explanation": "Algebraically identical.",
        }
        lines.append(json.dumps({
            "custom_id": item["custom_id"],
            "response": {
                "status_code": 200,
                "request_id": "req_test",
                "body": {
                    "output": [{"content": [{"type": "output_text", "text": json.dumps(review)}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            },
        }))
    records = scorer.process_batch_output(
        run_dir,
        ("\n".join(lines) + "\n").encode(),
        {item["custom_id"]: item for item in items},
        scorer.DEFAULT_MODEL,
        "medium",
    )
    assert len(records) == 2
    aggregate = json.loads((run_dir / "manual_solve_check_results.json").read_text())
    assert aggregate["classification_counts"] == {"exact": 2}
    assert aggregate["n_reviews"] == 2
