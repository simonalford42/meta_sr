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


def test_one_request_per_frontier_uses_openrouter_chat_schema(tmp_path):
    items = scorer.load_review_items(_make_run(tmp_path))
    assert len(items) == 4
    assert len({item["custom_id"] for item in items}) == 4
    assert {item["seed"] for item in items} == {10000, 10001}

    request = scorer.build_request(items[0], scorer.DEFAULT_MODEL, "medium", 1000)
    body = request["body"]
    assert set(request) == {"custom_id", "body"}
    assert body["model"] == "openai/gpt-5.6-terra"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}


def test_loads_empiricalbench_result_frontiers(tmp_path):
    run_dir = tmp_path / "empbench"
    run_dir.mkdir()
    (run_dir / "empbench_results.json").write_text(json.dumps({
        "protocol": {"datasets": ["empirical_hubble", "empirical_planck"]},
        "runs": [
            {
                "dataset": "empirical_hubble",
                "seed": 10000,
                "error": None,
                "frontier": [{"complexity": 3, "equation": "2*x0"}],
            },
            {
                "dataset": "empirical_planck",
                "seed": 10001,
                "error": "failed",
                "frontier": [],
            },
        ],
    }))

    items = scorer.load_review_items(run_dir)

    assert len(items) == 1
    assert items[0]["dataset"] == "empirical_hubble"
    assert items[0]["seed"] == 10000
    assert items[0]["custom_id"] == "e-t000000"


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
                    "id": "gen_test",
                    "choices": [{"message": {"content": json.dumps(review)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
            },
        }))
    records = scorer.process_batch_results(
        run_dir,
        [json.loads(line) for line in lines],
        {item["custom_id"]: item for item in items},
        scorer.DEFAULT_MODEL,
        "medium",
    )
    assert len(records) == 2
    aggregate = json.loads((run_dir / "manual_solve_check_results.json").read_text())
    assert aggregate["classification_counts"] == {"exact": 2}
    assert aggregate["n_reviews"] == 2


def test_run_submits_openrouter_batch_and_reads_inline_results(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path, n_seeds=1)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class FakeClient:
        requests = None

        def __init__(self, api_key, base_url):
            assert api_key == "test-key"
            assert base_url == "https://openrouter.ai/api"

        def json_request(self, method, path, payload=None):
            if method == "POST":
                assert path == "/beta/batches"
                assert payload["endpoint"] == "/v1/chat/completions"
                assert payload["model"] == scorer.DEFAULT_MODEL
                self.requests = payload["requests"]
                return {"id": "batch_test", "status": "validating"}
            assert method == "GET"
            assert path == "/beta/batches/batch_test"
            results = []
            for request in self.requests:
                review = {
                    "classification": "exact",
                    "best_frontier_indices": [0],
                    "matching_equation": "test equation",
                    "explanation": "Algebraically identical.",
                }
                results.append({
                    "custom_id": request["custom_id"],
                    "response": {
                        "status": 200,
                        "body": {
                            "id": "gen_test",
                            "choices": [{"message": {"content": json.dumps(review)}}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                        },
                    },
                })
            return {
                "id": "batch_test",
                "status": "completed",
                "request_counts": {"total": len(results), "completed": len(results), "failed": 0},
                "results": results,
            }

    monkeypatch.setattr(scorer, "OpenRouterHTTPClient", FakeClient)
    args = scorer.build_parser().parse_args([str(run_dir)])
    assert scorer.run(args) == 0
    state = json.loads((run_dir / "manual_solve_check" / "batch_state.json").read_text())
    assert state["provider"] == "openrouter"
    assert state["batch_id"] == "batch_test"
    assert (run_dir / "manual_solve_check" / "responses.json").exists()
