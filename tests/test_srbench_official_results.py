import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srbench_official_results import build_official_columns


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_eval(
    run_dir: Path,
    *,
    mode: str,
    source: str | None,
    entries: list[bool],
    merged: bool = False,
) -> None:
    manifest = {
        "mode": mode,
        "backend": "pysr",
        "method_meta": {"source": source} if source else {},
        "max_evals": 1_000_000,
        "n_runs": 10,
        "datasets": ["task"],
        "noise_levels": [0.0],
        "batches": [],
        "evaluation_types": ["ground_truth"],
        "merge_run_frontiers": merged,
    }
    results = {
        f"task|{index}|0": {
            "dataset": "task",
            "noise": 0.0,
            "present": True,
            "error": None,
            "solved": solved,
            **({"n_searches": 10} if merged else {}),
        }
        for index, solved in enumerate(entries)
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "srbench_full_results.json", {"results": results})


def _column(columns: list[dict], label: str) -> dict:
    return next(column for column in columns if column["label"] == label)


def test_standard_baseline_excludes_newer_merged_portfolio(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_eval(
        runs / "standard", mode="baseline", source=None,
        entries=[True, False],
    )
    _write_eval(
        runs / "merged", mode="baseline", source=None,
        entries=[True], merged=True,
    )

    column = _column(build_official_columns(runs, tmp_path), "PySR baseline")

    assert column["gt_completed"] == 2
    assert column["gt_rate"] == 0.5
    assert column["gt_10_restarts_completed"] == 1
    assert column["gt_10_restarts_rate"] == 1.0
    assert column["eval_ids"] == "standard,merged"


def test_nested_merged_portfolio_joins_exact_training_source(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    source = runs / "123"
    _write_json(source / "run_data.json", {"config": {"fitness_metric": "gt"}})
    _write_eval(
        runs / "standard", mode="evolve", source="runs/123",
        entries=[False, False],
    )
    _write_eval(
        source / "merged", mode="evolve", source="runs/123",
        entries=[True], merged=True,
    )

    column = _column(build_official_columns(runs, tmp_path), "PySR++ GT")

    assert column["gt_completed"] == 2
    assert column["gt_rate"] == 0.0
    assert column["gt_10_restarts_completed"] == 1
    assert column["gt_10_restarts_rate"] == 1.0
    assert column["eval_ids"] == "standard,123/merged"
