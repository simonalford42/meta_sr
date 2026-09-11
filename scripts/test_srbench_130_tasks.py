"""Regression coverage for the canonical SRBench 2021 task universe."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srbench_results_io as srio
from srbench_full_eval import load_evaluation_datasets
from inspect_srbench_results import inspect_run, summarize_run
from srbench_official_results import _ground_truth_stats, _split_performance


def args(**overrides):
    values = dict(datasets=None, srbench_2025=False, ground_truth=True,
                  black_box=False, split_file="splits/srbench_all.txt")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_and_explicit_evaluation_grids():
    tasks = load_evaluation_datasets(args())
    assert len(tasks) == 130
    assert sum(name.startswith("feynman") for name in tasks) == 116
    assert not set(tasks) & set(srio.UNSOLVABLE_TASKS)
    explicit = "feynman_I_26_2,strogatz_barmag1"
    assert load_evaluation_datasets(args(datasets=explicit)) == ["strogatz_barmag1"]
    assert len(load_evaluation_datasets(args(datasets=explicit, srbench_2025=True))) == 2
    assert len(load_evaluation_datasets(args(datasets=explicit, ground_truth=False, black_box=True))) == 2
    with pytest.raises(ValueError, match="No ground-truth tasks remain"):
        load_evaluation_datasets(args(datasets="feynman_I_26_2"))


@pytest.mark.parametrize("merged", [False, True])
def test_historical_results_excluded_from_scores_and_completion(tmp_path, merged):
    manifest = dict(datasets=["feynman_keep", *srio.UNSOLVABLE_TASKS],
                    n_datasets=4, seeds=[42, 43], n_runs=2, noise_levels=[0],
                    batches=[], merge_run_frontiers=merged)
    rows = {}
    for dataset in manifest["datasets"]:
        for seed in ([42] if merged else manifest["seeds"]):
            rows[srio.result_key(dataset, seed, 0)] = dict(
                dataset=dataset, seed=seed, noise=0, family="Feynman",
                present=dataset == "feynman_keep", error=None,
                solved=dataset == "feynman_keep", test_r2=1,
                runtime_seconds=1, solve_time=1,
                **({"n_searches": 2} if merged else {}))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "srbench_full_results.json").write_text(json.dumps({"results": rows}))
    row = summarize_run(tmp_path)
    expected = 1 if merged else 2
    assert row["complete"] and row["completed"] == row["total"] == expected
    assert row["all_pct"] == 1
    output = io.StringIO()
    with redirect_stdout(output):
        inspect_run(tmp_path, SimpleNamespace(show_missing=50, wandb=False))
    assert "STATS (1 tasks" in output.getvalue()
    assert "missing: 0" in output.getvalue()
    assert _ground_truth_stats(tmp_path, manifest) == (expected, 1, 1)
    # Completed inverse-trig failures must also be excluded from official test scores.
    for entry in rows.values():
        entry["present"] = True
    (tmp_path / "srbench_full_results.json").write_text(json.dumps({"results": rows}))
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "barely_unsolvable.txt").write_text("train\n")
    (splits / "barely_unsolvable_val2.txt").write_text("val\n")
    assert _split_performance(tmp_path, manifest, "gt", tmp_path) == (None, None, 1)
    filtered, _ = srio.standard_ground_truth_view(manifest, rows)
    assert filtered["n_datasets"] == 1 and manifest["n_datasets"] == 4
    assert srio.standard_ground_truth_view({**manifest, "srbench_edition": 2025}, rows)[1] == rows
