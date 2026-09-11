"""Regression tests for the compact comparison tables."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srbench_tables as tables
from srbench_official_results import build_official_columns
from inspect_srbench_results import main


def test_tables_cli(capsys):
    with patch.object(sys, "argv", ["inspect_srbench_results.py", "--tables", "--runs-root", "example"]):
        with patch.object(tables, "build_tables", return_value="Table 1\nTable 2") as build:
            main()
    build.assert_called_once_with("example")
    assert capsys.readouterr().out == "Table 1\nTable 2\n"


def write_run(path, *, solved=True, incomplete=False, **extra):
    path.mkdir(parents=True)
    manifest = dict(mode="baseline", backend="pysr", max_evals=1000000,
                    datasets=["train", "test", "feynman_I_26_2"], n_runs=10,
                    seeds=list(range(10)), noise_levels=[0, .001, .01, .1], batches=[])
    manifest.update(extra)
    (path / "manifest.json").write_text(json.dumps(manifest))
    rows = {}
    for dataset in manifest["datasets"]:
        for seed in manifest["seeds"]:
            for noise in manifest["noise_levels"]:
                rows[f"{dataset}|{seed}|{noise:g}"] = dict(
                    dataset=dataset, seed=seed, noise=noise, present=True, error=None,
                    solved=bool(solved and dataset == "train"), test_r2=.9)
    if incomplete:
        rows["train|0|0"]["present"] = False
    (path / "srbench_full_results.json").write_text(json.dumps({"results": rows}))


def write_splits(root):
    splits = root / "splits"
    splits.mkdir()
    (splits / "srbench_all.txt").write_text("train\ntest\nfeynman_I_26_2\n")
    (splits / "barely_unsolvable.txt").write_text("train\n")
    (splits / "barely_unsolvable_val2.txt").write_text("val\n")


def test_tables_group_order_subsets_budgets_and_missing(tmp_path):
    write_splits(tmp_path)
    runs = tmp_path / "runs"
    base, evolved = runs / "base", runs / "evolved"
    write_run(base)
    write_run(evolved)
    write_run(runs / "srbench_gt_baseline_90s", solved=False)
    write_run(runs / "709715/srbench_gt_90s", incomplete=True)
    columns = [dict(key="pysr_baseline", gt_path=str(base), bb_r2=.75,
                    bb_completed=1220),
               dict(key="pysrpp_gt", gt_path=str(evolved), training_id="709715")]
    with patch.object(tables, "build_official_columns", return_value=columns):
        output = tables.build_tables(runs, tmp_path)
    header = next(line for line in output.splitlines() if "Autoresearch" in line)
    assert [header.index(label) for label in ["PySR", "PySR++", "BasicSR", "BasicSR++", "HPO", "Autoresearch", "MDLFormer"]] == sorted(
        header.index(label) for label in ["PySR", "PySR++", "BasicSR", "BasicSR++", "HPO", "Autoresearch", "MDLFormer"])
    assert output.count("GT-R2") == 3
    assert "0.750" in output and "50.00%" in output
    # The supplied MDLFormer reference must not depend on local task counts.
    assert "40.50%" in output
    assert "without task-count rescaling" in output
    table2 = output.split("Table 2:")[1]
    train = next(line for line in table2.splitlines() if "Train tasks" in line)
    rest = next(line for line in table2.splitlines() if "Excluding train" in line)
    assert "n=1" in train and train.count("100.00%") == 2
    assert "n=1" in rest and rest.count("0.00%") == 2
    timed = next(line for line in table2.splitlines() if "90 seconds" in line)
    assert "0.00%" in timed and "TBD" in timed
    assert next(line for line in table2.splitlines() if "15 minutes" in line).count("TBD") == 2


def test_table1_does_not_replace_1m_with_newer_portfolio(tmp_path):
    write_splits(tmp_path)
    runs = tmp_path / "runs"
    base, portfolio = runs / "base", runs / "portfolio"
    write_run(base)
    write_run(portfolio, serial_restart_portfolio={"total_search_budget_seconds": 900})
    os.utime(base / "manifest.json", (1000, 1000))
    os.utime(portfolio / "manifest.json", (2000, 2000))
    ordinary = build_official_columns(runs, tmp_path)
    fixed = build_official_columns(runs, tmp_path, single_search_only=True)
    assert next(c for c in ordinary if c["key"] == "pysr_baseline")["gt_path"] == str(portfolio)
    assert next(c for c in fixed if c["key"] == "pysr_baseline")["gt_path"] == str(base)


def test_no_merged_portfolio_used_as_individual_searches(tmp_path):
    write_run(tmp_path / "run")
    path = tmp_path / "run/srbench_full_results.json"
    payload = json.loads(path.read_text())
    next(iter(payload["results"].values()))["n_searches"] = 10
    path.write_text(json.dumps(payload))
    assert tables._complete_gt(tmp_path / "run", {"train", "test"}) is None
