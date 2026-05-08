"""View execution traces from an evolve_pysr run, formatted as the prompt sees them.

Usage examples:
    python scripts/view_execution_trace.py runs/771876/traces/eval_0519
    python scripts/view_execution_trace.py runs/771876/traces/eval_0519 --dataset feynman_I_13_4
    python scripts/view_execution_trace.py runs/771876/traces/eval_0519 --run config000_run00
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_pysr import _load_execution_trace
from evolution_helpers import format_pareto_trace_for_task, load_task_formulas


_RUN_DIR_RE = re.compile(r"config\d+_run(\d+)$")


def iter_run_dirs(eval_dir: Path, run_filter: str | None):
    for run_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        if run_filter and run_dir.name != run_filter:
            continue
        yield run_dir


def iter_hof_csvs(run_dir: Path, dataset_filter: str | None):
    for csv_path in sorted(run_dir.glob("*_hof.csv")):
        dataset = csv_path.name[: -len("_hof.csv")]
        if dataset_filter and dataset_filter not in dataset:
            continue
        yield dataset, csv_path


def _run_index_from_dir(run_dir: Path) -> Optional[int]:
    m = _RUN_DIR_RE.match(run_dir.name)
    return int(m.group(1)) if m else None


def load_solve_lookup(eval_dir: Path) -> Dict[Tuple[str, int], bool]:
    """Build {(dataset_name, run_index): solved} from slurm_pysr/<eval>/results/.

    Sibling layout: runs/<id>/traces/eval_NNNN  ↔  runs/<id>/slurm_pysr/eval_NNNN/results/task_*.json
    Returns an empty dict if results directory is missing.
    """
    results_dir = eval_dir.parent.parent / "slurm_pysr" / eval_dir.name / "results"
    if not results_dir.is_dir():
        return {}
    out: Dict[Tuple[str, int], bool] = {}
    for task_path in results_dir.glob("task_*.json"):
        try:
            with open(task_path) as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        ds = r.get("dataset_name")
        ri = r.get("run_index")
        gt = r.get("gt_match_score")
        if ds is None or ri is None or gt is None:
            continue
        out[(ds, int(ri))] = float(gt) >= 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_dir", type=Path,
                    help="Path like runs/<id>/traces/eval_NNNN")
    ap.add_argument("--dataset", default=None,
                    help="Substring filter on dataset name")
    ap.add_argument("--run", default=None,
                    help="Exact configNNN_runMM subdir to view (default: all)")
    args = ap.parse_args()

    eval_dir = args.eval_dir.resolve()
    if not eval_dir.is_dir():
        ap.error(f"not a directory: {eval_dir}")

    # Collect all dataset names so we batch-load formulas once.
    datasets = set()
    for run_dir in iter_run_dirs(eval_dir, args.run):
        for dataset, _ in iter_hof_csvs(run_dir, args.dataset):
            datasets.add(dataset)
    formulas = load_task_formulas(sorted(datasets)) if datasets else {}

    # Cross-reference per-task gt_match_score from slurm_pysr/<eval>/results/.
    solve_lookup = load_solve_lookup(eval_dir)
    if not solve_lookup:
        print(f"[note: no slurm_pysr results dir found; solve status will be unknown]")

    for run_dir in iter_run_dirs(eval_dir, args.run):
        print(f"\n########## {run_dir.relative_to(eval_dir.parent.parent.parent)} ##########")
        run_index = _run_index_from_dir(run_dir)
        for dataset, csv_path in iter_hof_csvs(run_dir, args.dataset):
            milestones = _load_execution_trace([str(csv_path)])
            if not milestones:
                print(f"\n[no usable trace for {dataset}]")
                continue
            detail = {"execution_traces": [milestones]}
            solved: Optional[bool] = None
            if run_index is not None and (dataset, run_index) in solve_lookup:
                solved = solve_lookup[(dataset, run_index)]
            if solved is True:
                status = "SOLVED"
            elif solved is False:
                status = "UNSOLVED"
            else:
                status = "STATUS UNKNOWN"
            text = format_pareto_trace_for_task(detail, dataset, formulas.get(dataset, ""))
            if text:
                print()
                print(f">>> {status} <<<")
                print(text)


if __name__ == "__main__":
    main()
