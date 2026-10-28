"""Paper Table 1: SRBench black-box R2 and GT recovery for the 90-second evaluations.

GT is reported on held-out test tasks only (srbench_all minus unsolvable, train and val);
pass --all-tasks for the 130-task version. Black-box R2 uses all 122 tasks, none of
which were used for evolution or HPO. MDLformer GT is computed from its released trials.

    python scripts/paper_table1.py [--all-tasks]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srbench_results_io as srio
from inspect_srbench_results import _black_box_summary
from srbench_official_results import OFFICIAL_TRAIN_SPLIT, OFFICIAL_VAL_SPLIT

RUNS = ROOT / "runs"
NINETY = "srbench_full_9-22_10seed-90s"
ROWS = [  # (model, training metric, run dir)
    ("PySR", "", f"pysr-base-{NINETY}"),
    ("Evolved PySR", "R2", f"pysr-r2-120458-{NINETY}"),
    ("Evolved PySR", "GT-R2", f"pysr-gtr2-941339-{NINETY}"),
    ("Evolved PySR", "GT", f"pysr-gt-709715-{NINETY}"),
    ("BasicSR", "", f"fullsr-base-{NINETY}"),
    ("Evolved BasicSR", "R2", f"fullsr-r2-150812-{NINETY}"),
    ("Evolved BasicSR", "GT-R2", f"fullsr-gtr2-229869-{NINETY}"),
    ("Evolved BasicSR", "GT", f"fullsr-gt-225437-{NINETY}"),
    ("HPO applied to PySR", "R2", f"hpo-r2-555205-{NINETY}"),
    ("HPO applied to PySR", "GT-R2", f"hpo-gtr2-555206-{NINETY}"),
    ("HPO applied to PySR", "GT", f"hpo-gt-555204-{NINETY}"),
    ("Autoresearch applied to PySR", "GT",
     "autoresearch-apr9-82ea4c88-srbench_full_9-25_10seed-90s-retry"),
]
MDLFORMER_TRIALS = ROOT / "figures/srbench_figure3_extended/mdlformer_trials.csv.gz"


def _names(path):
    return {line.strip() for line in (ROOT / path).read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def gt_rate(run_dir, tasks):
    manifest = srio.load_manifest(run_dir)
    keyed = srio.load_keyed_results(run_dir) or srio.build_keyed_results(run_dir, manifest)
    manifest, keyed = srio.standard_ground_truth_view(manifest, keyed)
    rows = [row for row in keyed.values() if row.get("present") and row.get("error") is None
            and row["dataset"] in tasks]
    return 100 * sum(bool(row["solved"]) for row in rows) / len(rows), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all-tasks", action="store_true", help="GT over all 130 tasks")
    args = parser.parse_args()
    canonical = _names("splits/srbench_all.txt") - set(srio.UNSOLVABLE_TASKS)
    tasks = canonical if args.all_tasks else (
        canonical - _names(OFFICIAL_TRAIN_SPLIT) - _names(OFFICIAL_VAL_SPLIT))

    table = []
    for model, metric, name in ROWS:
        run_dir = RUNS / name
        gt, n = gt_rate(run_dir, tasks)
        completed, expected, bb = _black_box_summary(run_dir, srio.load_manifest(run_dir))
        if completed != expected:
            print(f"warning: {name} black box {completed}/{expected}", file=sys.stderr)
        if n != len(tasks) * 40:
            print(f"note: {name} GT {n}/{len(tasks) * 40} successful trials", file=sys.stderr)
        table.append((model, metric, f"{bb:.3f}", f"{gt:.1f}"))
    mdl = pd.read_csv(MDLFORMER_TRIALS)
    mdl = mdl[mdl.dataset.isin(tasks)]
    table.append(("MDLFormer", "GT", "--", f"{100 * mdl.symbolic_solution.mean():.1f}"))

    print(f"% GT over {len(tasks)} tasks; black box over 122 tasks.")
    print(r"\begin{tabular}{llrr}")
    print(r"\toprule")
    print(r"Model & Training metric & Black-box avg.\ $R^2$ & GT recovery (\%) \\")
    print(r"\midrule")
    previous = None
    for model, metric, bb, gt in table:
        metric = {"R2": "$R^2$", "GT-R2": "GT-$R^2$"}.get(metric, metric)
        shown = model if model != previous else ""
        print(f"{shown} & {metric} & {bb} & {gt} \\\\")
        previous = model
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
