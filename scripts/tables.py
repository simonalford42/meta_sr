"""Regenerate the paper's SRBench tables: figures/table_srbench.tex and table_srbench2.tex.

Ground-truth (GT) numbers are on held-out test tasks: srbench_all minus the three
unsolvable tasks, the train split and the val split (90 tasks). Black-box R2 uses all
122 tasks; no evolution or HPO run trained or validated on them.

    python scripts/tables.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srbench_results_io as srio
from inspect_srbench_results import _black_box_summary
from srbench_official_results import OFFICIAL_TRAIN_SPLIT, OFFICIAL_VAL_SPLIT

RUNS = ROOT / "runs"
OUT = ROOT / "figures"
NINETY = "srbench_full_9-22_10seed-90s"

# Table 1: 90-second evaluations. (model, training metric, run dir)
TABLE1_ROWS = [
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
MDLFORMER_TRIALS = OUT / "srbench_figure3_extended/mdlformer_trials.csv.gz"

# Table 2: PySR vs evolved PySR (GT, 709715). (label, baseline run, evolved run)
ONE_MILLION = ("runs/290227", "runs/973699")
TABLE2_BUDGETS = [
    ("1 million evaluations", *ONE_MILLION),
    ("90 seconds", f"runs/pysr-base-{NINETY}", f"runs/pysr-gt-709715-{NINETY}"),
    ("15 minutes", "runs/srbench_gt_baseline_15m_single", "runs/709715/srbench_gt_15m_single"),
    ("15 minutes + restarts", "runs/srbench_gt_baseline_15m_portfolio_1e6",
     "runs/709715/srbench_gt_15m_portfolio_1e6"),
]
METRIC_TEX = {"R2": "$R^2$", "GT-R2": "GT-$R^2$", "GT": "GT", "": ""}


def _names(path):
    return {line.strip() for line in (ROOT / path).read_text().splitlines()
            if line.strip() and not line.startswith("#")}


CANONICAL = _names("splits/srbench_all.txt") - set(srio.UNSOLVABLE_TASKS)
TRAIN = _names(OFFICIAL_TRAIN_SPLIT) & CANONICAL
VAL = _names(OFFICIAL_VAL_SPLIT) & CANONICAL
TEST = CANONICAL - TRAIN - VAL


def gt_trials(run_dir):
    """Successful GT trials, excluding the unsolvable tasks."""
    run_dir = ROOT / run_dir
    manifest = srio.load_manifest(run_dir)
    keyed = srio.load_keyed_results(run_dir) or srio.build_keyed_results(run_dir, manifest)
    manifest, keyed = srio.standard_ground_truth_view(manifest, keyed)
    return [row for row in keyed.values() if row.get("present") and row.get("error") is None]


def gt_rate(trials, tasks, noise=None):
    rows = [row for row in trials if row["dataset"] in tasks
            and (noise is None or float(row["noise"]) == noise)]
    expected = len(tasks) * (10 if noise is not None else 40)
    if len(rows) != expected:
        print(f"note: {len(rows)}/{expected} successful trials", file=sys.stderr)
    return 100 * sum(bool(row["solved"]) for row in rows) / len(rows)


def _bold(value, best):
    return rf"\bf {value}" if value == best else value


def table1():
    rows = []
    for model, metric, name in TABLE1_ROWS:
        run_dir = RUNS / name
        completed, expected, bb = _black_box_summary(run_dir, srio.load_manifest(run_dir))
        if completed != expected:
            print(f"warning: {name} black box {completed}/{expected}", file=sys.stderr)
        rows.append((model, metric, f"{bb:.3f}", f"{gt_rate(gt_trials(run_dir), TEST):.1f}"))
    mdl = pd.read_csv(MDLFORMER_TRIALS)
    mdl = mdl[mdl.dataset.isin(TEST)]
    rows.append(("MDLFormer", "GT", "--", f"{100 * mdl.symbolic_solution.mean():.1f}"))
    best_bb = max((r[2] for r in rows if r[2] != "--"), key=float)
    best_gt = max((r[3] for r in rows), key=float)

    groups = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    body = []
    for model, members in groups.items():
        if len(members) == 1:
            _, metric, bb, gt = members[0]
            body.append(f"{model} & {METRIC_TEX[metric]} & {_bold(bb, best_bb)} & {_bold(gt, best_gt)} \\\\")
        else:
            body.append(rf"\multirow[c]{{{len(members)}}}{{*}}{{{model}}}")
            width = max(len(METRIC_TEX[m[1]]) for m in members)
            body += [f"& {METRIC_TEX[metric]:<{width}} & {_bold(bb, best_bb)} & {_bold(gt, best_gt)} \\\\"
                     for _, metric, bb, gt in members]
        body += ["", r"\midrule"]
    del body[-2:]

    return rf"""\begin{{table}}[t]
\caption{{Results on SRBench, averaged over 10 seeds. Ground-truth recoveries are averaged over four noise levels (0, 0.001, 0.01, 0.1) and {len(TEST)} held-out test tasks, excluding the {len(TRAIN)} training and {len(VAL)} validation tasks used during evolution. Black box $R^2$ is averaged over 122 tasks. Improved versions of PySR and BasicSR are evaluated with a 90 second timeout.}}
\label{{tab:srbench}}
\centering
% \small
\scriptsize
\begin{{tabular}}{{llcc}}
\toprule

\multicolumn{{1}}{{l}}{{\multirow{{2}}{{*}}{{Model}}}} &
\multicolumn{{1}}{{l}}{{Training}} &
\multicolumn{{1}}{{c}}{{Black-box}} &
\multicolumn{{1}}{{c}}{{Ground-truth}} \\

& \multicolumn{{1}}{{l}}{{metric}}
 & \multicolumn{{1}}{{c}}{{average $R^2$}}
& \multicolumn{{1}}{{c}}{{average recovery rate (\%)}} \\

\midrule
""" + "\n".join(body) + r"""

\bottomrule
\end{tabular}
\end{table}
"""


def _pair(base, evolved):
    base, evolved = round(base, 1), round(evolved, 1)
    return f"{base:.1f} & {evolved:.1f} & {evolved - base:+.1f}"


def table2():
    budget_rows = [f"{label} & {_pair(gt_rate(gt_trials(base), TEST), gt_rate(gt_trials(evo), TEST))} \\\\"
                   for label, base, evo in TABLE2_BUDGETS]
    base, evo = (gt_trials(path) for path in ONE_MILLION)
    subset_rows = [f"{label} ($n={len(tasks)}$) & {_pair(gt_rate(base, tasks), gt_rate(evo, tasks))} \\\\"
                   for label, tasks in [("Train tasks", TRAIN), ("Val tasks", VAL),
                                        ("All tasks", CANONICAL)]]
    noise_rows = [f"% Noise {noise:g} & {_pair(gt_rate(base, CANONICAL, noise), gt_rate(evo, CANONICAL, noise))} \\\\"
                  for noise in [0, .001, .01, .1]]
    return rf"""\caption{{SRBench ground-truth recovery rates (\%) over 10 random seeds. Rows above the line use
the {len(TEST)} held-out test tasks; rows below use 1 million evaluations. $\Delta$ is Evolved PySR minus PySR.}}
\label{{tab:srbench2}}

% \scriptsize
\footnotesize
\begin{{tabular}}{{lccc}}
\toprule
& \multicolumn{{1}}{{c}}{{PySR}} & \multicolumn{{1}}{{c}}{{Evolved PySR}} & \multicolumn{{1}}{{c}}{{$\Delta$}} \\
\midrule
""" + "\n".join(budget_rows) + r"""
\midrule
""" + "\n".join(subset_rows) + r"""
% \midrule
""" + "\n".join(noise_rows) + r"""
\bottomrule
\end{tabular}"""


def main():
    for name, text in [("table_srbench.tex", table1()), ("table_srbench2.tex", table2())]:
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
