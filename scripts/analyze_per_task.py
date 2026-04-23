"""Print a per-task × per-generation table of best seeds-solved for a bundle run.

Reads run_data.json (produced by evolve_pysr.py) and, for each generation,
shows the maximum number of seeds that any bundle (population ∪ offspring)
solved on each task. "Solved" means gt_match_score >= 1.0 for that seed.

Usage:
    python analyze_per_task.py runs/726782/run_data.json
    python analyze_per_task.py runs/726782/run_data.json --show-bundle
    python analyze_per_task.py runs/726782/run_data.json --cumulative
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _bundle_display_name(bundle: Dict) -> str:
    parts = []
    for t in ("mutation", "survival", "selection"):
        op = (bundle.get("operators") or {}).get(t)
        parts.append(op.get("name") if op else "default")
    return " | ".join(parts)


def _n_solved(run_gt_scores: List[float]) -> int:
    return sum(1 for g in (run_gt_scores or []) if g >= 1.0)


def _best_solver_for_task(bundles: List[Dict], task_idx: int) -> Tuple[int, int, Optional[str]]:
    """Return (best_n_solved, total_seeds, winning_bundle_name) for a task."""
    best = -1
    total = 0
    winner = None
    for b in bundles:
        rd = b.get("result_details") or []
        if task_idx >= len(rd) or rd[task_idx] is None:
            continue
        gt = rd[task_idx].get("run_gt_scores") or []
        if not gt:
            continue
        n = _n_solved(gt)
        if n > best:
            best = n
            total = len(gt)
            winner = _bundle_display_name(b)
    if best < 0:
        return 0, 0, None
    return best, total, winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_data", type=Path, help="Path to run_data.json")
    parser.add_argument(
        "--show-bundle", action="store_true",
        help="Also print the winning bundle name per cell.",
    )
    parser.add_argument(
        "--cumulative", action="store_true",
        help="Use the max over all bundles seen up to and including each generation.",
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Prepend a 'Base' column using only the baseline bundle at gen 1.",
    )
    args = parser.parse_args()

    data = json.loads(args.run_data.read_text())
    generations = data.get("generations") or []
    if not generations:
        print("No generations in this run.")
        return

    dataset_names = (
        data.get("config", {}).get("dataset_names")
        or [rd.get("dataset", "?") for rd in (generations[0]["population"][0]["result_details"] or [])]
    )
    n_tasks = len(dataset_names)

    # Build per-gen bundle pools (population + offspring).
    gen_labels: List[str] = []
    gen_pools: List[List[Dict]] = []
    for g in generations:
        gen_labels.append(f"G{g['generation']}")
        pool = list(g.get("population") or []) + list(g.get("offspring") or [])
        gen_pools.append(pool)

    if args.cumulative:
        cum: List[Dict] = []
        cum_pools = []
        for p in gen_pools:
            cum = cum + p
            cum_pools.append(cum)
        gen_pools = cum_pools

    # Compute per-task per-gen results.
    # cells[task_idx][gen_idx] = (n_solved, total, winner)
    cells: List[List[Tuple[int, int, Optional[str]]]] = []
    totals_seen = 0
    for t in range(n_tasks):
        row = []
        for pool in gen_pools:
            n, tot, win = _best_solver_for_task(pool, t)
            row.append((n, tot, win))
            if tot > totals_seen:
                totals_seen = tot
        cells.append(row)

    # Header & widths.
    name_col_w = max(len(n) for n in dataset_names) if dataset_names else 10
    name_col_w = min(max(name_col_w, 8), 40)

    if args.show_bundle:
        # Build a bundle-name legend; render cells as "n/N (#id)".
        legend: Dict[str, int] = {}
        next_id = 1
        for row in cells:
            for _, _, win in row:
                if win and win not in legend:
                    legend[win] = next_id
                    next_id += 1

        cell_w = max(len(f"{totals_seen}/{totals_seen}") + 5, 8)
        header = f"{'Dataset'.ljust(name_col_w)}  " + "  ".join(lbl.rjust(cell_w) for lbl in gen_labels)
        print(header)
        print("-" * len(header))
        for t, ds_name in enumerate(dataset_names):
            cells_str = []
            for n, tot, win in cells[t]:
                if tot == 0:
                    cells_str.append("--".rjust(cell_w))
                else:
                    wid = legend.get(win, 0) if win else 0
                    tag = f"#{wid}" if wid else "  "
                    cells_str.append(f"{n}/{tot} {tag}".rjust(cell_w))
            print(f"{ds_name[:name_col_w].ljust(name_col_w)}  " + "  ".join(cells_str))

        print()
        print("Bundle legend:")
        for name, bid in sorted(legend.items(), key=lambda kv: kv[1]):
            print(f"  #{bid}: {name}")
    else:
        cell_w = max(len(f"{totals_seen}/{totals_seen}"), 5)
        header = f"{'Dataset'.ljust(name_col_w)}  " + "  ".join(lbl.rjust(cell_w) for lbl in gen_labels)
        print(header)
        print("-" * len(header))
        for t, ds_name in enumerate(dataset_names):
            cells_str = []
            for n, tot, _ in cells[t]:
                if tot == 0:
                    cells_str.append("--".rjust(cell_w))
                else:
                    cells_str.append(f"{n}/{tot}".rjust(cell_w))
            print(f"{ds_name[:name_col_w].ljust(name_col_w)}  " + "  ".join(cells_str))

    # Summary rows.
    print()
    per_gen_any = []
    per_gen_all = []
    per_gen_avg_best = []
    for g in range(len(gen_labels)):
        any_solved = sum(1 for t in range(n_tasks) if cells[t][g][0] >= 1)
        all_solved = sum(1 for t in range(n_tasks) if cells[t][g][1] > 0 and cells[t][g][0] == cells[t][g][1])
        # Average "seeds solved by the best bundle for that task" over all
        # tasks with at least one result in this generation.
        fracs = [cells[t][g][0] / cells[t][g][1] for t in range(n_tasks) if cells[t][g][1] > 0]
        avg_best = sum(fracs) / len(fracs) if fracs else 0.0
        per_gen_any.append(any_solved)
        per_gen_all.append(all_solved)
        per_gen_avg_best.append(avg_best)
    summary_label_w = max(name_col_w, len("avg best seeds-solved (fraction)"))
    cell_w = max(len(f"{n_tasks}/{n_tasks}"), len("0.000"), 5)
    print(f"{'tasks with >=1 seed solved'.ljust(summary_label_w)}  " +
          "  ".join(f"{v}/{n_tasks}".rjust(cell_w) for v in per_gen_any))
    print(f"{'tasks with all seeds solved'.ljust(summary_label_w)}  " +
          "  ".join(f"{v}/{n_tasks}".rjust(cell_w) for v in per_gen_all))
    print(f"{'avg best seeds-solved (fraction)'.ljust(summary_label_w)}  " +
          "  ".join(f"{v:.3f}".rjust(cell_w) for v in per_gen_avg_best))


if __name__ == "__main__":
    main()
