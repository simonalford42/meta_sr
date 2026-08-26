#!/usr/bin/env python3
"""Inspect a full-SRBench evaluation run.

Reads ``runs/<run-id>/`` (manifest.json + srbench_full_results.json, falling
back to re-joining the per-batch task artifacts) and prints:
  * X/Y runs completed (Y = tasks x seeds x noise levels) and what's missing.
  * solve rate (per-run and per-task-any) + mean solve time for
    srbench(all) / feynman / strogatz, broken down by noise level.

Usage:
    python inspect_srbench_results.py --run-id <run-id>
    python inspect_srbench_results.py --run-id <run-id> --exclude-unsolvable
    python inspect_srbench_results.py --run-id <run-id> --wandb
    python inspect_srbench_results.py --see-all
    python inspect_srbench_results.py --see-all --since 7
    python inspect_srbench_results.py --official
"""

import argparse
import json
import sys
import time
from pathlib import Path

import srbench_results_io as srio
from utils import load_dataset_names_from_split

# Default train / val splits used by evolve_pysr.py (its argparse defaults).
TRAIN_SPLIT = "splits/barely_unsolvable.txt"
VAL_SPLIT = "splits/barely_unsolvable_val2.txt"
CLEAN_TRAIN_SPLIT = "splits/train.txt"
CLEAN_VAL_SPLIT = "splits/val.txt"


def _load_split_set(split_file: str) -> "set[str] | None":
    try:
        return set(load_dataset_names_from_split(split_file))
    except Exception:
        return None


# Computed once; labels are the split-file stems (the "real names").
TRAIN_LABEL = Path(TRAIN_SPLIT).stem
VAL_LABEL = Path(VAL_SPLIT).stem
TRAIN_SET = _load_split_set(TRAIN_SPLIT)
VAL_SET = _load_split_set(VAL_SPLIT)
CLEAN_TRAIN_SET = _load_split_set(CLEAN_TRAIN_SPLIT)
CLEAN_VAL_SET = _load_split_set(CLEAN_VAL_SPLIT)


def subset_solve_rate(keyed: "dict", datasets: "set[str]",
                      noise: "float | None" = None) -> "float | None":
    """solve%/run over runs whose dataset is in ``datasets``.

    If ``noise`` is given, restrict to that noise level only; otherwise pool
    across all noise levels.
    """
    noise_str = srio.fmt_noise(noise) if noise is not None else None
    present = [e for e in keyed.values()
               if e.get("present") and e.get("error") is None
               and e["dataset"] in datasets
               and (noise_str is None or srio.fmt_noise(e["noise"]) == noise_str)]
    if not present:
        return None
    solved = sum(1 for e in present if e["solved"])
    return solved / len(present)


def find_full_srbench_runs(
    runs_root: "str | Path", since_days: "int | None" = None
) -> "list[Path]":
    """Return all run directories under ``runs_root`` that are full-SRBench runs.

    A full-SRBench run is identified by the ground-truth manifest grid or by a
    black-box evaluation marker written by ``srbench_full_eval.py``. If
    ``since_days`` is provided, only manifests modified within that many days
    are included.
    """
    runs_root = Path(runs_root)
    cutoff = time.time() - since_days * 24 * 60 * 60 if since_days is not None else None
    found = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        if cutoff is not None and manifest_path.stat().st_mtime < cutoff:
            continue
        try:
            manifest = srio.load_manifest(manifest_path.parent)
        except Exception:
            continue
        is_ground_truth = all(
            k in manifest for k in ("datasets", "noise_levels", "batches")
        )
        is_black_box = (
            "black_box" in (manifest.get("evaluation_types") or [])
            or "black_box" in manifest
        )
        if is_ground_truth or is_black_box:
            found.append(manifest_path.parent)
    return found


def bundle_id(manifest: dict) -> str:
    """Run-id of the bundle being evaluated (the ``method_meta.source`` run).

    Eval manifests record the operator/hparam bundle they were built from in
    ``method_meta.source`` (e.g. ``"runs/666285"`` or
    ``"runs/394789/best_params.json"``). Return just the run-id portion
    (``"666285"`` / ``"394789"``), or ``"-"`` when there is no source bundle
    (e.g. baseline runs).
    """
    source = (manifest.get("method_meta") or {}).get("source")
    if not source:
        return "-"
    parts = Path(source).parts
    # ".../runs/<id>/..." -> <id>; otherwise fall back to the first component.
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return Path(source).name


def _black_box_summary(run_dir: Path, manifest: dict) -> tuple[int, int, float | None]:
    """Return (completed trials, expected trials, mean best-per-trial test R²)."""
    path = run_dir / "srbench_black_box_results.json"
    black_box = manifest.get("black_box") or {}
    expected = int(black_box.get("n_datasets") or 0) * int(black_box.get("n_runs") or 0)
    if not path.exists():
        return 0, expected, None

    with open(path) as f:
        datasets = (json.load(f).get("datasets") or {})
    best_r2 = []
    completed = 0
    for stored_frontiers in datasets.values():
        # Before multi-trial black-box evaluation was added, each dataset was
        # stored as one frontier directly: ``[point, ...]``.  Current files
        # store a list of trial frontiers: ``[[point, ...], ...]``.  Normalize
        # the historical single-trial layout so old runs remain inspectable.
        if stored_frontiers and isinstance(stored_frontiers[0], dict):
            trials = [stored_frontiers]
        else:
            trials = stored_frontiers
        completed += len(trials)
        for frontier in trials:
            values = [point.get("test_r2") for point in frontier
                      if point.get("test_r2") is not None]
            if values:
                best_r2.append(max(values))
    mean_r2 = sum(best_r2) / len(best_r2) if best_r2 else None
    return completed, expected, mean_r2


def summarize_run(run_dir: Path) -> dict:
    """One row of summary stats, including partially completed evaluations."""
    manifest = srio.load_manifest(run_dir)
    noise_levels = manifest.get("noise_levels") or []

    keyed = srio.load_keyed_results(run_dir)
    if keyed is None and manifest.get("batches"):
        keyed = srio.build_keyed_results(run_dir, manifest)
    keyed = keyed or {}

    expected = srio.expected_keys(manifest) if manifest.get("datasets") else []
    present = sum(1 for k, e in keyed.items()
                  if e.get("present") and e.get("error") is None)
    bb_present, bb_expected, bb_r2 = _black_box_summary(run_dir, manifest)

    row = {
        "bundle": bundle_id(manifest),
        "mode": manifest.get("mode") or "-",
        "max_evals": manifest.get("max_evals"),
        "completed": present + bb_present,
        "total": len(expected) + bb_expected,
        "bb_r2": bb_r2,
    }
    row["complete"] = row["total"] > 0 and row["completed"] == row["total"]
    if not expected:
        return row

    metrics = srio.aggregate_metrics(keyed, noise_levels)
    # solve%/run and mean solve time for all / feynman / strogatz.
    for fam in ("all", "feynman", "strogatz"):
        m = metrics[fam]["all"]
        row[f"{fam}_pct"] = m["solve_rate_per_run"]
        row[f"{fam}_t"] = m["solve_time_mean_all"]
        row[f"{fam}_r2"] = m["test_r2_mean"]
    # solve%/run over all tasks at the noise=0 (clean) bucket only.
    m0 = metrics["all"].get(srio.fmt_noise(0))
    row["all0_pct"] = m0["solve_rate_per_run"] if m0 else None

    # Per-subset solve%/run: train, val, and the rest (srbench - train - val).
    if TRAIN_SET and VAL_SET:
        all_ds = {e["dataset"] for e in keyed.values()}
        rest_set = all_ds - TRAIN_SET - VAL_SET
        row["train_pct"] = subset_solve_rate(keyed, TRAIN_SET)
        row["val_pct"] = subset_solve_rate(keyed, VAL_SET)
        row["rest_pct"] = subset_solve_rate(keyed, rest_set)
        row["train0_pct"] = subset_solve_rate(keyed, TRAIN_SET, noise=0)
        row["val0_pct"] = subset_solve_rate(keyed, VAL_SET, noise=0)
    row["clean_train0_pct"] = (
        subset_solve_rate(keyed, CLEAN_TRAIN_SET, noise=0)
        if CLEAN_TRAIN_SET else None
    )
    row["clean_val0_pct"] = (
        subset_solve_rate(keyed, CLEAN_VAL_SET, noise=0)
        if CLEAN_VAL_SET else None
    )
    return row


def format_summary_table(rows: "list[dict]") -> str:
    """Render the --see-all summary rows as an aligned, boxed table."""
    def _pct(x):
        return f"{x*100:.1f}" if x is not None else "-"

    def _time(x):
        return f"{x:.1f}s" if x is not None else "-"

    def _r2(x):
        return f"{x:.3f}" if x is not None else "-"

    has_subset = any("train_pct" in r for r in rows)

    # (header, key-fn) for each column.
    cols = [
        ("bundle", lambda r: str(r["bundle"])),
        ("completed", lambda r: f"{r['completed']}/{r['total']}"),
        ("mode", lambda r: str(r["mode"])),
        ("max-evals", lambda r: (
            f"{r['max_evals']:,}" if r.get("max_evals") is not None else "-"
        )),
        ("bb_r2", lambda r: _r2(r.get("bb_r2"))),
        ("all%", lambda r: _pct(r.get("all_pct"))),
        ("all%(n0)", lambda r: _pct(r.get("all0_pct"))),
        ("all_r2", lambda r: _r2(r.get("all_r2"))),
        ("all_t", lambda r: _time(r.get("all_t"))),
        ("feyn%", lambda r: _pct(r.get("feynman_pct"))),
        ("feyn_t", lambda r: _time(r.get("feynman_t"))),
        ("strog%", lambda r: _pct(r.get("strogatz_pct"))),
        ("strog_t", lambda r: _time(r.get("strogatz_t"))),
    ]
    if has_subset:
        cols += [
            (TRAIN_LABEL, lambda r: _pct(r.get("train_pct"))),
            ("bu(n0)", lambda r: _pct(r.get("train0_pct"))),
            (VAL_LABEL, lambda r: _pct(r.get("val_pct"))),
            ("bu2(n0)", lambda r: _pct(r.get("val0_pct"))),
            ("train(n0)", lambda r: _pct(r.get("clean_train0_pct"))),
            ("val(n0)", lambda r: _pct(r.get("clean_val0_pct"))),
            ("rest", lambda r: _pct(r.get("rest_pct"))),
        ]

    headers = [h for h, _ in cols]
    table = [headers] + [[fn(r) for _, fn in cols] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def _fmt_row(cells):
        # First two columns left-aligned (text), the rest right-aligned (numbers).
        parts = [cells[i].ljust(widths[i]) if i in (0, 2) else cells[i].rjust(widths[i])
                 for i in range(len(cells))]
        return "│ " + " │ ".join(parts) + " │"

    def _rule(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    lines = [_rule("┌", "┬", "┐"), _fmt_row(headers), _rule("├", "┼", "┤")]
    lines += [_fmt_row(row) for row in table[1:]]
    lines.append(_rule("└", "┴", "┘"))
    return "\n".join(lines)


def inspect_run(run_dir: Path, args) -> None:
    """Print completion + stats (and optionally re-log to wandb) for one run."""
    manifest = srio.load_manifest(run_dir)
    noise_levels = manifest["noise_levels"]

    # Prefer the aggregated JSON; fall back to re-joining batch artifacts.
    keyed = srio.load_keyed_results(run_dir)
    if keyed is None:
        print("(srbench_full_results.json absent — rebuilding from batch artifacts)")
        keyed = srio.build_keyed_results(run_dir, manifest)

    # ---- completion ----
    expected = srio.expected_keys(manifest)
    n_expected = len(expected)
    present = {k for k, e in keyed.items() if e.get("present") and e.get("error") is None}
    errored = {k for k, e in keyed.items() if e.get("present") and e.get("error") is not None}
    missing = [k for k in expected if k not in keyed or not keyed[k].get("present")]

    print(f"Run: {run_dir}  (mode={manifest.get('mode')})")
    print(f"Grid: {manifest['n_datasets']} tasks x {manifest['n_runs']} seeds "
          f"x {len(noise_levels)} noise = {n_expected} runs")
    print(f"Completed: {len(present)}/{n_expected}   "
          f"(errored: {len(errored)}, missing: {len(missing)})")

    if missing:
        print(f"\nMissing ({len(missing)}):")
        for k in missing[:args.show_missing]:
            ds, seed, noise = k.split("|")
            print(f"  {ds}  seed={seed}  noise={noise}")
        if len(missing) > args.show_missing:
            print(f"  ... and {len(missing) - args.show_missing} more")

    # ---- stats ----
    metrics = srio.aggregate_metrics(keyed, noise_levels)
    print("\n" + "=" * 70)
    print("STATS (all 133 tasks)")
    print("=" * 70)
    print(srio.format_metrics_console(metrics, noise_levels))

    if args.exclude_unsolvable:
        metrics_excl = srio.aggregate_metrics(keyed, noise_levels, exclude_unsolvable=True)
        print("\n" + "=" * 70)
        print(f"STATS (excluding {len(srio.UNSOLVABLE_TASKS)} inverse-trig unsolvables)")
        print("=" * 70)
        print(srio.format_metrics_console(metrics_excl, noise_levels))

    if args.wandb:
        from wandb_utils import init_wandb, finish_wandb
        run = init_wandb(
            config={"mode": manifest.get("mode"), "run_id": run_dir.name,
                    "inspect": True},
            script_name="inspect_srbench_results.py",
            output_dir=str(run_dir),
            extra_tags=["srbench_inspect"],
        )
        srio.log_wandb_table_and_metrics(run, keyed, noise_levels)
        finish_wandb(run)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a full SRBench evaluation run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-id",
                      help="Run id / directory name under --runs-root.")
    mode.add_argument("--see-all", action="store_true",
                      help="Inspect every full-SRBench run under --runs-root.")
    mode.add_argument(
        "--official", action="store_true",
        help="Show the official baseline/HPO/PySR++/BasicSR++ comparison table.",
    )
    parser.add_argument("--since", type=int, metavar="NDAYS",
                        help="With --see-all, only include runs from the past NDAYS days.")
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--show-missing", type=int, default=50,
                        help="Max number of missing (task,seed,noise) triples to print.")
    parser.add_argument("--exclude-unsolvable", action="store_true",
                        help="Also report stats excluding the 3 inverse-trig tasks.")
    parser.add_argument("--wandb", action="store_true",
                        help="Re-log the results table + metrics to wandb.")
    args = parser.parse_args()

    if args.since is not None and args.since < 0:
        parser.error("--since must be non-negative")
    if args.since is not None and not args.see_all:
        parser.error("--since requires --see-all")

    if args.official:
        from srbench_official_results import build_official_table
        print(build_official_table(args.runs_root))
        return

    if args.see_all:
        run_dirs = find_full_srbench_runs(args.runs_root, since_days=args.since)
        if not run_dirs:
            suffix = f" from the past {args.since} day(s)" if args.since is not None else ""
            print(f"No full-SRBench runs found under {args.runs_root}{suffix}")
            sys.exit(1)
        rows = []
        for run_dir in run_dirs:
            try:
                summary = summarize_run(run_dir)
            except Exception as e:
                print(f"{run_dir.name}: ERROR {e}")
                continue
            rows.append(summary)
        # Keep completed runs first and move unfinished runs to the bottom.
        rows.sort(key=lambda row: not row["complete"])
        if rows:
            print(format_summary_table(rows))
        n_complete = sum(row["complete"] for row in rows)
        print(f"\n{n_complete}/{len(rows)} full-SRBench run(s) fully completed.")
        return

    if not args.run_id:
        parser.error("one of --run-id, --see-all, or --official is required")

    run_dir = Path(args.runs_root) / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        sys.exit(1)
    inspect_run(run_dir, args)


if __name__ == "__main__":
    main()
