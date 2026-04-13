"""Migrate legacy outputs/{evolve,openevolve,eval_pysr}_* dirs to runs/<jobid>.

Strategy:
- For each legacy dir, glob slurm_*.out to extract the SLURM job id.
- Dir with exactly one match: copy to runs/<jobid>, rename log to slurm.out.
- Dir with multiple matches: pick the smallest job id (parent job), same copy.
- Dir with no match: copy to runs/local_<original_name> (preserves history under runs/).
- If runs/<jobid> already exists: skip with a warning.
- Originals in outputs/ are left in place (copy, not move).

Run without flags for a dry run. Use --apply to actually copy.
"""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
RUNS = REPO / "runs"

PREFIXES = ("evolve_", "openevolve_", "eval_pysr_", "hpo_pysr_")
SLURM_LOG_RE = re.compile(r"slurm_(\d+)\.out$")

# Subdirectories holding regenerable worker caches / evaluation artifacts.
# These are excluded from the copy to keep the archive small — original data
# remains in outputs/ if ever needed.
EXCLUDE_DIRS = {
    "slurm_pysr",
    "pysr_eval",
    "final_eval",
    "TEMP_signal_diagnostics",
}


def _ignore(src, names):
    return [n for n in names if n in EXCLUDE_DIRS]


def find_jobid(d: Path) -> tuple[str | None, list[Path]]:
    logs = sorted(d.glob("slurm_*.out"))
    ids: list[tuple[int, Path]] = []
    for p in logs:
        m = SLURM_LOG_RE.search(p.name)
        if m:
            ids.append((int(m.group(1)), p))
    if not ids:
        return None, logs
    ids.sort()
    return str(ids[0][0]), [p for _, p in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually copy dirs (default: dry run)")
    args = ap.parse_args()

    dry = not args.apply
    RUNS.mkdir(exist_ok=True)

    candidates = [
        d for d in sorted(OUTPUTS.iterdir())
        if d.is_dir() and any(d.name.startswith(p) for p in PREFIXES)
    ]

    copied = copied_local = skipped_exists = 0
    multi_log_renamed = 0

    for d in candidates:
        jobid, logs = find_jobid(d)
        if jobid is None:
            dest = RUNS / f"local_{d.name}"
            if dest.exists():
                print(f"[skip:exists] {d.name} -> runs/{dest.name}")
                skipped_exists += 1
                continue
            print(f"[local]       {d.name} -> runs/{dest.name}")
            copied_local += 1
            if args.apply:
                shutil.copytree(str(d), str(dest), symlinks=True, ignore=_ignore)
            continue

        dest = RUNS / jobid
        if dest.exists():
            print(f"[skip:exists] {d.name} -> runs/{jobid}")
            skipped_exists += 1
            continue

        tag = f" ({len(logs)} slurm logs, picked min)" if len(logs) > 1 else ""
        print(f"[copy]        {d.name} -> runs/{jobid}{tag}")
        if len(logs) > 1:
            multi_log_renamed += 1

        if args.apply:
            shutil.copytree(str(d), str(dest), symlinks=True, ignore=_ignore)
            # Rename the canonical slurm log to slurm.out
            old_log = dest / f"slurm_{jobid}.out"
            new_log = dest / "slurm.out"
            if old_log.exists() and not new_log.exists():
                old_log.rename(new_log)
        copied += 1

    print()
    print(f"{'DRY RUN' if dry else 'APPLIED'}: "
          f"{copied} copied by jobid ({multi_log_renamed} had multiple logs), "
          f"{copied_local} copied as local_, "
          f"{skipped_exists} skipped (dest exists)")
    if dry:
        print("Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
