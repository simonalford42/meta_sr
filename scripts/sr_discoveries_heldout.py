"""Train / val / test split scores for the official SRBench evaluations.

train = splits/barely_unsolvable.txt, val = splits/barely_unsolvable_val2.txt,
test = every other SRBench ground-truth task (never seen during evolution).
Scores are GT solve rate per run (and clipped test R2), pooled over the four
noise levels, plus the noise-0-only solve rate. A dataset-level bootstrap gives
a 95% interval on the difference to the named baseline.

Usage:
    python scripts/sr_discoveries_heldout.py
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import srbench_results_io as srio  # noqa: E402
from utils import load_dataset_names_from_split  # noqa: E402

TRAIN = set(load_dataset_names_from_split("splits/barely_unsolvable.txt"))
VAL = set(load_dataset_names_from_split("splits/barely_unsolvable_val2.txt"))

# (label, run dir, baseline label)
GROUPS = {
    "1e6 evals, 10 seeds x 4 noise": [
        ("PySR base", "runs/290227", None),
        ("HPO GT", "runs/593871", "PySR base"),
        ("PySR++ GT", "runs/973699", "PySR base"),
        ("PySR++ GT-R2", "runs/139325", "PySR base"),
        ("PySR++ R2", "runs/606485", "PySR base"),
        ("BasicSR base", "runs/150814", None),
        ("BasicSR++ GT", "runs/249072", "BasicSR base"),
        ("BasicSR++ GT-R2", "runs/229869-srbench_full_9-21_10seed", "BasicSR base"),
        ("BasicSR++ R2", "runs/271624", "BasicSR base"),
    ],
    "90 s, 10 seeds x 4 noise": [
        ("PySR base", "runs/pysr-base-srbench_full_9-22_10seed-90s", None),
        ("HPO GT", "runs/hpo-gt-555204-srbench_full_9-22_10seed-90s", "PySR base"),
        ("PySR++ GT", "runs/pysr-gt-709715-srbench_full_9-22_10seed-90s", "PySR base"),
        ("  only loss", "runs/709715-only-loss-srbench_full_9-24_10seed-90s", "PySR base"),
        ("  only mutation", "runs/709715-only-mutation-srbench_full_9-24_10seed-90s", "PySR base"),
        ("  only selection", "runs/709715-only-selection-srbench_full_9-24_10seed-90s", "PySR base"),
        ("  only survival", "runs/709715-only-survival-srbench_full_9-24_10seed-90s", "PySR base"),
        ("PySR++ GT-R2", "runs/pysr-gtr2-941339-srbench_full_9-22_10seed-90s", "PySR base"),
        ("PySR++ R2", "runs/pysr-r2-120458-srbench_full_9-22_10seed-90s", "PySR base"),
        ("BasicSR base", "runs/fullsr-base-srbench_full_9-22_10seed-90s", None),
        ("BasicSR++ GT", "runs/fullsr-gt-225437-srbench_full_9-22_10seed-90s", "BasicSR base"),
        ("BasicSR++ GT-R2", "runs/fullsr-gtr2-229869-srbench_full_9-22_10seed-90s", "BasicSR base"),
        ("BasicSR++ R2", "runs/fullsr-r2-150812-srbench_full_9-22_10seed-90s", "BasicSR base"),
    ],
}


def load(run_dir):
    run_dir = ROOT / run_dir
    manifest = srio.load_manifest(run_dir)
    keyed = srio.load_keyed_results(run_dir)
    if keyed is None and manifest.get("batches"):
        keyed = srio.build_keyed_results(run_dir, manifest)
    manifest, keyed = srio.standard_ground_truth_view(manifest, keyed or {})
    # per dataset: list of (noise, solved, r2)
    per = defaultdict(list)
    n_expected = 0
    for e in keyed.values():
        n_expected += 1
        if not e.get("present") or e.get("error") is not None:
            continue
        r2 = e.get("test_r2")
        per[e["dataset"]].append((srio.fmt_noise(e.get("noise", 0)), float(bool(e.get("solved"))),
                                  None if r2 is None else max(0.0, float(r2))))
    n_present = sum(len(v) for v in per.values())
    return per, n_present, n_expected


def part(ds):
    return "train" if ds in TRAIN else "val" if ds in VAL else "test"


def rate(per, which, noise0=False, r2=False):
    vals = []
    for ds, rows in per.items():
        if which != "all" and part(ds) != which:
            continue
        for noise, solved, r in rows:
            if noise0 and float(noise) != 0.0:
                continue
            v = r if r2 else solved
            if v is not None:
                vals.append(v)
    return sum(vals) / len(vals) if vals else float("nan")


def boot_diff(per_a, per_b, which, n_boot=2000, seed=0):
    """95% CI of (a - b) solve rate, resampling datasets (paired)."""
    names = sorted(ds for ds in set(per_a) & set(per_b) if which == "all" or part(ds) == which)
    if not names:
        return (float("nan"), float("nan"))
    ma = {ds: (sum(s for _, s, _ in per_a[ds]), len(per_a[ds])) for ds in names}
    mb = {ds: (sum(s for _, s, _ in per_b[ds]), len(per_b[ds])) for ds in names}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [names[rng.randrange(len(names))] for _ in names]
        sa = sum(ma[d][0] for d in sample) / max(1, sum(ma[d][1] for d in sample))
        sb = sum(mb[d][0] for d in sample) / max(1, sum(mb[d][1] for d in sample))
        diffs.append(sa - sb)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def main():
    for title, rows in GROUPS.items():
        print(f"\n=== {title} ===")
        print(f"{'method':18s} {'done':>10s} | {'train':>6s} {'val':>6s} {'test':>6s} {'all':>6s} | "
              f"{'test n0':>7s} | {'test R2':>7s} | test diff vs base [95% CI]   | val diff [95% CI]")
        loaded = {}
        for label, run_dir, base in rows:
            try:
                per, n_present, n_expected = load(run_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"{label:18s} ERROR {exc}")
                continue
            loaded[label.strip()] = per
            line = (f"{label:18s} {n_present:>5d}/{n_expected:<4d} | "
                    f"{rate(per, 'train'):6.3f} {rate(per, 'val'):6.3f} {rate(per, 'test'):6.3f} {rate(per, 'all'):6.3f} | "
                    f"{rate(per, 'test', noise0=True):7.3f} | {rate(per, 'test', r2=True):7.3f} | ")
            if base and base in loaded:
                b = loaded[base]
                d = rate(per, "test") - rate(b, "test")
                lo, hi = boot_diff(per, b, "test")
                dv = rate(per, "val") - rate(b, "val")
                lov, hiv = boot_diff(per, b, "val")
                line += f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]      | {dv:+.3f} [{lov:+.3f}, {hiv:+.3f}]"
            print(line)


if __name__ == "__main__":
    main()
