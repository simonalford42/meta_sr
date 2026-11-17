"""Re-score SRBench black-box frontiers with a train-fit affine rescaling.

For every per-trial PySR frontier equation f in a run's black-box batch
(combined.json, full frontier), reproduce the exact 75/25 split, 10k train cap
and StandardScaler from parallel_eval_pysr.py, fit (a, b) by least squares on
the (scaled) training target, and report test R^2 of a + b*f in original units.
Also recomputes the raw (unscaled) test R^2 as a check against the stored value.

Usage:
  python scripts/blackbox_affine_r2.py RUN_DIR [RUN_DIR ...] \
      --datasets data/srbench_blackbox_e2e55.txt --out OUT.csv
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_srbench_dataset  # noqa: E402

NS = dict(sin=np.sin, cos=np.cos, exp=np.exp, log=np.log, sqrt=np.sqrt,
          square=np.square)


def black_box_batch(run_dir):
    for f in sorted(glob.glob(f"{run_dir}/slurm_pysr/*/combined.json")):
        rows = json.load(open(f))
        if rows and not rows[0]["dataset_name"].startswith(("feynman", "strogatz")):
            return rows
    raise FileNotFoundError(f"no black-box combined.json under {run_dir}")


def split(name, seed, max_train=10_000):
    X, y, _ = load_srbench_dataset(name, max_samples=None)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    np.random.seed(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.75, test_size=0.25,
                                          random_state=seed)
    if len(ytr) > max_train:
        keep = np.random.choice(len(ytr), max_train)
        Xtr, ytr = Xtr[keep], ytr[keep]
    xs, ys = StandardScaler(), StandardScaler()
    Xtr = xs.fit_transform(Xtr)
    Xte = xs.transform(Xte)
    ytr = ys.fit_transform(ytr.reshape(-1, 1)).ravel()
    return Xtr, ytr, Xte, yte, ys


def evaluate(eq, X):
    env = dict(NS, **{f"x{i}": X[:, i] for i in range(X.shape[1])})
    with np.errstate(all="ignore"):
        out = eval(eq, {"__builtins__": {}}, env)
    return np.broadcast_to(np.asarray(out, dtype=float), (len(X),)).copy()


def r2(y, pred):
    pred = np.clip(pred, -1e10, 1e10)
    return 1.0 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--datasets", default=None)
    p.add_argument("--base-seed", type=int, default=10_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    keep = set(Path(a.datasets).read_text().split()) if a.datasets else None

    out = []
    for run in a.runs:
        rows = [r for r in black_box_batch(run) if r.get("pareto_frontier")
                and (keep is None or r["dataset_name"] in keep)]
        rows.sort(key=lambda r: r["dataset_name"])
        cache = {}
        for r in rows:
            name, ri = r["dataset_name"], int(r.get("run_index") or 0)
            key = (name, ri)
            if key not in cache:
                cache = {key: split(name, a.base_seed + ri)}
            Xtr, ytr, Xte, yte, ys = cache[key]
            for pt in r["pareto_frontier"]:
                try:
                    ftr, fte = evaluate(pt["equation"], Xtr), evaluate(pt["equation"], Xte)
                except Exception:
                    continue
                raw = r2(yte, ys.inverse_transform(fte.reshape(-1, 1)).ravel())
                aff = np.nan
                if np.all(np.isfinite(ftr)):
                    fm = ftr.mean(); df = ftr - fm
                    sxx = float(df @ df)
                    b = float(df @ (ytr - ytr.mean())) / sxx if sxx > 0 else 0.0
                    b = b if np.isfinite(b) else 0.0
                    c = ytr.mean() - b * fm
                    aff = r2(yte, ys.inverse_transform((c + b * fte).reshape(-1, 1)).ravel())
                out.append(dict(run=Path(run).name, dataset=name, trial=ri,
                                complexity=pt["complexity"], stored_r2=pt["test_r2"],
                                raw_r2=raw, affine_r2=aff))
        print(f"{run}: {len(rows)} trials", flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
