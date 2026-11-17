"""Train / val / test solve rates of the val-selected bundle for each 90 s
reevaluation-ablation run (n3, n10, n3-reeval, n3-TTTS; evolution seeds 1-3).

For the bundle that srbench_full_eval picked (--select-by val), report:
  train_live     train GT-match rate during evolution (the n_runs-seed score used for selection)
  train_reeval   fresh-seed train GT-match rate during evolution (identification pass;
                 also the mean of the per-generation [train reeval] log lines for this bundle)
  val_evo        val GT-match rate during evolution (val_results, 10 runs)
  srb_train/val/test  full-SRBench eval (runs/*-srbench_gt_11-9_3seed-90s), 4 noise levels x 3 seeds;
                 *_n0 = noise 0 only (evolution runs at noise 0)

Usage: python scripts/reeval_ablation_train_val_test.py
"""
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1]
import inspect_srbench_results as I  # noqa: E402

srio = I.srio
METHODS = ["n3", "n10", "n3-reeval", "n3-ttts"]
# eval job -> earlier job(s) whose run.log holds the pre-resume history
PARENT = {"980597": "750247", "353972": "192862", "203388": "192863", "203389": "192864"}
REEVAL = re.compile(r"\[train reeval\] gen (\d+) (.+?): reeval GT match rate=([\d.]+) \(live=([\d.]+)")


def srbench_rates(eval_dir):
    man = srio.load_manifest(eval_dir)
    keyed = srio.load_keyed_results(eval_dir)
    man, keyed = srio.standard_ground_truth_view(man, keyed)
    alld = {e["dataset"] for e in keyed.values()}
    test = alld - I.TRAIN_SET - I.VAL_SET
    r = lambda s, n=None: 100 * I.subset_solve_rate(keyed, s, noise=n)
    return dict(srb_train=r(I.TRAIN_SET), srb_train_n0=r(I.TRAIN_SET, 0),
                srb_val=r(I.VAL_SET), srb_val_n0=r(I.VAL_SET, 0),
                srb_test=r(test), srb_test_n0=r(test, 0)), man


def evo_rates(job, man):
    run = json.loads((ROOT / "runs" / job / "run_data.json").read_text())
    meta = man["method_meta"]
    val = run["val_results"]
    name = max(val, key=lambda k: val[k]["avg_score"])
    assert abs(val[name]["avg_score"] - meta["val_score"]) < 1e-9, (job, name)
    assert name.split(" | ") == [op["name"] for op in meta["operators"]], (job, name)
    out = dict(bundle=name, train_live=100 * meta["train_score"], val_evo=100 * meta["val_score"])
    ident = {r["bundle_name"]: r for r in (run.get("identification") or {}).get("records", [])}
    out["train_reeval_ident"] = 100 * ident[name]["fresh_score"] if name in ident else None
    lines = []
    for j in [PARENT.get(job), job]:
        if j and (ROOT / "runs" / j / "run.log").exists():
            lines += [float(m[2]) for m in REEVAL.findall((ROOT / "runs" / j / "run.log").read_text())
                      if m[1] == name]
    out["train_reeval_gen"] = 100 * st.mean(lines) if lines else None
    out["n_reeval_lines"] = len(lines)
    return out


rows = {}
for d in sorted(glob.glob(str(ROOT / "runs/*-srbench_gt_11-9_3seed-90s"))):
    job, rest = Path(d).name.split("-", 1)
    method, seed = rest.replace("-srbench_gt_11-9_3seed-90s", "").rsplit("-s", 1)
    srb, man = srbench_rates(Path(d))
    rows[(method, int(seed))] = {**evo_rates(job, man), **srb, "job": job}

cols = ["train_live", "train_reeval_ident", "train_reeval_gen", "srb_train", "srb_train_n0",
        "val_evo", "srb_val", "srb_val_n0", "srb_test", "srb_test_n0"]
fmt = lambda x: "  -  " if x is None else f"{x:5.1f}"
print(f"{'run':16s}" + "".join(f"{c:>20s}" for c in cols) + "  reeval_lines  bundle")
for (m, s), r in sorted(rows.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
    print(f"{m + '-s' + str(s):16s}" + "".join(f"{fmt(r[c]):>20s}" for c in cols)
          + f"  {r['n_reeval_lines']:>12d}  {r['bundle']}")
print()
for m in METHODS:
    vals = {c: [rows[(m, s)][c] for s in (1, 2, 3) if rows[(m, s)][c] is not None] for c in cols}
    print(f"{m + ' mean':16s}" + "".join(
        f"{(f'{st.mean(v):5.1f}±{st.stdev(v):.1f}' if len(v) > 1 else fmt(v[0] if v else None)):>20s}"
        for v in vals.values()))
