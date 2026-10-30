"""Tables for reports/pom_wood2010.md: Ricker replicate study and blowfly fits."""
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRUTH = np.array([3.8, np.log(0.3), np.log(10.0)])


def ricker(run="runs/pom-wood-ricker", sl_from=None):
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, run, "replicates/seed*.json")))]
    if sl_from:  # take SL results from a rerun on the same datasets
        alt = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, sl_from, "replicates/seed*.json"))}
        reps = [dict(r, sl=alt[r["seed"]]["sl"]) for r in reps if r["seed"] in alt]
        run = f"{run} (SL from {sl_from})"
    if not reps:
        return
    TRUTH = np.array(reps[0].get("truth", [3.8, np.log(0.3), np.log(10.0)]))
    methods = [m for m in ["traj", "onestep", "sl", "pf"] if m in reps[0]]
    print(f"\n## {run}: {len(reps)} datasets, truth {np.round(TRUTH, 3)}\n")
    print("| method | param | mean est. | bias | sd | RMSE | median abs. err | 95% coverage |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    for m in methods:
        for j, p in enumerate(["log r", "log σ_e", "log φ"]):
            if m in ("traj", "onestep"):
                if j == 1:
                    continue
                est = np.array([r[m]["logr"] if j == 0 else np.log(r[m]["phi"]) for r in reps])
                # asymptotic LS SE: traj se index 0 = log r, 2 = log phi; onestep se 0 = log r, 1 = log phi
                k = 0 if j == 0 else (2 if m == "traj" else 1)
                se = np.array([r[m]["se"][k] for r in reps])
                cov = np.mean(np.abs(est - TRUTH[j]) <= 1.96 * se)
            else:
                est = np.array([r[m]["mean"][j] for r in reps])
                lo = np.array([r[m]["q025"][j] for r in reps])
                hi = np.array([r[m]["q975"][j] for r in reps])
                cov = np.mean((lo <= TRUTH[j]) & (TRUTH[j] <= hi))
            err = est - TRUTH[j]
            rmse = np.sqrt(np.mean(err ** 2))
            out[(m, p)] = rmse
            print(f"| {m} | {p} | {est.mean():.3f} | {err.mean():+.3f} | {est.std():.3f} | {rmse:.3f} | "
                  f"{np.median(np.abs(err)):.3f} | {cov:.0%} |")
    # paired comparison on log r
    e = {m: np.array([(r[m]["logr"] if m in ("traj", "onestep") else r[m]["mean"][0]) - TRUTH[0] for r in reps])
         for m in methods}
    for m in [m for m in methods if m != "sl"]:
        print(f"\n|err| SL < |err| {m} on {np.mean(np.abs(e['sl']) < np.abs(e[m])):.0%} of datasets")
    print("\nSL acceptance mean", np.mean([r["sl"]["acc"] for r in reps]))
    if "traj" not in methods:
        return
    print("traj fitted N_1 at lower bound (<=0.0011):", np.mean([r["traj"]["n_init"] < 0.0011 for r in reps]))
    print("traj R^2 (1-SSE/SST):", np.round(np.median([1 - r["traj"]["sse"] / (np.var(r["y"]) * len(r["y"])) for r in reps]), 3))
    print("onestep R^2:", np.round(np.median([1 - r["onestep"]["sse"] / (np.var(r["y"][1:]) * (len(r["y"]) - 1)) for r in reps]), 3))


def blow(fn):
    d = json.load(open(fn))
    print(f"\n### {os.path.basename(fn)}")
    print("data patterns:", {k: round(v) for k, v in d["data_patterns"].items()})
    if "truth" in d:
        print("truth:", {k: round(v, 3) for k, v in d["truth"].items()})
    print("SL posterior mean:", {k: round(v, 3) for k, v in d["sl"]["mean"].items()}, "acc", round(d["sl"]["acc"], 2))
    print("SL 95%:", {k: (round(d["sl"]["q025"][k], 3), round(d["sl"]["q975"][k], 3)) for k in d["sl"]["mean"]})
    if "onestep" in d:
        print("onestep:", {k: round(v, 3) for k, v in d["onestep"].items()})
    print("traj:", {k: round(v, 3) for k, v in d["traj"].items() if k != "sse_by_tau"})
    print("| fit | δ | P | N0 | τ | var_p, var_d | synth. loglik | stats in 95% band | |z|>3 stats | skeleton period / mean / min / max | stochastic median period / mean / sd / min / max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    pars = {"synthetic likelihood": d["sl"]["mean"], "one-step MSE": d.get("onestep"), "trajectory MSE": d["traj"],
            "truth": d.get("truth")}
    for e in d["eval"]:
        p = pars[e["name"]]
        sk, st = e["skeleton"], e["stoch_summary"]
        nz = sum(abs(v) > 3 for v in e["stat_z"].values())
        print(f"| {e['name']} | {p['delta']:.3f} | {p['P']:.2f} | {p['N0']:.0f} | {p['tau']:.1f} | "
              f"{e['noise'][0]:.2g}, {e['noise'][1]:.2g}{' (profiled)' if e.get('noise_profiled') else ''} | "
              f"{e['sl_loglik']:.1f} | {e['frac_stats_in_95']:.0%} | {nz}/23 | "
              f"{sk['period']:.0f} / {sk['mean']:.0f} / {sk['min']:.0f} / {sk['max']:.0f} | "
              f"{st['period']:.0f} / {st['mean']:.0f} / {st['sd']:.0f} / {st['min']:.0f} / {st['max']:.0f} |")
    for e in d["eval"]:
        bad = {k: round(v, 1) for k, v in e["stat_z"].items() if abs(v) > 3}
        print(f"  {e['name']} |z|>3: {bad}")


if __name__ == "__main__":
    ricker()
    ricker("runs/pom-wood-ricker-phi2")
    ricker("runs/pom-wood-ricker-phi2", sl_from="runs/pom-wood-ricker-phi2-mcwm")
    for f in sorted(glob.glob(os.path.join(ROOT, "runs/pom-wood-blowfly/*.json"))):
        blow(f)
