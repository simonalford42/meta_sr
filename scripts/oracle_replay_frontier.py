"""Full policy frontier for the oracle replay (pair average of runs 568245 +
568246, final generation). Grid:
  fixed    N_init in 1..10
  promote  N_init in 1..4  x  N_reeval in 1..9  (N_init + N_reeval <= 10)
  TTTS     N_init in 1..4  x  B in {5,10,20,40,60,80,100}
Reeval policies whose total seeds spent reach >= 75% of the oracle's (N_init=10)
are dropped; fixed-N is kept in full. Writes plots/oracle_replay/oracle_replay_frontier.json."""
import json, sys, time, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_replay as orr

BUDGET_FRAC = 0.75


def grid():
    P = []
    for n in range(1, 11):
        P.append((f"n{n}", "fixed", n, None, {"n_base": n, "reeval": "none"}, False))
    for n in range(1, 5):
        for r in range(1, 10):
            if n + r <= 10:
                P.append((f"n{n}->n{n+r}", f"promote n{n}", n, r,
                          {"n_base": n, "reeval": "promote", "n_to": n + r}, False))
    for n in range(1, 5):
        for B in (5, 10, 20, 40, 60, 80, 100):
            P.append((f"TTTS n{n} B={B}", f"ttts n{n}", n, B,
                      {"n_base": n, "reeval": "ttts", "B": B}, True))
    return P


def main():
    recs = {}
    for rid in orr.RUN_IDS:
        rec = orr.cached_bundle_records(rid)
        assert rec["max_recon_err"] < 1e-6
        orr._prep(rec["records"]); recs[rid] = rec["records"]
    out = {}
    for label, family, n, param, spec, stoch in grid():
        t0 = time.time()
        per_run = []
        for rid in orr.RUN_IDS:
            runs = []
            for s in (range(orr.N_POLICY_SEEDS) if stoch else range(1)):
                rng = np.random.default_rng(1000 * (zlib.crc32(label.encode()) % 997) + s)
                runs.append(orr.run_policy(recs[rid], spec, rng))
            per_run.append((np.mean([r["metric"][-1] for r in runs]),
                            np.mean([r["cum_seeds"][-1] for r in runs])))
        m = float(np.mean([p[0] for p in per_run])); sd = float(np.mean([p[1] for p in per_run]))
        out[label] = {"family": family, "n_init": n, "param": param, "metric": m, "seeds": sd}
        print(f"{label:16s} fam={family:12s} fit={m:.4f} seeds={sd:6.0f} ({time.time()-t0:.1f}s)", flush=True)
    oracle_seeds = out["n10"]["seeds"]
    # fixed-N kept in full (reference curve up to the oracle); reeval policies
    # filtered to < BUDGET_FRAC of the oracle's seeds.
    kept = {k: v for k, v in out.items()
            if v["family"] == "fixed" or v["seeds"] < BUDGET_FRAC * oracle_seeds}
    dropped = sorted(set(out) - set(kept))
    print(f"\nkept {len(kept)}/{len(out)}; dropped (>= {BUDGET_FRAC:.0%} of oracle seeds {oracle_seeds:.0f}): {dropped}")
    path = orr.OUT_DIR / "oracle_replay_frontier.json"
    json.dump({"budget_frac": BUDGET_FRAC, "oracle_seeds": oracle_seeds, "policies": kept},
              open(path, "w"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
