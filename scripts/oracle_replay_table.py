"""Compact oracle-replay table: fixed-n, promote (n1->n3, n3->n10), TTTS fixed-B,
dynamic B*, KG. Pair average (568245+568246), final generation. Reuses the
cached bundle records and run_policy from scripts/oracle_replay.py."""
import sys, time, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_replay as orr

POLICIES = [
    ("n1",          {"n_base": 1, "reeval": "none"}, False),
    ("n1->n3",      {"n_base": 1, "reeval": "promote", "n_to": 3}, False),
    ("n3",          {"n_base": 3, "reeval": "none"}, False),
    ("n3->n10",     {"n_base": 3, "reeval": "promote", "n_to": 10}, False),
    ("n10",         {"n_base": 10, "reeval": "none"}, False),
    ("TTTS n1 B=10", {"n_base": 1, "reeval": "ttts", "B": 10}, True),
    ("TTTS n1 B=20", {"n_base": 1, "reeval": "ttts", "B": 20}, True),
    ("TTTS n1 B=40", {"n_base": 1, "reeval": "ttts", "B": 40}, True),
    ("TTTS n1 B=60", {"n_base": 1, "reeval": "ttts", "B": 60}, True),
    ("TTTS n2 B=10", {"n_base": 2, "reeval": "ttts", "B": 10}, True),
    ("TTTS n2 B=20", {"n_base": 2, "reeval": "ttts", "B": 20}, True),
    ("TTTS n2 B=40", {"n_base": 2, "reeval": "ttts", "B": 40}, True),
    ("TTTS n2 B=60", {"n_base": 2, "reeval": "ttts", "B": 60}, True),
    ("TTTS n3 B=10", {"n_base": 3, "reeval": "ttts", "B": 10}, True),
    ("TTTS n3 B=20", {"n_base": 3, "reeval": "ttts", "B": 20}, True),
    ("TTTS n3 B=40", {"n_base": 3, "reeval": "ttts", "B": 40}, True),
    ("TTTS n3 B=60", {"n_base": 3, "reeval": "ttts", "B": 60}, True),
    ("TTTS B*",     {"n_base": 1, "reeval": "ttts_dyn"}, True),
    ("KG B=20",     {"n_base": 1, "reeval": "kg", "B": 20}, False),
    ("KG B=40",     {"n_base": 1, "reeval": "kg", "B": 40}, False),
    ("KG B=60",     {"n_base": 1, "reeval": "kg", "B": 60}, False),
]


def main():
    per_run = []
    for rid in orr.RUN_IDS:
        rec = orr.cached_bundle_records(rid)
        assert rec["max_recon_err"] < 1e-6
        records = rec["records"]; orr._prep(records)
        res = {}
        for label, spec, stoch in POLICIES:
            t0 = time.time()
            runs = []
            for s in (range(orr.N_POLICY_SEEDS) if stoch else range(1)):
                rng = np.random.default_rng(1000 * (zlib.crc32(label.encode()) % 997) + s)
                runs.append(orr.run_policy(records, spec, rng))
            res[label] = dict(
                metric=np.mean([r["metric"][-1] for r in runs]),
                seeds=np.mean([r["cum_seeds"][-1] for r in runs]),
                regret=np.mean([r["best_oracle"] - r["obs_argmax_oracle"] for r in runs]),
            )
            print(f"  [{rid}] {label:13s} fit={res[label]['metric']:.4f} "
                  f"seeds={res[label]['seeds']:.0f} regret={res[label]['regret']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        per_run.append(res)

    print("\nPair average (568245+568246), final generation\n")
    print("| policy | parent fitness | seeds spent | final-selection regret |")
    print("|---|---|---|---|")
    import json
    out = orr.OUT_DIR / "oracle_replay_table.json"
    json.dump({label: {k: float(np.mean([r[label][k] for r in per_run]))
                       for k in ("metric", "seeds", "regret")}
               for label, _, _ in POLICIES}, open(out, "w"), indent=1)
    print(f"saved {out}")
    rows = sorted(POLICIES, key=lambda p: -np.mean([r[p[0]]["metric"] for r in per_run]))
    for label, _, _ in rows:
        m = np.mean([r[label]["metric"] for r in per_run])
        sd = np.mean([r[label]["seeds"] for r in per_run])
        rg = np.mean([r[label]["regret"] for r in per_run])
        print(f"| {label} | {m:.3f} | {sd:.0f} | {rg:.3f} |")


if __name__ == "__main__":
    main()
