"""Print the evolution timeline of a slimmed run (see extract_evolution_history.py).

Usage:
    python scripts/summarize_evolution_history.py 709715 [--code NAME ...]

Shows, per generation, the best training bundle and which slot changed, then the
persisted validation / train-reeval records in submission order, then the
bundle the official evaluation selects (--select-by val).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis" / "sr_discoveries"


def bundle_name(b, slots):
    return " | ".join(str(b["ops"].get(s)) for s in slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--code", nargs="*", default=[], help="operator names to print code for")
    ap.add_argument("--offspring", action="store_true", help="list every offspring per generation")
    args = ap.parse_args()
    d = json.loads((SRC / f"{args.run}.json").read_text())
    ops = d["operators"]
    if args.code:
        for name in args.code:
            o = ops[name]
            print(f"##### {name}  slot={o['slot']} meta={ {k: v for k, v in o.items() if k not in ('code',)} }")
            print(o["code"])
        return
    gens = d["generations"]
    slots = list(gens[0]["population"][0]["ops"].keys())
    cfg = d["config"]
    print("run", d["run"], "| metric", cfg.get("fitness_metric"), "| slots", slots)
    print("baseline", {k: v for k, v in (d.get("baseline") or {}).items() if not isinstance(v, list)})
    prev = None
    for g in gens:
        allb = g["population"] + g["offspring"]
        scored = [b for b in allb if b["score"] is not None]
        best = max(g["population"], key=lambda b: (b["score"] if b["score"] is not None else -1))
        cur = best["ops"]
        changed = [s for s in slots if prev is None or prev.get(s) != cur.get(s)]
        off_scores = sorted((b["score"] for b in g["offspring"] if b["score"] is not None), reverse=True)
        print(f"gen {g['generation']:>2} mode={g.get('mutation_mode')} evolved={g.get('evolved_type')} "
              f"best={g.get('best_score')} pop_best={best['score']} n_seeds={best.get('seeds_evaluated')} "
              f"offspring_top={[round(x, 3) for x in off_scores[:3]]}")
        if changed:
            for s in changed:
                o = ops.get(cur[s], {})
                print(f"      {s}: {cur[s]}  (parent={o.get('parent_name')}, mode={o.get('mode')})")
        if args.offspring:
            for b in g["offspring"]:
                new = [s for s in slots if ops.get(b["ops"][s], {}).get("first_seen_gen") == g["generation"]
                       and ops.get(b["ops"][s], {}).get("first_seen_kind") == "offspring"]
                print(f"        off score={b['score']} new={[(s, b['ops'][s]) for s in new]}")
        prev = cur
    print("\nVAL RESULTS (submission order)")
    rows = sorted(d["val_results"].values(), key=lambda r: (r["gen_submitted"] if r["gen_submitted"] is not None else -1))
    best_val = max((r["val_avg"] for r in rows if r["val_avg"] is not None), default=None)
    for r in rows:
        mark = " <== val-selected" if r["val_avg"] is not None and r["val_avg"] == best_val else ""
        print(f"  gen={r['gen_submitted']} val={r['val_avg']} train_reeval={r['train_reeval_avg']} "
              f"train_at_submit={r['train_score_at_submit']}{mark}\n      {r['bundle_name']}")
    if d.get("best_bundle"):
        print("\nBEST_BUNDLE (train):", d["best_bundle"]["score"], bundle_name(d["best_bundle"], slots))
    ident = d.get("identification")
    if ident:
        print("IDENTIFICATION:", {k: v for k, v in ident.items() if k != "records"})


if __name__ == "__main__":
    main()
