"""Slim a (multi-GB) evolve_pysr / evolve_fullsr run_data.json down to lineage,
scores and operator code, for reading the evolution history by hand.

Usage:
    python scripts/extract_evolution_history.py runs/709715 [runs/941339 ...]

Writes analysis/sr_discoveries/<run>.json with:
  operators   name -> {slot, code, + any scalar metadata (generation, parent_name, mode)}
  generations per generation: best bundle, and for each population/offspring
              bundle its operator names, train score and per-dataset scores
  val_results bundle name -> val / train-reeval averages
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "sr_discoveries"


def slim_bundle(entry, operators, gen, kind):
    ops_key = "functions" if "functions" in entry else "operators"
    ops = {}
    for slot, op in (entry.get(ops_key) or {}).items():
        if not isinstance(op, dict):
            continue
        name = op.get("name")
        ops[slot] = name
        if name not in operators:
            meta = {k: v for k, v in op.items()
                    if k != "code" and isinstance(v, (str, int, float, bool, type(None)))}
            meta["slot"] = slot
            meta["code"] = op.get("code")
            meta["first_seen_gen"] = gen
            meta["first_seen_kind"] = kind
            operators[name] = meta
    details = []
    for d in entry.get("result_details") or []:
        details.append({
            "dataset": d.get("dataset"),
            "avg_gt": d.get("avg_gt"),
            "avg_r2": d.get("avg_r2"),
            "n_runs": len(d.get("run_gt_scores") or d.get("run_r2_scores") or []),
        })
    return {
        "ops": ops,
        "score": entry.get("score"),
        "seeds_evaluated": entry.get("seeds_evaluated"),
        "meta_mutation_counts": entry.get("meta_mutation_counts"),
        "best_hparams": entry.get("best_hparams"),
        "details": details,
    }


def main(run_dir):
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "run_data.json").read_text())
    operators = {}
    gens = []
    for g in data.get("generations", []):
        gi = g.get("generation")
        gens.append({
            "generation": gi,
            "evolved_type": g.get("evolved_type"),
            "mutation_mode": g.get("mutation_mode"),
            "population_type": g.get("population_type"),
            "best_name": g.get("best_name"),
            "best_score": g.get("best_score"),
            "population": [slim_bundle(e, operators, gi, "population") for e in g.get("population", [])],
            "offspring": [slim_bundle(e, operators, gi, "offspring") for e in g.get("offspring", [])],
        })
    best = data.get("best_bundle")
    val = {}
    for key, rec in (data.get("val_results") or {}).items():
        v = rec.get("val", rec) if isinstance(rec, dict) else {}
        tr = rec.get("train_reeval") if isinstance(rec, dict) else None
        val[key] = {
            "bundle_name": rec.get("bundle_name", key),
            "val_avg": v.get("avg_score") if isinstance(v, dict) else None,
            "val_vector": v.get("score_vector") if isinstance(v, dict) else None,
            "gen_submitted": (v.get("gen_submitted") if isinstance(v, dict) else None) or rec.get("gen_submitted"),
            "train_reeval_avg": tr.get("avg_score") if isinstance(tr, dict) else None,
            "train_reeval_vector": tr.get("score_vector") if isinstance(tr, dict) else None,
            "train_score_at_submit": tr.get("train_score_at_submit") if isinstance(tr, dict) else None,
            "extra_keys": [k for k in rec.keys()] if isinstance(rec, dict) else None,
        }
    cfg = dict(data.get("config") or {})
    out = {
        "run": run_dir.name,
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "config": cfg,
        "baseline": data.get("baseline"),
        "generations": gens,
        "best_bundle": slim_bundle(best, operators, None, "best_bundle") if best else None,
        "val_results": val,
        "identification": data.get("identification"),
        "operators": operators,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{run_dir.name}.json"
    path.write_text(json.dumps(out))
    print(f"{run_dir.name}: {len(gens)} gens, {len(operators)} operators -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(arg)
