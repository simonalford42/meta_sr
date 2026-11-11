#!/usr/bin/env python3
"""One-time conversion of LLM-SRBench LSR-Synth data to plain .npz + JSON.

LLM-SRBench (Shojaee et al., ICML 2025, arXiv:2504.10415) ships its data as a
gated HF dataset (nnheui/llm-srbench): per-domain parquet metadata + one
lsr_bench_data.hdf5 with /lsr_synth/<domain>/<name>/{train_data,id_test_data,
ood_test_data}; column 0 is the target (see baselines/llm-srbench-src/bench/
datamodules.py, pipelines.py).

Needs h5py + pyarrow, which the meta_sr env lacks, so run it in the small
`lsrbench_data` env:

    # after accepting the dataset terms on huggingface.co and `export HF_TOKEN=...`
    ~/.conda/envs/lsrbench_data/bin/python scripts/lsrsynth_prepare_data.py
    # or from the ungated third-party mirror (pkuHaowei/llm-srbench, parquet per domain)
    ~/.conda/envs/lsrbench_data/bin/python scripts/lsrsynth_prepare_data.py --mirror data/llm-srbench/mirror_pkuHaowei
    # or from an existing snapshot dir containing lsr_bench_data.hdf5 + data/*.parquet
    ~/.conda/envs/lsrbench_data/bin/python scripts/lsrsynth_prepare_data.py --snapshot DIR

Writes data/llm-srbench/lsr_synth/<domain>/<name>.npz (train, id_test, ood_test)
and data/llm-srbench/lsr_synth/manifest.json (symbols, descs, GT expression).
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "llm-srbench" / "lsr_synth"
DOMAINS = ("chem_react", "bio_pop_growth", "phys_osc", "matsci")
SPLITS = {"train": "train_data", "id_test": "id_test_data", "ood_test": "ood_test_data"}


def from_mirror(mirror, pq):
    """Ungated third-party repack (huggingface.co/datasets/pkuHaowei/llm-srbench)."""
    manifest = []
    for dom in DOMAINS:
        rows = pq.read_table(mirror / f"lsr_synth_{dom}.parquet").to_pylist()
        (OUT / dom).mkdir(parents=True, exist_ok=True)
        for r in rows:
            name = r["instance_id"].removeprefix(f"lsr_synth_{dom}_").upper()
            arrs = {}
            for k, (xi, yo) in {"train": ("train_input", "train_output"),
                                "id_test": ("test_input", "test_output"),
                                "ood_test": ("ood_input", "ood_output")}.items():
                if r.get(xi) is None:
                    continue
                X = np.asarray(r[xi], np.float64)
                y = np.asarray(r[yo], np.float64).reshape(-1, 1)
                arrs[k] = np.hstack([y, X])
            symbols = list(r["output_vars"]) + list(r["input_vars"])
            np.savez(OUT / dom / f"{name}.npz", **arrs)
            manifest.append({
                "domain": dom, "name": name, "instance_id": r["instance_id"],
                "symbols": symbols, "all_symbols": list(r["symbols"]),
                "description": r["description"], "expression": r["gt_expression"],
                "shapes": {k: list(a.shape) for k, a in arrs.items()},
                "source": "pkuHaowei/llm-srbench",
            })
        print(f"{dom}: {len(rows)} problems")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} problems -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None,
                    help="local dir with lsr_bench_data.hdf5 and data/*.parquet (skip download)")
    ap.add_argument("--mirror", default=None,
                    help="dir with ungated pkuHaowei/llm-srbench parquet files lsr_synth_<domain>.parquet")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    if args.mirror:
        return from_mirror(Path(args.mirror), pq)
    import h5py

    if args.snapshot:
        snap = Path(args.snapshot)
    else:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(repo_id="nnheui/llm-srbench", repo_type="dataset"))

    manifest = []
    with h5py.File(snap / "lsr_bench_data.hdf5", "r") as h5:
        for dom in DOMAINS:
            pqs = sorted((snap / "data").glob(f"lsr_synth_{dom}-*.parquet"))
            assert pqs, f"no parquet for {dom} in {snap/'data'}"
            rows = [r for p in pqs for r in pq.read_table(p).to_pylist()]
            (OUT / dom).mkdir(parents=True, exist_ok=True)
            for r in rows:
                g = h5[f"/lsr_synth/{dom}/{r['name']}"]
                arrs = {k: g[v][...].astype(np.float64) for k, v in SPLITS.items() if v in g}
                np.savez(OUT / dom / f"{r['name']}.npz", **arrs)
                manifest.append({
                    "domain": dom, "name": r["name"],
                    "symbols": list(r["symbols"]), "symbol_descs": list(r["symbol_descs"]),
                    "symbol_properties": list(r["symbol_properties"]),
                    "expression": r["expression"],
                    "shapes": {k: list(a.shape) for k, a in arrs.items()},
                })
            print(f"{dom}: {len(rows)} problems")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} problems -> {OUT}")


if __name__ == "__main__":
    main()
