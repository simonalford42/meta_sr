#!/usr/bin/env python3
"""Estimate the pooled per-seed noise sigma for a finished evolve_pysr run.

This reproduces the live `pooled_sigma` calculation (evolution_helpers.py) over
*all* bundles ever logged in a run's run_data.json — i.e. the same cumulative
within-bundle std the racing/smart-reeval code estimates on the fly, but
computed once over the whole run so it can be reused as a fixed sigma.

run_data.json files are large (multi-GB for racing runs), so bundles are
streamed one at a time with ijson and their result_details trimmed to just the
per-seed score arrays before anything is retained in memory.

Usage:
    python scripts/estimate_sigma.py [RUN_ID] [--fitness-metric gt|r2]
    python scripts/estimate_sigma.py 414990
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import ijson

# Import the production calculation so this stays in lock-step with evolution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evolution_helpers import pooled_sigma  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO / "runs"

OP_ORDER = ["mutation", "survival", "selection", "loss"]


class _SanitizingReader:
    """File-like wrapper that rewrites the non-standard JSON tokens Python's
    json.dump emits (NaN / Infinity / -Infinity) into valid numbers so strict
    streaming parsers (ijson/yajl) accept the file.

    Rewriting is letter-for-letter and only ever lands in numeric value
    positions or — harmlessly — inside string text we never read (we only
    consume numeric score arrays and operator names), so the document stays
    structurally valid either way. A 9-byte carry prevents splitting a token
    across read-chunk boundaries.
    """

    def __init__(self, raw, chunk=1 << 20):
        self.raw = raw
        self.chunk = chunk
        self.carry = b""   # held-back tail that may be a split token
        self.buf = b""     # sanitized bytes ready to hand out
        self.eof = False

    @staticmethod
    def _sub(b):
        return (b.replace(b"-Infinity", b"-0.0")
                 .replace(b"Infinity", b"0.0")
                 .replace(b"NaN", b"0.0"))

    def _fill(self):
        """Top up self.buf with at least one sanitized chunk (unless at EOF)."""
        while not self.buf and not self.eof:
            raw = self.raw.read(self.chunk)
            if not raw:
                self.eof = True
                if self.carry:
                    self.buf += self._sub(self.carry)
                    self.carry = b""
            else:
                data = self.carry + raw
                self.carry = data[-9:]          # avoid splitting a token
                self.buf += self._sub(data[:-9])

    def read(self, size=-1):
        if size is None or size < 0:
            parts = []
            while True:
                self._fill()
                if not self.buf:
                    break
                parts.append(self.buf)
                self.buf = b""
            return b"".join(parts)
        # Honor the requested size exactly — yajl's C backend keeps only the
        # first `size` bytes it asked for and discards the rest, so returning
        # an oversized chunk silently drops data and corrupts the parse.
        out = b""
        while len(out) < size:
            self._fill()
            if not self.buf:
                break
            take = size - len(out)
            out += self.buf[:take]
            self.buf = self.buf[take:]
        return out


def _display_name(operators: dict) -> str:
    """Mirror OperatorBundle.display_name from the serialized operators dict."""
    parts = []
    for t in OP_ORDER:
        op = (operators or {}).get(t)
        parts.append(op.get("name") if op else "default")
    return " | ".join(p if p else "default" for p in parts)


def _trim_details(result_details):
    """Keep only the per-seed score arrays pooled_sigma needs."""
    trimmed = []
    for d in result_details or []:
        # ijson yields numbers as decimal.Decimal; cast to float so the
        # production np.mean/np.var path matches the live calculation.
        trimmed.append({
            "run_r2_scores": [float(x) for x in (d.get("run_r2_scores") or [])],
            "run_gt_scores": [float(x) for x in (d.get("run_gt_scores") or [])],
        })
    return trimmed


def _read_config(path: Path):
    """Pull fitness_metric and n_runs from the (small, leading) config block."""
    cfg = {}
    with open(path, "rb") as fh:
        for prefix, event, val in ijson.parse(_SanitizingReader(fh)):
            if prefix.startswith("config.") and event in ("string", "number", "boolean"):
                cfg[prefix.split(".", 1)[1]] = val
            elif prefix == "generations" and event == "start_array":
                break  # config fully read; stop before the heavy part
    return cfg


def _collect_bundles(path: Path):
    """Stream every bundle (population + offspring across all generations),
    deduped by display_name keeping the entry with the most seeds (most
    complete result_details, matching how racing accumulates seeds)."""
    best_by_name = {}  # display_name -> (seeds, SimpleNamespace)
    for array in ("population", "offspring"):
        prefix = f"generations.item.{array}.item"
        with open(path, "rb") as fh:
            for bundle in ijson.items(_SanitizingReader(fh), prefix):
                seeds = int(bundle.get("seeds_evaluated") or 0)
                if seeds < 2:
                    continue  # can't contribute a within-bundle variance
                name = _display_name(bundle.get("operators"))
                prev = best_by_name.get(name)
                if prev is None or seeds > prev[0]:
                    obj = SimpleNamespace(
                        seeds_evaluated=seeds,
                        result_details=_trim_details(bundle.get("result_details")),
                    )
                    best_by_name[name] = (seeds, obj)
    return [obj for _, obj in best_by_name.values()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?", default="414990",
                    help="SLURM job id / run dir name under runs/ (default: 414990)")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT),
                    help="Directory containing run dirs (default: <repo>/runs)")
    ap.add_argument("--json-path", default=None,
                    help="Explicit path to run_data.json (overrides run_id lookup)")
    ap.add_argument("--fitness-metric", choices=["gt", "r2"], default=None,
                    help="Metric for per-seed scores (default: read from run config)")
    args = ap.parse_args()

    if args.json_path:
        path = Path(args.json_path)
    else:
        path = Path(args.runs_root) / args.run_id / "run_data.json"
    if not path.exists():
        ap.error(f"run_data.json not found: {path}")

    cfg = _read_config(path)
    metric = args.fitness_metric or cfg.get("fitness_metric") or "gt"
    print(f"run        : {args.run_id}")
    print(f"run_data   : {path}")
    print(f"metric     : {metric}"
          + ("" if args.fitness_metric else " (from config)"))
    print(f"n_runs     : {cfg.get('n_runs')}  reeval={cfg.get('reeval')}  racing={cfg.get('racing')}")
    print("streaming bundles (this reads the full run_data.json)...")

    bundles = _collect_bundles(path)
    sigma = pooled_sigma(bundles, metric)

    n_seeds = sum(b.seeds_evaluated for b in bundles)
    print(f"\nunique bundles with >=2 seeds : {len(bundles)}")
    print(f"total accumulated seeds        : {n_seeds}")
    print(f"\npooled sigma (metric={metric}) : {sigma:.6f}")


if __name__ == "__main__":
    main()
