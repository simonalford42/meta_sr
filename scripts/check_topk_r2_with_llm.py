#!/usr/bin/env python3
"""
Run SymPy+LLM symbolic matching on top-K highest-R^2 PySR results (1e8, zero noise).

Default behavior:
  - Input results: results/results_pysr_1e8/*.json
  - K: 50
  - Sequential processing
  - For each case:
      1) SymPy symbolic check (timeout default 10s)
      2) LLM equivalence check
      3) Combined match = sympy_match OR llm_match
      4) If sympy_match=True and llm_match=False -> raise RuntimeError
  - Print running timing stats (mean/min/max/std) for SymPy and LLM checks.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import check_pysr_symbolic_match, check_sympy_equivalence_with_llm
from utils import load_srbench_dataset, PMLB_PATH
from completions import get_usage, reset_usage


def get_var_names(dataset_name: str):
    """Load feature names from PMLB dataset header."""
    dataset_path = PMLB_PATH / dataset_name / f"{dataset_name}.tsv.gz"
    df = pd.read_csv(dataset_path, sep="\t", compression="gzip", nrows=0)
    return [c for c in df.columns if c != "target"]


def summarize_times(values):
    """Return mean/min/max/std for a list of times."""
    if not values:
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def fmt_stats(label: str, values):
    s = summarize_times(values)
    return (
        f"{label} mean/min/max/std="
        f"{s['mean']:.4f}/{s['min']:.4f}/{s['max']:.4f}/{s['std']:.4f}s"
    )


def load_topk_rows(results_dir: Path, k: int):
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        dataset = d.get("dataset")
        r2 = d.get("test_r2")
        best_equation = d.get("best_equation")
        if dataset is None or r2 is None or best_equation is None:
            continue
        try:
            r2 = float(r2)
        except Exception:
            continue
        if not math.isfinite(r2):
            continue
        rows.append(
            {
                "dataset": dataset,
                "r2": r2,
                "best_equation": str(best_equation),
            }
        )
    rows.sort(key=lambda x: x["r2"], reverse=True)
    return rows[: min(k, len(rows))]


def load_llm_false_datasets(jsonl_path: Path):
    """Load unique datasets where llm_match is False from a prior JSONL run."""
    out = []
    seen = set()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ds = rec.get("dataset")
            llm_match = rec.get("llm_match")
            if ds and llm_match is False and ds not in seen:
                seen.add(ds)
                out.append(ds)
    return out


def write_llm_parse_debug_dump(
    debug_dir: Path,
    case_index: int,
    dataset: str,
    r2: float,
    best_equation: str,
    ground_truth: str,
    var_names,
    sympy_result,
    llm_result,
    llm_model: str,
    llm_thinking_level: str,
):
    """Write a debug dump JSON for LLM parse failures."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{case_index:03d}_{dataset}_llm_parse_error.json"
    payload = {
        "case_index": case_index,
        "dataset": dataset,
        "r2": r2,
        "best_equation": best_equation,
        "ground_truth": ground_truth,
        "var_names": var_names,
        "llm_model": llm_model,
        "llm_thinking_level": llm_thinking_level,
        "sympy_result": sympy_result,
        "llm_result": llm_result,
        "notes": (
            "LLM response parsing failed. Inspect llm_result.raw_response and "
            "re-run one-off checks with this case."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def write_llm_disagreement_debug_dump(
    debug_dir: Path,
    case_index: int,
    dataset: str,
    r2: float,
    best_equation: str,
    ground_truth: str,
    var_names,
    sympy_result,
    llm_result,
    llm_model: str,
    llm_thinking_level: str,
):
    """Write a debug dump JSON for hard disagreements (sympy True, llm False with no llm error)."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{case_index:03d}_{dataset}_llm_disagreement.json"
    payload = {
        "case_index": case_index,
        "dataset": dataset,
        "r2": r2,
        "best_equation": best_equation,
        "ground_truth": ground_truth,
        "var_names": var_names,
        "llm_model": llm_model,
        "llm_thinking_level": llm_thinking_level,
        "sympy_result": sympy_result,
        "llm_result": llm_result,
        "notes": (
            "SymPy and LLM disagreed: sympy_match=True but llm_match=False "
            "with no LLM parse/API error."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Check top-K high-R^2 PySR results with SymPy+LLM symbolic matching."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/results_pysr_1e8"),
        help="Directory containing PySR result JSON files.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=130,
        help="Number of top-R^2 tasks to process.",
    )
    parser.add_argument(
        "--sympy-timeout-seconds",
        type=int,
        default=10,
        help="SymPy symbolic-check timeout in seconds.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-5.2",
        help="LLM model identifier.",
    )
    parser.add_argument(
        "--llm-thinking-level",
        type=str,
        default="high",
        help="LLM reasoning effort level.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=0,
        help="Max output tokens for LLM equivalence response. Use 0 for uncapped.",
    )
    parser.add_argument(
        "--raise-disagreement",
        action="store_true",
        help="Raise when sympy_match=True but llm_match=False with no LLM error.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable completion cache for LLM queries.",
    )
    parser.add_argument(
        "--save-jsonl",
        type=Path,
        default=Path("results/symbolic_checks_pysr_1e8/topk_best_sympy_llm_results.jsonl"),
        help="Path to save per-case results as JSONL.",
    )
    parser.add_argument(
        "--llm-debug-dir",
        type=Path,
        default=Path("results/symbolic_checks_pysr_1e8/llm_parse_debug"),
        help="Directory to save debug dumps when LLM response parsing fails.",
    )
    parser.add_argument(
        "--rerun-llm-false-jsonl",
        type=Path,
        default=None,
        help="If set, ignore top-K and rerun only datasets where llm_match=False in this JSONL.",
    )
    parser.add_argument(
        "--rerun-report-txt",
        type=Path,
        default=Path("results/symbolic_checks_pysr_1e8/llm_false_rerun_report.txt"),
        help="Text report path for rerun mode.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    rows = load_topk_rows(args.results_dir, args.top_k)
    if args.rerun_llm_false_jsonl is not None:
        llm_false_ds = set(load_llm_false_datasets(args.rerun_llm_false_jsonl))
        rows = [r for r in rows if r["dataset"] in llm_false_ds]
        if not rows:
            all_rows = load_topk_rows(args.results_dir, 10_000)
            rows = [r for r in all_rows if r["dataset"] in llm_false_ds]
        print(
            f"Rerun mode: loaded {len(llm_false_ds)} llm-false datasets from "
            f"{args.rerun_llm_false_jsonl}; {len(rows)} found in results dir."
        )
    if not rows:
        print("No valid result rows found.")
        return

    args.save_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_jsonl, "w"):
        pass

    sympy_times = []
    llm_times = []
    total_times = []
    llm_costs = []
    n_match = 0
    n_sympy_match = 0
    n_llm_match = 0
    rerun_report_lines = []
    reset_usage()

    print(f"Processing {len(rows)} tasks from {args.results_dir}")
    print(f"SymPy timeout: {args.sympy_timeout_seconds}s")
    llm_max_tokens = None if args.llm_max_tokens <= 0 else args.llm_max_tokens
    token_label = "uncapped" if llm_max_tokens is None else str(llm_max_tokens)
    print(f"LLM model: {args.llm_model} (thinking={args.llm_thinking_level}, max_tokens={token_label})")
    print("-" * 100)

    for i, row in enumerate(rows, 1):
        dataset = row["dataset"]
        r2 = row["r2"]
        best_equation = row["best_equation"]

        try:
            _, _, ground_truth = load_srbench_dataset(dataset)
            var_names = get_var_names(dataset)
        except Exception as e:
            result = {
                "index": i,
                "dataset": dataset,
                "r2": r2,
                "best_equation": best_equation,
                "error": f"dataset_load_error: {e}",
            }
            with open(args.save_jsonl, "a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[{i:02d}/{len(rows)}] {dataset:30s} R2={r2:.8f} ERROR loading dataset: {e}")
            continue

        case_start = time.perf_counter()
        sympy_start = time.perf_counter()
        sympy_result = check_pysr_symbolic_match(
            best_equation,
            ground_truth,
            var_names=var_names,
            timeout_seconds=args.sympy_timeout_seconds,
        )
        sympy_elapsed = time.perf_counter() - sympy_start

        llm_start = time.perf_counter()
        usage_before = get_usage()
        llm_result = check_sympy_equivalence_with_llm(
            predicted_expr=best_equation,
            ground_truth_expr=ground_truth,
            model=args.llm_model,
            thinking_level=args.llm_thinking_level,
            max_tokens=llm_max_tokens,
            use_cache=not args.no_cache,
        )
        usage_after = get_usage()
        llm_elapsed = time.perf_counter() - llm_start
        llm_cost = max(0.0, float(usage_after.get("total_cost", 0.0)) - float(usage_before.get("total_cost", 0.0)))
        llm_costs.append(llm_cost)

        sympy_match = bool(sympy_result.get("match", False))
        llm_match = bool(llm_result.get("llm_match", False))

        llm_error = str(llm_result.get("error") or "")
        llm_parse_error = "parse" in llm_error.lower()
        llm_debug_dump_path = None
        disagreement_debug_dump_path = None
        if llm_parse_error:
            llm_debug_dump_path = write_llm_parse_debug_dump(
                debug_dir=args.llm_debug_dir,
                case_index=i,
                dataset=dataset,
                r2=r2,
                best_equation=best_equation,
                ground_truth=ground_truth,
                var_names=var_names,
                sympy_result=sympy_result,
                llm_result=llm_result,
                llm_model=args.llm_model,
                llm_thinking_level=args.llm_thinking_level,
            )
            print(
                f"  [LLM PARSE ERROR] dataset={dataset} case={i} "
                f"dump={llm_debug_dump_path}"
            )

        llm_hard_no = (llm_result.get("error") in (None, "")) and (not llm_match)
        if sympy_match and llm_hard_no:
            disagreement_debug_dump_path = write_llm_disagreement_debug_dump(
                debug_dir=args.llm_debug_dir,
                case_index=i,
                dataset=dataset,
                r2=r2,
                best_equation=best_equation,
                ground_truth=ground_truth,
                var_names=var_names,
                sympy_result=sympy_result,
                llm_result=llm_result,
                llm_model=args.llm_model,
                llm_thinking_level=args.llm_thinking_level,
            )
            print(
                f"  [LLM DISAGREEMENT] dataset={dataset} case={i} "
                f"dump={disagreement_debug_dump_path}"
            )
            if args.raise_disagreement:
                raise RuntimeError(
                    "Unexpected disagreement: sympy_match=True but llm_match=False "
                    f"for dataset={dataset}, equation={best_equation}, ground_truth={ground_truth}. "
                    f"sympy_result={sympy_result}, llm_result={llm_result}"
                )

        combined_match = bool(sympy_match or llm_match)
        case_elapsed = time.perf_counter() - case_start

        sympy_times.append(sympy_elapsed)
        llm_times.append(llm_elapsed)
        total_times.append(case_elapsed)

        n_match += int(combined_match)
        n_sympy_match += int(sympy_match)
        n_llm_match += int(llm_match)

        result = {
            "index": i,
            "dataset": dataset,
            "r2": r2,
            "best_equation": best_equation,
            "ground_truth": ground_truth,
            "var_names": var_names,
            "sympy_match": sympy_match,
            "llm_match": llm_match,
            "combined_match": combined_match,
            "sympy_elapsed_seconds": sympy_elapsed,
            "llm_elapsed_seconds": llm_elapsed,
            "total_elapsed_seconds": case_elapsed,
            "sympy_result": sympy_result,
            "llm_result": llm_result,
            "llm_explanation": llm_result.get("reasoning", ""),
            "llm_cost_usd": llm_cost,
            "llm_parse_error": llm_parse_error,
            "llm_debug_dump_path": str(llm_debug_dump_path) if llm_debug_dump_path else None,
            "llm_disagreement": bool(sympy_match and llm_hard_no),
            "llm_disagreement_debug_dump_path": (
                str(disagreement_debug_dump_path) if disagreement_debug_dump_path else None
            ),
        }
        with open(args.save_jsonl, "a") as f:
            f.write(json.dumps(result) + "\n")

        if args.rerun_llm_false_jsonl is not None:
            rerun_report_lines.append("=" * 120)
            rerun_report_lines.append(f"dataset: {dataset} | R2={r2:.8f}")
            rerun_report_lines.append(f"sympy_match={sympy_match} | llm_match={llm_match} | combined={combined_match}")
            rerun_report_lines.append(f"llm_cost_usd={llm_cost:.6f}")
            rerun_report_lines.append(f"ground_truth: {ground_truth}")
            rerun_report_lines.append(f"best_equation: {best_equation}")
            rerun_report_lines.append(f"llm_explanation: {llm_result.get('reasoning', '')}")
            rerun_report_lines.append(f"llm_raw_response: {llm_result.get('raw_response', '')}")
            rerun_report_lines.append(f"sympy_result: {json.dumps(sympy_result, ensure_ascii=True)}")

        print(
            f"[{i:02d}/{len(rows)}] {dataset:30s} R2={r2:.8f} "
            f"sympy={sympy_match} llm={llm_match} combined={combined_match} | "
            f"{fmt_stats('sympy', sympy_times)} | "
            f"{fmt_stats('llm', llm_times)} | "
            f"{fmt_stats('total', total_times)} | "
            f"cost_this_query=${llm_cost:.6f}"
        )

    print("-" * 100)
    print(f"Completed {len(total_times)} cases (requested top-K={len(rows)}).")
    print(f"Combined matches: {n_match}/{len(total_times)}")
    print(f"SymPy matches:    {n_sympy_match}/{len(total_times)}")
    print(f"LLM matches:      {n_llm_match}/{len(total_times)}")
    print(fmt_stats("sympy", sympy_times))
    print(fmt_stats("llm", llm_times))
    print(fmt_stats("total", total_times))
    usage = get_usage()
    total_cost = float(usage.get("total_cost", 0.0))
    total_queries = int(usage.get("num_calls", 0))
    avg_cost_per_query = (total_cost / total_queries) if total_queries > 0 else 0.0
    avg_cost_per_case = (total_cost / len(total_times)) if total_times else 0.0
    print(
        f"LLM cost stats: total=${total_cost:.6f} across {total_queries} API queries; "
        f"avg/query=${avg_cost_per_query:.6f}; avg/case=${avg_cost_per_case:.6f}"
    )
    print(f"Saved JSONL: {args.save_jsonl}")
    if args.rerun_llm_false_jsonl is not None:
        args.rerun_report_txt.parent.mkdir(parents=True, exist_ok=True)
        args.rerun_report_txt.write_text("\n".join(rerun_report_lines) + "\n")
        print(f"Saved rerun report: {args.rerun_report_txt}")


if __name__ == "__main__":
    main()
