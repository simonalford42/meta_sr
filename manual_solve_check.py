#!/usr/bin/env python3
"""OpenRouter Batch API, one-frontier-per-request benchmark solve checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT_VERSION = 1
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}
DEFAULT_MODEL = "openai/gpt-5.6-terra"

# OpenRouter Batch prices in dollars per million tokens. Keep synchronized with
# https://openrouter.ai/api/v1/models. OpenAI cache writes are billed as ordinary
# input; cache reads receive the listed discount.
BATCH_PRICES = {
    "openai/gpt-5.6-luna": {"input": 0.10, "cached": 0.01, "write": 0.10, "output": 0.60},
    "openai/gpt-5.6-terra": {"input": 1.00, "cached": 0.10, "write": 1.00, "output": 6.00},
    "openai/gpt-5.6-sol": {"input": 1.00, "cached": 0.10, "write": 1.00, "output": 5.00},
}

TARGETS = {
    "first_principles_absorption": {
        "kind": "phenomenological",
        "target": "No unique ground-truth equation. Judge whether the frontier contains a simple compelling log/tanh-like empirical relationship.",
    },
    "first_principles_bode": {
        "kind": "phenomenological",
        "target": "No unique ground-truth equation. The conventional reference family is a = c0 + c1*exp(c2*n).",
    },
    "first_principles_hubble": {"kind": "ground_truth", "target": "v = c*D"},
    "first_principles_ideal_gas": {
        "kind": "ground_truth",
        "target": "logP = log(c*n*T/V) = c0 + log(n) + log(T) - log(V); the stored target is log pressure.",
    },
    "first_principles_kepler": {"kind": "ground_truth", "target": "P = c*a^(3/2)"},
    "first_principles_leavitt": {
        "kind": "ground_truth",
        "target": "M = c0 + c1*logP; the input feature is already log10(period).",
    },
    "first_principles_newton": {
        "kind": "ground_truth",
        "target": "logF = log(c*m1*m2/r^2) = c0 + log(m1) + log(m2) - 2*log(r); the stored target is log force.",
    },
    "first_principles_planck": {
        "kind": "ground_truth",
        "target": "logB = log(c0*nu^3/(exp(c1*nu/T)-1)); the stored target is log spectral radiance.",
    },
    "first_principles_rydberg": {
        "kind": "ground_truth",
        "target": "log(lambda) = log(c/(1/n1^2 - 1/n2^2)); the stored target is log wavelength.",
    },
    "first_principles_schechter": {
        "kind": "ground_truth",
        "target": "log(phi) = c0 + alpha*log(L) - L/c1; the stored target is log number density.",
    },
    "first_principles_supernovae_zr": {
        "kind": "ground_truth",
        "target": "flux = c0/(c1*exp(c2*t) + exp(-c3*t))",
    },
    "first_principles_tully_fisher": {
        "kind": "ground_truth",
        "target": "M = c0 + c1*log(DV), the magnitude-space form of L proportional to DV^2.5; the stored target is astronomical magnitude.",
    },
    "empirical_hubble": {"kind": "ground_truth", "target": "v = c*D"},
    "empirical_kepler": {"kind": "ground_truth", "target": "P = c*a^(3/2)"},
    "empirical_newton": {
        "kind": "ground_truth",
        "target": "logF = c0 + log(m1) + log(m2) - 2*log(r)",
    },
    "empirical_bode": {
        "kind": "ground_truth",
        "target": "loga = log(c0 + c1*exp(c2*n))",
    },
    "empirical_leavitt": {
        "kind": "ground_truth",
        "target": "M = c0 + c1*log(P)",
    },
    "empirical_schechter": {
        "kind": "ground_truth",
        "target": "logn = c0 + c1*log(L) + c2*L",
    },
    "empirical_ideal_gas": {
        "kind": "ground_truth",
        "target": "logP = c0 + log(n) + log(T) - log(V)",
    },
    "empirical_planck": {
        "kind": "ground_truth",
        "target": "logB = log(c0*nu^3/(exp(c1*nu/T)-1))",
    },
    "empirical_rydberg": {
        "kind": "ground_truth",
        "target": "log(lambda) = log(c/(1/n1^2 - 1/n2^2))",
    },
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["exact", "near", "miss", "phenomenological_match", "not_applicable", "error"],
        },
        "best_frontier_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "matching_equation": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
    },
    "required": ["classification", "best_frontier_indices", "matching_equation", "explanation"],
    "additionalProperties": False,
}

# Deliberately detailed and stable: the rubric is useful calibration material
# and makes the shared prefix large enough to qualify for GPT-5.6 caching.
RUBRIC = """You are an exacting symbolic-regression judge. Review one complete
Pareto frontier from one independent search seed. Compare every candidate with
the supplied accepted relationship using algebra and functional structure, not
fit quality alone. Numerical constants are freely fitted parameters: different
finite nonzero numerical values do not prevent an exact classification when
the variable dependence is identical. Algebraic rearrangements, commutation,
association, cancelling factors, integer powers, x*sqrt(x)=x^(3/2), and a
constant absorbed into another fitted constant are allowed.

Use `exact` only when at least one candidate has the accepted variable
dependence with no genuine nonconstant extra or missing term. A tiny coefficient
does not make a nonconstant term disappear. An expression that merely fits the
sampled range, a Taylor approximation, an asymptotic limit, or a transformed
version of the requested dependent variable is not exact unless the reference
explicitly says that transformed dependent variable is stored.

Use `near` when a candidate clearly contains the essential accepted structure
but has a genuine nonconstant extra/missing factor or term, or is a recognizable
asymptotic or numerical approximation. Use `miss` when no candidate contains
the accepted structure. For a phenomenological dataset, never use `exact`:
use `phenomenological_match` if the stated empirical family occurs,
`not_applicable` when the reference gives no judgeable family, and `miss` when a
judgeable stated family is absent. Use `error` only when the supplied frontier
is unusable.

Inspect all candidates, including simple early frontier rows. Prefer the
simplest exact candidate. For a near result, select the strongest structural
candidate rather than merely the highest-R2 candidate. Return frontier indices
exactly as supplied. `matching_equation` must reproduce the selected candidate
equation verbatim, not restate the reference. Return an empty index list and a
null matching equation for miss, not_applicable, or error. Keep the explanation
under 60 words and identify the decisive algebraic match or mismatch.

Calibration examples: c*x0 and 3.7*x0 are the same Hubble family. x0*sqrt(x0)
is exactly x0^(3/2). log(c*x0*x1/x2) is exact for a stored log-pressure target
but only near for an untransformed pressure target. c0+log(x0)+log(x1)-2*log(x2)
is the log of c*x0*x1/x2^2. Adding 0.0001*sin(x0) prevents exactness. A Wien-law
approximation is near, not exact, for Planck's law. A polynomial matching seven
planet observations is not exact Kepler recovery. A constant offset is allowed
only when the accepted family contains a free offset or when the benchmark's
equivalence rule explicitly permits it. Do not infer cancellation unless it is
algebraically valid for the entire input domain.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_review_items(run_dir: Path) -> list[dict[str, Any]]:
    empirical_path = run_dir / "empbench_results.json"
    if empirical_path.exists():
        payload = json.loads(empirical_path.read_text())
        unknown = sorted(set(payload.get("protocol", {}).get("datasets", [])) - set(TARGETS))
        if unknown:
            raise ValueError(f"Unsupported EmpiricalBench datasets: {unknown[:5]}")
        items = []
        for index, result in enumerate(payload.get("runs", [])):
            if result.get("error") or not result.get("frontier"):
                continue
            source_payload = {
                "dataset": result["dataset"],
                "seed": int(result["seed"]),
                "noise": 0.0,
                "frontier": [
                    {
                        "frontier_index": frontier_index,
                        "complexity": row.get("complexity"),
                        "equation": row.get("equation"),
                    }
                    for frontier_index, row in enumerate(result["frontier"])
                ],
            }
            source_hash = hashlib.sha256(
                json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            items.append({
                **source_payload,
                "source_hash": source_hash,
                "custom_id": f"e-t{index:06d}",
            })
        return items

    manifest = json.loads((run_dir / "manifest.json").read_text())
    unknown = sorted(set(manifest.get("datasets", [])) - set(TARGETS))
    if unknown:
        raise ValueError(
            f"API manual solve check does not support datasets: {unknown[:5]}"
        )
    items = []
    for batch_index, batch in enumerate(manifest.get("batches", [])):
        batch_dir = run_dir / batch["batch_dir"]
        tasks = json.loads((batch_dir / "tasks.json").read_text())
        for task_index, task in enumerate(tasks):
            result_path = batch_dir / "results" / f"task_{task_index:06d}.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            if result.get("error") or not result.get("pareto_frontier"):
                continue
            dataset = task["dataset_name"]
            frontier = [
                {
                    "frontier_index": index,
                    "complexity": row.get("complexity"),
                    "equation": row.get("equation"),
                }
                for index, row in enumerate(result["pareto_frontier"])
            ]
            source_payload = {
                "dataset": dataset,
                "seed": int(task["seed"]) + int(task.get("run_index", 0)),
                "noise": float(task.get("target_noise", 0.0)),
                "frontier": frontier,
            }
            source_hash = hashlib.sha256(
                json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            items.append({
                **source_payload,
                "source_hash": source_hash,
                "custom_id": f"b{batch_index:03d}-t{task_index:06d}",
            })
    return items


def developer_prompt(dataset: str) -> str:
    reference = TARGETS[dataset]
    return (
        RUBRIC
        + "\nDataset-specific reference follows.\n"
        + f"Dataset: {dataset}\n"
        + f"Reference type: {reference['kind']}\n"
        + f"Accepted relationship: {reference['target']}\n"
    )


def build_request(item: dict[str, Any], model: str, reasoning_effort: str,
                  max_output_tokens: int) -> dict[str, Any]:
    shared = developer_prompt(item["dataset"])
    changing = json.dumps({
        "dataset": item["dataset"],
        "seed": item["seed"],
        "noise": item["noise"],
        "frontier": item["frontier"],
    }, separators=(",", ":"))
    return {
        "custom_id": item["custom_id"],
        "body": {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "messages": [
                {"role": "system", "content": shared},
                {"role": "user", "content": changing},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "srbench_frontier_review",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                },
            },
            "provider": {"require_parameters": True},
            "max_tokens": max_output_tokens,
        },
    }


def estimate_cost_upper(requests: list[dict[str, Any]], model: str,
                        max_output_tokens: int) -> dict[str, Any]:
    prices = BATCH_PRICES[model]
    # Math-heavy JSON tokenizes less efficiently than prose. Two characters per
    # token is intentionally conservative. Assume no cache hits for the guard.
    estimated_input_tokens = sum(
        (len(json.dumps(request["body"], separators=(",", ":"))) + 1) // 2
        for request in requests
    )
    maximum_output_tokens = len(requests) * max_output_tokens
    dollars = (
        estimated_input_tokens * prices["input"]
        + maximum_output_tokens * prices["output"]
    ) / 1_000_000
    return {
        "method": "conservative_chars_div_2_no_cache",
        "estimated_input_tokens": estimated_input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "maximum_cost_usd": dollars,
    }


class OpenRouterHTTPClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, data: bytes | None = None,
                 content_type: str | None = None) -> tuple[bytes, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"OpenRouter API {method} {path} failed ({exc.code}): {detail}") from exc

    def json_request(self, method: str, path: str, payload: Any | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        raw, _ = self._request(method, path, data, "application/json" if data else None)
        return json.loads(raw)

def _extract_output_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content
    for output in body.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    raise ValueError("response contains no output_text")


def calculate_cost(usage: dict[str, Any], model: str) -> float:
    prices = BATCH_PRICES[model]
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    writes = int(details.get("cache_write_tokens") or 0)
    ordinary = max(0, input_tokens - cached - writes)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return (
        ordinary * prices["input"]
        + cached * prices["cached"]
        + writes * prices["write"]
        + output_tokens * prices["output"]
    ) / 1_000_000


def _validate_review(review: dict[str, Any], frontier_size: int) -> None:
    classification = review.get("classification")
    allowed = set(REVIEW_SCHEMA["properties"]["classification"]["enum"])
    if classification not in allowed:
        raise ValueError(f"invalid classification {classification!r}")
    indices = review.get("best_frontier_indices")
    if not isinstance(indices, list) or any(
        not isinstance(index, int) or index < 0 or index >= frontier_size
        for index in indices
    ):
        raise ValueError("invalid best_frontier_indices")
    if not isinstance(review.get("explanation"), str):
        raise ValueError("explanation must be a string")
    equation = review.get("matching_equation")
    if equation is not None and not isinstance(equation, str):
        raise ValueError("matching_equation must be a string or null")


def write_aggregate(run_dir: Path, review_records: list[dict[str, Any]],
                    model: str, reasoning_effort: str) -> None:
    review_records.sort(key=lambda row: (row["dataset"], row["seed"], row["noise"]))
    counts = Counter(row["classification"] for row in review_records)
    total_cost = sum(float(row.get("cost_usd") or 0.0) for row in review_records)
    payload = {
        "format_version": 1,
        "reviewer": "openrouter-batch-api",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "run_dir": str(run_dir.resolve()),
        "n_reviews": len(review_records),
        "classification_counts": dict(counts),
        "total_cost_usd": total_cost,
        "reviews": review_records,
    }
    write_json_atomic(run_dir / "manual_solve_check_results.json", payload)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_records:
        by_dataset[row["dataset"]].append(row)
    lines = [
        "# API manual solve check", "",
        f"Model: `{model}`; reasoning: `{reasoning_effort}`; cost: `${total_cost:.4f}`", "",
        "| Dataset | Exact | Near | Phenomenological | Miss | N/A | Error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, rows in sorted(by_dataset.items()):
        c = Counter(row["classification"] for row in rows)
        lines.append(
            f"| {dataset} | {c['exact']} | {c['near']} | "
            f"{c['phenomenological_match']} | {c['miss']} | "
            f"{c['not_applicable']} | {c['error']} |"
        )
    lines.extend(["", "## Per-frontier evidence", ""])
    for dataset, rows in sorted(by_dataset.items()):
        lines.extend([f"### {dataset}", ""])
        for row in rows:
            indices = ", ".join(map(str, row["best_frontier_indices"])) or "—"
            equation = row["matching_equation"] or "—"
            lines.append(
                f"- Seed {row['seed']}: **{row['classification']}** — "
                f"frontier `{indices}` — `{equation}` — {row['explanation']}"
            )
        lines.append("")
    (run_dir / "manual_solve_check_results.md").write_text("\n".join(lines) + "\n")


def process_batch_results(run_dir: Path, results: list[dict[str, Any]],
                          items_by_id: dict[str, dict[str, Any]], model: str,
                          reasoning_effort: str) -> list[dict[str, Any]]:
    review_dir = run_dir / "manual_solve_check" / "reviews"
    records = []
    seen = set()
    for envelope in results:
        custom_id = envelope["custom_id"]
        item = items_by_id[custom_id]
        seen.add(custom_id)
        response = envelope.get("response") or {}
        body = response.get("body") or {}
        try:
            status_code = response.get("status_code", response.get("status", 200 if body else 0))
            if envelope.get("error") or int(status_code) != 200:
                raise ValueError(str(envelope.get("error") or body)[:1000])
            review = json.loads(_extract_output_text(body))
            _validate_review(review, len(item["frontier"]))
        except Exception as exc:
            review = {
                "classification": "error",
                "best_frontier_indices": [],
                "matching_equation": None,
                "explanation": f"API response validation failed: {exc}"[:500],
            }
        usage = body.get("usage") or {}
        record = {
            "dataset": item["dataset"], "seed": item["seed"], "noise": item["noise"],
            **review,
            "model": model, "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION, "source_hash": item["source_hash"],
            "custom_id": custom_id,
            "api_request_id": response.get("request_id") or body.get("id"),
            "usage": usage, "cost_usd": calculate_cost(usage, model), "reviewed_at": _now(),
        }
        write_json_atomic(review_dir / f"{custom_id}.json", record)
        records.append(record)
    for custom_id, item in items_by_id.items():
        if custom_id in seen:
            continue
        record = {
            "dataset": item["dataset"], "seed": item["seed"], "noise": item["noise"],
            "classification": "error", "best_frontier_indices": [],
            "matching_equation": None, "explanation": "Batch output omitted this request.",
            "model": model, "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION, "source_hash": item["source_hash"],
            "custom_id": custom_id, "api_request_id": None, "usage": {},
            "cost_usd": 0.0, "reviewed_at": _now(),
        }
        write_json_atomic(review_dir / f"{custom_id}.json", record)
        records.append(record)
    write_aggregate(run_dir, records, model, reasoning_effort)
    return records


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    items = load_review_items(run_dir)
    if not items:
        raise SystemExit(f"No usable Pareto frontiers found under {run_dir}")
    work_dir = run_dir / "manual_solve_check"
    state_path = work_dir / "batch_state.json"
    items_by_id = {item["custom_id"]: item for item in items}

    state = json.loads(state_path.read_text()) if state_path.exists() and not args.force else None
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY is required (it is never written to disk or logs)")

    requests = [
        build_request(item, args.model, args.reasoning_effort, args.max_output_tokens)
        for item in items
    ]
    estimate = estimate_cost_upper(requests, args.model, args.max_output_tokens)
    request_sources = [
        {"custom_id": item["custom_id"], "source_hash": item["source_hash"]}
        for item in items
    ]
    if state is not None:
        expected_state = {
            "provider": "openrouter",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "requests": request_sources,
        }
        mismatches = [
            key for key, expected in expected_state.items()
            if state.get(key) != expected
        ]
        if mismatches:
            raise SystemExit(
                "Existing Batch state does not match this invocation/source data "
                f"({', '.join(mismatches)}). Resume with the original arguments, or "
                "use --force to explicitly submit a new paid Batch."
            )
    print(
        f"Prepared {len(requests)} independent frontier reviews; conservative "
        f"maximum ${estimate['maximum_cost_usd']:.2f} (guard ${args.max_cost:.2f}).",
        flush=True,
    )
    if estimate["maximum_cost_usd"] > args.max_cost:
        raise SystemExit("Refusing paid submission: estimated maximum exceeds --max-cost")

    batch_payload = {
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "requests": requests,
    }
    input_path = work_dir / "batch_input.json"
    write_json_atomic(input_path, batch_payload)
    if args.dry_run:
        print(f"Dry run only; wrote {input_path}")
        return 0

    client = OpenRouterHTTPClient(api_key, args.base_url)
    if state is None:
        batch = client.json_request("POST", "/beta/batches", batch_payload)
        state = {
            "format_version": 1, "provider": "openrouter",
            "created_at": _now(), "run_dir": str(run_dir),
            "model": args.model, "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens, "cost_estimate": estimate,
            "batch_id": batch["id"], "status": batch.get("status"),
            "requests": request_sources,
        }
        write_json_atomic(state_path, state)
        print(f"Submitted OpenRouter Batch {batch['id']} with {len(requests)} requests.", flush=True)

    while True:
        batch = client.json_request("GET", f"/beta/batches/{state['batch_id']}")
        state.update({
            "status": batch.get("status"), "checked_at": _now(),
            "request_counts": batch.get("request_counts"),
        })
        write_json_atomic(state_path, state)
        print(f"Batch {state['batch_id']}: {state['status']} {state.get('request_counts') or ''}", flush=True)
        if state["status"] in TERMINAL_BATCH_STATES:
            break
        if args.no_wait:
            print("Exiting after submission/status check; rerun the same command to resume.")
            return 0
        time.sleep(args.poll_seconds)

    if state["status"] != "completed":
        raise SystemExit(f"Batch ended with status {state['status']}; inspect {state_path}")
    results = batch.get("results")
    if not isinstance(results, list):
        raise SystemExit("Completed OpenRouter Batch has no inline results")
    write_json_atomic(work_dir / "responses.json", {"results": results})
    records = process_batch_results(
        run_dir, results, items_by_id, args.model, args.reasoning_effort
    )
    total = sum(row["cost_usd"] for row in records)
    print(f"Wrote {len(records)} reviews; measured Batch cost ${total:.4f}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", choices=sorted(BATCH_PRICES), default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default="https://openrouter.ai/api", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_output_tokens <= 0 or args.max_cost < 0 or args.poll_seconds <= 0:
        raise SystemExit("token, cost, and polling limits must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
