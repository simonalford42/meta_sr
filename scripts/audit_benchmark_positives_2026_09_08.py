#!/usr/bin/env python3
"""Reproduce the selected-equation audit of the six September 8 setups.

Reads archived review requests; makes no API calls and submits no jobs.
Original reviews are preserved. N/M reviews are copied without re-evaluation.
The symbolic checks substantiate a human review of every selected E/P equation;
they are specific to these datasets and are not a general equation judge.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/benchmark_positive_audit_2026-09-08.json"
SETUPS = [
    ("emp_base8", "EmpiricalBench baseline / 8 cores / single",
     "runs/empiricalbench_9-8_baseline_8core_l1_60m_no_warmup"),
    ("srb_base8", "SRBench2 baseline / 8 cores / single",
     "runs/srbench2_9-8_gt_baseline_8core_l1_60m_no_warmup"),
    ("srb_base1", "SRBench2 baseline / 1 core / single",
     "runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup"),
    ("srb_base_port", "SRBench2 baseline / 1 core / portfolio",
     "runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup"),
    ("srb_evo1", "SRBench2 evolved 709715 / 1 core / single",
     "runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup"),
    ("srb_evo_port", "SRBench2 evolved 709715 / 1 core / portfolio",
     "runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup"),
]
PHEN = {"first_principles_absorption", "first_principles_bode"}
POSITIVE = {"exact", "phenomenological_match"}
RYDBERG_NEAR = {("srb_base8", 10000), ("srb_base8", 10002), ("srb_evo_port", 10004)}
X = sp.symbols("x0:3", positive=True)
LOCALS = {str(x): x for x in X} | {
    "square": lambda x: x**2, "cube": lambda x: x**3,
    "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
}


def constant(e):
    return not sp.simplify(e).free_symbols


def zero(e):
    return sp.simplify(e) == 0


def same_gradient(e, target):
    # Work with exact rational decimal constants: never snap a fitted value to 1.
    return all(zero(sp.diff(e, x) - sp.diff(target, x)) for x in X)


def exponential_family(e, require_offset=False):
    """Recognize a + b*exp(k*x), including a=0 for the broad P rubric."""
    if e.free_symbols != {X[0]}:
        return False
    d = sp.diff(e, X[0])
    k = sp.simplify(sp.diff(d, X[0]) / d)
    if not constant(k) or zero(k):
        return False
    offset = sp.simplify(e - d / k)
    return constant(offset) and (not require_offset or not zero(offset))


def check_equation(dataset, equation):
    e = sp.sympify(equation, locals=LOCALS, rational=True)
    task = dataset.removeprefix("empirical_").removeprefix("first_principles_")
    x, y, z = X
    if task == "hubble":
        ok = e.free_symbols == {x} and constant(e / x) and not zero(e / x)
        reason = "Proportional to distance: c*x0."
    elif task == "kepler":
        ok = e.free_symbols == {x} and constant(e / x**sp.Rational(3, 2))
        reason = "Proportional to x0^(3/2) on the physical positive domain."
    elif task == "leavitt":
        basis = sp.log(x) if dataset.startswith("empirical_") else x
        slope = sp.simplify(sp.diff(e, x) / sp.diff(basis, x))
        ok = e.free_symbols == {x} and constant(slope) and not zero(slope)
        reason = "Affine in period's logarithm (SRBench2 input is already log period)."
    elif task == "ideal_gas":
        ok = same_gradient(e, sp.log(x*y/z))
        reason = "Equals log(x0*x1/x2) plus a fitted constant."
    elif task == "newton":
        ok = same_gradient(e, sp.log(y*z/x**2))
        reason = "Equals log(x1*x2/x0^2) plus a fitted constant; x0 is distance."
    elif task == "tully_fisher":
        slope = sp.simplify(x * sp.diff(e, x))
        ok = e.free_symbols == {x} and constant(slope) and not zero(slope)
        reason = "Affine in log(rotational velocity), the accepted magnitude-space family."
    elif task == "schechter":
        a = sp.simplify(-x**2 * sp.diff(e, x, 2))
        b = sp.simplify(sp.diff(e, x) - a/x)
        ok = (e.free_symbols == {x} and constant(a) and constant(b)
              and not zero(a) and not zero(b))
        reason = "c0 + c1*log(x0) + c2*x0, with both nonconstant terms present."
    elif task == "rydberg":
        target = -sp.log(1/x**2 - 1/y**2)
        ok = same_gradient(e, target)
        reason = ("Equals -log(1/x0^2 - 1/x1^2) plus a fitted constant."
                  if ok else "Near: fitted relative coefficient differs from the required 1; "
                  "it cannot be absorbed into the overall Rydberg constant.")
    elif task == "supernovae_zr":
        terms = sp.expand(1/e).as_ordered_terms()
        slopes = [sp.simplify(sp.diff(t, x)/t) for t in terms]
        ok = (e.free_symbols == {x} and len(terms) == 2
              and all(constant(k) for k in slopes) and slopes[0]*slopes[1] < 0)
        reason = "Reciprocal of a sum of two exponentials with opposite-sign rates."
    elif task == "bode" and dataset.startswith("empirical_"):
        logs = list(e.atoms(sp.log))
        ok = (len(logs) == 1 and constant(e - logs[0])
              and exponential_family(logs[0].args[0], require_offset=True))
        reason = "log(c0 + c1*exp(c2*x0)); outer constants absorb into inner amplitudes."
    elif task == "bode":
        ok = exponential_family(e)
        reason = ("Broad P rubric only: exponential family, possibly with zero offset. "
                  "Not counted as exact recovery; this does not validate the full nonzero-offset Bode law.")
    elif task == "absorption":
        logs = list(e.atoms(sp.log))
        ok = False
        if len(logs) == 1:
            l = logs[0]
            slope = sp.simplify(sp.diff(e, x)/sp.diff(l, x))
            arg = l.args[0]
            ok = (constant(slope) and not zero(slope)
                  and (arg == x or exponential_family(arg, require_offset=True)
                       or exponential_family(1/arg, require_offset=True)))
        reason = ("Broad P rubric only: log-like empirical family. No unique exact target "
                  "or quantitative fit threshold was supplied, so this is not a validated solve.")
    else:
        raise ValueError(f"Uninspected positive task: {dataset}")
    return bool(ok), reason


def write_report(output):
    codes = {"exact": "E", "near": "N", "miss": "M", "phenomenological_match": "P"}
    lines = [
        "# EmpiricalBench and SRBench2 results — September 8, 2026",
        "",
        "Audited September 9: all **516 original E/P judgments**, covering **518 selected "
        "equations**, were inspected and checked against their archived frontier indices. "
        "Algebraic checks use exact decimal constants, without rounding fitted coefficients "
        "to theoretical values. The other 129 N/M judgments were retained without re-evaluation; "
        "unselected frontier equations were not searched for alternative exact matches.",
        "",
        "All setups used 60 minutes per seed, at most 1,000 samples, and no max-size warmup. "
        "EmpiricalBench used 5 seeds per dataset; SRBench2 used 10. Portfolio searches restart "
        "after 1 million evaluations within a shared 60-minute budget. Single SRBench2 "
        "searches have a 1-billion-evaluation cap. SRBench2 used zero added noise.",
        "",
        "**Exact recovery** means the accepted functional form up to its free constants, "
        "not numerical prediction accuracy. Outer scale/offset is allowed where present in "
        "the accepted family; fixed powers and relative coefficients are not freely adjustable. "
        "For example, Leavitt's linear form and Tully–Fisher's log-linear form may have a zero "
        "intercept. An expression with large or poorly fitted constants can still be a "
        "structural match under this definition.",
        "",
        "**Exact tasks** counts distinct datasets with at least one accepted E seed. "
        "**P is excluded from solved counts**: absorption and SRBench2 Bode have only broad "
        "phenomenological-family criteria. Thus SRBench2 exact-recovery denominators are "
        "10 reference-equation tasks / 100 seeds, with 2 phenomenological tasks / 20 seeds "
        "reported separately. EmpiricalBench has 9 declared reference-equation tasks / 45 seeds.",
        "",
        "## Audited summary",
        "",
        "The baseline uses L1 loss. Evolved PySR uses the validation-selected bundle from "
        "run `709715`, including its evolved loss.",
        "",
        "| Benchmark / setup | Exact tasks | Exact seeds |",
        "|---|---:|---:|",
    ]
    for s in output["setups"]:
        lines.append(f"| {s['label']} | {s['exact_tasks']}/{s['known_tasks']} | "
                     f"{s['exact_seeds']}/{s['known_seeds']} |")
    lines += ["", "Classification counts below are dataset–seed counts. SRBench2 rows sum "
              "to 120, including the 20 phenomenological cases.", "",
              "| Benchmark / setup | Exact | Near | Miss | P |",
              "|---|---:|---:|---:|---:|"]
    for s in output["setups"]:
        c = s["counts"]
        lines.append(f"| {s['label']} | {c.get('exact', 0)} | {c.get('near', 0)} | "
                     f"{c.get('miss', 0)} | {c.get('phenomenological_match', 0)} |")
    lines += [
        "", "## Audit changes and interpretation", "",
        "- **11 EmpiricalBench P → E:** Bode (4), Kepler (1), Leavitt (2), Schechter (4). "
        "The submitted references designate these tasks as ground truth. Their selected "
        "equations match the accepted families; the reviewer incorrectly inferred a "
        "phenomenological category from the word ‘empirical’.",
        "- **Three Rydberg E → N:** baseline 8 cores, seeds 10000 and 10002; evolved "
        "portfolio, seed 10004. The required form is `C - log(1/n1^2 - 1/n2^2)`. "
        "The selected expressions introduce relative coefficients `0.9991896950043856`, "
        "`1.000811129807545`, and `1.0001309684893134`. These alter the inverse-square "
        "difference and cannot be absorbed into the overall constant. Close is near, not exact.",
        "- **Other selected E equations pass.** In particular, the supernova expressions "
        "`exp(a*t)/(exp(t)+b)` are algebraically `1/(exp((1-a)*t)+b*exp(-a*t))`, "
        "which is the accepted two-exponential denominator family. Log-pressure/log-force "
        "targets and the different Leavitt input representations were accounted for.",
        "- **P remains a broad family tag, not a validated solve.** Only SRBench2 absorption "
        "and Bode retain P. Absorption's rubric accepts log-like forms without a unique "
        "target or fit threshold. All 20 evolved Bode selections lack an additive offset "
        "(19 are bare `exp(x0)`). They satisfy the broad exponential-family interpretation, "
        "but do not demonstrate recovery of the full nonzero-offset Bode equation. This "
        "ambiguity is why P is shown separately rather than included in solved counts; "
        "no stricter phenomenological success criterion is retroactively imposed here.",
        "",
        "This is a selected-equation false-positive audit. A downgraded witness does not "
        "prove that no other equation on its frontier is exact. Original review files are "
        "unchanged; this report supersedes its earlier unaudited summary.",
        "", "## Per-task results", "",
        "Each character represents one seed in ascending order: **10000–10004** for "
        "EmpiricalBench and **10000–10009** for SRBench2. **E** = exact, **N** = near, "
        "**M** = miss, **P** = broad phenomenological-family tag. A dash in the exact-seeds "
        "column means the task is excluded from exact-recovery scoring.", "",
    ]
    for s in output["setups"]:
        lines += ["### " + s["label"], "", "| Task | Exact seeds | Per-seed audit |",
                  "|---|---:|---|"]
        by_task = defaultdict(list)
        for r in output["records"]:
            if r["setup"] == s["key"]:
                by_task[r["dataset"]].append(r)
        for ds, rows in sorted(by_task.items()):
            rows.sort(key=lambda r: r["seed"])
            n = 5 if s["key"].startswith("emp_") else 10
            assert [r["seed"] for r in rows] == list(range(10000, 10000+n))
            sequence = "".join(codes[r["audited"]] for r in rows)
            exact = "—" if ds in PHEN else f"{sequence.count('E')}/{n}"
            task = ds.removeprefix("empirical_").removeprefix("first_principles_").replace("_", " ")
            lines.append(f"| {task} | {exact} | `{sequence}` |")
        lines.append("")
    lines += ["## Audit trail and sources", "",
              "- [Per-review audit decisions and source hashes](benchmark_positive_audit_2026-09-08.json)",
              "- [Reproducible audit script](../scripts/audit_benchmark_positives_2026_09_08.py)",
              "- [Reference definitions and original rubric](../manual_solve_check.py)", "",
              "Original review aggregates:", ""]
    for s in output["setups"]:
        lines.append(f"- [{s['label']}](../{s['source']})")
    (ROOT / "analysis/benchmark_results_2026-09-08.md").write_text("\n".join(lines) + "\n")


def main():
    output = {"scope": "Selected equations for every original E/P review; original N/M not re-evaluated.",
              "exact_definition": "Accepted functional form up to the reference's free constants, "
              "including outer scale/offset where allowed; fixed coefficient ratios remain fixed. "
              "Not a claim of numerical prediction accuracy.",
              "phenomenological_definition": "Original broad family-membership rubric, kept separate "
              "from exact recovery and excluded from headline solved counts.",
              "setups": [], "records": []}
    reviewed = references = 0
    for key, label, rel in SETUPS:
        run = ROOT / rel
        source = run / "manual_solve_check_results.json"
        payload = json.loads(source.read_text())
        requests = json.loads((run / "manual_solve_check/batch_input.json").read_text())["requests"]
        items = {r["custom_id"]: json.loads(next(
            m["content"] for m in r["body"]["messages"] if m["role"] == "user")) for r in requests}
        records = []
        for original in payload["reviews"]:
            ds, seed, old = original["dataset"], original["seed"], original["classification"]
            rec = {"setup": key, "dataset": ds, "seed": seed, "original": old,
                   "audited": old, "selected_equation": original["matching_equation"],
                   "reference_indices": original["best_frontier_indices"],
                   "checked": old in POSITIVE, "reason": "Original N/M retained; not re-evaluated."}
            if old in POSITIVE:
                reviewed += 1
                item = items[original["custom_id"]]
                assert (item["dataset"], item["seed"]) == (ds, seed)
                frontier = {r["frontier_index"]: r["equation"] for r in item["frontier"]}
                selected = [frontier[j] for j in original["best_frontier_indices"]]
                assert original["matching_equation"] in selected
                checked = [check_equation(ds, e) for e in selected]
                references += len(checked)
                ok = any(c[0] for c in checked)
                expected_near = ds.endswith("_rydberg") and (key, seed) in RYDBERG_NEAR
                assert ok != expected_near, (key, ds, seed, selected, checked)
                rec["audited"] = "near" if expected_near else (
                    "phenomenological_match" if ds in PHEN else "exact")
                rec["reason"] = next((reason for passed, reason in checked if passed), checked[0][1])
                if old == "phenomenological_match" and ds not in PHEN:
                    rec["reason"] = "P -> E: reference is ground_truth. " + rec["reason"]
                rec["selected_candidates"] = selected
            records.append(rec)
        counts = Counter(r["audited"] for r in records)
        known = [r for r in records if r["dataset"] not in PHEN]
        tasks = {r["dataset"] for r in known}
        solved_tasks = {r["dataset"] for r in known if r["audited"] == "exact"}
        by_task = defaultdict(list)
        for r in records:
            by_task[r["dataset"]].append(r)
        summary = {"key": key, "label": label, "source": str(source.relative_to(ROOT)),
                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "counts": dict(counts), "total": len(records),
                   "exact_tasks": len(solved_tasks), "known_tasks": len(tasks),
                   "exact_seeds": counts["exact"], "known_seeds": len(known),
                   "per_task": {ds: dict(Counter(r["audited"] for r in rows))
                                for ds, rows in sorted(by_task.items())}}
        output["setups"].append(summary)
        output["records"].extend(records)
        print(key, counts, f"exact tasks={len(solved_tasks)}/{len(tasks)}",
              f"exact seeds={counts['exact']}/{len(known)}", flush=True)
    output["positive_judgments_checked"] = reviewed
    output["selected_equations_checked"] = references
    output["changes"] = [r for r in output["records"] if r["original"] != r["audited"]]
    assert reviewed == 516 and references == 518 and len(output["records"]) == 645
    assert len(output["changes"]) == 14
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    write_report(output)
    print(f"Wrote {OUT.relative_to(ROOT)}; {reviewed} positive reviews, "
          f"{references} selected equations, {len(output['changes'])} changed labels.")


if __name__ == "__main__":
    main()
