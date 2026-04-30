#!/usr/bin/env python3
"""Generate operator proposal diversity reports for run 947961."""

from __future__ import annotations

import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "947961"
RUN_DIR = ROOT / "runs" / RUN_ID
OP_DIR = RUN_DIR / "operators"
LOG_PATH = ROOT / "out" / f"{RUN_ID}.out"

ALL_SUMMARY = ROOT / f"{RUN_ID}_operator_proposal_summaries.txt"
EXPLORE_SUMMARY = ROOT / f"{RUN_ID}_operator_proposal_summaries_explore_only.txt"
INITIAL_ANALYSIS = ROOT / f"{RUN_ID}_initial_explore_operator_diversity.md"
EXPLORE_ANALYSIS = ROOT / f"{RUN_ID}_explore_operator_diversity.md"

TYPES = ("mutation", "survival", "selection")

CATEGORY_ORDER = [
    "mutation: data-aware constant/local fit",
    "mutation: residual/feature-guided construction",
    "mutation: rational/denominator/physics motif",
    "mutation: algebraic rewrite/simplification",
    "mutation: subtree reuse/recombination/symmetry",
    "mutation: generic structural wrapper/operator edit",
    "selection: Pareto/AFPO multi-objective",
    "selection: lexicase/epsilon-lexicase",
    "selection: rank/roulette/Boltzmann/global sampling",
    "selection: complexity niche/rarity/diversity",
    "selection: age/recency-biased",
    "survival: Pareto/crowding/niche preservation",
    "survival: redundancy/stepping-stone pruning",
    "survival: age-regularized/worst-oldest",
    "survival: worst-cost/reverse tournament",
    "survival: bloat/complexity culling",
]


def parse_modes() -> dict[tuple[int, int], str]:
    """Return {(generation, index): mode} from the run log."""
    modes: dict[tuple[int, int], str] = {}
    current_gen: int | None = None
    gen_re = re.compile(r"^Generation\s+(\d+)/")
    created_re = re.compile(r"^\s*Created:\s+(\S+)\s+\(mode=(\w+),")
    init_re = re.compile(r"^\s*(mutation|survival|selection):\s+(\S+_init_(\d+))\s+\(model=")

    for line in LOG_PATH.read_text().splitlines():
        gen_match = gen_re.match(line)
        if gen_match:
            current_gen = int(gen_match.group(1))
            continue

        init_match = init_re.match(line)
        if init_match:
            idx = int(init_match.group(3))
            modes[(0, idx)] = "explore"
            continue

        created_match = created_re.match(line)
        if not created_match or current_gen is None:
            continue
        name, mode = created_match.groups()
        name_match = re.search(r"_gen(\d+)_(\d+)$", name)
        if not name_match:
            continue
        gen = int(name_match.group(1))
        idx = int(name_match.group(2))
        modes[(gen, idx)] = mode

    return modes


def parse_operator_path(path: Path) -> tuple[int, str, int]:
    match = re.match(r"gen(\d+)_(mutation|survival|selection)(\d+)\.jl$", path.name)
    if not match:
        raise ValueError(f"Unexpected operator filename: {path}")
    return int(match.group(1)), match.group(2), int(match.group(3))


def extract_docstring(code: str) -> str:
    match = re.match(r'\s*"""(.*?)"""', code, re.S)
    if not match:
        return ""
    return textwrap.dedent(match.group(1)).strip()


def summarize_docstring(doc: str) -> str:
    if not doc:
        return "No docstring found."
    lines = [line.strip() for line in doc.splitlines()]
    body_lines = []
    in_signature = True
    for line in lines:
        if in_signature:
            if not line:
                in_signature = False
            continue
        if not line:
            if body_lines:
                break
            continue
        if line.startswith(("Steps:", "Steps when", "1.", "- ")):
            break
        body_lines.append(line)
    text = " ".join(body_lines) or " ".join(line for line in lines if line)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("e.g.", "e.g").replace("i.e.", "i.e")
    sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    summary = sentence_match.group(1) if sentence_match else text
    if len(summary) > 300:
        summary = summary[:299].rstrip() + "..."
    return summary


def load_records() -> list[dict[str, object]]:
    modes = parse_modes()
    records = []
    for path in sorted(OP_DIR.glob("gen*_*.jl"), key=lambda p: parse_operator_path(p)):
        gen, op_type, idx = parse_operator_path(path)
        code = path.read_text()
        doc = extract_docstring(code)
        mode = modes.get((gen, idx), "unknown")
        records.append(
            {
                "generation": gen,
                "operator_type": op_type,
                "index": idx,
                "mode": mode,
                "path": path,
                "docstring": doc,
                "summary": summarize_docstring(doc),
            }
        )
    return records


def write_summary(records: list[dict[str, object]], path: Path, explore_only: bool = False) -> None:
    selected = [r for r in records if not explore_only or r["mode"] == "explore"]
    by_gen: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in selected:
        by_gen[int(record["generation"])].append(record)

    out = []
    for gen in sorted(by_gen):
        out.append(f"GENERATION {gen}:")
        for record in sorted(by_gen[gen], key=lambda r: (TYPES.index(str(r["operator_type"])), int(r["index"]))):
            out.append(
                f"{record['operator_type']}: {record['mode']}:\t {record['summary']}"
            )
        out.append("")
    path.write_text("\n".join(out).rstrip() + "\n")


def phrase_counts(records: list[dict[str, object]]) -> Counter[str]:
    buckets = Counter()
    for record in records:
        text = f"{record['path'].name} {record['summary']}".lower()
        if any(x in text for x in ("least-squares", "constant", "newton", "affine fit")):
            buckets["data-fit/constant optimization"] += 1
        if any(x in text for x in ("correlation", "residual", "feature", "dataset", "data-aware")):
            buckets["data-aware residual/correlation"] += 1
        if any(x in text for x in ("pareto", "age", "novelty", "rarity", "diversity", "crowd")):
            buckets["selection/survival diversity pressure"] += 1
        if any(x in text for x in ("distribute", "fold", "factor", "rewrite", "simplif")):
            buckets["algebraic rewrite/simplification"] += 1
        if any(x in text for x in ("rational", "ratio", "denominator", "reciprocal")):
            buckets["rational/denominator structure"] += 1
    return buckets


def record_text(record: dict[str, object]) -> str:
    return f"{record['path'].name} {record['summary']} {record['docstring']}".lower()


def primary_text(record: dict[str, object]) -> str:
    return f"{record['path'].name} {record['summary']}".lower()


def has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def has_age_terms(text: str) -> bool:
    return bool(re.search(r"\b(age|aged|birth|youth|younger|newer|recency|oldest|older)\b|age[-_/]|age_fitness", text))


def primary_category(record: dict[str, object]) -> str:
    text = primary_text(record)
    op_type = str(record["operator_type"])

    if op_type == "mutation":
        if has_any(text, ("fold", "distribute", "factor", "rewrite", "simplif", "identity", "annihilation", "cancellation", "refactor")):
            return "mutation: algebraic rewrite/simplification"
        if has_any(text, ("reuse", "recombination", "crossover", "copy", "duplicate", "repeated", "symmetry", "symmetric", "building block", "combine")):
            return "mutation: subtree reuse/recombination/symmetry"
        if has_any(text, ("rational", "denominator", "reciprocal", "sqrt", "lorentz", "pade", "padé", "ratio", "feedback", "one-plus", "1 / sqrt", "numerator over denominator", "physics motif")):
            return "mutation: rational/denominator/physics motif"
        if has_any(text, ("least-squares", "least squares", "gauss", "newton", "constant-refinement", "constant refinement", "affine correction", "affine calibration", "closed-form", "calibrate", "local fit", "leaf-refinement", "constant leaf", "constant node")):
            return "mutation: data-aware constant/local fit"
        if has_any(text, ("residual", "feature", "correlation", "boosting", "data-aware", "dataset.x", "dataset.y", "linear readout")):
            return "mutation: residual/feature-guided construction"
        return "mutation: generic structural wrapper/operator edit"

    if op_type == "selection":
        if has_any(text, ("lexicase", "epsilon-lexicase", "macro-lexicase")):
            return "selection: lexicase/epsilon-lexicase"
        if has_any(text, ("boltzmann", "softmax", "roulette", "rank-based", "ranking", "borda", "global rank", "fitness-proportionate", "linear ranking")):
            return "selection: rank/roulette/Boltzmann/global sampling"
        if has_any(text, ("pareto", "afpo", "multi-objective", "dominance", "frontier")):
            return "selection: Pareto/AFPO multi-objective"
        if has_any(text, ("complexity", "niche", "rarity", "novelty", "diversity", "underrepresented", "frequency")):
            return "selection: complexity niche/rarity/diversity"
        if has_age_terms(text):
            return "selection: age/recency-biased"
        return "selection: complexity niche/rarity/diversity"

    if op_type == "survival":
        if has_any(text, ("pareto", "crowd", "niche", "diversity", "front", "overcrowded", "complexity bin")):
            return "survival: Pareto/crowding/niche preservation"
        if has_any(text, ("redundant", "stepping stone", "stepping-stone", "loss-complexity landscape", "weak trade-offs", "unpromising")):
            return "survival: redundancy/stepping-stone pruning"
        if has_any(text, ("bloat", "highest complexity", "most complex", "complexity culling", "strictly controlling equation bloat")):
            return "survival: bloat/complexity culling"
        if has_age_terms(text):
            return "survival: age-regularized/worst-oldest"
        if has_any(text, ("reverse tournament", "anti-tournament", "inverse tournament", "worst", "highest cost", "worst-fitness", "worst fitness")):
            return "survival: worst-cost/reverse tournament"
        return "survival: Pareto/crowding/niche preservation"

    return "uncategorized"


def secondary_tags(record: dict[str, object]) -> set[str]:
    text = record_text(record)
    tags = set()
    tests = {
        "data-aware": ("data-aware", "dataset.x", "dataset.y", "residual", "correlation", "least-squares", "least squares", "gauss", "newton"),
        "constant fitting": ("constant", "least-squares", "least squares", "gauss", "newton", "affine calibration", "affine correction"),
        "rational/denominator": ("rational", "denominator", "reciprocal", "ratio", "sqrt", "lorentz", "pade", "padé"),
        "algebraic rewrite": ("fold", "distribute", "factor", "rewrite", "simplif", "identity", "cancellation", "refactor"),
        "subtree reuse": ("reuse", "recombination", "crossover", "copy", "duplicate", "repeated", "building block", "combine"),
        "Pareto/frontier": ("pareto", "frontier", "multi-objective", "dominance", "afpo"),
        "complexity diversity": ("complexity", "niche", "rarity", "novelty", "diversity", "crowd", "underrepresented", "frequency"),
        "age/recency": ("birth", "oldest", "older", "younger", "newer", "recency", "youth", "age-", "age_fitness", " age "),
        "lexicase": ("lexicase",),
        "rank/softmax": ("rank", "roulette", "boltzmann", "softmax", "borda", "fitness-proportionate"),
        "worst/reverse survival": ("reverse tournament", "anti-tournament", "inverse tournament", "highest cost", "worst"),
        "bloat pressure": ("bloat", "highest complexity", "most complex"),
    }
    for tag, needles in tests.items():
        if tag == "age/recency" and has_age_terms(text):
            tags.add(tag)
        elif tag != "age/recency" and has_any(text, needles):
            tags.add(tag)
    return tags


def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "0.0%"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def write_explore_analysis(records: list[dict[str, object]]) -> None:
    explore = [r for r in records if r["mode"] == "explore"]
    for record in explore:
        record["primary_category"] = primary_category(record)
        record["secondary_tags"] = secondary_tags(record)

    total = len(explore)
    by_type = Counter(str(r["operator_type"]) for r in explore)
    by_category = Counter(str(r["primary_category"]) for r in explore)
    by_tag = Counter(tag for r in explore for tag in r["secondary_tags"])
    by_gen = Counter(int(r["generation"]) for r in explore)

    category_rows = []
    for category in CATEGORY_ORDER:
        count = by_category.get(category, 0)
        if count:
            type_counts = Counter(str(r["operator_type"]) for r in explore if r["primary_category"] == category)
            category_rows.append([
                category,
                count,
                pct(count, total),
                ", ".join(f"{t}={type_counts[t]}" for t in TYPES if type_counts[t]),
            ])

    type_rows = []
    for op_type in TYPES:
        type_total = by_type[op_type]
        cats = Counter(str(r["primary_category"]) for r in explore if r["operator_type"] == op_type)
        top_cats = ", ".join(f"{cat.replace(op_type + ': ', '')}: {n}" for cat, n in cats.most_common(5))
        type_rows.append([op_type, type_total, pct(type_total, total), top_cats])

    tag_rows = [[tag, count, pct(count, total)] for tag, count in by_tag.most_common()]

    phase_defs = [
        ("initial", 0, 0),
        ("early", 1, 10),
        ("middle", 11, 30),
        ("late", 31, 50),
    ]
    phase_rows = []
    for label, start, end in phase_defs:
        phase = [r for r in explore if start <= int(r["generation"]) <= end]
        cats = Counter(str(r["primary_category"]) for r in phase)
        type_counts = Counter(str(r["operator_type"]) for r in phase)
        top = ", ".join(f"{cat.split(': ', 1)[1]} ({n})" for cat, n in cats.most_common(4))
        phase_rows.append([
            f"{label} ({start}" + (f"-{end}" if start != end else "") + ")",
            len(phase),
            ", ".join(f"{t}={type_counts[t]}" for t in TYPES if type_counts[t]),
            top,
        ])

    examples = defaultdict(list)
    for record in explore:
        category = str(record["primary_category"])
        if len(examples[category]) < 3:
            examples[category].append(record)

    out = [
        "# Explore Proposal Diversity Across All Generations",
        "",
        f"Run: `{RUN_ID}`",
        f"Source summary: `{EXPLORE_SUMMARY.relative_to(ROOT)}`",
        "",
        "## High-Level Stats",
        "",
        f"Total explore proposals: {total} across {len(by_gen)} generations.",
        "",
    ]
    out.extend(markdown_table(["operator type", "count", "share", "largest buckets"], type_rows))
    out.extend([
        "",
        "## Primary Buckets",
        "",
        "Each proposal is assigned one primary bucket based on its operator name and one-line summary. Secondary tags below capture overlap using the fuller docstrings.",
        "",
    ])
    out.extend(markdown_table(["primary bucket", "count", "share", "operator types"], category_rows))
    out.extend([
        "",
        "## Secondary Theme Tags",
        "",
    ])
    out.extend(markdown_table(["theme tag", "proposals", "share"], tag_rows))
    out.extend([
        "",
        "## Generation Phases",
        "",
    ])
    out.extend(markdown_table(["phase", "count", "operator mix", "top buckets"], phase_rows))
    out.extend([
        "",
        "## Interpretation",
        "",
        "Across the full run, explore proposals are much more diverse than the initial population, but the diversity is uneven. Mutation exploration repeatedly returns to data-aware local improvement, denominator/rational forms, and subtree reuse. Those are useful families, yet many proposals are variants of the same recipe: evaluate current predictions, compute a residual or fitted scalar, then wrap or retune the current tree.",
        "",
        "Selection exploration is dominated by multi-objective parent choice. Pareto/AFPO, complexity rarity, and age/recency pressure appear in many combinations; lexicase and rank/softmax policies provide some independent variety, but there are fewer genuinely different selection mechanisms than the proposal count suggests.",
        "",
        "Survival exploration is broad in replacement target criteria but still clusters around a few themes: Pareto/crowding protection, removing redundant members in loss-complexity space, age-regularized worst-oldest variants, and reverse tournament/worst-cost replacement. This is healthier than the initial population, where survival was barely represented.",
        "",
        "The clearest gap is not raw quantity; it is orthogonality. Many proposals share the same ingredients under different names. Future prompts should explicitly ask for categories that are underrepresented or missing from the current generation rather than asking generally for a new operator.",
        "",
        "## Bucket Examples",
        "",
    ])

    for category in CATEGORY_ORDER:
        if category not in examples:
            continue
        out.append(f"### {category}")
        out.append("")
        for record in examples[category]:
            rel = Path(record["path"]).relative_to(ROOT)
            out.append(f"- gen {record['generation']} `{rel}`: {record['summary']}")
        out.append("")

    out.extend([
        "## Prompting Recommendations",
        "",
        "- Track accepted proposal buckets in the prompt context and ask the model to choose an underfilled bucket before writing code.",
        "- Use explicit per-generation quotas, not just operator-type quotas: for mutation, reserve slots for data-aware fitting, algebraic rewrite, subtree recombination, rational/physics motifs, and one deliberately non-local structural move.",
        "- For selection, separate Pareto, lexicase, rank/softmax, and archive/niche sampling prompts so every proposal does not collapse into Pareto tournament plus novelty wording.",
        "- For survival, distinguish replacement target families: oldest, worst cost, bloat, crowding, redundancy, and Pareto-front protection. Ask for the one least represented in recent generations.",
        "- Add a novelty check to the prompt: require the proposed operator to name the nearest prior bucket and explain the concrete behavioral difference from prior proposals.",
    ])

    EXPLORE_ANALYSIS.write_text("\n".join(out).rstrip() + "\n")


def write_initial_analysis(records: list[dict[str, object]]) -> None:
    initial = [
        r
        for r in records
        if r["generation"] == 0 and r["mode"] == "explore"
    ]
    initial = sorted(initial, key=lambda r: (TYPES.index(str(r["operator_type"])), int(r["index"])))
    counts_by_type = Counter(str(r["operator_type"]) for r in initial)
    buckets = phrase_counts(initial)

    out = [
        "# Initial Explore Operator Diversity",
        "",
        f"Run: `{RUN_ID}`",
        f"Explore-only summary source: `{EXPLORE_SUMMARY.relative_to(ROOT)}`",
        "",
        "## Initial Population Operators",
        "",
    ]

    for record in initial:
        rel_path = Path(record["path"]).relative_to(ROOT)
        out.extend(
            [
                f"### {record['operator_type']} init {record['index']}",
                "",
                f"- Source: `{rel_path}`",
                f"- One-line summary: {record['summary']}",
                "",
                "Docstring:",
                "",
                "```text",
                str(record["docstring"]).rstrip(),
                "```",
                "",
            ]
        )

    out.extend(
        [
            "## Diversity Analysis",
            "",
            f"The initial explore set contains {len(initial)} proposed operators: "
            + ", ".join(f"{count} {op_type}" for op_type, count in sorted(counts_by_type.items()))
            + ". This is not balanced across operator types: mutation dominates the initial pool, selection gets moderate coverage, and survival is represented by only one proposal.",
            "",
            "The mutation proposals have meaningful implementation diversity, but they cluster around local improvement rather than broad structural exploration. Several operators tune or refit constants, one performs algebraic distribution, and another applies an affine subtree fit. These are practical hill-climbing moves, but they mostly preserve the existing expression scaffold.",
            "",
            "The selection proposals are more strategically diverse: they use novelty, Pareto ranking, rarity, and epsilon-style acceptance to change parent pressure. They are all variants of tournament/Pareto selection, so the conceptual spread is narrower than the keyword diversity suggests.",
            "",
            "The survival side is underexplored in the initial population. With only one survival proposal, the run starts with little variation in replacement pressure, archive maintenance, age handling, or diversity-preserving survivor choice.",
            "",
            "Detected theme counts in the initial set:",
        ]
    )
    for theme, count in buckets.most_common():
        out.append(f"- {theme}: {count}")

    out.extend(
        [
            "",
            "Prompting implications:",
            "",
            "- Explicitly budget initial explore proposals by operator type, for example equal quotas for mutation, selection, and survival.",
            "- Ask for at least one proposal in each structural family: data-aware mutation, algebraic rewrite, rational/denominator construction, subtree recombination, parent selection, and survivor replacement.",
            "- For selection/survival prompts, discourage small variations of tournament plus Pareto language unless the implementation changes the actual selection pressure.",
            "- For mutation prompts, require some proposals that add genuinely new expression topology rather than only optimizing constants around the current tree.",
        ]
    )

    INITIAL_ANALYSIS.write_text("\n".join(out).rstrip() + "\n")


def main() -> None:
    records = load_records()
    write_summary(records, ALL_SUMMARY)
    write_summary(records, EXPLORE_SUMMARY, explore_only=True)
    write_initial_analysis(records)
    write_explore_analysis(records)
    print(ALL_SUMMARY)
    print(EXPLORE_SUMMARY)
    print(INITIAL_ANALYSIS)
    print(EXPLORE_ANALYSIS)


if __name__ == "__main__":
    main()
