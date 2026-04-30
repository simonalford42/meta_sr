#!/usr/bin/env python3
"""Generate human-readable lineage reports for run 947961."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "out" / "947961.out"
RUN = ROOT / "runs" / "947961"
LINEAGE_OUT = ROOT / "947961_lineages.txt"
BEST_OPS_OUT = ROOT / "947961_best_operator_code.md"

BASELINE = {
    "mutation": "add_constant_offset",
    "survival": "age_regularized_survival",
    "selection": "tournament_selection",
}
TYPES = ("mutation", "survival", "selection")


def clean_id(text: str) -> str:
    return text.replace("_", "-")


def component_id(name: str) -> str:
    if name in BASELINE.values():
        return "0-0"
    m = re.search(r"_init_(\d+)$", name)
    if m:
        return f"1-{int(m.group(1)) + 1}"
    m = re.search(r"_gen(\d+)_(\d+)$", name)
    if m:
        # User-facing convention: baseline is 0, initial population is 1,
        # logged generation 1 is generation 2, etc.
        return f"{int(m.group(1)) + 1}-{int(m.group(2))}"
    return "unknown"


def operator_file(name: str, op_type: str) -> Path | None:
    m = re.search(r"_init_(\d+)$", name)
    if m:
        path = RUN / "operators" / f"gen0_{op_type}{m.group(1)}.jl"
        return path if path.exists() else None
    m = re.search(r"_gen(\d+)_(\d+)$", name)
    if m:
        path = RUN / "operators" / f"gen{m.group(1)}_{op_type}{m.group(2)}.jl"
        return path if path.exists() else None
    return None


class Bundle:
    def __init__(self, comps, score=None, stage="", line_no=0):
        self.comps = tuple(comps)
        self.score = score
        self.stage = stage
        self.line_no = line_no
        self.ids = tuple(component_id(c) for c in comps)
        self.new_positions = [
            i
            for i, cid in enumerate(self.ids)
            if cid != "0-0" and (stage == "initial" and cid.startswith("1-") or stage.startswith("gen"))
        ]
        self.primary_pos = self._primary_pos()
        self.bundle_id = self.ids[self.primary_pos] if self.primary_pos is not None else "0-0"
        self.parent_key = None
        self.parent_id = None

    def _primary_pos(self):
        if self.stage == "baseline":
            return None
        if self.stage == "initial":
            ids = [i for i, cid in enumerate(self.ids) if cid.startswith("1-")]
            return ids[0] if ids else None
        m = re.match(r"gen(\d+)$", self.stage)
        if m:
            prefix = f"{int(m.group(1)) + 1}-"
            ids = [i for i, cid in enumerate(self.ids) if cid.startswith(prefix)]
            return ids[0] if ids else None
        return None

    def key(self):
        return self.comps

    def component_text(self):
        return f"({self.ids[0]}, {self.ids[1]}, {self.ids[2]})"

    def names_text(self):
        return " | ".join(self.comps)

    def summary_line(self):
        parent = self.parent_id or "?"
        score = f" score={self.score:.4f}" if self.score is not None else ""
        return f"{self.bundle_id} -> {self.component_text()} parent={parent}{score} :: {self.names_text()}"


def parse_log():
    bundles = []
    frontier_by_gen = defaultdict(list)
    avg_by_gen = defaultdict(list)
    best_by_gen = {}
    created = {}
    current_gen = None
    in_initial_eval = False

    avg_re = re.compile(r"^\s*Avg\s+([0-9.]+)\s+(.+?):\s+\[")
    frontier_re = re.compile(r"^\s*\[frontier\]\s+(.+?):\s+score=([0-9.]+)")
    best_initial_re = re.compile(r"^Best initial bundle:\s+(.+?)\s+\(score:\s+([0-9.]+)\)")
    best_re = re.compile(r"^\s*Best:\s+(.+?)\s+\(score:\s+([0-9.]+)\)")
    gen_re = re.compile(r"^Generation\s+(\d+)/")
    created_re = re.compile(r"^\s*Created:\s+(\S+)\s+\(mode=(\w+), model=([^)]+)\)")

    with LOG.open() as f:
        for line_no, line in enumerate(f, 1):
            m = gen_re.match(line)
            if m:
                current_gen = int(m.group(1))
                in_initial_eval = False
                continue
            if "Evaluating initial population" in line:
                in_initial_eval = True
                current_gen = None
                continue
            if line.startswith("Evaluating baseline"):
                bundles.append(Bundle(tuple(BASELINE[t] for t in TYPES), 0.5200, "baseline", line_no))
                continue
            m = created_re.match(line)
            if m and current_gen is not None:
                created[m.group(1)] = {
                    "logged_generation": current_gen,
                    "line": line_no,
                    "mode": m.group(2),
                    "model": m.group(3),
                }
                continue
            m = avg_re.match(line)
            if m:
                comps = [part.strip() for part in m.group(2).split("|")]
                if len(comps) == 3:
                    stage = "initial" if in_initial_eval else f"gen{current_gen}"
                    b = Bundle(comps, float(m.group(1)), stage, line_no)
                    bundles.append(b)
                    avg_by_gen[0 if in_initial_eval else current_gen].append(b)
                continue
            m = frontier_re.match(line)
            if m and current_gen is not None:
                comps = [part.strip() for part in m.group(1).split("|")]
                if len(comps) == 3:
                    frontier_by_gen[current_gen].append(Bundle(comps, float(m.group(2)), f"frontier{current_gen}", line_no))
                continue
            m = best_initial_re.match(line)
            if m:
                comps = tuple(part.strip() for part in m.group(1).split("|"))
                best_by_gen[0] = (comps, float(m.group(2)))
                continue
            m = best_re.match(line)
            if m and current_gen is not None:
                comps = tuple(part.strip() for part in m.group(1).split("|"))
                best_by_gen[current_gen] = (comps, float(m.group(2)))

    return bundles, avg_by_gen, frontier_by_gen, best_by_gen, created


def add_parent_candidate(bundle, seen, by_other_components):
    seen[bundle.key()] = bundle
    for pos in range(3):
        other = tuple(c for i, c in enumerate(bundle.comps) if i != pos)
        by_other_components[(pos, other)].append(bundle)


def assign_parent_from_prior(bundle, seen, by_other_components):
    if bundle.primary_pos is None:
        return
    pos = bundle.primary_pos
    other = tuple(c for i, c in enumerate(bundle.comps) if i != pos)
    candidates = by_other_components[(pos, other)]
    parent = candidates[-1] if candidates else None
    if parent is None:
        parent = seen.get(tuple(BASELINE[t] if i == pos else c for i, c in enumerate(bundle.comps)))
    bundle.parent_key = parent.key() if parent else None
    bundle.parent_id = parent.bundle_id if parent else "?"


def assign_parents(bundles, avg_by_gen):
    # Parent lookup is generation-buffered: all offspring in the same generation
    # are siblings, so one cannot be the direct ancestor of another.
    seen = {}
    by_other_components = defaultdict(list)

    baseline = [b for b in bundles if b.stage == "baseline"]
    for b in baseline:
        b.parent_id = "none"
        add_parent_candidate(b, seen, by_other_components)

    for b in avg_by_gen.get(0, []):
        b.parent_id = "0-0"
        b.parent_key = tuple(BASELINE[t] for t in TYPES)
    for b in avg_by_gen.get(0, []):
        add_parent_candidate(b, seen, by_other_components)

    for gen in sorted(g for g in avg_by_gen if g > 0):
        generation_bundles = avg_by_gen[gen]
        for b in generation_bundles:
            assign_parent_from_prior(b, seen, by_other_components)
        for b in generation_bundles:
            add_parent_candidate(b, seen, by_other_components)


def bundle_lookup(bundles):
    latest = {}
    for b in bundles:
        latest[b.key()] = b
    return latest


def trace_lineage(bundle, latest):
    chain = []
    seen = set()
    cur = bundle
    while cur is not None and cur.key() not in seen:
        seen.add(cur.key())
        chain.append(cur)
        cur = latest.get(cur.parent_key) if cur.parent_key else None
    return list(reversed(chain))


def changed_position(prev, cur):
    changed = [i for i, (a, b) in enumerate(zip(prev.ids, cur.ids)) if a != b]
    return changed[0] if len(changed) == 1 else None


def simplify_lineage(chain):
    if len(chain) <= 2:
        return chain

    simplified = [chain[0]]
    pending = chain[1]
    pending_pos = changed_position(chain[0], chain[1])

    for cur in chain[2:]:
        cur_pos = changed_position(pending, cur)
        if pending_pos is not None and cur_pos == pending_pos:
            pending = cur
        else:
            simplified.append(pending)
            pending = cur
            pending_pos = cur_pos

    simplified.append(pending)
    return simplified


def write_lineages(bundles, avg_by_gen, frontier_by_gen, best_by_gen):
    latest = bundle_lookup(bundles)
    out = []
    out.append("Lineage report for run 947961")
    out.append("")
    out.append("ID convention used here:")
    out.append("- 0-0 = baseline component.")
    out.append("- 1-N = initial-population suggestion init_(N-1).")
    out.append("- G-I = logged generation (G-1), suggestion I. Example: logged gen21_5 is shown as 22-5.")
    out.append("- Bundle ID = the ID of the component changed to create that bundle.")
    out.append("- Direct ancestors are inferred from the evaluated bundle stream in 947961.out by matching the unchanged components and the prior bundle that supplied them.")
    out.append("")
    out.append("Component order: (mutation, survival, selection)")
    out.append("")
    out.append("Best Bundle Lineages")
    for gen in sorted(best_by_gen):
        comps, score = best_by_gen[gen]
        b = latest.get(comps)
        out.append("")
        out.append(f"Generation {gen} best score={score:.4f}: {b.bundle_id if b else '?'} -> {Bundle(comps).component_text()}")
        out.append(f"  operators: {' | '.join(comps)}")
        if b:
            chain = trace_lineage(b, latest)
            simple_chain = simplify_lineage(chain)
            out.append("  lineage: " + " -> ".join(c.component_text() for c in chain))
            out.append("  simplified lineage: " + " -> ".join(c.component_text() for c in simple_chain))
            out.append("  direct ancestor: " + (b.parent_id or "none"))
        else:
            out.append("  lineage: not found among parsed bundle evaluations")
    out.append("")
    out.append("Evolution Summary")
    out.append("Each generation lists carried-in frontier/population bundles first, then new offspring bundles from that generation.")
    for gen in sorted(set(avg_by_gen) | set(frontier_by_gen)):
        title = "Initial population" if gen == 0 else f"Generation {gen}"
        out.append("")
        out.append(title)
        rows = []
        if gen > 0:
            rows.extend(frontier_by_gen.get(gen - 1, [])[:10])
        rows.extend(avg_by_gen.get(gen, []))
        for b in rows:
            if b.stage.startswith("frontier"):
                score = f" score={b.score:.4f}" if b.score is not None else ""
                out.append(f"  carried {b.component_text()}{score} :: {b.names_text()}")
            else:
                out.append("  " + b.summary_line())
    LINEAGE_OUT.write_text("\n".join(out) + "\n")


def write_best_operator_code(bundles, best_by_gen):
    latest = bundle_lookup(bundles)
    out = []
    out.append("# Best Operator Code Over Time")
    out.append("")
    out.append("This file records the best bundle whenever the best bundle changes. Code blocks are included for the non-baseline operators in that best bundle.")
    out.append("")
    prev_comps = None
    emitted = set()
    for gen in sorted(best_by_gen):
        comps, score = best_by_gen[gen]
        if comps == prev_comps:
            continue
        b = latest.get(comps)
        out.append(f"## Generation {gen}: score {score:.4f}")
        if b:
            out.append(f"- Bundle ID: `{b.bundle_id}`")
            out.append(f"- Direct ancestor: `{b.parent_id or 'none'}`")
            out.append(f"- Components: `{b.component_text()}`")
        out.append(f"- Operators: `{' | '.join(comps)}`")
        out.append("")
        for op_type, name in zip(TYPES, comps):
            cid = component_id(name)
            if cid == "0-0":
                continue
            key = (op_type, name)
            path = operator_file(name, op_type)
            out.append(f"### {op_type}: `{name}` ({cid})")
            if key in emitted:
                out.append("_Code already shown above._")
                out.append("")
                continue
            if path and path.exists():
                out.append(f"Source: `{path.relative_to(ROOT)}`")
                out.append("")
                out.append("```julia")
                out.append(path.read_text().rstrip())
                out.append("```")
            else:
                out.append("_No saved operator source file found._")
            out.append("")
            emitted.add(key)
        prev_comps = comps
    BEST_OPS_OUT.write_text("\n".join(out).rstrip() + "\n")


def main():
    bundles, avg_by_gen, frontier_by_gen, best_by_gen, _created = parse_log()
    assign_parents(bundles, avg_by_gen)
    write_lineages(bundles, avg_by_gen, frontier_by_gen, best_by_gen)
    write_best_operator_code(bundles, best_by_gen)
    print(LINEAGE_OUT)
    print(BEST_OPS_OUT)


if __name__ == "__main__":
    main()
