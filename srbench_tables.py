"""Compact SRBench method and compute-budget tables for the inspector."""
import math
from pathlib import Path

import srbench_results_io as srio
from srbench_official_results import build_official_columns, BLACK_BOX_TOTAL


# User-supplied benchmark reference, not recomputed from the released trial snapshot.
# Keep this value unchanged when local SRBench task denominators change.
MDLFORMER_REFERENCE_GT = 0.405

GROUPS = [
    ("PySR", [("", "pysr_baseline")]),
    ("PySR++", [("R2", "pysrpp_r2"), ("GT-R2", "pysrpp_gt_r2"), ("GT", "pysrpp_gt")]),
    ("BasicSR", [("", "basicsr_baseline")]),
    ("BasicSR++", [("R2", "basicsrpp_r2"), ("GT-R2", "basicsrpp_gt_r2"), ("GT", "basicsrpp_gt")]),
    ("HPO", [("R2", "hpo_r2"), ("GT-R2", "hpo_gt_r2"), ("GT", "hpo_gt")]),
    ("Autoresearch", [("", "autoresearch_gt")]),
    ("MDLFormer", [("", "mdlformer")]),
]


def _names(path):
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def _format(value, *, rate=True):
    if value is None or not math.isfinite(value):
        return "TBD"
    if rate:
        return f"{100 * value:.2f}%"
    return f"{value:.3e}" if abs(value) >= 10000 else f"{value:.3f}"


def _grid(groups, rows, *, subheaders=True):
    """Render grouped column headings; None rows insert a horizontal divider."""
    widths = [max(len(sub), 6, *(len(row[i]) for row in rows if row is not None))
              for i, sub in enumerate(sub for _, subs in groups for sub in subs)]
    spans = []
    offset = 0
    for label, subs in groups:
        width = sum(widths[offset:offset+len(subs)]) + 3*len(subs)-1
        if len(label)+2 > width:
            widths[offset] += len(label)+2-width
            width = len(label)+2
        spans.append(width)
        offset += len(subs)
    rule = "+" + "+".join("-"*(width+2) for width in widths) + "+"
    lines = ["+" + "+".join("-"*span for span in spans) + "+",
             "|" + "|".join(label.center(span) for (label, _), span in zip(groups, spans)) + "|"]
    if subheaders:
        lines.append("| " + " | ".join(sub.center(width) for sub, width in zip(
            (sub for _, subs in groups for sub in subs), widths)) + " |")
    lines.append(rule)
    for row in rows:
        if row is None:
            lines.append(rule)
        else:
            lines.append("| " + " | ".join(
                value.ljust(width) if i == 0 else value.rjust(width)
                for i, (value, width) in enumerate(zip(row, widths))) + " |")
    lines.append(rule)
    return "\n".join(lines)


def _complete_gt(path, canonical):
    """Return a full ten-seed/four-noise grid, or None for unfinished/missing runs."""
    if path is None:
        return None
    path = Path(path)
    if not (path / "manifest.json").exists():
        return None
    manifest = srio.load_manifest(path)
    keyed = srio.load_keyed_results(path)
    if keyed is None and manifest.get("batches"):
        keyed = srio.build_keyed_results(path, manifest)
    manifest, keyed = srio.standard_ground_truth_view(manifest, keyed or {})
    if (set(manifest.get("datasets", [])) != canonical
            or len(manifest.get("seeds", [])) != 10
            or set(manifest.get("noise_levels", [])) != {0, .001, .01, .1}
            or any("n_searches" in entry for entry in keyed.values())):
        return None
    expected = srio.expected_keys(manifest, keyed)
    if len(expected) != len(canonical)*40 or not all(
        keyed.get(key, {}).get("present") and keyed[key].get("error") is None
        for key in expected
    ):
        return None
    return [keyed[key] for key in expected]


def _rate(rows, datasets=None, noise=None):
    if rows is None:
        return None
    values = [bool(row["solved"]) for row in rows
              if (datasets is None or row["dataset"] in datasets)
              and (noise is None or float(row["noise"]) == noise)]
    return sum(values)/len(values) if values else None


def build_tables(runs_root="runs", project_root=None):
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parent
    runs_root = Path(runs_root)
    canonical = _names(project_root / "splits/srbench_all.txt") - set(srio.UNSOLVABLE_TASKS)
    train = _names(project_root / "splits/barely_unsolvable.txt") & canonical
    columns = {column["key"]: column for column in build_official_columns(
        runs_root, project_root, single_search_only=True)}
    cache = {}

    def complete(path):
        key = str(path) if path is not None else None
        if key not in cache:
            cache[key] = _complete_gt(path, canonical)
        return cache[key]

    keys = [key for _, subs in GROUPS for _, key in subs]
    bb, gt = [], []
    for key in keys:
        column = columns.get(key, {})
        bb.append(_format(column.get("bb_r2") if column.get("bb_completed") == BLACK_BOX_TOTAL
                          else None, rate=False))
        gt.append(_format(MDLFORMER_REFERENCE_GT if key == "mdlformer"
                          else _rate(complete(column.get("gt_path")))))
    table1 = _grid([("", [""])] + [(label, [sub for sub, _ in subs]) for label, subs in GROUPS],
                   [["SRBench black box (R2)", *bb], ["SRBench ground truth", *gt]])

    baseline = columns.get("pysr_baseline", {})
    evolved = columns.get("pysrpp_gt", {})
    evolved_dir = runs_root / evolved.get("training_id", "-")
    budgets = [
        ("1 million evaluations", baseline.get("gt_path"), evolved.get("gt_path")),
        ("90 seconds", runs_root / "srbench_gt_baseline_90s", evolved_dir / "srbench_gt_90s"),
        ("15 minutes", runs_root / "srbench_gt_baseline_15m_single", evolved_dir / "srbench_gt_15m_single"),
        ("15m-portfolio", runs_root / "srbench_gt_baseline_15m_portfolio_1e6", evolved_dir / "srbench_gt_15m_portfolio_1e6"),
    ]
    rows = [[label, _format(_rate(complete(base))), _format(_rate(complete(evo)))]
            for label, base, evo in budgets]
    base_rows, evo_rows = complete(baseline.get("gt_path")), complete(evolved.get("gt_path"))
    rows.append(None)
    for label, datasets, noise in [
        (f"Train tasks (n={len(train)})", train, None),
        (f"Excluding train tasks (n={len(canonical-train)})", canonical-train, None),
        *[(f"Noise {noise:g}", None, noise) for noise in [0, .001, .01, .1]],
    ]:
        rows.append([label, _format(_rate(base_rows, datasets, noise)),
                     _format(_rate(evo_rows, datasets, noise))])
    table2 = _grid([("", [""]), ("PySR", [""]), ("Evolved", [""])], rows, subheaders=False)
    return ("Table 1: 1M-evaluation method comparison\n" + table1
            + "\nBlack box: mean test R2. Local ground truth: solve rate over 130 tasks and four noise levels."
            + "\nMDLFormer GT: supplied reference score (40.5%), without task-count rescaling; BB unavailable."
            + "\n\nTable 2: PySR vs PySR++ GT (evolved=" + evolved.get("training_id", "TBD") + ")\n"
            + table2 + "\nBreakdown below the divider uses 1M evaluations; 10 seeds."
            + "\nTBD = unavailable or incomplete local evaluation.")
