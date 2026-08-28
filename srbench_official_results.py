#!/usr/bin/env python3
"""Build the transposed comparison table used by ``--official``.

The table is discovered from SRBench evaluation manifests.  In particular,
``method_meta.source`` joins an evaluation back to the exact HPO/evolution
run that produced its configuration, which prevents results from different
training runs from being combined merely because they share a method name.
"""

import filecmp
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import srbench_results_io as srio


GT_TOTAL = 5_320
BLACK_BOX_TOTAL = 1_220
ONE_MILLION = 1_000_000
TEN_MILLION = 10_000_000

# (internal key, display label, method family, training objective)
OFFICIAL_COLUMNS = (
    ("pysr_baseline", "PySR baseline", "pysr_baseline", None),
    ("basicsr_baseline", "BasicSR baseline", "basicsr_baseline", None),
    ("hpo_gt", "HPO GT", "hpo", "gt"),
    ("pysrpp_gt", "PySR++ GT", "pysrpp", "gt"),
    ("basicsrpp_gt", "BasicSR++ GT", "basicsrpp", "gt"),
    ("hpo_gt_r2", "HPO GT-R2", "hpo", "gt-r2"),
    ("pysrpp_gt_r2", "PySR++ GT-R2", "pysrpp", "gt-r2"),
    ("basicsrpp_gt_r2", "BasicSR++ GT-R2", "basicsrpp", "gt-r2"),
    ("hpo_r2", "HPO R2", "hpo", "r2"),
    ("pysrpp_r2", "PySR++ R2", "pysrpp", "r2"),
    ("basicsrpp_r2", "BasicSR++ R2", "basicsrpp", "r2"),
)

SPLIT_ABBREVIATIONS = {
    "barely_unsolvable.txt": "bu.txt",
    "barely_unsolvable_val.txt": "bu_val.txt",
    "barely_unsolvable_val2.txt": "bu_val2.txt",
}


def _method_family(manifest: dict) -> Optional[str]:
    mode = manifest.get("mode")
    backend = manifest.get("backend")
    if mode == "fullsr_baseline" or (mode == "baseline" and backend == "fullsr"):
        return "basicsr_baseline"
    if mode == "baseline":
        return "pysr_baseline"
    if mode == "hpo":
        return "hpo"
    if mode == "evolve_fullsr":
        return "basicsrpp"
    if mode == "evolve":
        return "basicsrpp" if backend == "fullsr" else "pysrpp"
    return None


def _has_ground_truth(run_dir: Path, manifest: dict) -> bool:
    return bool(manifest.get("datasets")) or (run_dir / "srbench_full_results.json").exists()


def _has_black_box(run_dir: Path, manifest: dict) -> bool:
    return (
        "black_box" in (manifest.get("evaluation_types") or [])
        or bool(manifest.get("black_box"))
        or (run_dir / "srbench_black_box_results.json").exists()
    )


def _source_path(source: str, runs_root: Path, project_root: Path) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            suffix = parts[index + 2:]
            return runs_root / parts[index + 1] / Path(*suffix)
    return project_root / path


def _source_dir(source: str, runs_root: Path, project_root: Path) -> Path:
    path = _source_path(source, runs_root, project_root)
    return path.parent if path.is_file() else path


def _read_config_prefix(path: Path) -> dict:
    """Read the leading top-level ``config`` without loading a huge run_data."""
    if not path.exists():
        return {}
    decoder = json.JSONDecoder()
    text = ""
    with open(path) as handle:
        while len(text) < 4 * 1024 * 1024:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            text += chunk
            marker = text.find('"config"')
            if marker < 0:
                continue
            colon = text.find(":", marker)
            if colon < 0:
                continue
            value_start = colon + 1
            while value_start < len(text) and text[value_start].isspace():
                value_start += 1
            try:
                value, _ = decoder.raw_decode(text, value_start)
            except json.JSONDecodeError:
                continue
            return value if isinstance(value, dict) else {}
    return {}


def _load_final_summary(source_dir: Path) -> dict:
    path = source_dir / "final_eval_summary.json"
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _summary_score(summary: dict, split: Optional[str], index: int) -> Optional[float]:
    if not split:
        return None
    stem = Path(split).stem
    candidates = [summary.get(stem)]
    # New HPO summaries use the generic key ``val`` for their held-out split.
    if index == 1:
        candidates.append(summary.get("val"))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("avg_score", candidate.get("avg_r2"))
        if value is not None:
            return float(value)
    return None


def _copied_slurm_id(source_dir: Path, project_root: Path) -> str:
    """Recover an HPO driver ID by matching its copied ``slurm.out`` file."""
    copied = source_dir / "slurm.out"
    if not copied.exists():
        return "-"
    try:
        size = copied.stat().st_size
    except OSError:
        return "-"
    for candidate in sorted((project_root / "out").glob("*.out")):
        try:
            if candidate.stat().st_size != size:
                continue
            if filecmp.cmp(copied, candidate, shallow=False):
                return candidate.stem
        except OSError:
            continue
    return "-"


def _training_metadata(
    source: Optional[str], runs_root: Path, project_root: Path
) -> dict:
    if not source:
        return {
            "training_id": "-", "metric": None,
            "train_set": "-", "val_set": "-",
            "train_perf": None, "val_perf": None,
        }

    source_dir = _source_dir(source, runs_root, project_root)
    summary = _load_final_summary(source_dir)
    config = _read_config_prefix(source_dir / "run_data.json")
    splits = summary.get("splits") or []
    train_set = config.get("split") or (splits[0] if splits else None)
    val_set = config.get("val_split") or (splits[1] if len(splits) > 1 else None)
    metric = summary.get("fitness_metric") or config.get("fitness_metric")

    parts = Path(source).parts
    if "runs" in parts and parts.index("runs") + 1 < len(parts):
        training_id = parts[parts.index("runs") + 1]
    else:
        training_id = _copied_slurm_id(source_dir, project_root)

    return {
        "training_id": training_id,
        "metric": metric,
        "train_set": Path(train_set).name if train_set else "-",
        "val_set": Path(val_set).name if val_set else "-",
        "train_perf": _summary_score(summary, train_set, 0),
        "val_perf": _summary_score(summary, val_set, 1),
    }


def _black_box_stats(run_dir: Path, manifest: dict) -> tuple[int, Optional[float]]:
    path = run_dir / "srbench_black_box_results.json"
    if not path.exists():
        return 0, None
    try:
        with open(path) as handle:
            datasets = (json.load(handle).get("datasets") or {})
    except (OSError, json.JSONDecodeError):
        return 0, None

    completed = 0
    best_r2 = []
    for stored_frontiers in datasets.values():
        if stored_frontiers and isinstance(stored_frontiers[0], dict):
            trials = [stored_frontiers]
        else:
            trials = stored_frontiers or []
        completed += len(trials)
        for frontier in trials:
            values = [
                point.get("test_r2") for point in frontier
                if point.get("test_r2") is not None
            ]
            if values:
                best_r2.append(max(values))
    mean_r2 = sum(best_r2) / len(best_r2) if best_r2 else None
    return completed, mean_r2


def _ground_truth_stats(
    run_dir: Path, manifest: dict
) -> tuple[int, Optional[float], Optional[float]]:
    """Return completed runs, per-run solve rate, and any-seed solve rate.

    The any-seed rate treats each (dataset, noise) pair as one task and marks
    it solved when at least one seed solved it.
    """
    keyed = srio.load_keyed_results(run_dir)
    if keyed is None and manifest.get("batches"):
        keyed = srio.build_keyed_results(run_dir, manifest)
    present = [
        entry for entry in (keyed or {}).values()
        if entry.get("present") and entry.get("error") is None
    ]
    if not present:
        return 0, None, None
    solved = sum(bool(entry.get("solved")) for entry in present)
    tasks = {}
    for entry in present:
        task = (entry.get("dataset"), srio.fmt_noise(entry.get("noise", 0)))
        tasks[task] = tasks.get(task, False) or bool(entry.get("solved"))
    any_seed_rate = sum(tasks.values()) / len(tasks)
    return len(present), solved / len(present), any_seed_rate


def _pick_evaluation(
    records: Iterable[dict], budget: int, result_type: str
) -> Optional[dict]:
    if result_type == "gt":
        candidates = [
            record for record in records
            if record["manifest"].get("max_evals") == budget
            and _has_ground_truth(record["run_dir"], record["manifest"])
        ]
    else:
        candidates = [
            record for record in records
            if record["manifest"].get("max_evals") == budget
            and _has_black_box(record["run_dir"], record["manifest"])
        ]
    return max(candidates, key=lambda record: record["mtime"], default=None)


def _source_recency(records: Iterable[dict]) -> float:
    primary = [
        record["mtime"] for record in records
        if record["manifest"].get("max_evals") == ONE_MILLION
    ]
    return max(primary or [record["mtime"] for record in records])


def _column_from_records(records: list[dict], training: dict) -> dict:
    gt = _pick_evaluation(records, ONE_MILLION, "gt")
    black_box = _pick_evaluation(records, ONE_MILLION, "black_box")
    gt_10m = _pick_evaluation(records, TEN_MILLION, "gt")

    gt_completed, gt_rate, gt_any_seed_rate = (
        _ground_truth_stats(gt["run_dir"], gt["manifest"])
        if gt else (0, None, None)
    )
    bb_completed, bb_r2 = (
        _black_box_stats(black_box["run_dir"], black_box["manifest"])
        if black_box else (0, None)
    )
    _, gt_10m_rate, _ = (
        _ground_truth_stats(gt_10m["run_dir"], gt_10m["manifest"])
        if gt_10m else (0, None, None)
    )

    contributing = []
    for record in (gt, black_box, gt_10m):
        if record and record["run_dir"].name not in contributing:
            contributing.append(record["run_dir"].name)

    metadata_record = gt or black_box or gt_10m
    method_meta = (
        (metadata_record["manifest"].get("method_meta") or {})
        if metadata_record else {}
    )
    train_perf = method_meta.get("train_score")
    val_perf = method_meta.get("val_score")
    if train_perf is None:
        train_perf = training.get("train_perf")
    if val_perf is None:
        val_perf = training.get("val_perf")

    return {
        **training,
        "eval_ids": ",".join(contributing) or "-",
        "train_perf": train_perf,
        "val_perf": val_perf,
        "bb_r2": bb_r2,
        "gt_rate": gt_rate,
        "gt_any_seed_rate": gt_any_seed_rate,
        "gt_10m_rate": gt_10m_rate,
        "gt_completed": gt_completed,
        "bb_completed": bb_completed,
    }


def build_official_columns(
    runs_root: "str | Path", project_root: "str | Path | None" = None
) -> list[dict]:
    """Discover and summarize the eleven official comparison columns."""
    runs_root = Path(runs_root)
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parent
    grouped: Dict[tuple, list[dict]] = {}
    training_cache: Dict[str, dict] = {}

    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            with open(manifest_path) as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        family = _method_family(manifest)
        if family is None:
            continue
        run_dir = manifest_path.parent
        if not (_has_ground_truth(run_dir, manifest) or _has_black_box(run_dir, manifest)):
            continue

        source = (manifest.get("method_meta") or {}).get("source")
        if family.endswith("baseline"):
            source_key = family
            training = _training_metadata(None, runs_root, project_root)
        else:
            if not source:
                continue
            source_key = source
            if source not in training_cache:
                training_cache[source] = _training_metadata(
                    source, runs_root, project_root
                )
            training = training_cache[source]
        metric = training.get("metric")
        key = (family, metric, source_key)
        grouped.setdefault(key, []).append({
            "run_dir": run_dir,
            "manifest": manifest,
            "mtime": manifest_path.stat().st_mtime,
            "training": training,
        })

    blank = {
        "training_id": "-", "eval_ids": "-", "train_set": "-", "val_set": "-",
        "train_perf": None, "val_perf": None, "bb_r2": None,
        "gt_rate": None, "gt_any_seed_rate": None, "gt_10m_rate": None,
        "gt_completed": 0, "bb_completed": 0,
    }
    columns = []
    for key, label, family, metric in OFFICIAL_COLUMNS:
        candidates = [
            records for (candidate_family, candidate_metric, _), records in grouped.items()
            if candidate_family == family and candidate_metric == metric
        ]
        if candidates:
            records = max(candidates, key=_source_recency)
            values = _column_from_records(records, records[0]["training"])
        else:
            values = dict(blank)
        columns.append({"key": key, "label": label, **values})
    return columns


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 10_000:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _fmt_rate(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "-"


def format_official_table(columns: list[dict]) -> str:
    """Render official columns with requested metadata/results as rows."""
    def split_name(column: dict, key: str) -> str:
        return SPLIT_ABBREVIATIONS.get(column[key], column[key])

    rows = [
        ("training slurm", lambda column: column["training_id"]),
        ("SRBench eval slurm(s)", lambda column: column["eval_ids"]),
        ("train set", lambda column: split_name(column, "train_set")),
        ("val set", lambda column: split_name(column, "val_set")),
        ("train perf", lambda column: _fmt_score(column["train_perf"])),
        ("val perf", lambda column: _fmt_score(column["val_perf"])),
        ("SRBench BB R2", lambda column: _fmt_score(column["bb_r2"])),
        ("SRBench GT solve (all)", lambda column: _fmt_rate(column["gt_rate"])),
        ("SRBench GT solve (any seed)",
         lambda column: _fmt_rate(column["gt_any_seed_rate"])),
        ("SRBench GT solve (all, 10M)", lambda column: _fmt_rate(column["gt_10m_rate"])),
        ("GT completed", lambda column: f"{column['gt_completed']}/{GT_TOTAL}"),
        ("BB completed", lambda column: f"{column['bb_completed']}/{BLACK_BOX_TOTAL}"),
    ]
    table = [["metric"] + [column["label"] for column in columns]]
    table += [
        [label] + [str(formatter(column)) for column in columns]
        for label, formatter in rows
    ]
    widths = [max(len(row[index]) for row in table) for index in range(len(table[0]))]

    def rule(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def render(cells: list[str]) -> str:
        parts = [cells[0].ljust(widths[0])]
        parts += [cells[index].rjust(widths[index]) for index in range(1, len(cells))]
        return "│ " + " │ ".join(parts) + " │"

    lines = [rule("┌", "┬", "┐"), render(table[0]), rule("├", "┼", "┤")]
    lines += [render(row) for row in table[1:]]
    lines.append(rule("└", "┴", "┘"))
    used_splits = {
        column[key]
        for column in columns
        for key in ("train_set", "val_set")
        if column[key] in SPLIT_ABBREVIATIONS
    }
    legend = "; ".join(
        f"{SPLIT_ABBREVIATIONS[name]} = {name}" for name in SPLIT_ABBREVIATIONS
        if name in used_splits
    )
    table_text = "\n".join(lines)
    return f"Split key: {legend}\n{table_text}" if legend else table_text


def build_official_table(
    runs_root: "str | Path", project_root: "str | Path | None" = None
) -> str:
    return format_official_table(build_official_columns(runs_root, project_root))
