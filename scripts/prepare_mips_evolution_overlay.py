#!/usr/bin/env python3
"""Create one artifact-root overlay for original and refined MIPS relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mips_tasks import (  # noqa: E402
    component_artifact_path,
    parse_dataset_name,
    relation_artifact_path,
    task_artifact_dir,
)


DEFAULT_SPLIT = ROOT / "splits" / "mips_sr_targets_plus_refined.txt"
DEFAULT_BASE_ROOT = ROOT / "outputs" / "mips_transition_tables"
DEFAULT_REFINED_ROOT = ROOT / "outputs" / "mips_refined_six_artifacts" / "train"
DEFAULT_REFINED_SPLIT = (
    ROOT / "outputs" / "mips_refined_six_artifacts" / "pysr_components.txt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "mips_evolution_51_artifacts"


def read_split(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Split contains duplicate dataset names: {path}")
    return names


def validate_component(dataset_name: str, root: Path) -> Path:
    component = parse_dataset_name(dataset_name)
    component_path = component_artifact_path(component, root)
    if component_path.is_file():
        with np.load(component_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        if not bool(metadata["deterministic"]):
            raise ValueError(f"Refusing contradictory relation {dataset_name}")
        return component_path
    relation_path = relation_artifact_path(component.task, component.kind, root)
    if not relation_path.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_name} in {root}; checked {component_path} and "
            f"{relation_path}"
        )
    with np.load(relation_path, allow_pickle=False) as payload:
        relation_metadata = json.loads(str(payload["metadata_json"].item()))
    if component.index >= int(relation_metadata["target_count"]):
        raise IndexError(
            f"{dataset_name} exceeds target_count="
            f"{relation_metadata['target_count']} in {relation_path}"
        )
    metadata = relation_metadata["components"][component.index]
    if not bool(metadata["deterministic"]):
        raise ValueError(f"Refusing contradictory relation {dataset_name}")
    return relation_path


def ensure_task_link(destination: Path, source: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(
                f"Refusing to replace mismatched symlink {destination} -> "
                f"{destination.resolve()} (expected {source})"
            )
        return
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing non-symlink artifact path {destination}"
        )
    destination.symlink_to(source, target_is_directory=True)


def prepare(
    split: Path,
    base_root: Path,
    refined_root: Path,
    refined_split: Path,
    output_root: Path,
) -> dict[str, object]:
    names = read_split(split)
    refined_names = set(read_split(refined_split))
    unexpected = sorted(refined_names - set(names))
    if unexpected:
        raise ValueError(f"Refined split contains names absent from union: {unexpected}")

    output_tasks = output_root / "tasks"
    output_tasks.mkdir(parents=True, exist_ok=True)
    task_sources: dict[str, Path] = {}
    source_counts = {"base": 0, "refined": 0}
    for name in names:
        component = parse_dataset_name(name)
        source_label = "refined" if name in refined_names else "base"
        source_root = refined_root if source_label == "refined" else base_root
        source_task = task_artifact_dir(component.task, source_root)
        previous = task_sources.setdefault(component.task, source_task.resolve())
        if previous != source_task.resolve():
            raise ValueError(
                f"Task {component.task} would mix artifact roots: {previous} and "
                f"{source_task.resolve()}"
            )
        validate_component(name, source_root)
        source_counts[source_label] += 1

    for task, source_task in sorted(task_sources.items()):
        ensure_task_link(output_tasks / task, source_task)

    manifest: dict[str, object] = {
        "split": str(split.resolve()),
        "dataset_count": len(names),
        "rnn_task_count": len(task_sources),
        "base_relation_count": source_counts["base"],
        "refined_relation_count": source_counts["refined"],
        "base_root": str(base_root.resolve()),
        "refined_root": str(refined_root.resolve()),
        "output_root": str(output_root.resolve()),
        "tasks": {task: str(source) for task, source in sorted(task_sources.items())},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--refined-root", type=Path, default=DEFAULT_REFINED_ROOT)
    parser.add_argument("--refined-split", type=Path, default=DEFAULT_REFINED_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = prepare(
        args.split,
        args.base_root,
        args.refined_root,
        args.refined_split,
        args.output_root,
    )
    print(
        f"Prepared {manifest['dataset_count']} scalar relations from "
        f"{manifest['rnn_task_count']} RNN tasks at {manifest['output_root']} "
        f"({manifest['base_relation_count']} base + "
        f"{manifest['refined_relation_count']} refined)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
