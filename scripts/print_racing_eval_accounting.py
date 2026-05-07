#!/usr/bin/env python3
"""Print generation-level racing eval accounting for a meta-SR run.

Default target is runs/666286.  Counts named "evals" in the table are PySR
task evals: one bundle/config evaluated on one dataset for one seed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "666286"

GEN_HEADER_RE = re.compile(r"^Generation (?P<gen>\d+)/")
GEN_COMPLETE_RE = re.compile(r"^Generation (?P<gen>\d+) complete:")
RACING_RE = re.compile(
    "Racing gen (?P<gen>\\d+): \\u03bb=(?P<lambda>\\d+), .* "
    "(?P<qualifiers>\\d+)/(?P<archive>\\d+) archive bundles qualify"
)
GENERATED_RE = re.compile(r"Generated (?P<count>\d+) offspring bundles")
WAIT_RE = re.compile(
    r"Racing: waiting on (?P<extras>\d+) extra-runs \+ "
    r"(?P<offspring>\d+) offspring batches"
)
RESULT_RE = re.compile(
    r"^\s+\[(?P<kind>extras|offspring)\]\s+Avg\s+\S+\s+"
    r"(?P<name>.+?):\s+\(seeds=(?P<seeds>\d+)\)"
)


@dataclass
class EvalResult:
    name: str
    seeds: int


@dataclass
class Generation:
    gen: int
    lambda_value: int | None = None
    qualifiers: int | None = None
    archive_size: int | None = None
    generated_offspring: int | None = None
    wait_extras: int | None = None
    wait_offspring: int | None = None
    complete: bool = False
    extras: list[EvalResult] = field(default_factory=list)
    offspring: list[EvalResult] = field(default_factory=list)


def normalize_name(name: str) -> str:
    return " ".join(name.split())


def generation_for(gens: dict[int, Generation], gen: int) -> Generation:
    if gen not in gens:
        gens[gen] = Generation(gen=gen)
    return gens[gen]


def parse_log(log_path: Path) -> dict[int, Generation]:
    gens: dict[int, Generation] = {}
    current_gen: int | None = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := GEN_HEADER_RE.search(line):
            current_gen = int(m.group("gen"))
            generation_for(gens, current_gen)

        if m := RACING_RE.search(line):
            current_gen = int(m.group("gen"))
            gen = generation_for(gens, current_gen)
            gen.lambda_value = int(m.group("lambda"))
            gen.qualifiers = int(m.group("qualifiers"))
            gen.archive_size = int(m.group("archive"))

        if current_gen is None:
            continue

        gen = generation_for(gens, current_gen)

        if m := GENERATED_RE.search(line):
            gen.generated_offspring = int(m.group("count"))

        if m := WAIT_RE.search(line):
            gen.wait_extras = int(m.group("extras"))
            gen.wait_offspring = int(m.group("offspring"))

        if m := RESULT_RE.search(line):
            entry = EvalResult(
                name=normalize_name(m.group("name")),
                seeds=int(m.group("seeds")),
            )
            if m.group("kind") == "extras":
                gen.extras.append(entry)
            else:
                gen.offspring.append(entry)

        if m := GEN_COMPLETE_RE.search(line):
            generation_for(gens, int(m.group("gen"))).complete = True

    return gens


def load_config(run_dir: Path) -> dict:
    run_data_path = run_dir / "run_data.json"
    if not run_data_path.exists():
        return {}
    with run_data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("config") or {}


def int_config(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    return default if value is None else int(value)


def offspring_count(gen: Generation) -> int:
    if gen.generated_offspring is not None:
        return gen.generated_offspring
    if gen.wait_offspring is not None:
        return gen.wait_offspring
    return len(gen.offspring)


def reeval_count(gen: Generation) -> int:
    if gen.wait_extras is not None:
        return gen.wait_extras
    if gen.qualifiers is not None:
        return gen.qualifiers
    return len(gen.extras)


def count_made_next_gen(gen: Generation, next_gen: Generation | None) -> int | None:
    if next_gen is None:
        return None
    if not next_gen.extras:
        if reeval_count(next_gen) == 0:
            return 0
        return None

    offspring_names = Counter(item.name for item in gen.offspring)
    next_extras = Counter(item.name for item in next_gen.extras)
    return sum(min(count, next_extras[name]) for name, count in offspring_names.items())


def table_rows(gens: dict[int, Generation], config: dict) -> Iterable[list[str]]:
    n_datasets = int_config(config, "n_datasets", len(config.get("dataset_names") or []))
    n_runs = int_config(config, "n_runs", 1)
    n_extra_runs = int_config(config, "n_extra_runs", 0)
    last_seeds_by_name: dict[str, int] = {}

    for gen_no in sorted(gens):
        gen = gens[gen_no]
        lambda_value = gen.lambda_value or 1

        if gen.offspring:
            offspring_init_evals = sum(item.seeds * n_datasets for item in gen.offspring)
        else:
            offspring_init_evals = (
                offspring_count(gen) * n_runs * lambda_value * n_datasets
            )

        if gen.extras:
            reeval_evals = 0
            for item in gen.extras:
                previous = last_seeds_by_name.get(item.name, n_runs)
                added = max(0, item.seeds - previous)
                reeval_evals += added * n_datasets
                last_seeds_by_name[item.name] = max(previous, item.seeds)
        else:
            reeval_evals = (
                reeval_count(gen) * n_extra_runs * lambda_value * n_datasets
            )

        for item in gen.offspring:
            previous = last_seeds_by_name.get(item.name, 0)
            last_seeds_by_name[item.name] = max(previous, item.seeds)

        total_evals = offspring_init_evals + reeval_evals
        made_next = count_made_next_gen(gen, gens.get(gen_no + 1))
        status = "complete" if gen.complete else "incomplete"

        yield [
            str(gen_no),
            status,
            str(lambda_value),
            f"{offspring_init_evals:,}",
            "?" if made_next is None else str(made_next),
            str(reeval_count(gen)),
            f"{reeval_evals:,}",
            f"{total_evals:,}",
        ]


def print_table(run_dir: Path) -> None:
    config = load_config(run_dir)
    gens = parse_log(run_dir / "run.log")

    n_datasets = int_config(config, "n_datasets", len(config.get("dataset_names") or []))

    print(f"# Racing eval accounting: {run_dir}")
    print(f"# task eval = 1 bundle/config x 1 dataset x 1 seed; datasets={n_datasets}")
    print()
    print(
        "| gen | status | lambda | offspring_init_evals | "
        "offspring_made_next_gen | reeval_bundles | reeval_evals | "
        "total_evals |"
    )
    print("|---:|:---|---:|---:|---:|---:|---:|---:|")
    for row in table_rows(gens, config):
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-generation racing/eval accounting for run 666286."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory to analyze (default: {DEFAULT_RUN_DIR})",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "run.log").exists():
        raise SystemExit(f"Missing run log: {run_dir / 'run.log'}")
    print_table(run_dir)


if __name__ == "__main__":
    main()
