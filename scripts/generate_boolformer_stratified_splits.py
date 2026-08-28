#!/usr/bin/env python3
"""Generate deterministic, nuisance-balanced Boolformer train/val splits.

The final test set intentionally remains an IID draw.  Train and validation
instead use the same stratification recipe with disjoint target IDs so that a
small meta-learning split does not accidentally omit low-sample or noisy tasks.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from boolformer_tasks import (
    BOOLFORMER_FLIP_PROBS,
    BOOLFORMER_N_POINTS,
    BOOLFORMER_TRAJECTORY_FLIP_PROBS,
    get_boolformer_noisy_metadata,
)


N_SUPPORTS = 6
N_QUANTILE_BINS = 5


@dataclass(frozen=True)
class Candidate:
    task_id: str
    support: int
    n_points_idx: int
    flip_idx: int
    trajectory_idx: int
    inactive_bin: int
    complexity: int
    complexity_bin: int = 0


def _index_of(value: float | int, choices: tuple[float | int, ...]) -> int:
    return min(range(len(choices)), key=lambda i: abs(float(choices[i]) - float(value)))


def _candidate_pool(candidates_per_support: int) -> list[Candidate]:
    raw: list[Candidate] = []
    complexities: dict[int, list[int]] = defaultdict(list)
    for support in range(1, N_SUPPORTS + 1):
        for candidate_idx in range(candidates_per_support):
            task_id = f"strat_s{support}_{candidate_idx:05d}"
            meta = get_boolformer_noisy_metadata(task_id)
            complexity = int(meta["complexity"])
            complexities[support].append(complexity)
            raw.append(Candidate(
                task_id=task_id,
                support=support,
                n_points_idx=_index_of(meta["n_points"], BOOLFORMER_N_POINTS),
                flip_idx=_index_of(meta["flip_prob"], BOOLFORMER_FLIP_PROBS),
                trajectory_idx=_index_of(
                    meta["trajectory_flip_prob"],
                    BOOLFORMER_TRAJECTORY_FLIP_PROBS,
                ),
                inactive_bin=min(
                    N_QUANTILE_BINS - 1,
                    int(meta["n_inactive_vars"]) * N_QUANTILE_BINS // 121,
                ),
                complexity=complexity,
            ))

    ordered = {support: sorted(values) for support, values in complexities.items()}
    result = []
    for candidate in raw:
        values = ordered[candidate.support]
        # Empirical within-support quantiles retain the generator's natural
        # complexity distribution while ensuring both splits span it.
        rank = bisect.bisect_left(values, candidate.complexity)
        complexity_bin = min(N_QUANTILE_BINS - 1, rank * N_QUANTILE_BINS // len(values))
        result.append(Candidate(**{
            **candidate.__dict__, "complexity_bin": complexity_bin,
        }))
    return result


def _cyclic_distance(actual: int, desired: int, size: int) -> float:
    delta = abs(actual - desired)
    return min(delta, size - delta) / (size // 2)


def _match_cost(candidate: Candidate, trajectory: int, inactive: int, complexity: int) -> float:
    return (
        _cyclic_distance(candidate.trajectory_idx, trajectory, len(BOOLFORMER_TRAJECTORY_FLIP_PROBS))
        + abs(candidate.inactive_bin - inactive) / (N_QUANTILE_BINS - 1)
        + 0.75 * abs(candidate.complexity_bin - complexity) / (N_QUANTILE_BINS - 1)
    )


def select_split(
    pool: list[Candidate],
    size: int,
    split_offset: int,
    used: set[str],
) -> list[Candidate]:
    if size % (N_SUPPORTS * len(BOOLFORMER_N_POINTS)):
        raise ValueError("split size must be divisible by 30")
    repeats = size // (N_SUPPORTS * len(BOOLFORMER_N_POINTS))
    grouped: dict[tuple[int, int, int], list[Candidate]] = defaultdict(list)
    for candidate in pool:
        grouped[(candidate.support, candidate.n_points_idx, candidate.flip_idx)].append(candidate)

    selected: list[Candidate] = []
    trajectory_counts: Counter[int] = Counter()
    inactive_counts: Counter[int] = Counter()
    complexity_counts: Counter[int] = Counter()
    trajectory_ideal = size / len(BOOLFORMER_TRAJECTORY_FLIP_PROBS)
    quantile_ideal = size / N_QUANTILE_BINS
    template_idx = 0
    for support in range(1, N_SUPPORTS + 1):
        for repeat in range(repeats):
            for n_points_idx in range(len(BOOLFORMER_N_POINTS)):
                # A Latin-cycle assignment makes every support contain every
                # noise level equally often without requiring the full 6x5x5 grid.
                flip_idx = (
                    n_points_idx + 2 * repeat + support + split_offset
                ) % len(BOOLFORMER_FLIP_PROBS)
                trajectory = (4 * template_idx + split_offset) % len(
                    BOOLFORMER_TRAJECTORY_FLIP_PROBS
                )
                inactive = (2 * template_idx + split_offset) % N_QUANTILE_BINS
                complexity = (3 * template_idx + support + split_offset) % N_QUANTILE_BINS
                choices = [
                    candidate
                    for candidate in grouped[(support, n_points_idx, flip_idx)]
                    if candidate.task_id not in used
                ]
                if not choices:
                    raise RuntimeError(
                        "candidate pool exhausted for "
                        f"support={support}, n_points={BOOLFORMER_N_POINTS[n_points_idx]}, "
                        f"flip_prob={BOOLFORMER_FLIP_PROBS[flip_idx]}"
                    )
                chosen = min(
                    choices,
                    key=lambda candidate: (
                        _match_cost(candidate, trajectory, inactive, complexity)
                        + 1.5 * trajectory_counts[candidate.trajectory_idx] / trajectory_ideal
                        + 1.5 * inactive_counts[candidate.inactive_bin] / quantile_ideal
                        + complexity_counts[candidate.complexity_bin] / quantile_ideal,
                        candidate.task_id,
                    ),
                )
                selected.append(chosen)
                used.add(chosen.task_id)
                trajectory_counts[chosen.trajectory_idx] += 1
                inactive_counts[chosen.inactive_bin] += 1
                complexity_counts[chosen.complexity_bin] += 1
                template_idx += 1
    return selected


def _write_manifest(path: Path, candidates: list[Candidate]) -> None:
    path.write_text("".join(f"boolformer_noisy:{candidate.task_id}\n" for candidate in candidates))


def _summarize(label: str, candidates: list[Candidate]) -> None:
    print(f"{label}: {len(candidates)} tasks")
    for field in ("support", "n_points_idx", "flip_idx", "trajectory_idx", "inactive_bin", "complexity_bin"):
        counts = Counter(getattr(candidate, field) for candidate in candidates)
        print(f"  {field}: {dict(sorted(counts.items()))}")
    complexities = [candidate.complexity for candidate in candidates]
    print(
        f"  complexity: mean={sum(complexities) / len(complexities):.2f}, "
        f"range={min(complexities)}..{max(complexities)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--candidates-per-support", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    args = parser.parse_args()

    pool = _candidate_pool(args.candidates_per_support)
    used: set[str] = set()
    train = select_split(pool, args.train_size, split_offset=0, used=used)
    val = select_split(pool, args.val_size, split_offset=1, used=used)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_dir / "boolformer_noisy_stratified_train.txt", train)
    _write_manifest(args.output_dir / "boolformer_noisy_stratified_val.txt", val)
    _summarize("train", train)
    _summarize("validation", val)


if __name__ == "__main__":
    main()
