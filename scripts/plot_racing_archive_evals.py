#!/usr/bin/env python3
"""Plot racing archive score vs evaluation count for each generation.

For each racing generation, the plot shows the archive state at the start of
that generation, before the selected extra evaluations are merged in.

Color:
  red  = currently in population
  blue = in archive but not in population
  yellow outline = selected for extra evaluations this generation

For each distinct seed-eval count N, the rightmost point gets a horizontal
error bar showing +/- score_std_one_eval / sqrt(N).  The default one-seed
standard deviation is 0.07.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "666286"

SCORE_RE = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)"
INITIAL_AVG_RE = re.compile(rf"^\s+Avg\s+(?P<score>{SCORE_RE})\s+(?P<name>.+?):")
RESULT_RE = re.compile(
    rf"^\s+\[(?P<kind>extras|offspring)\]\s+Avg\s+"
    rf"(?P<score>{SCORE_RE})\s+(?P<name>.+?):\s+\(seeds=(?P<seeds>\d+)\)"
)
RACING_RE = re.compile(
    r"Racing gen (?P<gen>\d+): [λ\u03bb]=(?P<lambda>\d+), "
    r"σ̂=(?P<sigma>[0-9.]+), "
    r"(?P<qualifiers>\d+)/(?P<archive>\d+) archive bundles qualify "
    r"for extras \(cap=(?P<cap>\d+) seeds, \+(?P<extra>\d+) per qualifier\)"
)


@dataclass
class BundleState:
    name: str
    score: float
    seeds: int


@dataclass
class RacingMeta:
    gen: int
    lambda_value: int
    sigma: float
    expected_qualifiers: int
    expected_archive_size: int
    cap: int
    extra: int


@dataclass
class RunTrace:
    initial_archive: list[BundleState]
    racing: dict[int, RacingMeta]
    updates: dict[int, dict[str, list[BundleState]]]


def normalize_name(name: str) -> str:
    return " ".join(name.split())


def parse_float(text: str) -> float | None:
    value = float(text)
    return value if math.isfinite(value) else None


def load_config(run_dir: Path) -> dict:
    with (run_dir / "run_data.json").open(encoding="utf-8") as f:
        return (json.load(f).get("config") or {})


def parse_run_log(run_dir: Path, initial_seeds: int) -> RunTrace:
    initial_archive: list[BundleState] = []
    racing: dict[int, RacingMeta] = {}
    updates: dict[int, dict[str, list[BundleState]]] = {}

    in_initial_eval = False
    current_gen: int | None = None

    for line in (run_dir / "run.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if "Evaluating initial population" in line:
            in_initial_eval = True
            continue
        if line.startswith("Best initial bundle:"):
            in_initial_eval = False
            continue

        if in_initial_eval:
            if m := INITIAL_AVG_RE.match(line):
                score = parse_float(m.group("score"))
                if score is not None:
                    initial_archive.append(
                        BundleState(
                            name=normalize_name(m.group("name")),
                            score=score,
                            seeds=initial_seeds,
                        )
                    )
            continue

        if m := RACING_RE.search(line):
            current_gen = int(m.group("gen"))
            racing[current_gen] = RacingMeta(
                gen=current_gen,
                lambda_value=int(m.group("lambda")),
                sigma=float(m.group("sigma")),
                expected_qualifiers=int(m.group("qualifiers")),
                expected_archive_size=int(m.group("archive")),
                cap=int(m.group("cap")),
                extra=int(m.group("extra")),
            )
            updates.setdefault(current_gen, {"extras": [], "offspring": []})
            continue

        if current_gen is None:
            continue

        if m := RESULT_RE.match(line):
            score = parse_float(m.group("score"))
            if score is None:
                continue
            updates.setdefault(current_gen, {"extras": [], "offspring": []})
            updates[current_gen][m.group("kind")].append(
                BundleState(
                    name=normalize_name(m.group("name")),
                    score=score,
                    seeds=int(m.group("seeds")),
                )
            )

    return RunTrace(initial_archive=initial_archive, racing=racing, updates=updates)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def qualification_prob(mu_i: float, n_i: int, mu_p: float, n_p: int, sigma: float) -> float:
    if sigma <= 0 or n_i <= 0 or n_p <= 0:
        return 1.0 if mu_i >= mu_p else 0.0
    se = math.sqrt(sigma**2 / n_i + sigma**2 / n_p)
    if se <= 0:
        return 1.0 if mu_i >= mu_p else 0.0
    return normal_cdf((mu_i - mu_p) / se)


def ranked_indices(archive: list[BundleState]) -> list[int]:
    return sorted(range(len(archive)), key=lambda i: archive[i].score, reverse=True)


def select_population_indices(
    archive: list[BundleState],
    population_size: int,
) -> set[int]:
    return set(ranked_indices(archive)[:population_size])


def select_qualifying_indices(
    archive: list[BundleState],
    population_size: int,
    meta: RacingMeta,
    qual_threshold: float = 0.05,
    max_qualifiers_factor: int = 2,
) -> set[int]:
    ranked = ranked_indices(archive)
    max_qualifiers = max_qualifiers_factor * population_size

    if len(ranked) < population_size:
        candidates = [
            idx for idx in ranked
            if min(meta.extra, meta.cap - archive[idx].seeds) > 0
        ]
        candidates.sort(key=lambda idx: archive[idx].seeds)
        return set(candidates[:max_qualifiers])

    pth = archive[ranked[population_size - 1]]
    candidates: list[int] = []
    for idx in ranked:
        bundle = archive[idx]
        if bundle.seeds >= meta.cap:
            continue
        prob = qualification_prob(
            bundle.score,
            bundle.seeds,
            pth.score,
            pth.seeds,
            meta.sigma,
        )
        if prob < qual_threshold:
            continue
        if min(meta.extra, meta.cap - bundle.seeds) <= 0:
            continue
        candidates.append(idx)

    candidates.sort(key=lambda idx: archive[idx].seeds)
    return set(candidates[:max_qualifiers])


def update_existing_bundle(archive: list[BundleState], updated: BundleState) -> bool:
    for bundle in archive:
        if bundle.name == updated.name:
            bundle.score = updated.score
            bundle.seeds = updated.seeds
            return True
    return False


def apply_generation_updates(
    archive: list[BundleState],
    updates: dict[str, list[BundleState]],
) -> None:
    for updated in updates.get("extras", []):
        if not update_existing_bundle(archive, updated):
            print(f"WARNING: extras bundle not found in archive: {updated.name}")
    archive.extend(updates.get("offspring", []))


def plot_generation(
    archive: list[BundleState],
    population_indices: set[int],
    qualifier_indices: set[int],
    meta: RacingMeta,
    run_id: str,
    out_path: Path,
    score_std_one_eval: float,
) -> None:
    xs = [b.score for b in archive]
    ys = [b.seeds for b in archive]
    facecolors = [
        "#d62728" if i in population_indices else "#1f77b4"
        for i in range(len(archive))
    ]
    edgecolors = [
        "#f2c300" if i in qualifier_indices else "#262626"
        for i in range(len(archive))
    ]
    linewidths = [2.2 if i in qualifier_indices else 0.35 for i in range(len(archive))]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        xs,
        ys,
        s=58,
        c=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=0.88,
    )

    # Show score uncertainty without cluttering the whole cloud: for every
    # evaluation count N, annotate the highest-score bundle at that N.
    rightmost_by_eval_count: dict[int, tuple[int, BundleState]] = {}
    for idx, bundle in enumerate(archive):
        current = rightmost_by_eval_count.get(bundle.seeds)
        if current is None or bundle.score > current[1].score:
            rightmost_by_eval_count[bundle.seeds] = (idx, bundle)

    for idx, bundle in sorted(rightmost_by_eval_count.values(), key=lambda item: item[1].seeds):
        if bundle.seeds <= 0:
            continue
        xerr = score_std_one_eval / math.sqrt(bundle.seeds)
        ax.errorbar(
            bundle.score,
            bundle.seeds,
            xerr=xerr,
            fmt="none",
            ecolor="#111111",
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            alpha=0.85,
            zorder=4,
        )

    if xs:
        x_pad = max(0.02, (max(xs) - min(xs)) * 0.08)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    if ys:
        y_pad = max(1.0, (max(ys) - min(ys)) * 0.12)
        ax.set_ylim(max(0, min(ys) - y_pad), max(ys) + y_pad)

    ax.set_xlabel("bundle score")
    ax.set_ylabel("bundle-seed evals")
    ax.set_title(
        f"Run {run_id} racing gen {meta.gen}: "
        f"archive={len(archive)}, population={len(population_indices)}, "
        f"considered={len(qualifier_indices)}"
    )
    ax.grid(alpha=0.25)

    legend_items = [
        Line2D(
            [0], [0], marker="o", linestyle="", label="in population",
            markerfacecolor="#d62728", markeredgecolor="#262626", markersize=8,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="", label="not in population",
            markerfacecolor="#1f77b4", markeredgecolor="#262626", markersize=8,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="", label="considered for more evals",
            markerfacecolor="white", markeredgecolor="#f2c300",
            markeredgewidth=2.2, markersize=8,
        ),
        Line2D(
            [0], [0], color="#111111", linestyle="-",
            label=f"+/- {score_std_one_eval:g}/sqrt(N) score std",
        ),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plots(run_dir: Path, out_dir: Path, score_std_one_eval: float) -> list[Path]:
    config = load_config(run_dir)
    population_size = int(config.get("population_size") or 10)
    initial_seeds = int(config.get("n_runs") or 1)
    run_id = run_dir.name

    trace = parse_run_log(run_dir, initial_seeds=initial_seeds)
    if not trace.initial_archive:
        raise RuntimeError(f"No initial archive entries parsed from {run_dir / 'run.log'}")

    archive = [BundleState(b.name, b.score, b.seeds) for b in trace.initial_archive]
    written: list[Path] = []
    for gen in sorted(trace.racing):
        meta = trace.racing[gen]
        population_indices = select_population_indices(archive, population_size)
        qualifier_indices = select_qualifying_indices(archive, population_size, meta)

        if len(archive) != meta.expected_archive_size:
            print(
                f"WARNING gen {gen}: reconstructed archive has {len(archive)} "
                f"but log says {meta.expected_archive_size}"
            )
        if len(qualifier_indices) != meta.expected_qualifiers:
            print(
                f"WARNING gen {gen}: selected {len(qualifier_indices)} qualifiers "
                f"but log says {meta.expected_qualifiers}"
            )

        out_path = out_dir / f"gen_{gen}.png"
        plot_generation(
            archive=archive,
            population_indices=population_indices,
            qualifier_indices=qualifier_indices,
            meta=meta,
            run_id=run_id,
            out_path=out_path,
            score_std_one_eval=score_std_one_eval,
        )
        written.append(out_path)

        apply_generation_updates(archive, trace.updates.get(gen, {}))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot racing archive score vs eval count for each generation."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory to analyze (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: plots/<run_id>_racing)",
    )
    parser.add_argument(
        "--score-std-one-eval",
        type=float,
        default=0.07,
        help="One-seed score standard deviation for error bars (default: 0.07)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "plots" / f"{run_dir.name}_racing"
    else:
        out_dir = out_dir.resolve()

    written = make_plots(
        run_dir,
        out_dir,
        score_std_one_eval=args.score_std_one_eval,
    )
    print(f"Wrote {len(written)} plots to {out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
