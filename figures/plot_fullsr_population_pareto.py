#!/usr/bin/env python3
"""Render fixed-axis population Pareto frames from FullSR or PySR snapshots."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evolution_helpers import code_loc


def bundle_loc(bundle):
    if bundle.get("raw_module_body"):
        return code_loc(bundle["raw_module_body"])
    components = bundle.get("functions", bundle.get("operators"))
    if components is None:
        raise ValueError("Bundle contains neither functions nor operators")
    return sum(code_loc(fn.get("code", ""))
               for fn in components.values() if fn)


def frontier(points):
    """Nondominated coordinates: minimize LOC and maximize saved score."""
    result = []
    best = -math.inf
    for loc, score in sorted(set(points), key=lambda p: (p[0], -p[1])):
        if score > best:
            result.append((loc, score))
            best = score
    return result


def limits(values, minimum_pad):
    low, high = min(values), max(values)
    pad = max(minimum_pad, (high - low) * 0.07)
    return [low - pad, high + pad]


def render_sequence(out, populations, run_name, simplify_start, fitness_metric):
    out.mkdir(parents=True, exist_ok=True)
    all_points = [p for points in populations.values() for p in points]
    xlim = limits([p[0] for p in all_points], 5)
    ylim = limits([p[1] for p in all_points], 0.002)
    for gen, points in populations.items():
        front = frontier(points)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.84)
        ax.scatter(*zip(*points), s=65, color="#3578b5", alpha=0.75,
                   label=f"Selected population (n={len(points)})", zorder=3)
        ax.plot(*zip(*front), "o-", color="#d64c3b", linewidth=2,
                markersize=5, label="Population Pareto frontier", zorder=4)
        ax.set(xlim=xlim, ylim=ylim, xlabel="LOC (excluding blanks, comments, and docstrings)",
               ylabel=f"Training score ({fitness_metric}; higher is better)")
        ax.set_title(f"Generation {gen:03d}  |  " +
                     ("Simplification" if gen >= simplify_start else "Original run"), pad=12)
        fig.suptitle(run_name, fontsize=14, y=0.96)
        ax.grid(alpha=0.18)
        ax.legend(loc="lower right", fontsize=9)
        fig.text(0.12, 0.04, "Lower LOC and higher score are better. Frontier uses this generation's population.",
                 fontsize=9, color="#555555")
        fig.savefig(out / f"generation_{gen:03d}.png")
        plt.close(fig)
    return {"generations": list(populations), "xlim": xlim, "ylim": ylim}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--title", help="Display name for a continuation stored under a job ID")
    parser.add_argument("--simplify-start", type=int, default=31)
    args = parser.parse_args()
    title = args.title or args.run.name
    out = args.out_dir or Path(__file__).resolve().parent / f"{args.run.name}_population_pareto"
    data = json.loads((args.run / "run_data.json").read_text())
    fitness_metric = data.get("config", {}).get("fitness_metric", "saved fitness")
    populations = {}
    for generation in data["generations"]:
        points = [(bundle_loc(b), float(b["score"])) for b in generation["population"]
                  if b.get("score") is not None and math.isfinite(float(b["score"]))]
        if not points:
            raise ValueError(f"Empty population at generation {generation['generation']}")
        populations[int(generation["generation"])] = points
    populations = dict(sorted(populations.items()))
    metadata = {"source": str(args.run / "run_data.json"), "title": title,
                "fitness_metric": fitness_metric, "sequences": {}}
    metadata["sequences"]["full_history"] = render_sequence(
        out / "full_history", populations, title, args.simplify_start, fitness_metric)
    simplify = {g: p for g, p in populations.items() if g >= args.simplify_start - 1}
    if simplify:
        metadata["sequences"]["simplification_zoom"] = render_sequence(
            out / "simplification_zoom", simplify, title, args.simplify_start, fitness_metric)
    (out / "axis_limits.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (out / "population.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "population_index", "loc", "train_score", "on_frontier"])
        for gen, points in populations.items():
            front = set(frontier(points))
            for i, (loc, score) in enumerate(points):
                writer.writerow([gen, i, loc, score, (loc, score) in front])
    (out / "README.md").write_text(
        f"# {title}: population Pareto evolution\n\n"
        f"Source: `{args.run / 'run_data.json'}`. "
        f"Saved generations {min(populations)}–{max(populations)}.\n\n"
        "`full_history/` contains every saved generation. `simplification_zoom/` starts "
        "with the generation preceding simplification. Each sequence has fixed x and y "
        "limits across every image; the zoom sequence uses tighter limits. Images sort "
        "chronologically by filename.\n\n"
        "Blue points are the selected population after survival selection. Red points "
        "are its Pareto frontier (minimize LOC, maximize score), with lines as visual "
        "guides. This is not a cumulative archive frontier. Overlapping coordinates "
        "may hide duplicate individuals. Scores are saved training `score` values "
        f"(fitness metric `{fitness_metric}`), not held-out validation scores. LOC uses the project's "
        "`evolution_helpers.code_loc`: excludes blanks, comments, and docstrings, "
        "counting raw module body when present, otherwise all policy functions or operators.\n\n"
        "`population.csv` contains all plotted points; `axis_limits.json` records limits.\n\n"
        "Regenerate from the repository root:\n\n```bash\n"
        f"python figures/plot_fullsr_population_pareto.py {args.run}"
        f" --out-dir {out} --simplify-start {args.simplify_start}"
        f" --title '{title}'\n```\n")
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out.parent))
    print(json.dumps(metadata, indent=2))
    print(f"Saved images to {out}\nArchive: {archive}")


if __name__ == "__main__":
    main()
