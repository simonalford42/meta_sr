#!/usr/bin/env python3
"""Plot seed-level NeuronBench errors for PySR and run 708907.

Primary artifacts
-----------------
* Baseline PySR: ``runs/190178/neuron_results.json`` (six worlds, five seeds).
* Evolved held-out worlds: ``runs/708907/neuron_full_eval/neuron_results.json``.
* Evolved training world: the first five final reevaluation equations in
  ``runs/708907/run_data.json``.  Their NRMSE is recomputed on the independent
  16,384-state Z-rebound test set used by :class:`NeuronBenchDomain`.

Run 708907 used one training world (Z-rebound), uninformative prompts, and no
execution feedback. Colors encode a binary numerical classification:
recovered <= 1e-6 and failure > 1e-6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains import NeuronBenchDomain  # noqa: E402


FIGURE_DIR = Path(__file__).resolve().parent
BASELINE_RESULTS = ROOT / "runs/190178/neuron_results.json"
EVOLVED_RESULTS = ROOT / "runs/708907/neuron_full_eval/neuron_results.json"
EVOLUTION_RUN = ROOT / "runs/708907/run_data.json"

WORLDS = (
    "z_rebound",
    "h_sag",
    "na_fatigue",
    "ca_rebound",
    "d_type",
    "textbook_M",
)
WORLD_LABELS = {
    "z_rebound": "Z-rebound",
    "h_sag": "H-sag",
    "na_fatigue": "Na-fatigue",
    "ca_rebound": "Ca-rebound",
    "d_type": "D-type",
    "textbook_M": "Textbook M",
}
COLORS = {
    "recovered": "#2E7D32",
    "failure": "#B2182B",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def outcome(nrmse: float) -> str:
    if nrmse <= NeuronBenchDomain.RECOVERED_NRMSE:
        return "recovered"
    return "failure"


def baseline_values() -> dict[str, list[float]]:
    payload = read_json(BASELINE_RESULTS)
    values: dict[str, list[float]] = {}
    for world in WORLDS:
        rows = sorted(
            (row for row in payload["runs"] if row["world"] == world),
            key=lambda row: row["seed"],
        )
        if len(rows) != 5:
            raise ValueError(f"Expected five baseline fits for {world}, found {len(rows)}")
        values[world] = [float(row["best_nrmse"]) for row in rows]
    return values


def evaluate_expressions_on_z_rebound(expressions: list[str]) -> list[float]:
    _, _, X_test, y_test, _ = NeuronBenchDomain().load_train_validation("z_rebound")
    variables = sp.symbols(f"x0:{X_test.shape[1]}")
    locals_by_name = {str(variable): variable for variable in variables}
    denominator = max(
        float(np.sqrt(np.mean(np.asarray(y_test, dtype=float) ** 2))),
        float(np.finfo(float).tiny),
    )

    values = []
    for expression_string in expressions:
        expression = sp.sympify(expression_string, locals=locals_by_name)
        function = sp.lambdify(variables, expression, modules="numpy")
        prediction = np.asarray(
            function(*[X_test[:, index] for index in range(X_test.shape[1])]),
            dtype=float,
        )
        if prediction.ndim == 0:
            prediction = np.full_like(y_test, float(prediction))
        prediction = prediction.reshape(-1)
        if prediction.shape != y_test.shape or not np.all(np.isfinite(prediction)):
            raise ValueError("A Z-rebound reevaluation equation produced invalid predictions")
        rmse = float(np.sqrt(np.mean((prediction - y_test) ** 2)))
        values.append(rmse / denominator)
    return values


def evolved_values() -> dict[str, list[float]]:
    run_data = read_json(EVOLUTION_RUN)
    config = run_data["config"]
    if config.get("dataset_names") != ["z_rebound"]:
        raise ValueError("Run 708907 no longer identifies Z-rebound as its sole training world")
    if config.get("uninformative_prompts") is not True:
        raise ValueError("Run 708907 is not marked as an uninformative-prompt run")
    if int(config.get("execution_feedback_n", -1)) != 0:
        raise ValueError("Run 708907 used execution feedback")

    details = run_data["best_bundle"]["result_details"]
    training_detail = next(item for item in details if item["dataset"] == "z_rebound")
    expressions = training_detail["run_best_equations"][:5]
    if len(expressions) != 5 or any(expression is None for expression in expressions):
        raise ValueError("Expected five usable final reevaluation equations for Z-rebound")

    values = {"z_rebound": evaluate_expressions_on_z_rebound(expressions)}
    held_out = read_json(EVOLVED_RESULTS)
    for world in WORLDS[1:]:
        rows = sorted(
            (row for row in held_out["runs"] if row["world"] == world),
            key=lambda row: row["seed"],
        )
        if len(rows) != 5:
            raise ValueError(f"Expected five evolved fits for {world}, found {len(rows)}")
        values[world] = [float(row["best_nrmse"]) for row in rows]
    return values


def count_outcomes(values: dict[str, list[float]]) -> dict[str, int]:
    counts = {name: 0 for name in COLORS}
    for world_values in values.values():
        for value in world_values:
            counts[outcome(value)] += 1
    return counts


def make_figure(output_stem: Path) -> None:
    baseline = baseline_values()
    evolved = evolved_values()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.2, 4.5))

    # Mark the boundary between the one evolution task and five transfer tasks.
    ax.axvline(0.50, color="#8A8A8A", linewidth=0.8, zorder=1)

    method_specs = (
        ("PySR", baseline, -0.16, "o"),
        ("Evolved PySR", evolved, 0.16, "*"),
    )
    seed_offsets = np.linspace(-0.055, 0.055, 5)
    for world_index, world in enumerate(WORLDS):
        for _, method_values, method_offset, marker in method_specs:
            for seed_offset, value in zip(seed_offsets, method_values[world]):
                ax.scatter(
                    world_index + method_offset + seed_offset,
                    value,
                    s=38 if marker == "o" else 92,
                    marker=marker,
                    facecolor=COLORS[outcome(value)],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=3,
                )

    ax.set_yscale("log")
    ax.set_ylim(3e-13, 3e-2)
    ax.set_xlim(-0.55, len(WORLDS) - 0.45)
    ax.set_ylabel("Held-out NRMSE")
    ax.set_xticks(range(len(WORLDS)), [WORLD_LABELS[world] for world in WORLDS])
    ax.tick_params(axis="x", length=0, pad=7)
    ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(
        0.0,
        1.025,
        "training task",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        3.0,
        1.025,
        "held-out tasks",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=COLORS["recovered"], markeredgecolor="white",
               label="Recovered"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=COLORS["failure"], markeredgecolor="white",
               label="Failure"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.5,
               markerfacecolor="#555555", markeredgecolor="#555555",
               label="PySR"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=9,
               markerfacecolor="#555555", markeredgecolor="#555555",
               label="Evolved PySR"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=4,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.4,
        borderaxespad=0.6,
    )

    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.94))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"PySR outcomes: {count_outcomes(baseline)}")
    print(f"Evolved PySR outcomes: {count_outcomes(evolved)}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=FIGURE_DIR / "neuronbench_uninformative_all_fits",
        help="Output path without a file extension",
    )
    args = parser.parse_args()
    make_figure(args.output_stem)


if __name__ == "__main__":
    main()
