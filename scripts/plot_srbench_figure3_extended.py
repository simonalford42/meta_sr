#!/usr/bin/env python3
"""Extend SRBench 2021 Figure 3 with local methods and released MDLformer trials.

Uses the original notebook's aggregation: mean over seeds within dataset/noise,
then mean over datasets, then call the actual SRBench notebook plotting function (including its CIs).
Inputs are existing artifacts; no evaluations or SLURM jobs are launched.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srbench_results_io as srio

OUT = ROOT / "reports/srbench_figure3_extended"
LOCAL = {
    "PySR": "290227",
    "BasicSR": "150814",
    "Evolved BasicSR": "271625",
    "Evolved PySR": "973699",
}
MDL = "MDLformer"
NOISE = [0, .001, .01, .1]
MARKERS = ["o", "s", "x", "+"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    original_path = ROOT / "srbench/results/symbolic_dataset_results_sum.csv.gz"
    original = pd.read_csv(original_path)
    original = original.rename(columns={"algorithm": "method", "data_group": "family",
                                        "target_noise": "noise", "symbolic_solution_rate": "rate",
                                        "random_state_repeats": "n_seeds"})
    columns = ["method", "dataset", "family", "noise", "rate", "n_seeds"]
    original = original[columns].copy()
    original["source"] = "SRBench 2021"
    shared = set(original.dataset)
    assert len(shared) == 130
    frames = [original]
    provenance = {
        "original_paper": "https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper-round1.pdf",
        "original_notebook": "srbench/postprocessing/groundtruth_results.ipynb",
        "original_results_sha256": sha(original_path),
        "srbench_commit": subprocess.check_output(["git", "-C", str(ROOT / "srbench"), "rev-parse", "HEAD"], text=True).strip(),
        "local_runs": {},
        "mdlformer": json.loads((OUT / "mdlformer_source.json").read_text()),
        "aggregation": "Mean over available seeds per dataset/noise, then equally weighted mean over datasets",
        "uncertainty": "Seaborn 95% CI, 1000 dataset-bootstrap samples; RNG seed 20260910",
        "plotting": "Execute the compare function directly from the local SRBench groundtruth_results.ipynb",
        "seaborn_version": sns.__version__,
        "missing_results": "No imputation; retain available trials and dataset/noise cells; counts saved in summaries",
    }
    for method, run in LOCAL.items():
        directory = ROOT / "runs" / run
        manifest = srio.load_manifest(directory)
        keyed = srio.load_keyed_results(directory)
        assert manifest["max_evals"] == 1_000_000
        assert len(keyed) == 5320
        rows = []
        for entry in keyed.values():
            if not entry.get("present") or entry.get("error") is not None:
                raise ValueError(f"Incomplete run {run}: {entry['dataset']}")
            rows.append(dict(method=method, dataset=entry["dataset"], family=entry["family"],
                             noise=float(entry["noise"]), solved=int(entry["solved"])))
        rates = pd.DataFrame(rows).groupby(["method", "dataset", "family", "noise"], as_index=False).agg(
            rate=("solved", "mean"), n_seeds=("solved", "size"))
        assert (rates.n_seeds == 10).all()
        rates["source"] = f"Local runs/{run}"
        frames.append(rates)
        provenance["local_runs"][method] = dict(path=f"runs/{run}", manifest=manifest,
                                               results_sha256=sha(directory / "srbench_full_results.json"))
    mdl = pd.read_csv(OUT / "mdlformer_trials.csv.gz")
    assert mdl.symbolic_solution.notna().all()
    assert not mdl.duplicated(["dataset", "random_state", "target_noise"]).any()
    mdl = mdl.rename(columns={"data_group": "family", "target_noise": "noise"})
    rates = mdl.groupby(["dataset", "family", "noise"], as_index=False).agg(
        rate=("symbolic_solution", "mean"), n_seeds=("symbolic_solution", "size"))
    rates["method"] = MDL
    rates["source"] = "SR4MDL author release"
    frames.append(rates)
    all_rates = pd.concat(frames, ignore_index=True)
    assert all_rates.rate.between(0, 1).all()
    assert set(all_rates.noise) == set(NOISE)
    assert not all_rates.duplicated(["method", "dataset", "noise"]).any()
    provenance["excluded_from_shared_figure"] = sorted(set(all_rates.dataset) - shared)
    return all_rates, shared, provenance


def render(data, stem):
    # Execute the actual notebook function, not a reimplementation of its style.
    notebook_path = ROOT / "srbench/postprocessing/groundtruth_results.ipynb"
    notebook = json.loads(notebook_path.read_text())
    function_source = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def compare(" not in source:
            continue
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "compare")
        function_source = ast.get_source_segment(source, function)
        break
    if function_source is None:
        raise ValueError("Cannot find the SRBench notebook's compare function")
    (OUT / "srbench_compare_source.py.txt").write_text(function_source + "\n")
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    captured = []

    def save(grid, name):
        grid.tight_layout()
        captured.append(grid)

    namespace = dict(sns=sns, np=np, plt=plt, save=save)
    exec(compile(function_source, str(notebook_path), "exec"), namespace)
    plot_data = data.rename(columns={"method": "algorithm", "family": "data_group",
                                     "noise": "target_noise"}).copy()
    plot_data["symbolic_solution_rate_(%)"] = plot_data.rate * 100
    namespace["compare"](
        df_compare=plot_data, x="symbolic_solution_rate_(%)", est=np.mean,
        orient="h", kind="point", join=False, markers=MARKERS,
        hue="target_noise", col="data_group", col_order=["Feynman", "Strogatz"],
        hue_order=NOISE, seed=20260910, n_boot=1000,
        # Explicit discrete palette preserves old Seaborn's categorical noise colors.
        palette=dict(zip(NOISE, sns.color_palette("flare_r", 4))),
        # Enlarge the canvas for extra rows and longer method labels.
        height=max(5, 5 * data.method.nunique() / 14),
        aspect=.8,
    )
    assert len(captured) == 1
    grid = captured[0]
    for family, ax in grid.axes_dict.items():
        n_tasks = data.loc[data.family == family, "dataset"].nunique()
        ax.set_title(f"{family} (n={n_tasks})")
    for ax in grid.axes.flat:
        for label in ax.get_yticklabels():
            if label.get_text() in LOCAL:
                label.set_fontweight("bold")
    grid.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        path = OUT / f"{stem}.{extension}"
        grid.figure.savefig(path, dpi=400, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(grid.figure)
    summaries = []
    for (method, family, noise), group in data.groupby(["method", "family", "noise"]):
        values = group.rate.to_numpy() * 100
        boot = sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000, seed=20260910)
        low, high = np.percentile(boot, [2.5, 97.5])
        summaries.append(dict(method=method, family=family, noise=noise,
                              rate_pct=values.mean(), ci95_low=low, ci95_high=high,
                              n_tasks=len(group), n_trials=int(group.n_seeds.sum())))
    pd.DataFrame(summaries).to_csv(OUT / f"{stem}_summary.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rates, shared, provenance = collect()
    rates.to_csv(OUT / "dataset_rates.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    render(rates[rates.dataset.isin(shared)], "figure3_extended")
    render(rates[rates.method.isin(set(LOCAL) | {MDL})], "five_methods_133_tasks")
    print(OUT / "figure3_extended.png")


if __name__ == "__main__":
    main()
