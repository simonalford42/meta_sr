#!/usr/bin/env python3
"""Extend SRBench 2021 Figure 3 with local methods and released MDLformer trials.

Uses the original notebook's aggregation: mean over seeds within dataset/noise,
then mean over datasets, with percentile bootstrap 95% CIs across datasets.
Inputs are existing artifacts; no evaluations or SLURM jobs are launched.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srbench_results_io as srio

OUT = ROOT / "reports/srbench_figure3_extended"
LOCAL = {
    "PySR": "290227",
    "BasicSR": "150814",
    "Evolved BasicSR (GT-R2)": "271625",
    "Evolved PySR (709715)": "973699",
}
MDL = "MDLformer (SR4MDL)"
NOISE = [0, .001, .01, .1]
COLORS = ["#75316e", "#aa3a6e", "#d65a61", "#ec976e"]
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
        "uncertainty": "95% percentile CI from 10000 dataset-bootstrap samples; RNG seed 20260910",
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


def render(data, stem, subtitle, footnote):
    rng = np.random.default_rng(20260910)
    # Same overall ordering criterion as the original notebook: average noise-level means.
    order = data.groupby(["method", "noise"]).rate.mean().groupby("method").mean().sort_values(ascending=False).index.tolist()
    added = set(LOCAL) | {MDL}
    height = max(6.2, len(order) * .48 + 2.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, height), sharey=True)
    fig.subplots_adjust(left=.245, right=.975, top=1-1.25/height,
                        bottom=1.45/height, wspace=.10)
    summaries = []
    for ax, family in zip(axes, ["Feynman", "Strogatz"]):
        n_tasks = data.loc[data.family == family, "dataset"].nunique()
        ax.set_title(f"{family} · {n_tasks} tasks", fontsize=12, pad=12)
        for i, method in enumerate(order):
            if method in added:
                ax.axhspan(i-.45, i+.45, color="#edf3f8", zorder=0)
            for j, noise in enumerate(NOISE):
                group = data[(data.method == method) & (data.family == family) & (data.noise == noise)]
                values = group.rate.to_numpy() * 100
                if not len(values):
                    continue
                rate = values.mean()
                boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
                low, high = np.quantile(boot, [.025, .975])
                ax.errorbar(rate, i + (j-1.5)*.16, xerr=[[rate-low], [high-rate]],
                            fmt=MARKERS[j], color=COLORS[j], ms=4.6, capsize=2,
                            elinewidth=1.0, markeredgewidth=1.1, zorder=3)
                summaries.append(dict(method=method, family=family, noise=noise,
                                      rate_pct=rate, ci95_low=low, ci95_high=high,
                                      n_tasks=len(group), n_trials=int(group.n_seeds.sum())))
        ax.set(xlim=(-2, 102), xticks=[0, 25, 50, 75, 100], xlabel="Symbolic solution rate (%)")
        ax.set_yticks(range(len(order)), order)
        ax.set_ylim(len(order)-.6, -.6)
        ax.grid(axis="x", color="#dce0e5", linewidth=.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, pad=8)
    for label in axes[0].get_yticklabels():
        if label.get_text() in added:
            label.set_fontweight("bold")
            label.set_color("#234965")
    fig.suptitle("SRBench ground-truth recovery", fontsize=18, fontweight="bold", y=1-.2/height)
    fig.text(.5, 1-.65/height, subtitle, ha="center", fontsize=10, color="#4b5563")
    handles = [Line2D([0], [0], color=c, marker=m, linestyle="none", markersize=6,
                      label=f"{n:g}") for n, c, m in zip(NOISE, COLORS, MARKERS)]
    fig.legend(handles=handles, title="Target noise", loc="center", ncol=4,
               frameon=False, bbox_to_anchor=(.61, .68/height))
    fig.text(.5, .2/height, footnote, ha="center", va="center", fontsize=8, color="#4b5563")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(OUT / f"{stem}_summary.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rates, shared, provenance = collect()
    rates.to_csv(OUT / "dataset_rates.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    render(rates[rates.dataset.isin(shared)], "figure3_extended",
           "Original 14 methods + five additions · shared 130-task benchmark",
           "Points: mean over tasks after averaging seeds. Bars: 95% dataset-bootstrap CIs.\n"
           "Highlighted rows: added methods. Historical and local evaluation protocols differ; see README.")
    render(rates[rates.method.isin(set(LOCAL) | {MDL})], "five_methods_133_tasks",
           "Five added methods · full 133-task benchmark",
           "Points: mean over tasks after averaging seeds. Bars: 95% dataset-bootstrap CIs.\n"
           "Local methods: 10 seeds, 1M evaluations. MDLformer: author-released available trials.")
    print(OUT / "figure3_extended.png")


if __name__ == "__main__":
    main()
