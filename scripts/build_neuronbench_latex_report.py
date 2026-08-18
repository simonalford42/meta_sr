#!/usr/bin/env python3
"""Build the publication-style LaTeX report for the NeuronBench PySR study."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "reports" / "neuronbench_pysr_fully_observable.results.json"
DEFAULT_TEX = ROOT / "reports" / "neuronbench_pysr_fully_observable.tex"
ASSET_DIR = ROOT / "reports" / "neuronbench_pysr_fully_observable_assets"
WORLDS = ("z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M")
WORLD_LABELS = {
    "z_rebound": r"Z-rebound",
    "h_sag": r"H-sag",
    "na_fatigue": r"Na-fatigue",
    "ca_rebound": r"Ca-rebound",
    "d_type": r"D-type",
    "textbook_M": r"Textbook-M",
}
METHODS = ("baseline", "evolved_538190")
METHOD_LABELS = {"baseline": "Vanilla PySR", "evolved_538190": "Evolved 538190"}
COLORS = {"baseline": "#1768AC", "evolved_538190": "#D1495B"}
WORLD_COLORS = dict(zip(WORLDS, plt.get_cmap("tab10").colors[:6]))


def load_results(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload["results"]
    if len(results) != 36:
        raise ValueError(f"Expected 36 result records, found {len(results)}")
    return payload, results


def sci(value: float, digits: int = 2) -> str:
    if value == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"${coefficient:.{digits}f}\!\times\!10^{{{exponent}}}$"


def escape_tex(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def frontier_envelope(result: Dict[str, Any], max_complexity: int = 35) -> np.ndarray:
    rows = sorted(result["frontier"], key=lambda row: int(row["complexity"]))
    output = np.full(max_complexity, np.nan)
    best = float("inf")
    cursor = 0
    for complexity in range(1, max_complexity + 1):
        while cursor < len(rows) and int(rows[cursor]["complexity"]) <= complexity:
            best = min(best, float(rows[cursor]["test_nrmse"]))
            cursor += 1
        if math.isfinite(best):
            output[complexity - 1] = best
    return output


def plot_frontier_grid(results: Sequence[Dict[str, Any]], output: Path) -> None:
    index = {(r["method"], r["world"], int(r["seed"])): r for r in results}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharex=True, sharey=True)
    x = np.arange(1, 36)
    for ax, world in zip(axes.flat, WORLDS):
        for method in METHODS:
            curves = np.vstack([frontier_envelope(index[(method, world, seed)]) for seed in (0, 1, 2)])
            median = np.nanmedian(curves, axis=0)
            low = np.nanmin(curves, axis=0)
            high = np.nanmax(curves, axis=0)
            ax.fill_between(x, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
            ax.plot(x, median, color=COLORS[method], linewidth=2.0, label=METHOD_LABELS[method])
        gt_complexity = int(index[("baseline", world, 0)]["ground_truth_complexity"])
        ax.axvline(gt_complexity, color="#333333", linestyle="--", linewidth=1.0)
        ax.axhline(1e-6, color="#2A9D8F", linestyle=":", linewidth=1.2)
        ax.axhline(1e-3, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_title(WORLD_LABELS[world], fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 2)
        ax.set_xlim(1, 35)
        ax.grid(True, which="major", color="#D8D8D8", linewidth=0.55)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
    for ax in axes[-1, :]:
        ax.set_xlabel("maximum expression complexity")
    for ax in axes[:, 0]:
        ax.set_ylabel("best held-out NRMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.0),
        plt.Line2D([0], [0], color="#2A9D8F", linestyle=":", linewidth=1.2),
    ])
    labels.extend(["ground-truth complexity", r"recovery threshold ($10^{-6}$)"])
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Pareto-frontier accuracy by NeuronBench world", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_paired(results: Sequence[Dict[str, Any]], output: Path) -> None:
    index = {(r["method"], r["world"], int(r["seed"])): r for r in results}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    markers = {0: "o", 1: "s", 2: "^"}
    values: List[float] = []
    for world in WORLDS:
        for seed in (0, 1, 2):
            baseline = float(index[("baseline", world, seed)]["best_frontier"]["test_nrmse"])
            evolved = float(index[("evolved_538190", world, seed)]["best_frontier"]["test_nrmse"])
            values.extend([baseline, evolved])
            ax.scatter(
                baseline, evolved, marker=markers[seed], s=54,
                color=WORLD_COLORS[world], edgecolor="white", linewidth=0.6, zorder=3,
            )
    low, high = min(values) / 2.0, max(values) * 2.0
    ax.plot([low, high], [low, high], "--", color="#333333", linewidth=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel("Vanilla PySR: best frontier NRMSE")
    ax.set_ylabel("Evolved 538190: best frontier NRMSE")
    ax.grid(True, which="both", color="#E5E5E5", linewidth=0.55)
    world_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=WORLD_COLORS[w],
                   markeredgecolor="white", markersize=7, label=WORLD_LABELS[w])
        for w in WORLDS
    ]
    seed_handles = [
        plt.Line2D([0], [0], marker=markers[s], color="#555555", linestyle="none",
                   markersize=6, label=f"seed {s}") for s in (0, 1, 2)
    ]
    first = ax.legend(handles=world_handles, loc="upper left", fontsize=8, ncol=2, frameon=True)
    ax.add_artist(first)
    ax.legend(handles=seed_handles, loc="lower right", fontsize=8, frameon=True)
    ax.text(0.97, 0.06, "below diagonal: evolved better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="#444444")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def result_index(results: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    return {(r["method"], r["world"], int(r["seed"])): r for r in results}


def world_summary_rows(results: Sequence[Dict[str, Any]]) -> str:
    index = result_index(results)
    rows = []
    for world in WORLDS:
        b = np.asarray([index[("baseline", world, seed)]["best_frontier"]["test_nrmse"] for seed in (0, 1, 2)])
        e = np.asarray([index[("evolved_538190", world, seed)]["best_frontier"]["test_nrmse"] for seed in (0, 1, 2)])
        winner = "Vanilla" if np.median(b) < np.median(e) else "Evolved"
        gt_complexity = int(index[("baseline", world, 0)]["ground_truth_complexity"])
        rows.append(
            f"{WORLD_LABELS[world]} & {gt_complexity} & {sci(float(np.min(b)))} & {sci(float(np.median(b)))} "
            f"& {sci(float(np.min(e)))} & {sci(float(np.median(e)))} & {winner} \\\\"
        )
    return "\n".join(rows)


def detailed_rows(results: Sequence[Dict[str, Any]]) -> str:
    index = result_index(results)
    rows = []
    for world in WORLDS:
        values = []
        for method in METHODS:
            for seed in (0, 1, 2):
                record = index[(method, world, seed)]
                nrmse = float(record["best_frontier"]["test_nrmse"])
                text = sci(nrmse, 2)
                if nrmse <= 1e-6:
                    text = r"\cellcolor{green!16}\textbf{" + text + "}"
                elif nrmse <= 1e-3:
                    text = r"\cellcolor{blue!10}" + text
                values.append(text)
        rows.append(f"{WORLD_LABELS[world]} & " + " & ".join(values) + " \\\\")
    return "\n".join(rows)


def render_tex(payload: Dict[str, Any], results: Sequence[Dict[str, Any]], tex_path: Path) -> None:
    aggregate = payload["aggregate"]
    baseline = aggregate["methods"]["baseline"]
    evolved = aggregate["methods"]["evolved_538190"]
    paired = aggregate["paired"]
    index = result_index(results)
    recovered = index[("baseline", "na_fatigue", 1)]
    recovered_eq = recovered["best_frontier"]
    evolved_case = index[("evolved_538190", "d_type", 2)]
    evolved_eq = evolved_case["best_frontier"]

    text = rf"""\documentclass[10pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{microtype}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs,array,multirow}}
\usepackage[table]{{xcolor}}
\usepackage{{siunitx}}
\usepackage{{pdflscape}}
\usepackage{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\usepackage{{parskip}}

\definecolor{{navy}}{{HTML}}{{153B5B}}
\definecolor{{blue}}{{HTML}}{{1768AC}}
\definecolor{{red}}{{HTML}}{{D1495B}}
\definecolor{{teal}}{{HTML}}{{2A9D8F}}
\definecolor{{softgray}}{{HTML}}{{F3F5F7}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=blue,pdftitle={{PySR on Fully Observable NeuronBench}}}}
\captionsetup{{font=small,labelfont=bf}}
\setlist[itemize]{{leftmargin=1.4em,itemsep=0.25em,topsep=0.25em}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\color{{navy}} PySR on Fully Observable NeuronBench}}
\fancyhead[R]{{\small\color{{gray}} 1M evaluations, 3 seeds}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}
\newcommand{{\methodA}}{{\textcolor{{blue}}{{Vanilla PySR}}}}
\newcommand{{\methodB}}{{\textcolor{{red}}{{Evolved 538190}}}}

\begin{{document}}

\begin{{center}}
{{\color{{navy}}\LARGE\bfseries PySR on Fully Observable NeuronBench}}\\[0.55em]
{{\large A controlled symbolic-regression comparison of vanilla PySR and evolved run 538190}}\\[0.8em]
{{\small Six deterministic worlds $\times$ two methods $\times$ three seeds; $10^6$ maximum evaluations per fit}}\\[0.25em]
{{\small Report generated 18 August 2026}}
\end{{center}}

\vspace{{0.6em}}
\noindent\colorbox{{softgray}}{{\parbox{{0.96\linewidth}}{{
\textbf{{Bottom line.}} Vanilla PySR produced the only numerically recovered vector field
(Na-fatigue, seed 1; held-out NRMSE {sci(float(recovered_eq['test_nrmse']))}), but neither
method achieved literal symbolic identity. Across all 18 paired fits the result was a
\textbf{{9--9 tie}}. Evolved 538190 was modestly better on affine-calibrated shape error,
as expected from its affine-invariant loss, but it did not improve full physical-dynamics recovery.
}}}}

\section{{Question and reduction}}

NeuronBench\footnote{{\url{{https://github.com/murphyk/neuronbench}}, pinned at
\texttt{{c354622458c460b419cab821d482c879f0578377}}. The benchmark is introduced in
\url{{https://arxiv.org/abs/2608.09696}}.}} normally combines active experiment design,
partially observed gating state, and mechanistic model discovery. This study deliberately removes
the first two difficulties. For every deterministic world, the regressor observes injected current
$I_{{\rm ext}}$, membrane voltage $V$, and every composite channel-open fraction $\phi_c$ from the
benchmark's Eq.~(32). The target is the exact voltage vector field
\begin{{equation}}
\dot V = I_{{\rm ext}} - \sum_c g_c\,\phi_c\,(V-E_c).
\label{{eq:vectorfield}}
\end{{equation}}
The result is six ordinary noiseless scalar SR problems. The composite features remain mechanistic:
for example, $\phi_{{\rm Na}}=m_{{\rm Na}}^3h_{{\rm Na}}$ and
$\phi_{{\rm K}}=n_{{\rm K}}^4$; in Na-fatigue,
$\phi_{{\rm Na}}=m_{{\rm Na}}^3h_{{\rm Na}}s_{{\rm Na}}$ includes the slow inactivation state.

\begin{{table}}[h]
\centering
\caption{{World-specific term added to the common Na/K/leak current balance.}}
\small
\begin{{tabular}}{{lll}}
\toprule
World & Additional observable / state change & Additional term in $\dot V$ \\
\midrule
Z-rebound & $\phi_Z=m_Z^2h_Z$ & $-4\phi_Z(V-120)$ \\
H-sag & $\phi_h=m_h$ & $-5\phi_h(V+30)$ \\
Na-fatigue & $\phi_{{\rm Na}}$ includes $s_{{\rm Na}}$ & none beyond modified $\phi_{{\rm Na}}$ \\
Ca-rebound & $\phi_T=m_T^2h_T$ & $-3.2\phi_T(V-120)$ \\
D-type & $\phi_D=m_Dh_D$ & $-9\phi_D(V+77)$ \\
Textbook-M & $\phi_M=m_M$ & $-2.5\phi_M(V+77)$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Experimental protocol}}

\begin{{itemize}}
\item \textbf{{Data.}} 1,024 training and 16,384 held-out scrambled-Sobol states per world,
covering $I_{{\rm ext}}\in[-40,20]$, $V\in[-95,60]$, and each $\phi_c\in[0,1]$.
Targets are analytic evaluations of Eq.~\eqref{{eq:vectorfield}}---there is no noise or numerical differentiation.
\item \textbf{{Shared search settings.}} Binary operators $\{{+,-,\times\}}$, no unary operators,
15 populations of 33 members, maximum size 35, Float64 arithmetic, deterministic serial execution,
and $10^6$ maximum evaluations. Target RMS scaling is inverted before every reported error.
\item \textbf{{Only treatment difference.}} Evolved 538190 replaces PySR's mutation, selection,
survival, and loss with the saved bundle from \texttt{{runs/538190}}. All data, seeds, and search
hyperparameters are otherwise identical.
\item \textbf{{Assessment.}} For each run, the reported error is the lowest held-out NRMSE attained
anywhere on the discovered Pareto frontier (an oracle assessment of the frontier, not test-set model selection).
Numerical recovery means NRMSE $\leq10^{{-6}}$; near-exact means $\leq10^{{-3}}$.
Exact symbolic equality is checked separately with SymPy.
\end{{itemize}}

\section{{Aggregate results}}

\begin{{table}}[h]
\centering
\caption{{Overall comparison across 18 fits per method. Lower NRMSE is better.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & Recovered & Exact symbolic & Median raw NRMSE & Median affine-calibrated NRMSE \\
\midrule
\methodA & {baseline['recovered']}/18 & {baseline['symbolic_exact']}/18 & {sci(float(baseline['median_nrmse']))} & {sci(float(baseline['median_affine_calibrated_nrmse']))} \\
\methodB & {evolved['recovered']}/18 & {evolved['symbolic_exact']}/18 & {sci(float(evolved['median_nrmse']))} & {sci(float(evolved['median_affine_calibrated_nrmse']))} \\
\bottomrule
\end{{tabular}}
\end{{table}}

The raw comparison does not support an overall improvement from 538190. Vanilla has the lower
median raw error ({sci(float(baseline['median_nrmse']))} versus {sci(float(evolved['median_nrmse']))})
and the only strict recovery. The paired win count is exactly
{paired['evolved_wins']} evolved to {paired['baseline_wins']} vanilla. The secondary calibrated-shape
metric favors 538190 ({sci(float(evolved['median_affine_calibrated_nrmse']))} versus
{sci(float(baseline['median_affine_calibrated_nrmse']))}), which is consistent with its custom loss:
that loss scores an affine correction $a f(x)+b$ internally, while the physical equation must itself
recover scale and offset.

\begin{{table}}[h]
\centering
\caption{{Best and median raw frontier error over three seeds. ``Winner'' compares medians.}}
\small
\begin{{tabular}}{{lrrrrrl}}
\toprule
& GT & \multicolumn{{2}}{{c}}{{Vanilla PySR}} & \multicolumn{{2}}{{c}}{{Evolved 538190}} & \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}
World & complexity & best & median & best & median & winner \\
\midrule
{world_summary_rows(results)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[p]
\centering
\includegraphics[width=0.83\linewidth]{{neuronbench_pysr_fully_observable_assets/paired_raw.pdf}}
\caption{{Per-seed comparison of the best equation anywhere on each discovered frontier.
Markers encode seed and colors encode world. The 9--9 split around the diagonal shows that
neither method is consistently better.}}
\end{{figure}}

\begin{{landscape}}
\begin{{figure}}[p]
\centering
\includegraphics[width=0.97\linewidth]{{neuronbench_pysr_fully_observable_assets/frontier_grid.pdf}}
\caption{{Held-out accuracy along the discovered Pareto frontiers. Solid curves are the median
best-so-far envelope over three seeds; shaded bands span the seedwise minimum to maximum.
The vertical dashed line is the reference ground-truth complexity. The horizontal teal line
is the strict numerical-recovery threshold. Na-fatigue is the only world to cross it.}}
\end{{figure}}
\end{{landscape}}

\section{{What the one recovered run found}}

Vanilla PySR's Na-fatigue seed-1 frontier contains a complexity-35 expression with held-out
NRMSE {sci(float(recovered_eq['test_nrmse']), 3)}. It is not literally identical under symbolic
simplification, but expanding its redundant tree exposes an extremely accurate coefficient match:
\begin{{align}}
\dot V_{{\rm true}} ={{}}& I_{{\rm ext}} -36\phi_KV-2772\phi_K-120\phi_{{\rm Na}}V
 +6000\phi_{{\rm Na}}-0.3V-16.32,\\
\widehat{{\dot V}} ={{}}& 1.000045I_{{\rm ext}}-35.999901\phi_KV-2771.9985\phi_K
 -119.99995\phi_{{\rm Na}}V \notag\\
&+6000.0037\phi_{{\rm Na}}-0.300082V-16.322334.
\end{{align}}
Thus the numerical recovery is scientifically meaningful, although the search used 12 more nodes
than the compact reference expression (35 versus 23) and did not reproduce exact floating constants.

\begin{{table}}[h]
\centering
\caption{{Expanded coefficients for the recovered Na-fatigue run.}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Term & truth & discovered & relative error \\
\midrule
$I_{{\rm ext}}$ & $1$ & $1.000045$ & $4.5\times10^{{-5}}$ \\
$\phi_KV$ & $-36$ & $-35.999901$ & $2.8\times10^{{-6}}$ \\
$\phi_K$ & $-2772$ & $-2771.9985$ & $5.4\times10^{{-7}}$ \\
$\phi_{{\rm Na}}V$ & $-120$ & $-119.99995$ & $4.2\times10^{{-7}}$ \\
$\phi_{{\rm Na}}$ & $6000$ & $6000.0037$ & $6.2\times10^{{-7}}$ \\
$V$ & $-0.3$ & $-0.300082$ & $2.7\times10^{{-4}}$ \\
constant & $-16.32$ & $-16.322334$ & $1.4\times10^{{-4}}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Best evolved near-miss}}

The strongest evolved result is D-type seed 2 (NRMSE {sci(float(evolved_eq['test_nrmse']), 3)}).
After expansion,
\begin{{align}}
\widehat{{\dot V}}_{{538190}} ={{}}&0.997137 I_{{\rm ext}}-9.18239\phi_DV-695.786\phi_D
-36.1745\phi_KV-2775.07\phi_K \notag\\
&-120.193\phi_{{\rm Na}}V+5997.49\phi_{{\rm Na}}-11.5219.
\end{{align}}
It identifies all three gated-current structures and gets their coefficients close, but it omits the
leak slope $-0.3V$ and shifts the constant (truth $-16.32$). This is exactly the distinction between
shape recovery and a correct physical vector field that the raw-versus-calibrated evaluation is meant
to expose.

\section{{Conclusions and limitations}}

\begin{{enumerate}}[leftmargin=1.6em,itemsep=0.35em]
\item \textbf{{Vanilla PySR can solve at least one simplified NeuronBench world.}} It numerically
recovers Na-fatigue in one of three seeds, with coefficient errors mostly below $10^{{-4}}$ relative.
\item \textbf{{Run 538190 does not improve strict recovery at this budget.}} It has zero strict
recoveries, a slightly worse raw median, and an exactly tied paired win count.
\item \textbf{{538190 does improve affine-calibrated shape modestly.}} This aligns with its evolved
loss rather than demonstrating better recovery of physically scaled dynamics.
\item \textbf{{This is intentionally not the original benchmark task.}} Composite open fractions
are supplied, all states are sampled directly, and the study learns only $\dot V$. It does not test
active design, recovery of gate ODEs, latent-state inference, or trajectory forecasting.
\item \textbf{{Three seeds are enough for a demo, not a fine-grained ranking.}} The broad seed bands
and 9--9 paired split argue against claiming a small algorithmic advantage either way.
\end{{enumerate}}

\begin{{landscape}}
\section{{Appendix: all 36 best-frontier errors}}
\begin{{table}}[h]
\centering
\caption{{Held-out raw NRMSE of the closest equation anywhere on each run's frontier.
Green marks strict recovery; blue marks near-exact ($\leq10^{{-3}}$).}}
\normalsize
\setlength{{\tabcolsep}}{{9pt}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{Vanilla PySR}} & \multicolumn{{3}}{{c}}{{Evolved 538190}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
World & seed 0 & seed 1 & seed 2 & seed 0 & seed 1 & seed 2 \\
\midrule
{detailed_rows(results)}
\bottomrule
\end{{tabular}}
\end{{table}}

\vspace{{1em}}
\begin{{minipage}}{{0.9\linewidth}}
\subsection*{{Reproducibility}}
{{\small NeuronBench was installed into the \texttt{{meta\_sr}} conda environment at the pinned commit above.
The complete frontiers and metrics are in
\path{{reports/neuronbench_pysr_fully_observable.results.json}}. The experiment runner is
\path{{scripts/neuronbench_fully_observable.py}} and this report is regenerated by
\path{{scripts/build_neuronbench_latex_report.py}}. SLURM array job 142502 completed all 36 tasks
without a nonzero exit or traceback.}}
\end{{minipage}}
\end{{landscape}}

\end{{document}}
"""
    tex_path.write_text(text, encoding="utf-8")


def compile_tex(tex_path: Path) -> Path:
    command = ["tectonic", tex_path.name]
    subprocess.run(command, cwd=tex_path.parent, check=True)
    return tex_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    payload, results = load_results(args.results)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plot_frontier_grid(results, ASSET_DIR / "frontier_grid.pdf")
    plot_paired(results, ASSET_DIR / "paired_raw.pdf")
    render_tex(payload, results, args.tex)
    print(f"Wrote {args.tex}")
    if not args.no_compile:
        pdf = compile_tex(args.tex)
        print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
