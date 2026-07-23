"""Reusable machinery for the per-GENERATION symbolic-regression comparison
figures (the "gen axis" family: n1 vs n3 vs n10, smart-n1 vs n3, single best run).

This module holds only the *plotting machinery* -- data loading, per-gen
aggregation, and a configurable render(). The actual figure definitions (which
run ids, which panels, which titles) live in the companion notebook
gen_axis_plots.ipynb so each figure can be browsed and re-run individually.

Data layer: build_run_entry / build_wandb_index / col_stats come from
plot_eval_axis_comparison.py (which reconstructs per-gen metrics from each run's
run_data.json and fetches val-eval series from wandb).

Typical use (from the notebook):

    import gen_axis_plots as gp
    api = gp.get_api()
    widx = gp.build_wandb_index()
    n1 = gp.load_method(api, widx, [89281, 825769, ...], "n1")
    gp.render([("n1", n1, gp.COLOR(0), "o")], gp.OUTDIR / "fig.png",
              "title", panels=gp.PANELS_COMPARISON)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_eval_axis_comparison import (  # noqa: E402
    REPO, Method, Variant, build_run_entry, build_wandb_index, col_stats,
    plot_variant,
)

# Default output directory for the whole family (kept where the old scripts wrote).
OUTDIR = REPO / "plots" / "gen_axis_n1_vs_n3"

# Convenience: tab10 color by index, matching the old scripts' choices.
_CMAP = plt.get_cmap("tab10")


def COLOR(i):
    """tab10 color by index (0=blue, 2=green, 3=red, ...)."""
    return _CMAP(i)


# ---- panel presets -------------------------------------------------------
# Each panel is (by_gen key, title, kind). kind is one of:
#   "line"  -> per-gen line (with std band when a method has >1 seed). Keys that
#              start with "val_" additionally get a marker ONLY at generations
#              where a real val submission happened (the flat forward-filled
#              stretches are drawn as a bare line).
#   "bar"   -> stacked train-score decomposition, one column per method.
#   "table" -> train-score decomposition as a small table (best for one method).

PANELS_COMPARISON = [
    ("best",                    "Best offspring score",                             "line"),
    ("avg_pop",                 "Average population score (train)",                 "line"),
    ("avg_off",                 "Average offspring score (train)",                  "line"),
    ("val_train_avg_score",     "Re-eval train score",                              "line"),
    ("val_avg_score",           "Val score",                                        "line"),
    ("val_winners_curse_delta", "Winners curse (train score - reeval train score)", "line"),
    ("val_overfit",             "Overfitting (reeval train score - val score)",     "line"),
    ("train_decomp",            "Train score decomposition (final)",                "bar"),
]

# Single-run variant: best-so-far -> best-given-current-posteriors, plus a
# re-evals-per-generation panel, and the decomposition as a table.
PANELS_SINGLE = [
    ("best_current",            "Best score (current posteriors)",                  "line"),
    ("avg_pop",                 "Average population score (train)",                 "line"),
    ("avg_off",                 "Average offspring score (train)",                  "line"),
    ("val_train_avg_score",     "Re-eval train score",                              "line"),
    ("val_avg_score",           "Val score",                                        "line"),
    ("gen_reevals",             "Re-evals per generation",                          "line"),
    ("val_winners_curse_delta", "Winners curse (train score - reeval train score)", "line"),
    ("val_overfit",             "Overfitting (reeval train score - val score)",     "line"),
    ("train_decomp",            "Train score decomposition (final)",                "table"),
]


def get_api():
    """A wandb API handle (loads ~/.netrc creds)."""
    return wandb.Api()


def load_method(api, wandb_index, ids, tag):
    """Build a per-seed plot entry for each run id and tag the group.

    Returns a list of run entries (one per loaded seed). Each entry's by_gen also
    gets a "best" series (best-so-far cumulative max) on the gen grid so it can be
    plotted like the other per-gen keys.
    """
    entries = []
    label_counts: dict = {}
    for rid in ids:
        entry = build_run_entry(api, wandb_index, f"runs/{rid}", label_counts)
        if entry is None:
            continue
        entry["by_gen"]["best"] = dict(
            zip(entry["gen"].tolist(), entry["best"].tolist()))
        entries.append(entry)
    print(f"{tag}: loaded {len(entries)}/{len(ids)} seeds")
    return entries


def method_curve_by_gen(method_runs, key):
    """Mean/std of a metric across seeds on the shared generation grid.

    Returns (gens, ymean, ystd) with x = the generation number itself. Gens where
    no seed has a value are dropped.
    """
    gens = sorted({g for r in method_runs for g in r["by_gen"].get(key, {})})
    if not gens:
        return np.array([]), np.array([]), np.array([])
    ymat = np.full((len(method_runs), len(gens)), np.nan)
    for i, r in enumerate(method_runs):
        vals = r["by_gen"].get(key, {})
        for j, g in enumerate(gens):
            v = vals.get(g)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ymat[i, j] = v
    ymean, ystd = col_stats(ymat)
    x = np.array(gens, dtype=float)
    mask = ~np.isnan(ymean)
    return x[mask], ymean[mask], ystd[mask]


def _method_last(method_runs, key):
    """Mean across seeds of each seed's final (max-gen) value for `key`."""
    vals = []
    for r in method_runs:
        d = r["by_gen"].get(key, {})
        if d:
            vals.append(d[max(d)])
    return float(np.mean(vals)) if vals else 0.0


def _render_decomp_bar(ax, methods, title):
    """Stacked decomposition of each method's mean final train score:
    train = val_avg_score + overfitting + winner's curse. One column per method."""
    names = [m[0] for m in methods]
    xpos = np.arange(len(methods))
    val = np.array([_method_last(mr, "val_avg_score") for _, mr, *_ in methods])
    over = np.array([_method_last(mr, "val_overfit") for _, mr, *_ in methods])
    wc = np.array([_method_last(mr, "val_winners_curse_delta") for _, mr, *_ in methods])
    ax.bar(xpos, val, color="#4c72b0", label="val_avg_score")
    ax.bar(xpos, over, bottom=val, color="#dd8452", label="overfitting")
    ax.bar(xpos, wc, bottom=val + over, color="#55a868", label="winner's curse")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylabel("score")
    ax.legend(fontsize=7)


def _render_decomp_table(ax, methods, title):
    """Final-value decomposition as a table (uses the FIRST method). Cleaner than
    a one-column stacked bar when there's a single run to describe."""
    runs = methods[0][1]
    val = _method_last(runs, "val_avg_score")
    over = _method_last(runs, "val_overfit")
    wc = _method_last(runs, "val_winners_curse_delta")
    rows = [
        ("Val score",        f"{val:.3f}"),
        ("+ Overfitting",    f"{over:.3f}"),
        ("+ Winner's curse", f"{wc:.3f}"),
        ("= Train score",    f"{val + over + wc:.3f}"),
    ]
    ax.axis("off")
    ax.set_title(title)
    tbl = ax.table(cellText=rows, colLabels=["component", "score"],
                   cellLoc="left", colLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for c in range(2):
        tbl[(len(rows), c)].set_text_props(fontweight="bold")


def _api_cached():
    """Lazily-built (and reused) wandb API handle + run index for quick_plot."""
    global _API, _WIDX
    if _API is None:
        _API = get_api()
    if _WIDX is None:
        _WIDX = build_wandb_index()
    return _API, _WIDX


_API = None
_WIDX = None
_MARKERS = ["o", "s", "^", "D", "*", "v", "P"]


def _group_label(rids):
    """Auto label for a group of ids: the ids themselves when there are few."""
    if len(rids) == 1:
        return f"run {rids[0]}"
    if len(rids) <= 3:
        return "avg " + "/".join(str(r) for r in rids)
    return f"avg {rids[0]}+{len(rids) - 1} more"


def _as_groups(ids):
    """Normalize quick_plot's `ids` into an ordered {label: [run ids]} mapping.

    A bare id is its own curve; a nested list is one curve averaged over those
    seeds. A dict gives explicit labels (its values follow the same rule, so a
    flat list under a label is averaged).
    """
    if isinstance(ids, dict):
        return {lbl: (list(v) if isinstance(v, (list, tuple)) else [v])
                for lbl, v in ids.items()}
    groups = {}
    for item in ids:
        rids = list(item) if isinstance(item, (list, tuple)) else [item]
        groups[_group_label(rids)] = rids
    return groups


def quick_plot(ids, name, gen_axis=True, title=None, panels=None):
    """One-shot figure for an ad-hoc set of run ids -- paste ids, get a PNG.

    ids:      list of run ids (== slurm job ids). Each bare id is drawn as its
              OWN line: [883427, 883429] -> two lines. Nest a list to average
              seeds into one mean +/- std curve instead:
                  [[883427, 883429], [883428, 883430]]  -> two averaged curves
                  [[883427, 883429], 538190]            -> one avg + one line
              Or pass a dict {label: ids} to name the curves explicitly (a flat
              list under a label is averaged, same as a nested list).
    name:     output file stem.
    gen_axis: True  -> per-GENERATION panels (eval cost factored out),
                       saved to plots/gen_axis_n1_vs_n3/gen_axis_<name>.png
              False -> cumulative-EVAL-axis panels,
                       saved to plots/eval_axis_comparison/eval_axis_comparison_<name>.png
    title:    suptitle (defaults to the curve labels + axis).
    panels:   gen-axis only; defaults to PANELS_SINGLE for a lone run, else
              PANELS_COMPARISON.

    Returns the matplotlib Figure (already saved; shows inline in the notebook).
    """
    groups = _as_groups(ids)
    api, widx = _api_cached()
    axis_desc = "per generation" if gen_axis else "per total evaluation"
    # Spell out the ids for each curve, except where the auto label already has
    # them (e.g. "run 883427").
    desc = ";  ".join(lbl if all(str(r) in lbl for r in rids) else f"{lbl}: {rids}"
                      for lbl, rids in groups.items())
    suptitle = title or f"{name} — {axis_desc}\n{desc}"
    style = [(COLOR(i % 10), _MARKERS[i % len(_MARKERS)])
             for i in range(len(groups))]

    if gen_axis:
        methods = [(lbl, load_method(api, widx, rids, lbl), *style[i])
                   for i, (lbl, rids) in enumerate(groups.items())]
        if panels is None:
            n_runs = sum(len(runs) for _, runs, *_ in methods)
            panels = PANELS_SINGLE if n_runs == 1 else PANELS_COMPARISON
        return render(methods, OUTDIR / f"gen_axis_{name}.png", suptitle,
                      panels=panels)

    # Eval axis: reuse the Variant renderer from plot_eval_axis_comparison.
    # Always aggregate mode -- a single-id group is already a plain line there,
    # and a multi-id group is exactly the mean +/- std band we want.
    variant = Variant(
        name=name,
        title=suptitle,
        individual=False,
        methods=[Method(lbl, style[i][0], list(rids), style[i][1])
                 for i, (lbl, rids) in enumerate(groups.items())],
    )
    return plot_variant(api, widx, variant)


def render(methods, out_path, suptitle, panels=PANELS_COMPARISON, save=True):
    """Render the per-generation panels for `methods` and (optionally) save.

    methods: list of (name, runs, color, marker). Each run-list is a load_method
        result; it's aggregated to per-gen mean +/- std (line panels) and final
        value (decomposition panel).
    panels:  a list of (key, title, kind) -- see PANELS_COMPARISON / PANELS_SINGLE.
    save:    write out_path (PNG). The figure is always returned and shown inline.

    Returns the matplotlib Figure.
    """
    ncols = 3
    nrows = -(-len(panels) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.3 * nrows), squeeze=False)
    axflat = axes.flatten()

    for i, (key, title, kind) in enumerate(panels):
        ax = axflat[i]
        if kind == "bar":
            _render_decomp_bar(ax, methods, title)
            continue
        if kind == "table":
            _render_decomp_table(ax, methods, title)
            continue
        for name, runs, color, mk in methods:
            if not runs:
                continue
            x, ymean, ystd = method_curve_by_gen(runs, key)
            if x.size == 0:
                continue
            n = len(runs)
            lbl = name if n == 1 else f"{name} (n={n})"
            if key.startswith("val_"):
                # Step function that only changes on a real val submission: draw
                # the line through the held-constant stretches, mark ONLY gens
                # where some seed actually re-evaluated.
                ax.plot(x, ymean, "-", color=color, linewidth=1.7, label=lbl)
                gen_to_y = {int(g): y for g, y in zip(x, ymean)}
                ev = sorted({g for r in runs
                             for g in r["by_gen"].get(f"{key}__eval", {})}
                            & set(gen_to_y))
                ax.plot(ev, [gen_to_y[g] for g in ev], mk, color=color,
                        markersize=5, linestyle="None")
            else:
                ax.plot(x, ymean, f"-{mk}", color=color, markersize=5,
                        linewidth=1.7, label=lbl)
            if n > 1:
                ax.fill_between(x, ymean - ystd, ymean + ystd, color=color,
                                alpha=0.18, linewidth=0)
        if key in ("val_overfit", "val_winners_curse_delta"):
            ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("re-evals" if key == "gen_reevals" else "score")
        ax.grid(True, alpha=0.3)

    for j in range(len(panels), len(axflat)):
        axflat[j].axis("off")
    axflat[0].legend(loc="best", fontsize=9)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if save:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        print(f"saved: {out_path}")
    return fig
