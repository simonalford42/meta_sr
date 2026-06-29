"""Plot 7 evolve_pysr runs on a common 'total evaluations (seeds, incl. reeval)'
x-axis. These runs predate the new step semantics, so the x-axis is reconstructed
from run_data.json (sum of max seeds_evaluated across all unique bundles seen so
far). val_eval/avg_score is fetched via the wandb public API.
"""
import json
import os
import pickle
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb

# 6/26/26 — budget-matched smart vs no-reeval comparison (submit_jobs.sh block).
# All 5 seeds (0-4) complete. Two groups, each overlaying the 5 "no reeval" seeds
# against the 5 "smart reeval" seeds for a fixed per-gen eval budget B. Smart runs
# are drawn with star markers. (slurm job id == run dir id.)
# NB seed-0 of the "no reeval" rows uses the resubmits 89281/89282 — the originals
# (825765/825766) died at gen 2 from a mid-run code edit (r2_frontier_score kwarg).
GROUPS = {
    # B=20 evals/gen: none n1o20 (20) vs smart n1o5 (--max-runs-per-generation 20)
    "n1_none_vs_smart": [
        89281, 825769, 825773, 825777, 825781,   # nn_n1o20   seeds 0,1,2,3,4
        825767, 825771, 825775, 825779, 825783,   # sm_n1o5b20 seeds 0,1,2,3,4
    ],
    # B=60 evals/gen: none n3o20 (60) vs smart n3o5 (--max-runs-per-generation 60)
    "n3_none_vs_smart": [
        89282, 825770, 825774, 825778, 825782,    # nn_n3o20   seeds 0,1,2,3,4
        825768, 825772, 825776, 825780, 825784,    # sm_n3o5b60 seeds 0,1,2,3,4
    ],
}



REPO = Path("/home/sca63/meta_sr")
CACHE_DIR = REPO / "plots" / ".cache" / "eval_axis_comparison"


def is_smart(label: str) -> bool:
    """Smart-reeval runs are named either '...smart' (older runs) or 'sm_...'
    (the 6/26 budget-matched comparison)."""
    lab = label.lower()
    return "smart" in lab or lab.startswith("sm_")


def build_wandb_index() -> Dict[str, str]:
    """Map 'runs/<id>' -> wandb run id by scanning the local wandb/ cache.

    Each offline run dir (wandb/run-<timestamp>-<wandbid>/) records the
    runs/<id> it wrote to in its files/output.log. If several wandb dirs
    reference the same run dir (e.g. a resume), keep the most recent one.
    """
    index: Dict[str, Tuple[str, str]] = {}  # runs/<id> -> (timestamp, wandb_id)
    for wd in glob(str(REPO / "wandb" / "run-*")):
        m = re.search(r"run-([0-9_]+)-([a-z0-9]+)$", os.path.basename(wd))
        if not m:
            continue
        ts, wid = m.group(1), m.group(2)
        olog = Path(wd) / "files" / "output.log"
        if not olog.exists():
            continue
        try:
            text = olog.read_text(errors="ignore")
        except OSError:
            continue
        for rid in set(re.findall(r"runs/([0-9]+(?:_\d+)?)", text)):
            key = f"runs/{rid}"
            if key not in index or index[key][0] < ts:
                index[key] = (ts, wid)
    return {k: v[1] for k, v in index.items()}


def resolve_wandb_id(rdir: str, index: Dict[str, str]) -> Optional[str]:
    """Resolve runs/<id> -> wandb run id, preferring the local wandb cache and
    falling back to parsing runs/<id>/slurm.out."""
    if rdir in index:
        return index[rdir]
    slurm = REPO / rdir / "slurm.out"
    if slurm.exists():
        m = re.search(r"meta-sr/runs/([a-z0-9]+)", slurm.read_text(errors="ignore"))
        if m:
            return m.group(1)
    return None


def bundle_key(b: dict) -> str:
    """Stable identifier for a bundle (mirrors OperatorBundle.display_name)."""
    parts = []
    for t in ["mutation", "survival", "selection", "loss"]:
        op = (b.get("operators") or {}).get(t)
        parts.append(op["name"] if op else "default")
    return " | ".join(parts)


def per_gen_metrics(run_data: dict) -> Dict[str, np.ndarray]:
    """Reconstruct per-gen metrics + cumulative seed-evals for one run.

    Cumulative evals at end-of-gen k = sum over all unique bundles seen by then
    of their max seeds_evaluated (matches the new step axis: 1 SLURM run = 1 seed
    for 1 bundle).
    """
    gens = run_data["generations"]
    cfg = run_data["config"]
    n_runs = int(cfg.get("n_runs", 1))
    pop_size = int(cfg.get("population_size", 0))

    best_seeds: Dict[str, int] = {}
    gen_nums, cum_evals, cum_reevals = [], [], []
    avg_pop, avg_off, best_so_far, gen_best = [], [], [], []

    best = float("-inf")
    for g in gens:
        # Update seed counts from this gen's population (carries cumulative seeds
        # via racing/smart extras) and freshly-evaluated offspring.
        for b in g.get("population", []) + g.get("offspring", []):
            k = bundle_key(b)
            s = int(b.get("seeds_evaluated") or 0)
            if s > best_seeds.get(k, 0):
                best_seeds[k] = s

        pop_scores = [b["score"] for b in g.get("population", []) if b.get("score") is not None]
        off_scores = [b["score"] for b in g.get("offspring", []) if b.get("score") is not None]
        gb = g.get("best_score")
        if gb is not None and gb > best:
            best = gb

        gen_nums.append(int(g["generation"]))
        cum_evals.append(sum(best_seeds.values()))
        # Reevaluations = seeds beyond the first per bundle (only smart/racing
        # runs reevaluate, so this stays ~0 for plain runs).
        cum_reevals.append(sum(best_seeds.values()) - len(best_seeds))
        avg_pop.append(np.mean(pop_scores) if pop_scores else np.nan)
        avg_off.append(np.mean(off_scores) if off_scores else np.nan)
        gen_best.append(gb if gb is not None else np.nan)
        best_so_far.append(best if best != float("-inf") else np.nan)

    # Synthesize an "initial pop" point (before gen 1's offspring). Best we can
    # do from run_data alone: take gen 1's population, treat as the post-init
    # state. Cumulative evals = pop_size * n_runs (initial-pop seeds).
    init_pop = gens[0].get("population", []) if gens else []
    init_scores = [b["score"] for b in init_pop if b.get("score") is not None]
    if init_scores:
        gen_nums.insert(0, 0)
        cum_evals.insert(0, pop_size * n_runs)
        # Initial pop = pop_size bundles each on n_runs seeds -> reevals beyond
        # the first per bundle = pop_size * (n_runs - 1).
        cum_reevals.insert(0, pop_size * max(n_runs - 1, 0))
        avg_pop.insert(0, float(np.mean(init_scores)))
        avg_off.insert(0, np.nan)
        b0 = float(np.max(init_scores))
        gen_best.insert(0, b0)
        best_so_far.insert(0, b0)

    # Per-gen reevals = increase in cumulative reevals from the prior gen.
    cr = np.array(cum_reevals, dtype=float)
    gen_reevals = np.concatenate([cr[:1], np.diff(cr)]) if cr.size else cr

    return {
        "gen": np.array(gen_nums),
        "cum_evals": np.array(cum_evals),
        "gen_reevals": gen_reevals,
        "avg_pop": np.array(avg_pop, dtype=float),
        "avg_off": np.array(avg_off, dtype=float),
        "best": np.array(best_so_far, dtype=float),
        "gen_best": np.array(gen_best, dtype=float),
    }


def cached_per_gen_metrics(rdir: str) -> Dict[str, np.ndarray]:
    """per_gen_metrics() with a disk cache keyed on run_data.json's mtime+size.

    run_data.json files are huge (100s of MB-1GB) and take ~4s each to parse,
    while the reconstructed metrics are tiny. Cache the metrics so repeat plots
    are fast; the cache auto-invalidates when run_data.json changes.
    """
    rd_path = REPO / rdir / "run_data.json"
    st = rd_path.stat()
    rid = rdir.replace("/", "_")
    cache_file = CACHE_DIR / f"{rid}_{int(st.st_mtime)}_{st.st_size}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        # Recompute if the cache predates a newly-added metric key.
        if "gen_reevals" in cached:
            return cached
    with open(rd_path) as f:
        data = json.load(f)
    m = per_gen_metrics(data)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Drop stale cache entries for this run (older mtime/size keys).
    for old in CACHE_DIR.glob(f"{rid}_*.pkl"):
        old.unlink()
    with open(cache_file, "wb") as f:
        pickle.dump(m, f)
    return m


# Val-eval series fetched from wandb: (short_name, wandb_metric_key). Each is
# plotted on its own panel as scatter vs gen_submitted -> cumulative evals.
# Some keys only exist on newer runs; missing ones come back as empty series.
VAL_METRICS = [
    ("avg_score",           "val_eval/avg_score"),
    ("train_avg_score",     "val_eval/train_avg_score"),
    ("winners_curse_delta", "val_eval/train_winners_curse_delta"),
]


def fetch_run(api: wandb.Api, wandb_id: str) -> Tuple[Optional[str], Dict[str, List[Tuple[int, float]]]]:
    """Return (run_name, {short_name: [(gen_submitted, value), ...]}) for a run.

    run_name is the wandb run's display name (used as the plot label). Each
    metric in VAL_METRICS is fetched independently -- mixing a missing key into
    a single history() call makes wandb return zero rows. The result is cached
    to disk keyed by the run's last history step so it auto-invalidates when the
    run logs more. On failure returns (None, {})."""
    try:
        run = api.run(f"simon-alford/meta-sr/{wandb_id}")
    except Exception as e:
        print(f"  [warn] wandb fetch failed for {wandb_id}: {e}")
        return None, {}
    step = getattr(run, "lastHistoryStep", None)
    cache_file = CACHE_DIR / f"wandb_{wandb_id}_{step}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    series: Dict[str, List[Tuple[int, float]]] = {}
    for short, key in VAL_METRICS:
        pts = []
        for row in run.history(keys=["val_eval/gen_submitted", key], pandas=False):
            g = row.get("val_eval/gen_submitted")
            v = row.get(key)
            if g is not None and v is not None:
                pts.append((int(g), float(v)))
        series[short] = pts
    result = (run.name, series)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for old in CACHE_DIR.glob(f"wandb_{wandb_id}_*.pkl"):
        old.unlink()
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def forward_fill(pts: List[Tuple[int, float]], gen_grid: List[int],
                 gen_to_cum: Dict[int, float]) -> List[Tuple[float, float]]:
    """Hold each submitted val value constant until the next submission.

    Validation only runs when a new best operator appears, so the val series are
    sparse step functions. Forward-filling over the run's gen grid means a run
    that never finds a new best after init (a single submission) shows a
    horizontal line at that value across the whole eval axis instead of a lone
    point; runs with several submissions become proper step curves.
    """
    if not pts:
        return []
    sub = dict(pts)  # gen_submitted -> value (later dup-gen submission wins)
    sub_gens = sorted(sub)
    out: List[Tuple[float, float]] = []
    cur = None
    j = 0
    for g in sorted(gen_grid):
        while j < len(sub_gens) and sub_gens[j] <= g:
            cur = sub[sub_gens[j]]
            j += 1
        if cur is not None:
            out.append((gen_to_cum.get(g, np.nan), cur))
    return out


def forward_fill_by_gen(pts: List[Tuple[int, float]],
                        gen_grid: List[int]) -> Dict[int, float]:
    """Like forward_fill but keyed by generation (gen -> held value), so seeds
    can be averaged on a common per-gen grid regardless of their (differing)
    cumulative-eval x-positions."""
    if not pts:
        return {}
    sub = dict(pts)
    sub_gens = sorted(sub)
    out: Dict[int, float] = {}
    cur = None
    j = 0
    for g in sorted(gen_grid):
        while j < len(sub_gens) and sub_gens[j] <= g:
            cur = sub[sub_gens[j]]
            j += 1
        if cur is not None:
            out[g] = cur
    return out


def col_stats(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column nanmean/nanstd, returning NaN (no warning) for all-NaN cols."""
    counts = np.sum(~np.isnan(mat), axis=0)
    mean = np.full(mat.shape[1], np.nan)
    std = np.full(mat.shape[1], np.nan)
    valid = counts > 0
    if valid.any():
        mean[valid] = np.nanmean(mat[:, valid], axis=0)
        std[valid] = np.nanstd(mat[:, valid], axis=0)
    return mean, std


def method_curve(method_runs: List[dict], key: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a metric across seeds of one method onto a shared per-gen grid.

    Returns (x, ymean, ystd) where x is the mean cumulative-eval position across
    seeds at each gen and ymean/ystd are the mean/std of the metric. Gens where
    no seed has a value are dropped.
    """
    gens = sorted({g for r in method_runs for g in r["cum_by_gen"]})
    if not gens:
        return np.array([]), np.array([]), np.array([])
    xmat = np.full((len(method_runs), len(gens)), np.nan)
    ymat = np.full((len(method_runs), len(gens)), np.nan)
    for i, r in enumerate(method_runs):
        cum = r["cum_by_gen"]
        vals = r["by_gen"].get(key, {})
        for j, g in enumerate(gens):
            if g in cum:
                xmat[i, j] = cum[g]
            v = vals.get(g)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ymat[i, j] = v
    xmean, _ = col_stats(xmat)
    ymean, ystd = col_stats(ymat)
    mask = ~np.isnan(ymean)
    return xmean[mask], ymean[mask], ystd[mask]


def build_run_entry(api: wandb.Api, wandb_index: Dict[str, str], rdir: str,
                    label_counts: Dict[str, int],
                    label_override: Optional[str] = None) -> Optional[dict]:
    """Load one run's reconstructed metrics + wandb val series into a plot entry
    with per-gen indexing (so seeds can be averaged on a common grid). Returns
    None if the run dir is missing."""
    rd_path = REPO / rdir / "run_data.json"
    if not rd_path.exists():
        print(f"[skip] {rdir}: missing {rd_path}")
        return None
    m = cached_per_gen_metrics(rdir)
    wid = resolve_wandb_id(rdir, wandb_index)
    name, series = (None, {})
    if wid is None:
        print(f"  [warn] could not resolve wandb id for {rdir}; "
              f"plotting train curves only")
    else:
        name, series = fetch_run(api, wid)
    # Label = explicit override, else wandb run name, else the run dir.
    label = label_override or name or Path(rdir).name
    # De-duplicate identical names so runs don't collide / overwrite.
    label_counts[label] = label_counts.get(label, 0) + 1
    if label_counts[label] > 1:
        label = f"{label}#{label_counts[label]}"
    gen_grid = m["gen"].tolist()
    gen_to_cum = dict(zip(gen_grid, m["cum_evals"].tolist()))
    entry = {"label": label, **m}
    entry["cum_by_gen"] = dict(zip(gen_grid, m["cum_evals"].tolist()))
    by_gen: Dict[str, Dict[int, float]] = {}
    for lk in ("avg_pop", "avg_off", "gen_best", "gen_reevals"):
        by_gen[lk] = dict(zip(gen_grid, m[lk].tolist()))
    # Each val series is forward-filled across the run's gens so it spans the
    # whole eval axis (constant line when there's never a new best), rather than
    # collapsing to a single point.
    for short, pts in series.items():
        ff = forward_fill_by_gen(pts, gen_grid)
        by_gen[f"val_{short}"] = ff
        entry[f"val_{short}"] = [(gen_to_cum.get(g, np.nan), v)
                                 for g, v in sorted(ff.items())]
    # Overfitting over time = val/train_avg_score - val/avg_score. Positive
    # => train beats val (overfit).
    tr = forward_fill_by_gen(series.get("train_avg_score", []), gen_grid)
    vl = forward_fill_by_gen(series.get("avg_score", []), gen_grid)
    shared = sorted(set(tr) & set(vl))
    by_gen["val_overfit"] = {g: tr[g] - vl[g] for g in shared}
    entry["val_overfit"] = [(gen_to_cum.get(g, np.nan), tr[g] - vl[g])
                            for g in shared]
    entry["by_gen"] = by_gen
    print(f"{label}: {len(m['gen'])} gen points, "
          f"{len(entry.get('val_avg_score', []))} val points, "
          f"max cum_evals={int(m['cum_evals'].max())}")
    return entry


def plot_group(api: wandb.Api, wandb_index: Dict[str, str],
               run_dirs: List[str], group_name: str,
               extra_refs: Optional[List[Tuple[str, str]]] = None,
               suffix: str = ""):
    """Plot one comparison group. extra_refs is a list of (label, run_dir)
    single-run reference curves (e.g. a best-models run) overlaid as dashed
    lines with no std band; suffix is appended to the output filename."""
    runs: List[dict] = []  # ordered, one entry per plotted run
    label_counts: Dict[str, int] = {}
    for rdir in run_dirs:
        entry = build_run_entry(api, wandb_index, rdir, label_counts)
        if entry is not None:
            runs.append(entry)
    # Reference runs (plotted individually, no aggregation / band).
    ref_entries: List[dict] = []
    for ref_label, ref_dir in (extra_refs or []):
        entry = build_run_entry(api, wandb_index, ref_dir, label_counts,
                                label_override=ref_label)
        if entry is not None:
            ref_entries.append(entry)

    # Panels: "val" panels are points (x already mapped to cumulative evals);
    # "line" panels are arrays vs cum_evals.
    panels = [
        ("avg_pop",                  "Train avg population score",          "line"),
        ("avg_off",                  "Train avg offspring score",           "line"),
        ("gen_best",                 "Best score (current best in population)", "line"),
        ("val_avg_score",            "Val eval / avg_score",                "val"),
        ("val_train_avg_score",      "Val eval / train_avg_score",          "val"),
        ("val_winners_curse_delta",  "Val eval / train_winners_curse_delta", "val"),
        ("val_overfit",             "Overfitting (val train_avg - val avg)", "val"),
        ("gen_reevals",             "Reevaluations this gen (smart runs)",  "reeval"),
        ("train_decomp",            "Train score decomposition (latest)",  "bar"),
    ]
    # Aggregate seeds into two methods: no-reeval (dots) vs smart-reeval (stars).
    # Each method curve is the per-gen mean across its seeds, with a light ±std
    # band. Smart runs are detected via is_smart() on the label.
    none_runs = [d for d in runs if not is_smart(d["label"])]
    smart_runs = [d for d in runs if is_smart(d["label"])]
    cmap = plt.get_cmap("tab10")
    # (name, runs, color, marker, linestyle)
    methods = [
        ("no reeval", none_runs, cmap(0), "o", "-"),
        ("smart reeval", smart_runs, cmap(1), "*", "-"),
    ]
    # Each reference run is its own single-run "method": a dashed line, distinct
    # color/square marker, and no std band (n=1).
    ref_colors = [cmap(2), cmap(3), cmap(4)]
    for i, ref in enumerate(ref_entries):
        methods.append((ref["label"], [ref], ref_colors[i % len(ref_colors)], "s", "--"))

    ncols = 3
    nrows = -(-len(panels) // ncols)  # ceil
    # sharex=False: most panels share the eval axis (set per-panel below), but
    # the bar panel uses a categorical x and must not be locked to it.
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.3 * nrows), sharex=False)
    axflat = axes.flatten()

    def method_last(method_runs, key):
        """Mean across seeds of each seed's final (max-gen) value for `key`."""
        vals = []
        for r in method_runs:
            d = r["by_gen"].get(key, {})
            if d:
                vals.append(d[max(d)])
        return float(np.mean(vals)) if vals else 0.0

    for i, (key, title, kind) in enumerate(panels):
        ax = axflat[i]
        if kind == "bar":
            # Stacked decomposition of each method's mean final train score:
            # train = val_avg_score + overfitting + winner's curse. One column
            # per method; components stacked val (bottom) -> overfit -> WC (top).
            names = [m[0] for m in methods]
            xpos = np.arange(len(methods))
            val = np.array([method_last(mr, "val_avg_score") for _, mr, *_ in methods])
            over = np.array([method_last(mr, "val_overfit") for _, mr, *_ in methods])
            wc = np.array([method_last(mr, "val_winners_curse_delta") for _, mr, *_ in methods])
            ax.bar(xpos, val, color="#4c72b0", label="val_avg_score")
            ax.bar(xpos, over, bottom=val, color="#dd8452", label="overfitting")
            ax.bar(xpos, wc, bottom=val + over, color="#55a868", label="winner's curse")
            ax.set_xticks(xpos)
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylabel("score")
            ax.legend(fontsize=7)
            continue
        for name, method_runs, mcolor, mk, ls in methods:
            # The reeval panel only applies to smart runs (others never reeval).
            if kind == "reeval" and "smart" not in name.lower():
                continue
            if not method_runs:
                continue
            x, ymean, ystd = method_curve(method_runs, key)
            if x.size == 0:
                continue
            n_seeds = len(method_runs)
            ms = 8 if mk == "*" else (6 if kind == "val" else 4)
            lbl = f"{name} (n={n_seeds})" if n_seeds > 1 else name
            ax.plot(x, ymean, f"{ls}{mk}", color=mcolor,
                    label=lbl, markersize=ms, linewidth=1.6)
            # Std band only for aggregated (multi-seed) methods; refs are n=1.
            if n_seeds > 1:
                ax.fill_between(x, ymean - ystd, ymean + ystd, color=mcolor,
                                alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("count" if kind == "reeval" else "score")
        # ax.set_xlim(right=2000)
    if "val_overfit" in [p[0] for p in panels]:
        axflat[[p[0] for p in panels].index("val_overfit")].axhline(
            0, color="gray", lw=0.8, ls="--")
    # Hide any unused axes and label the bottom-most used axis in each column.
    for j in range(len(panels), len(axflat)):
        axflat[j].axis("off")
    for c in range(ncols):
        used = [i for i in range(len(panels)) if i % ncols == c]
        # Skip the bar panel -- its x is categorical, not the eval axis.
        if used and panels[max(used)][2] != "bar":
            axflat[max(used)].set_xlabel("Total evaluations (seeds, incl. reeval)")
    axflat[0].legend(loc="best", fontsize=8, ncol=2)

    ref_note = (" + " + ", ".join(r["label"] for r in ref_entries)) if ref_entries else ""
    fig.suptitle(f"Evolve-PySR: {group_name} (smart vs no reeval{ref_note}) on shared "
                 f"'total evals' axis", fontsize=12)
    fig.tight_layout()
    out = REPO / "plots" / "eval_axis_comparison" / f"eval_axis_comparison_{group_name}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved: {out}")


def main():
    api = wandb.Api()
    wandb_index = build_wandb_index()
    # Best-models reference (538190 = n3o10 smart, opus/gpt/gemini/grok ensemble,
    # 30 gens) overlaid on every panel to gauge best- vs weak-model evolution.
    REF = [("538190 (best, n3o10smart)", "runs/538190")]
    for group_name, ids in GROUPS.items():
        print(f"\n=== {group_name} ===")
        run_dirs = [f"runs/{r}" for r in ids]
        plot_group(api, wandb_index, run_dirs, group_name)
        plot_group(api, wandb_index, run_dirs, group_name,
                   extra_refs=REF, suffix="_with538190")


if __name__ == "__main__":
    main()
