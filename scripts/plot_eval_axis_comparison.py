"""Registry of evolve_pysr comparison plots on a common 'total evaluations
(seeds, incl. reeval)' x-axis.

The x-axis is reconstructed from each run's run_data.json (sum of max
seeds_evaluated across all unique bundles seen so far), and val_eval metrics are
fetched via the wandb public API. Runs that pay more evals per generation (e.g.
n3, which evaluates every offspring on 3 seeds) are pushed right on this axis
relative to cheaper runs producing the same #offspring/gen.

HOW THIS SCRIPT IS ORGANIZED
----------------------------
Every comparison is a `Variant` in the `VARIANTS` list below. A variant is a set
of `Method`s (each a named group of seed run-ids, with a color/marker) plus a
flag for how to draw them:
  * individual=False (default): aggregate each method's seeds into a per-gen
    mean +/- std band. Use for >=3 seeds.
  * individual=True: draw every seed as its own line (per-seed line style),
    no band. Use when there are too few seeds (n<=2) for a band to mean much.
A method with a single run is always a plain line with no band (e.g. a reference
run overlaid as a dashed line).

ADD, DON'T REPLACE. To plot something new, append a new Variant rather than
editing an old one -- this keeps a record of everything we've plotted and its
output PNG (eval_axis_comparison_<name>.png) around as evidence. Each variant
writes to its own file, so old plots are never clobbered.

Run all variants:            python scripts/plot_eval_axis_comparison.py
Run specific ones by name:   python scripts/plot_eval_axis_comparison.py n1_vs_n3_eval_axis
"""
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb

REPO = Path("/home/sca63/meta_sr")
CACHE_DIR = REPO / "plots" / ".cache" / "eval_axis_comparison"

_C = plt.get_cmap("tab10")  # 0 blue, 1 orange, 2 green, 3 red, 4 purple


@dataclass
class Method:
    """A named group of seed run-ids drawn as one curve (or one line per seed).

    run_ids are run-dir ids (== slurm job ids for these runs). marker/linestyle
    style the aggregated curve; in individual mode the color is shared across the
    method's seeds and line styles distinguish them.
    """
    label: str
    color: object
    run_ids: List[int]
    marker: str = "o"
    linestyle: str = "-"


@dataclass
class Variant:
    """One comparison plot: a set of methods + how to draw them (see module doc).
    Renders to plots/eval_axis_comparison/eval_axis_comparison_<name>.png.
    """
    name: str
    title: str
    methods: List[Method]
    individual: bool = False


# --- Registry of comparison plots. Append new entries; don't edit old ones. ----
# NB seed-0 of the 6/24 "no reeval" rows uses the resubmits 89281/89282 -- the
# originals (825765/825766) died at gen 2 from a mid-run code edit.
VARIANTS: List[Variant] = [
    # 7/01/26 (NEW): n1 vs n3, both reeval=none / offspring=20, on the cumulative
    # EVAL axis. Same #offspring/gen for both, but n3 evaluates each offspring on
    # 3 seeds so it pays 3x eval cost -- on this axis n3's curve is stretched to
    # the right. Companion scripts/gen_axis_plots.ipynb shows the SAME runs
    # per-generation (cost factored out); this asks whether n3's cleaner signal
    # is worth its 3x eval budget.
    Variant(
        name="n1_vs_n3_eval_axis",
        title=("n1 vs n3 (both reeval=none, offspring=20) on shared 'total evals' "
               "axis\nsame #offspring/gen; n3 evaluates each on 3 seeds -> 3x eval "
               "cost/gen"),
        methods=[
            Method("n1 (n_runs 1)", _C(0), [89281, 825769, 825773, 825777, 825781], "o"),
            Method("n3 (n_runs 3)", _C(3), [89282, 825770, 825774, 825778, 825782], "s"),
        ],
    ),
    # 7/01/26: best-models n1, no-reeval ("none") vs smart-TTTS ("ttts"), budget-
    # matched ~20 evals/gen (none: offspring 20; ttts: offspring 5 + reeval,
    # --max-runs-per-generation 20). Two seeds each -> every seed its own line
    # (no band): none blue, ttts orange. The ttts runs (492224/492225) were still
    # running when plotted, so their curves are shorter.
    Variant(
        name="bestmodels_n1_none_vs_ttts",
        title="Evolve-PySR: best-models n1, none (blue) vs ttts (orange), all seeds",
        individual=True,
        methods=[
            Method("none", "tab:blue",   [397148, 397145], "o"),
            Method("ttts", "tab:orange", [492224, 492225], "o"),
        ],
    ),
    # 6/26/26 budget-matched no-reeval vs smart-reeval (5 seeds each), with the
    # best-models run 538190 (n3o10 smart) overlaid as a dashed reference. B is the
    # per-gen eval budget: B=20 (n1) and B=60 (n3). Smart runs use star markers.
    Variant(
        name="n1_none_vs_smart",
        title=("Evolve-PySR: n1 none vs smart reeval (B=20 evals/gen) + 538190 ref "
               "on shared 'total evals' axis"),
        methods=[
            Method("no reeval",    _C(0), [89281, 825769, 825773, 825777, 825781], "o"),
            Method("smart reeval", _C(1), [825767, 825771, 825775, 825779, 825783], "*"),
            Method("538190 (best, n3o10smart)", _C(2), [538190], "D", "--"),
        ],
    ),
    Variant(
        name="n3_none_vs_smart",
        title=("Evolve-PySR: n3 none vs smart reeval (B=60 evals/gen) + 538190 ref "
               "on shared 'total evals' axis"),
        methods=[
            Method("no reeval",    _C(0), [89282, 825770, 825774, 825778, 825782], "o"),
            Method("smart reeval", _C(1), [825768, 825772, 825776, 825780, 825784], "*"),
            Method("538190 (best, n3o10smart)", _C(2), [538190], "D", "--"),
        ],
    ),
]


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
    # best_current = best score among the *current* population under its current
    # (possibly re-evaluated) posterior scores. Unlike best_so_far (a cumulative
    # all-time max), this can DROP when a re-eval lowers the top candidate and no
    # survivor exceeds it.
    best_current = []

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
        best_current.append(max(pop_scores) if pop_scores else np.nan)

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
        best_current.insert(0, b0)

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
        "best_current": np.array(best_current, dtype=float),
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
        if "best_current" in cached:
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
    for lk in ("avg_pop", "avg_off", "gen_best", "gen_reevals", "best_current"):
        by_gen[lk] = dict(zip(gen_grid, m[lk].tolist()))
    # Each val series is forward-filled across the run's gens so it spans the
    # whole eval axis (constant line when there's never a new best), rather than
    # collapsing to a single point.
    grid_set = set(gen_grid)
    for short, pts in series.items():
        ff = forward_fill_by_gen(pts, gen_grid)
        by_gen[f"val_{short}"] = ff
        entry[f"val_{short}"] = [(gen_to_cum.get(g, np.nan), v)
                                 for g, v in sorted(ff.items())]
        # Companion "__eval" dict: value ONLY at gens where a real val submission
        # actually happened (val runs only on a new best), vs the forward-filled
        # repeats above. Lets plots mark real evaluations and merely draw the line
        # through the held-constant stretches. dict(pts) => last dup-gen wins.
        by_gen[f"val_{short}__eval"] = {g: v for g, v in dict(pts).items()
                                        if g in grid_set}
    # Overfitting over time = val/train_avg_score - val/avg_score. Positive
    # => train beats val (overfit).
    tr = forward_fill_by_gen(series.get("train_avg_score", []), gen_grid)
    vl = forward_fill_by_gen(series.get("avg_score", []), gen_grid)
    shared = sorted(set(tr) & set(vl))
    by_gen["val_overfit"] = {g: tr[g] - vl[g] for g in shared}
    entry["val_overfit"] = [(gen_to_cum.get(g, np.nan), tr[g] - vl[g])
                            for g in shared]
    # Overfit is "really" re-evaluated when either train or val is submitted.
    tr_ev = {g for g, _ in dict(series.get("train_avg_score", [])).items()}
    vl_ev = {g for g, _ in dict(series.get("avg_score", [])).items()}
    ev_gens = (tr_ev | vl_ev) & grid_set
    by_gen["val_overfit__eval"] = {g: by_gen["val_overfit"][g]
                                   for g in ev_gens if g in by_gen["val_overfit"]}
    entry["by_gen"] = by_gen
    print(f"{label}: {len(m['gen'])} gen points, "
          f"{len(entry.get('val_avg_score', []))} val points, "
          f"max cum_evals={int(m['cum_evals'].max())}")
    return entry


def panel_series(entry: dict, key: str, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract NaN-filtered (x, y) arrays for one panel from a run entry.
    'line'/'reeval' panels read the per-gen metric arrays vs cum_evals; 'val'
    panels read the pre-mapped (cum_evals, value) point lists."""
    if kind in ("line", "reeval"):
        x = np.asarray(entry["cum_evals"], dtype=float)
        y = np.asarray(entry[key], dtype=float)
        m = ~np.isnan(x) & ~np.isnan(y)
        return x[m], y[m]
    pts = entry.get(key, [])
    if not pts:
        return np.array([]), np.array([])
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    return x[m], y[m]


def plot_variant(api: wandb.Api, wandb_index: Dict[str, str], variant: Variant):
    """Render one Variant onto the shared cumulative-eval x-axis and save it.

    Aggregate mode (variant.individual=False): each method is its per-gen mean
    +/- std band across its seeds. Individual mode: every seed is its own line
    (method color, per-seed line style), no band. A method with a single run is
    always a plain line with no band regardless of mode (e.g. a reference run).
    """
    label_counts: Dict[str, int] = {}
    # (method, [entries]) preserving method order.
    loaded: List[Tuple[Method, List[dict]]] = []
    for meth in variant.methods:
        entries: List[dict] = []
        for rid in meth.run_ids:
            # Individual mode uses the per-seed label on the plot; aggregate mode
            # only labels the method, so let the run's wandb name through for logs.
            override = f"{meth.label} ({rid})" if variant.individual else None
            entry = build_run_entry(api, wandb_index, f"runs/{rid}", label_counts,
                                    label_override=override)
            if entry is not None:
                entries.append(entry)
        loaded.append((meth, entries))

    # Panels: "val" panels are (x already mapped to cum evals, value) points;
    # "line"/"reeval" panels are per-gen metric arrays vs cum_evals; "bar" is the
    # categorical train-score decomposition.
    panels = [
        ("avg_pop",                  "Train avg population score",              "line"),
        ("avg_off",                  "Train avg offspring score",               "line"),
        ("gen_best",                 "Best score (current best in population)", "line"),
        ("val_avg_score",            "Val eval / avg_score",                    "val"),
        ("val_train_avg_score",      "Val eval / train_avg_score",              "val"),
        ("val_winners_curse_delta",  "Val eval / train_winners_curse_delta",    "val"),
        ("val_overfit",              "Overfitting (val train_avg - val avg)",   "val"),
        ("gen_reevals",              "Extra seed-evals this gen (beyond 1/bundle)", "reeval"),
        ("train_decomp",             "Train score decomposition (latest)",      "bar"),
    ]

    # Within an individual-mode method, seed 0 is solid, seed 1 dashed, etc.
    linestyles = ["-", "--", ":", "-."]
    ncols = 3
    nrows = -(-len(panels) // ncols)  # ceil
    # sharex=False: eval-axis panels are labeled per-column below; the bar panel
    # is categorical and must not be locked to the eval axis.
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.3 * nrows),
                             sharex=False, squeeze=False)
    axflat = axes.flatten()

    def method_last(entries, key):
        """Mean across a method's seeds of each seed's final (max-gen) value."""
        vals = []
        for e in entries:
            d = e["by_gen"].get(key, {})
            if d:
                vals.append(d[max(d)])
        return float(np.mean(vals)) if vals else 0.0

    for i, (key, title, kind) in enumerate(panels):
        ax = axflat[i]
        if kind == "bar":
            # Stacked decomposition of each method's mean final train score:
            # train = val_avg_score + overfitting + winner's curse. One column
            # per method; components stacked val (bottom) -> overfit -> WC (top).
            names = [m.label for m, _ in loaded]
            xpos = np.arange(len(loaded))
            val = np.array([method_last(e, "val_avg_score") for _, e in loaded])
            over = np.array([method_last(e, "val_overfit") for _, e in loaded])
            wc = np.array([method_last(e, "val_winners_curse_delta") for _, e in loaded])
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
        for meth, entries in loaded:
            if not entries:
                continue
            if variant.individual:
                for si, entry in enumerate(entries):
                    x, y = panel_series(entry, key, kind)
                    if x.size == 0:
                        continue
                    ls = linestyles[si % len(linestyles)]
                    ms = 6 if kind == "val" else 4
                    ax.plot(x, y, ls + meth.marker, color=meth.color,
                            label=entry["label"], markersize=ms, linewidth=1.6)
            else:
                x, ymean, ystd = method_curve(entries, key)
                if x.size == 0:
                    continue
                n = len(entries)
                ms = 8 if meth.marker == "*" else (6 if kind == "val" else 4)
                lbl = f"{meth.label} (n={n})" if n > 1 else meth.label
                ax.plot(x, ymean, meth.linestyle + meth.marker, color=meth.color,
                        label=lbl, markersize=ms, linewidth=1.6)
                # Std band only for aggregated (multi-seed) methods; refs are n=1.
                if n > 1:
                    ax.fill_between(x, ymean - ystd, ymean + ystd,
                                    color=meth.color, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("count" if kind == "reeval" else "score")
    keys = [p[0] for p in panels]
    if "val_overfit" in keys:
        axflat[keys.index("val_overfit")].axhline(0, color="gray", lw=0.8, ls="--")
    # Hide any unused axes and label the bottom-most eval-axis panel per column.
    for j in range(len(panels), len(axflat)):
        axflat[j].axis("off")
    for c in range(ncols):
        used = [i for i in range(len(panels)) if i % ncols == c]
        if used and panels[max(used)][2] != "bar":
            axflat[max(used)].set_xlabel("Total evaluations (seeds, incl. reeval)")
    axflat[0].legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(variant.title, fontsize=12)
    fig.tight_layout()
    out = (REPO / "plots" / "eval_axis_comparison" /
           f"eval_axis_comparison_{variant.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved: {out}")
    # Returned (not closed) so notebook callers can display it inline; main()
    # closes it since it renders every variant in one go.
    return fig


def main():
    # No args -> render every variant (keeps all evidence fresh); otherwise render
    # only the named variants, e.g. `... n1_vs_n3_eval_axis`.
    names = sys.argv[1:]
    by_name = {v.name: v for v in VARIANTS}
    if names:
        unknown = [n for n in names if n not in by_name]
        if unknown:
            print(f"unknown variant(s): {unknown}\navailable: {list(by_name)}")
        selected = [by_name[n] for n in names if n in by_name]
    else:
        selected = VARIANTS
    if not selected:
        return
    api = wandb.Api()
    wandb_index = build_wandb_index()
    for v in selected:
        print(f"\n=== {v.name} ===")
        plt.close(plot_variant(api, wandb_index, v))


if __name__ == "__main__":
    main()
