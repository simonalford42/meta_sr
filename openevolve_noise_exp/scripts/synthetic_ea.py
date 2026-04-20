"""Synthetic evolutionary-search testbed for studying evaluation noise.

Task: evolve a D-digit genome (each digit 0..B-1) to maximize a multimodal
fitness landscape. Each digit maps linearly to a real coordinate in [lo, hi];
the true fitness is −f(x) for a user-chosen multimodal f (default: Rastrigin).

Search: (μ+λ) evolution strategy with random digit-flip mutation and
noisy-fitness truncation selection. Noise is per-evaluation Gaussian with
stddev σ; each candidate is evaluated `num_trials` times and the mean
is used for selection — so effective noise is σ/√num_trials per eval.

The x-axis of all cost plots is cumulative #trials = generation_evals × num_trials,
so higher num_trials costs more per generation but gives a cleaner signal.

Usage:
  # Single run:
  python scripts/synthetic_ea.py run --budget 5000 --num-trials 3 --noise 5.0 \
      --seed 0 --out outputs/synthetic/demo
  # Sweep:
  python scripts/synthetic_ea.py sweep --out-root outputs/synthetic/run1 \
      --budget 5000 --noises 0 1 5 20 --num-trials 1 3 10 --seeds 0..19
  # Plot:
  python scripts/synthetic_ea.py plot --sweep-dir outputs/synthetic/run1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np


# ------------------------- landscapes -------------------------

def rastrigin(x: np.ndarray) -> float:
    """Classic multimodal; global min at x=0, f=0. Many local minima every 1.0."""
    A = 10.0
    return float(A * len(x) + np.sum(x * x - A * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Nearly-flat outer region with a deep spike at 0."""
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    d = len(x)
    s1 = float(np.sum(x * x))
    s2 = float(np.sum(np.cos(c * x)))
    return -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e


def sphere(x: np.ndarray) -> float:
    """Trivial unimodal; sanity check."""
    return float(np.sum(x * x))


LANDSCAPES: dict[str, Callable[[np.ndarray], float]] = {
    "rastrigin": rastrigin,
    "ackley": ackley,
    "sphere": sphere,
}


# ------------------------- genome/task -------------------------

@dataclass
class Task:
    landscape: str = "rastrigin"
    dim: int = 4              # number of real coordinates
    base: int = 10            # digits per coord
    digits_per_coord: int = 3 # resolution: 10**3 = 1000 levels per coord
    lo: float = -5.12
    hi: float = 5.12

    @property
    def genome_length(self) -> int:
        return self.dim * self.digits_per_coord

    def decode(self, genome: np.ndarray) -> np.ndarray:
        """Map digit-vector (len D*K) to real vector (len D) in [lo, hi]."""
        g = genome.reshape(self.dim, self.digits_per_coord)
        # Each row is K digits in base B. Max integer = B^K - 1.
        powers = self.base ** np.arange(self.digits_per_coord - 1, -1, -1)
        ints = (g * powers).sum(axis=1)
        frac = ints / (self.base ** self.digits_per_coord - 1)
        return self.lo + frac * (self.hi - self.lo)

    def true_fitness(self, genome: np.ndarray) -> float:
        """We maximize true_fitness; so return −f(x)."""
        x = self.decode(genome)
        return -LANDSCAPES[self.landscape](x)

    def random_genome(self, rng) -> np.ndarray:
        return rng.integers(0, self.base, size=self.genome_length, dtype=np.int32)


# ------------------------- EA -------------------------

@dataclass
class ESConfig:
    mu: int = 10
    lam: int = 20
    mutation_flips_mean: float = 1.0   # Poisson-distributed n_flips per mutation
    num_trials: int = 1
    noise_std: float = 0.0
    budget_trials: int = 5000
    seed: int = 0


def mutate(genome: np.ndarray, base: int, mean_flips: float, rng) -> np.ndarray:
    """Flip Poisson(mean_flips) random digits to a random new value (resampled)."""
    out = genome.copy()
    n_flips = max(1, int(rng.poisson(mean_flips)))
    n_flips = min(n_flips, len(out))
    idx = rng.choice(len(out), size=n_flips, replace=False)
    # pick a new digit different from current — otherwise no-op
    for i in idx:
        new = int(rng.integers(0, base - 1))
        if new >= out[i]:
            new += 1
        out[i] = new
    return out


def noisy_eval(task: Task, genome: np.ndarray, n_trials: int, noise_std: float, rng) -> Tuple[float, float]:
    """Return (noisy_mean, true_fitness)."""
    true = task.true_fitness(genome)
    if n_trials <= 0:
        return true, true
    if noise_std <= 0:
        return true, true
    eps = rng.normal(0.0, noise_std, size=n_trials)
    return true + float(eps.mean()), true


def run_es(task: Task, cfg: ESConfig, log_interval: int = 1) -> dict:
    """Run (μ+λ) ES. Returns a trajectory log."""
    rng = np.random.default_rng(cfg.seed)
    # init μ parents
    pop = [task.random_genome(rng) for _ in range(cfg.mu)]
    pop_noisy = []
    pop_true = []
    trials_used = 0
    for g in pop:
        nz, tr = noisy_eval(task, g, cfg.num_trials, cfg.noise_std, rng)
        pop_noisy.append(nz)
        pop_true.append(tr)
        trials_used += max(1, cfg.num_trials)

    # global trackers
    best_true_ever = max(pop_true)
    best_noisy_ever = max(pop_noisy)
    # true score of the program currently declared best by noisy score
    declared_idx = int(np.argmax(pop_noisy))
    true_of_declared_best = pop_true[declared_idx]

    traj = []

    def log_step(generation):
        # pop-level aggregates
        declared_idx = int(np.argmax(pop_noisy))
        traj.append({
            "generation": generation,
            "cum_trials": trials_used,
            "best_true_ever": best_true_ever,
            "best_noisy_ever": best_noisy_ever,
            "true_of_declared_best": pop_true[declared_idx],
            "noisy_of_declared_best": pop_noisy[declared_idx],
            "pop_best_true": float(max(pop_true)),
            "pop_mean_true": float(np.mean(pop_true)),
        })

    log_step(0)

    generation = 0
    while trials_used < cfg.budget_trials:
        generation += 1
        # generate λ offspring
        offspring = []
        o_noisy = []
        o_true = []
        for _ in range(cfg.lam):
            p = rng.integers(cfg.mu)
            child = mutate(pop[p], task.base, cfg.mutation_flips_mean, rng)
            nz, tr = noisy_eval(task, child, cfg.num_trials, cfg.noise_std, rng)
            offspring.append(child)
            o_noisy.append(nz)
            o_true.append(tr)
            trials_used += max(1, cfg.num_trials)
            if tr > best_true_ever:
                best_true_ever = tr
            if nz > best_noisy_ever:
                best_noisy_ever = nz

        # plus-selection: combine and take top μ by NOISY score
        all_g = pop + offspring
        all_noisy = pop_noisy + o_noisy
        all_true = pop_true + o_true
        order = np.argsort(all_noisy)[::-1]
        sel = order[: cfg.mu]
        pop = [all_g[i] for i in sel]
        pop_noisy = [all_noisy[i] for i in sel]
        pop_true = [all_true[i] for i in sel]

        if generation % log_interval == 0:
            log_step(generation)

    # final
    log_step(generation)

    return {
        "task": asdict(task),
        "cfg": asdict(cfg),
        "trajectory": traj,
    }


# ------------------------- CLI -------------------------

def cmd_run(args):
    task = Task(landscape=args.landscape, dim=args.dim,
                digits_per_coord=args.digits_per_coord, lo=args.lo, hi=args.hi)
    cfg = ESConfig(mu=args.mu, lam=args.lam, mutation_flips_mean=args.flips,
                   num_trials=args.num_trials, noise_std=args.noise,
                   budget_trials=args.budget, seed=args.seed)
    result = run_es(task, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(result))
    last = result["trajectory"][-1]
    print(f"[run] seed={cfg.seed} noise={cfg.noise_std} N={cfg.num_trials} "
          f"gens={last['generation']} trials={last['cum_trials']} "
          f"best_true={last['best_true_ever']:.4f} "
          f"true_of_declared={last['true_of_declared_best']:.4f}")


def _parse_seeds(spec: str) -> List[int]:
    """Accept '0..19' ranges or '0,1,2,3' lists."""
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def cmd_sweep(args):
    seeds = _parse_seeds(args.seeds) if isinstance(args.seeds, str) else args.seeds
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = []
    for noise in args.noises:
        for n in args.num_trials:
            for s in seeds:
                configs.append((noise, n, s))

    print(f"[sweep] total runs: {len(configs)} out_root={out_root}")
    t0 = time.time()
    for i, (noise, n, s) in enumerate(configs, 1):
        task = Task(landscape=args.landscape, dim=args.dim,
                    digits_per_coord=args.digits_per_coord, lo=args.lo, hi=args.hi)
        cfg = ESConfig(mu=args.mu, lam=args.lam, mutation_flips_mean=args.flips,
                       num_trials=n, noise_std=noise, budget_trials=args.budget, seed=s)
        result = run_es(task, cfg)
        tag = f"n{n}_sigma{noise}_seed{s}"
        (out_root / f"{tag}.json").write_text(json.dumps(result))
        last = result["trajectory"][-1]
        print(f"  [{i:3d}/{len(configs)}] {tag:24s} best_true={last['best_true_ever']:.3f} "
              f"true_decl={last['true_of_declared_best']:.3f}")
    print(f"[sweep] done in {time.time() - t0:.1f}s")


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_dir = Path(args.sweep_dir)
    results_dir = sweep_dir / "results"
    results_dir.mkdir(exist_ok=True)

    runs = []
    for f in sorted(sweep_dir.glob("*.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    if not runs:
        print(f"[plot] no runs found in {sweep_dir}")
        return 1

    # collect trajectories
    rows = []
    for r in runs:
        cfg = r["cfg"]
        for step in r["trajectory"]:
            rows.append({
                "noise": float(cfg["noise_std"]),
                "n_trials": int(cfg["num_trials"]),
                "seed": int(cfg["seed"]),
                **step,
            })
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "summary.csv", index=False)

    # final-value summary
    finals = df.sort_values("generation").groupby(["noise", "n_trials", "seed"]).tail(1)
    agg = finals.groupby(["noise", "n_trials"]).agg(
        mean_best_true=("best_true_ever", "mean"),
        std_best_true=("best_true_ever", "std"),
        mean_true_of_declared=("true_of_declared_best", "mean"),
        mean_noisy_of_declared=("noisy_of_declared_best", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    agg["lucky_inflation"] = agg["mean_noisy_of_declared"] - agg["mean_true_of_declared"]
    agg.to_csv(results_dir / "final_summary.csv", index=False)
    print(agg.to_string(index=False))

    # helper: interpolate running-max onto common log-spaced grid per config
    def interp(sub, x_grid, ycol):
        curves = []
        for _, g in sub.groupby("seed"):
            g = g.sort_values("cum_trials")
            xs = g["cum_trials"].to_numpy()
            ys = g[ycol].to_numpy()
            if len(xs) == 0:
                continue
            # forward-fill step function on log grid
            interp = np.full_like(x_grid, np.nan, dtype=float)
            for i, x in enumerate(x_grid):
                mask = xs <= x
                if mask.any():
                    interp[i] = ys[mask][-1]
            curves.append(interp)
        if not curves:
            return None, None
        arr = np.vstack(curves)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    def plot_metric(df, ycol, ylabel, fname, x_scale="log"):
        noises = sorted(df["noise"].unique())
        ntrials = sorted(df["n_trials"].unique())
        ncols = min(3, len(noises))
        nrows = (len(noises) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), sharey=True, squeeze=False)
        axes = axes.ravel()
        x_max = int(df["cum_trials"].max())
        if x_scale == "log":
            x_grid = np.logspace(0, np.log10(x_max), 200)
        else:
            x_grid = np.linspace(1, x_max, 200)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ntrials)))
        for ax in axes[len(noises):]:
            ax.axis("off")
        for ax, sigma in zip(axes, noises):
            for c, n in zip(colors, ntrials):
                sub = df[(df["noise"] == sigma) & (df["n_trials"] == n)]
                if sub.empty:
                    continue
                mean, std = interp(sub, x_grid, ycol)
                if mean is None:
                    continue
                ax.plot(x_grid, mean, color=c, lw=2, label=f"N={n}")
                ax.fill_between(x_grid, mean - std, mean + std, color=c, alpha=0.18)
            ax.set_title(f"σ = {sigma}")
            ax.set_xlabel("cumulative trials")
            if x_scale == "log":
                ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_ylabel(ylabel)
        axes[0].legend(loc="lower right", fontsize=9)
        n_seeds = df["seed"].nunique()
        fig.suptitle(f"{ylabel} — {n_seeds} seeds / config")
        fig.tight_layout()
        out = results_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {out}")

    plot_metric(df, "best_true_ever", "best true fitness ever seen", "best_true_ever.png")
    plot_metric(df, "true_of_declared_best", "true fitness of declared-best", "true_of_declared.png")
    plot_metric(df, "noisy_of_declared_best", "noisy (reported) best", "noisy_best.png")

    # also plot linear-x variants
    plot_metric(df, "best_true_ever", "best true fitness ever seen", "best_true_ever_linx.png", x_scale="linear")
    plot_metric(df, "true_of_declared_best", "true fitness of declared-best", "true_of_declared_linx.png", x_scale="linear")

    # inflation
    df2 = df.copy()
    df2["inflation"] = df2["noisy_of_declared_best"] - df2["true_of_declared_best"]
    plot_metric(df2, "inflation", "noisy_best − true_of_declared (inflation)", "lucky_inflation.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common_task = argparse.ArgumentParser(add_help=False)
    common_task.add_argument("--landscape", default="rastrigin", choices=list(LANDSCAPES))
    common_task.add_argument("--dim", type=int, default=4)
    common_task.add_argument("--digits-per-coord", type=int, default=3)
    common_task.add_argument("--lo", type=float, default=-5.12)
    common_task.add_argument("--hi", type=float, default=5.12)
    common_task.add_argument("--mu", type=int, default=10)
    common_task.add_argument("--lam", type=int, default=20)
    common_task.add_argument("--flips", type=float, default=1.0)

    pr = sub.add_parser("run", parents=[common_task])
    pr.add_argument("--noise", type=float, required=True)
    pr.add_argument("--num-trials", type=int, default=1)
    pr.add_argument("--budget", type=int, default=5000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", type=str, required=True)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sweep", parents=[common_task])
    ps.add_argument("--out-root", type=str, required=True)
    ps.add_argument("--noises", type=float, nargs="+", required=True)
    ps.add_argument("--num-trials", type=int, nargs="+", required=True)
    ps.add_argument("--seeds", type=str, default="0..19", help="'a..b' inclusive or 'a,b,c'")
    ps.add_argument("--budget", type=int, default=5000)
    ps.set_defaults(func=cmd_sweep)

    pp = sub.add_parser("plot")
    pp.add_argument("--sweep-dir", type=str, required=True)
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
