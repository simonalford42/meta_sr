"""Synthetic EA with pluggable trial-allocation policies.

Task + mutation + ES scaffolding identical to ``synthetic_ea.py``; the difference
is that here each candidate carries its own history of noisy samples, and a
``Policy`` decides how to spend the trial budget each generation. All policies
operate on the same compute budget (total trial units) so we can fairly
compare:

    FixedN              — spend n trials on each new candidate. No re-eval.
    ReEvalElites        — evolve_pysr's "racing": accumulate extra trials on
                          every current elite each generation.
    SuccessiveHalving   — cheap eval all offspring, keep top-frac, add trials,
                          repeat until μ survive. (Hoeffding-style racing.)
    ConditionalReEval   — cheap eval first; only candidates above a threshold
                          (median of current elites) get confirmation trials.

All policies run until ``cum_trials >= budget``.

Usage:
  python scripts/synthetic_policies.py run --policy racing --noise 5 \
      --budget 10000 --seed 0 --out outputs/synthetic_pol/demo
  python scripts/synthetic_policies.py sweep --out-root outputs/synthetic_pol/r1 \
      --budget 10000 --noises 0 1 5 20 50 \
      --policies fixed1 fixed3 fixed10 fixed30 race1_2 race1_5 halving cond
  python scripts/synthetic_policies.py plot --sweep-dir outputs/synthetic_pol/r1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


# ------------------------- landscapes -------------------------

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * len(x) + np.sum(x * x - A * np.cos(2.0 * np.pi * x)))


LANDSCAPES: dict[str, Callable[[np.ndarray], float]] = {"rastrigin": rastrigin}


# ------------------------- task -------------------------

@dataclass
class Task:
    landscape: str = "rastrigin"
    dim: int = 4
    base: int = 10
    digits_per_coord: int = 3
    lo: float = -5.12
    hi: float = 5.12
    # "Discovery difficulty" knob: clip fitness at -plateau, so everything
    # worse than -plateau looks identical. Smaller plateau = harder discovery
    # (must be within a tight neighborhood of the optimum before the EA sees
    # any gradient). Default float("inf") = no clipping = standard Rastrigin.
    plateau: float = float("inf")

    @property
    def genome_length(self) -> int:
        return self.dim * self.digits_per_coord

    def decode(self, genome: np.ndarray) -> np.ndarray:
        g = genome.reshape(self.dim, self.digits_per_coord)
        powers = self.base ** np.arange(self.digits_per_coord - 1, -1, -1)
        ints = (g * powers).sum(axis=1)
        frac = ints / (self.base ** self.digits_per_coord - 1)
        return self.lo + frac * (self.hi - self.lo)

    def true_fitness(self, genome: np.ndarray) -> float:
        raw = LANDSCAPES[self.landscape](self.decode(genome))  # ≥ 0, 0 at optimum
        if np.isfinite(self.plateau):
            raw = min(raw, self.plateau)
        return -raw

    def random_genome(self, rng) -> np.ndarray:
        return rng.integers(0, self.base, size=self.genome_length, dtype=np.int32)


def mutate(genome: np.ndarray, base: int, mean_flips: float, rng) -> np.ndarray:
    out = genome.copy()
    n = max(1, int(rng.poisson(mean_flips)))
    n = min(n, len(out))
    idx = rng.choice(len(out), size=n, replace=False)
    for i in idx:
        new = int(rng.integers(0, base - 1))
        if new >= out[i]:
            new += 1
        out[i] = new
    return out


# ------------------------- candidate & evaluation -------------------------

class Candidate:
    """A candidate with a cached true fitness and a growing list of noisy draws."""

    __slots__ = ("genome", "true_fit", "noisy_samples", "_task", "_noise")

    def __init__(self, genome: np.ndarray, task: Task, noise_std: float):
        self.genome = genome
        self.true_fit = task.true_fitness(genome)
        self.noisy_samples: List[float] = []
        self._task = task
        self._noise = noise_std

    def add_trials(self, k: int, rng) -> int:
        """Append k noisy samples. Returns k (= trials consumed)."""
        if self._noise > 0 and k > 0:
            self.noisy_samples.extend((self.true_fit + rng.normal(0, self._noise, size=k)).tolist())
        else:
            self.noisy_samples.extend([self.true_fit] * k)
        return k

    @property
    def n_evals(self) -> int:
        return len(self.noisy_samples)

    @property
    def noisy_mean(self) -> float:
        return float(np.mean(self.noisy_samples)) if self.noisy_samples else -np.inf


# ------------------------- policies -------------------------

@dataclass
class PolicyBase:
    name: str

    def eval_initial(self, pop: List[Candidate], rng) -> int:
        raise NotImplementedError

    def advance_generation(
        self, elites: List[Candidate], offspring: List[Candidate], rng
    ) -> Tuple[List[Candidate], int]:
        """Evaluate offspring + (optionally) top-up elites; return (new_elites, trials_spent)."""
        raise NotImplementedError


@dataclass
class FixedN(PolicyBase):
    n: int = 1
    mu: int = 10
    name: str = "fixed"

    def eval_initial(self, pop, rng):
        spent = 0
        for c in pop:
            spent += c.add_trials(self.n, rng)
        return spent

    def advance_generation(self, elites, offspring, rng):
        spent = 0
        for c in offspring:
            spent += c.add_trials(self.n, rng)
        pool = elites + offspring
        pool.sort(key=lambda c: -c.noisy_mean)
        return pool[: self.mu], spent


@dataclass
class RaceUniform(PolicyBase):
    """evolve_pysr-style racing: every generation, eval ALL members (elites + offspring)
    with k fresh trials. Offspring enter with 0 prior samples; elites accumulate across
    generations, so survivors' score estimates converge to the true mean."""
    k: int = 2
    mu: int = 10
    name: str = "race_uniform"

    def eval_initial(self, pop, rng):
        spent = 0
        for c in pop:
            spent += c.add_trials(self.k, rng)
        return spent

    def advance_generation(self, elites, offspring, rng):
        spent = 0
        for c in elites + offspring:
            spent += c.add_trials(self.k, rng)
        pool = elites + offspring
        pool.sort(key=lambda c: -c.noisy_mean)
        return pool[: self.mu], spent


@dataclass
class RaceAsym(PolicyBase):
    """Asymmetric racing: n_new trials on each new offspring (enough to estimate),
    k_refresh additional trials on each current elite."""
    n_new: int = 3
    k_refresh: int = 2
    mu: int = 10
    name: str = "race_asym"

    def eval_initial(self, pop, rng):
        spent = 0
        for c in pop:
            spent += c.add_trials(self.n_new, rng)
        return spent

    def advance_generation(self, elites, offspring, rng):
        spent = 0
        for c in offspring:
            spent += c.add_trials(self.n_new, rng)
        for c in elites:
            spent += c.add_trials(self.k_refresh, rng)
        pool = elites + offspring
        pool.sort(key=lambda c: -c.noisy_mean)
        return pool[: self.mu], spent


@dataclass
class SuccessiveHalving(PolicyBase):
    """Classic successive halving on offspring: cheap eval all, rank, keep top half,
    give survivors more trials, repeat until only μ remain. Elites are not re-evaluated
    here (their scores are frozen from when they entered the pool)."""
    n_init: int = 1
    n_per_round: int = 2
    mu: int = 10
    name: str = "halving"

    def eval_initial(self, pop, rng):
        spent = 0
        for c in pop:
            spent += c.add_trials(self.n_init + self.n_per_round, rng)
        return spent

    def advance_generation(self, elites, offspring, rng):
        spent = 0
        survivors = list(offspring)
        # round 0: cheap eval
        for c in survivors:
            spent += c.add_trials(self.n_init, rng)
        # halve until we have ≤ mu offspring
        while len(survivors) > self.mu:
            survivors.sort(key=lambda c: -c.noisy_mean)
            k = max(self.mu, len(survivors) // 2)
            survivors = survivors[:k]
            # winners get another round of trials
            for c in survivors:
                spent += c.add_trials(self.n_per_round, rng)
        pool = elites + survivors
        pool.sort(key=lambda c: -c.noisy_mean)
        return pool[: self.mu], spent


@dataclass
class ConditionalReEval(PolicyBase):
    n_cheap: int = 1
    n_confirm: int = 3
    mu: int = 10
    name: str = "cond"

    def eval_initial(self, pop, rng):
        spent = 0
        for c in pop:
            spent += c.add_trials(self.n_cheap + self.n_confirm, rng)  # seed with full budget
        return spent

    def advance_generation(self, elites, offspring, rng):
        spent = 0
        for c in offspring:
            spent += c.add_trials(self.n_cheap, rng)
        # threshold = median of current elite noisy means
        if elites:
            thr = float(np.median([c.noisy_mean for c in elites]))
        else:
            thr = -np.inf
        for c in offspring:
            if c.noisy_mean > thr:
                spent += c.add_trials(self.n_confirm, rng)
        pool = elites + offspring
        pool.sort(key=lambda c: -c.noisy_mean)
        return pool[: self.mu], spent


POLICY_SPECS = {
    # label              -> (class, kwargs)
    "fixed1":   (FixedN,           dict(n=1)),
    "fixed3":   (FixedN,           dict(n=3)),
    "fixed10":  (FixedN,           dict(n=10)),
    "fixed30":  (FixedN,           dict(n=30)),
    # evolve_pysr-style racing: everyone gets k trials per gen (elites accumulate)
    "race_k1":  (RaceUniform,      dict(k=1)),
    "race_k2":  (RaceUniform,      dict(k=2)),
    "race_k3":  (RaceUniform,      dict(k=3)),
    # Asymmetric: more trials on new offspring, cheaper top-up of elites
    "race_asym3_2": (RaceAsym,     dict(n_new=3, k_refresh=2)),
    # Successive halving on offspring
    "halving":  (SuccessiveHalving, dict(n_init=1, n_per_round=2)),
    # Cheap-then-confirm
    "cond1_3":  (ConditionalReEval, dict(n_cheap=1, n_confirm=3)),
    "cond1_9":  (ConditionalReEval, dict(n_cheap=1, n_confirm=9)),
}


def make_policy(label: str, mu: int) -> PolicyBase:
    cls, kw = POLICY_SPECS[label]
    return cls(name=label, mu=mu, **kw)


# ------------------------- EA driver -------------------------

@dataclass
class ESConfig:
    mu: int = 10
    lam: int = 20
    mutation_flips_mean: float = 1.0
    noise_std: float = 0.0
    budget_trials: int = 10000
    seed: int = 0
    policy: str = "fixed1"


def run_es(task: Task, cfg: ESConfig, log_every: int = 1) -> dict:
    rng = np.random.default_rng(cfg.seed)
    policy = make_policy(cfg.policy, cfg.mu)

    # Init μ parents, evaluate per policy rules
    pop = [Candidate(task.random_genome(rng), task, cfg.noise_std) for _ in range(cfg.mu)]
    trials_used = policy.eval_initial(pop, rng)

    # Global trackers
    best_true_ever = max(c.true_fit for c in pop)
    declared = max(pop, key=lambda c: c.noisy_mean)
    traj = []

    def snap(gen: int):
        decl = max(pop, key=lambda c: c.noisy_mean)
        traj.append({
            "generation": gen,
            "cum_trials": trials_used,
            "best_true_ever": best_true_ever,
            "true_of_declared_best": decl.true_fit,
            "noisy_of_declared_best": decl.noisy_mean,
            "pop_best_true": float(max(c.true_fit for c in pop)),
            "pop_mean_true": float(np.mean([c.true_fit for c in pop])),
            "pop_mean_nevals": float(np.mean([c.n_evals for c in pop])),
            "declared_n_evals": decl.n_evals,
        })

    snap(0)

    gen = 0
    while trials_used < cfg.budget_trials:
        gen += 1
        offspring = []
        for _ in range(cfg.lam):
            p = int(rng.integers(cfg.mu))
            child_genome = mutate(pop[p].genome, task.base, cfg.mutation_flips_mean, rng)
            offspring.append(Candidate(child_genome, task, cfg.noise_std))

        new_pop, spent = policy.advance_generation(pop, offspring, rng)
        trials_used += spent
        pop = new_pop
        best_true_ever = max(best_true_ever, max(c.true_fit for c in pop + offspring))
        if gen % log_every == 0:
            snap(gen)

    snap(gen)

    return {
        "task": asdict(task),
        "cfg": asdict(cfg),
        "trajectory": traj,
    }


# ------------------------- CLI -------------------------

def cmd_run(args):
    task = Task(plateau=args.plateau)
    cfg = ESConfig(mu=args.mu, lam=args.lam, noise_std=args.noise,
                   budget_trials=args.budget, seed=args.seed, policy=args.policy)
    result = run_es(task, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(result))
    last = result["trajectory"][-1]
    print(f"[run] policy={cfg.policy} noise={cfg.noise_std} plateau={args.plateau} "
          f"seed={cfg.seed} gens={last['generation']} trials={last['cum_trials']} "
          f"best_true={last['best_true_ever']:.3f} "
          f"true_of_decl={last['true_of_declared_best']:.3f} "
          f"decl_n_evals={last['declared_n_evals']}")


def _parse_seeds(spec: str) -> List[int]:
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def cmd_sweep(args):
    seeds = _parse_seeds(args.seeds)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    plateaus = args.plateaus if args.plateaus else [float("inf")]
    configs = []
    for policy in args.policies:
        for noise in args.noises:
            for plateau in plateaus:
                for s in seeds:
                    configs.append((policy, noise, plateau, s))

    print(f"[sweep] total runs: {len(configs)}")
    t0 = time.time()
    for i, (policy, noise, plateau, s) in enumerate(configs, 1):
        task = Task(plateau=plateau)
        cfg = ESConfig(mu=args.mu, lam=args.lam, noise_std=noise,
                       budget_trials=args.budget, seed=s, policy=policy)
        result = run_es(task, cfg)
        # result's task dict already records plateau
        plateau_tag = f"plat{plateau}" if np.isfinite(plateau) else "platinf"
        tag = f"{policy}_sigma{noise}_{plateau_tag}_seed{s}"
        (out_root / f"{tag}.json").write_text(json.dumps(result, default=str))
        last = result["trajectory"][-1]
        if i % 50 == 0 or i == len(configs):
            print(f"  [{i:4d}/{len(configs)}] {tag:45s} best_true={last['best_true_ever']:.3f} "
                  f"decl_n={last['declared_n_evals']}")
    print(f"[sweep] done in {time.time() - t0:.1f}s")


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

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

    rows = []
    for r in runs:
        cfg = r["cfg"]
        task = r.get("task", {})
        plateau = task.get("plateau", float("inf"))
        try:
            plateau = float(plateau)
        except (TypeError, ValueError):
            plateau = float("inf")
        for step in r["trajectory"]:
            rows.append({
                "policy": cfg["policy"],
                "noise": float(cfg["noise_std"]),
                "plateau": plateau,
                "seed": int(cfg["seed"]),
                **step,
            })
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "summary.csv", index=False)

    # final summary
    finals = df.sort_values("generation").groupby(["policy", "noise", "plateau", "seed"]).tail(1)
    agg = finals.groupby(["policy", "noise", "plateau"]).agg(
        mean_best_true=("best_true_ever", "mean"),
        std_best_true=("best_true_ever", "std"),
        mean_true_of_declared=("true_of_declared_best", "mean"),
        mean_noisy_of_declared=("noisy_of_declared_best", "mean"),
        mean_declared_n_evals=("declared_n_evals", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    agg["lucky_inflation"] = agg["mean_noisy_of_declared"] - agg["mean_true_of_declared"]
    agg.to_csv(results_dir / "final_summary.csv", index=False)
    print(agg.to_string(index=False))

    def interp(sub, x_grid, ycol):
        curves = []
        for _, g in sub.groupby("seed"):
            g = g.sort_values("cum_trials")
            xs = g["cum_trials"].to_numpy()
            ys = g[ycol].to_numpy()
            if len(xs) == 0:
                continue
            a = np.full_like(x_grid, np.nan, dtype=float)
            for i, x in enumerate(x_grid):
                m = xs <= x
                if m.any():
                    a[i] = ys[m][-1]
            curves.append(a)
        if not curves:
            return None, None
        arr = np.vstack(curves)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    def plot_metric(df, ycol, ylabel, fname, x_scale="log"):
        noises = sorted(df["noise"].unique())
        policies = sorted(df["policy"].unique())
        ncols = min(3, len(noises))
        nrows = (len(noises) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                                 sharey=True, squeeze=False)
        axes = axes.ravel()
        for ax in axes[len(noises):]:
            ax.axis("off")
        x_max = int(df["cum_trials"].max())
        x_grid = np.logspace(0, np.log10(x_max), 200) if x_scale == "log" \
                 else np.linspace(1, x_max, 200)
        colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
        for ax, sigma in zip(axes, noises):
            for c, pol in zip(colors, policies):
                sub = df[(df["noise"] == sigma) & (df["policy"] == pol)]
                if sub.empty:
                    continue
                mean, std = interp(sub, x_grid, ycol)
                if mean is None:
                    continue
                ax.plot(x_grid, mean, color=c, lw=1.8, label=pol)
                ax.fill_between(x_grid, mean - std, mean + std, color=c, alpha=0.12)
            ax.set_title(f"σ = {sigma}")
            ax.set_xlabel("cumulative trials")
            if x_scale == "log":
                ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_ylabel(ylabel)
        axes[0].legend(loc="lower right", fontsize=8, ncol=2)
        n_seeds = df["seed"].nunique()
        fig.suptitle(f"{ylabel} — {n_seeds} seeds/config")
        fig.tight_layout()
        out = results_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {out}")

    # produce one set of plots per plateau (or one set if only one plateau present)
    for plateau_val, df_p in df.groupby("plateau"):
        suffix = "" if (df["plateau"].nunique() == 1) else f"_plat{plateau_val}"
        plot_metric(df_p, "best_true_ever", "best true fitness ever seen", f"best_true_ever{suffix}.png")
        plot_metric(df_p, "true_of_declared_best", "true fitness of declared-best", f"true_of_declared{suffix}.png")
        df2 = df_p.copy()
        df2["inflation"] = df2["noisy_of_declared_best"] - df2["true_of_declared_best"]
        plot_metric(df2, "inflation", "noisy_best − true_of_declared (inflation)", f"lucky_inflation{suffix}.png")
        plot_metric(df_p, "declared_n_evals", "# evaluations of declared-best", f"declared_n_evals{suffix}.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mu", type=int, default=10)
    common.add_argument("--lam", type=int, default=20)
    common.add_argument("--budget", type=int, default=10000)

    pr = sub.add_parser("run", parents=[common])
    pr.add_argument("--policy", choices=list(POLICY_SPECS.keys()), required=True)
    pr.add_argument("--noise", type=float, required=True)
    pr.add_argument("--plateau", type=float, default=float("inf"),
                    help="clip fitness at -plateau; smaller = harder discovery (default: inf)")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", type=str, required=True)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sweep", parents=[common])
    ps.add_argument("--out-root", type=str, required=True)
    ps.add_argument("--policies", nargs="+", default=list(POLICY_SPECS.keys()))
    ps.add_argument("--noises", type=float, nargs="+", required=True)
    ps.add_argument("--plateaus", type=float, nargs="*", default=None,
                    help="fitness-clip levels; smaller = harder")
    ps.add_argument("--seeds", type=str, default="0..19")
    ps.set_defaults(func=cmd_sweep)

    pp = sub.add_parser("plot")
    pp.add_argument("--sweep-dir", type=str, required=True)
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
