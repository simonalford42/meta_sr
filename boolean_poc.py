"""Proof-of-concept: evolving PySR for the Boolean-synthesis domain.

Compares three conditions on the held-out IWLS 2020 benchmark (test minterms):

  1. ``baseline`` - default Boolean-PySR (no custom operator, default hparams).
  2. ``hpo``      - best of a small hyperparameter search, tuned on the IWLS
                    *validation* minterms (no custom operator).
  3. ``evolved``  - default hparams + an LLM-evolved custom mutation operator
                    (evolved on synthetic Boolean tasks by ``evolve_boolean.py``).

For every IWLS function we fit PySR on its *train* minterms and score accuracy
on its *test* minterms (the contest's generalization metric). HPO tuning uses
the *validation* minterms so the *test* numbers stay honest.

Everything runs locally on a persistent pool of Julia-backed workers (no SLURM).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from boolean_eval import BooleanEvaluator, BooleanJob, default_n_workers
from boolean_pysr import get_boolean_pysr_kwargs
from boolean_tasks import load_iwls_task, tractable_iwls_ids
from evolve_boolean import BooleanEvolver, EvolveConfig

# A small, tractable IWLS subset (low input width => Boolean-expression search
# has a real chance). Overridable via --iwls-ids.
DEFAULT_IWLS_IDS = ["ex41", "ex40", "ex73", "ex75", "ex77", "ex30"]


def hpo_config_grid() -> List[Dict[str, Any]]:
    """A small hyperparameter grid for the HPO baseline (pysr_kwargs overrides)."""
    grid = [
        {"maxsize": 20, "niterations": 60},
        {"maxsize": 30, "niterations": 60},
        {"maxsize": 40, "niterations": 60},
        {"maxsize": 30, "populations": 30, "niterations": 60},
        {"maxsize": 30, "niterations": 100},
        {"maxsize": 30, "niterations": 60, "parsimony": 0.001},
    ]
    return grid


@dataclass
class MethodResult:
    name: str
    per_task: Dict[str, dict] = field(default_factory=dict)
    mean_test_acc: float = 0.0
    test_solve_rate: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BooleanPOC:
    def __init__(self, args):
        self.args = args
        self.iwls_ids = args.iwls_ids or DEFAULT_IWLS_IDS
        self.n_workers = args.workers or default_n_workers()
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sample = args.iwls_samples
        self.results: Dict[str, MethodResult] = {}

    # --- IWLS task loading -------------------------------------------------
    def _iwls_pair(self, ex: str, eval_split: str):
        """(train_task, eval_task) for one IWLS function; eval on given split."""
        train = load_iwls_task(ex, "train", max_samples=self.sample, seed=0)
        ev = load_iwls_task(ex, eval_split, max_samples=self.sample, seed=1)
        return train, ev

    def _eval_method_on_split(
        self, evaluator: BooleanEvaluator, name: str,
        pysr_kwargs: Dict[str, Any], mutation_code: Optional[Dict[str, str]],
        eval_split: str,
    ) -> MethodResult:
        jobs = []
        for ex in self.iwls_ids:
            train, ev = self._iwls_pair(ex, eval_split)
            jobs.append(BooleanJob(
                train_task=train, eval_task=ev,
                custom_mutation_code=mutation_code,
                custom_mutation_weight=self.args.mutation_weight,
                pysr_kwargs=pysr_kwargs, seed=self.args.seed, tag=ex,
            ))
        done = evaluator.run(jobs)
        res = MethodResult(name=name)
        accs, solves = [], []
        for ex, job in zip(self.iwls_ids, done):
            r = job.result or {}
            acc = float(r.get("eval_acc") or 0.0)
            solved = bool(r.get("eval_solved"))
            res.per_task[ex] = {
                "test_acc": acc, "test_solved": solved,
                "train_acc": float(r.get("train_acc") or 0.0),
                "best_equation": r.get("best_equation", ""),
                "best_complexity": r.get("best_complexity"),
                "error": r.get("error"),
            }
            accs.append(acc)
            solves.append(solved)
        res.mean_test_acc = float(np.mean(accs)) if accs else 0.0
        res.test_solve_rate = float(np.mean(solves)) if solves else 0.0
        return res

    # --- conditions --------------------------------------------------------
    def run_baseline(self, evaluator: BooleanEvaluator) -> MethodResult:
        print("\n=== [baseline] default Boolean-PySR ===", flush=True)
        kwargs = {"niterations": self.args.niterations, "maxsize": 30}
        return self._eval_method_on_split(evaluator, "baseline", kwargs, None, "test")

    def run_hpo(self, evaluator: BooleanEvaluator) -> MethodResult:
        print("\n=== [hpo] tuning hyperparameters on IWLS validation ===", flush=True)
        grid = hpo_config_grid()
        val_scores = []
        for gi, cfg in enumerate(grid):
            r = self._eval_method_on_split(evaluator, f"hpo_cfg{gi}", cfg, None, "validation")
            val_scores.append((r.mean_test_acc, gi, cfg))
            print(f"  cfg{gi} {cfg} -> val_acc={r.mean_test_acc:.3f}", flush=True)
        val_scores.sort(reverse=True)
        best_val, best_gi, best_cfg = val_scores[0]
        print(f"  best hpo cfg{best_gi}: {best_cfg} (val_acc={best_val:.3f})", flush=True)
        # Report the tuned config on TEST.
        res = self._eval_method_on_split(evaluator, "hpo", best_cfg, None, "test")
        res.extra = {"best_config": best_cfg, "val_acc": best_val, "grid_val": [
            {"cfg": c, "val_acc": s} for s, _, c in val_scores]}
        return res

    def run_evolved(self, evaluator: BooleanEvaluator, mutation_code: Dict[str, str]) -> MethodResult:
        print("\n=== [evolved] default hparams + evolved mutation operator ===", flush=True)
        kwargs = {"niterations": self.args.niterations, "maxsize": 30}
        res = self._eval_method_on_split(evaluator, "evolved", kwargs, mutation_code, "test")
        res.extra = {"mutation_name": list(mutation_code.keys())[0]}
        return res

    # --- orchestration -----------------------------------------------------
    def run(self):
        t0 = time.time()
        print(f"POC: IWLS subset={self.iwls_ids}, workers={self.n_workers}, "
              f"samples/split={self.sample}", flush=True)

        # 1) Evolve a mutation operator on synthetic tasks (own worker pool).
        mutation_code = None
        if not self.args.skip_evolve:
            ecfg = EvolveConfig(
                n_generations=self.args.evolve_generations,
                population_size=self.args.evolve_population,
                n_offspring=self.args.evolve_offspring,
                niterations=self.args.evolve_niterations,
                n_workers=self.n_workers,
                model=self.args.model,
                reasoning_effort=self.args.effort,
                out_dir=str(self.out_dir / "evolve"),
                seed=self.args.seed,
            )
            best = BooleanEvolver(ecfg).run()
            mutation_code = best.mutation_code
            if mutation_code is None:
                # Overall best was plain baseline; report the best mutation-bearing
                # operator instead so the evolved condition is still meaningful.
                eop = Path(ecfg.out_dir) / "best_evolved_operator.json"
                if eop.exists():
                    data = json.loads(eop.read_text())
                    mutation_code = data.get("custom_mutation_code")
                    print(f"[evolved] baseline won evolution; using best mutation-bearing "
                          f"operator {list((mutation_code or {}).keys())}", flush=True)
                else:
                    print("[warn] evolution produced no custom operator at all; "
                          "evolved == baseline", flush=True)
        else:
            # Load a previously-evolved operator.
            path = Path(self.args.evolved_operator)
            data = json.loads(path.read_text())
            mutation_code = data.get("custom_mutation_code")
            print(f"[evolved] loaded operator from {path}: {list((mutation_code or {}).keys())}", flush=True)

        # 2) Evaluate all three conditions on IWLS (shared worker pool).
        with BooleanEvaluator(n_workers=self.n_workers) as evaluator:
            self.results["baseline"] = self.run_baseline(evaluator)
            self.results["hpo"] = self.run_hpo(evaluator)
            if mutation_code:
                self.results["evolved"] = self.run_evolved(evaluator, mutation_code)

        self._report(time.time() - t0)

    def _report(self, dt: float):
        print("\n" + "=" * 70, flush=True)
        print(f"POC RESULTS  (IWLS test minterms, {len(self.iwls_ids)} functions)", flush=True)
        print("=" * 70, flush=True)
        header = f"{'method':<10} {'mean_test_acc':>14} {'test_solve_rate':>16}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for name in ("baseline", "hpo", "evolved"):
            if name in self.results:
                r = self.results[name]
                print(f"{name:<10} {r.mean_test_acc:>14.4f} {r.test_solve_rate:>16.3f}", flush=True)
        # Per-task breakdown.
        print("\nPer-task test accuracy:", flush=True)
        cols = [n for n in ("baseline", "hpo", "evolved") if n in self.results]
        print(f"{'task':<8} " + " ".join(f"{c:>10}" for c in cols), flush=True)
        for ex in self.iwls_ids:
            row = f"{ex:<8} "
            for c in cols:
                row += f"{self.results[c].per_task.get(ex, {}).get('test_acc', 0.0):>10.4f} "
            print(row, flush=True)

        payload = {
            "iwls_ids": self.iwls_ids,
            "n_workers": self.n_workers,
            "samples_per_split": self.sample,
            "runtime_seconds": dt,
            "results": {n: {
                "mean_test_acc": r.mean_test_acc,
                "test_solve_rate": r.test_solve_rate,
                "per_task": r.per_task,
                "extra": r.extra,
            } for n, r in self.results.items()},
            "config": vars(self.args),
        }
        (self.out_dir / "poc_results.json").write_text(json.dumps(payload, indent=2))
        print(f"\n[poc] saved -> {self.out_dir}/poc_results.json  ({dt:.0f}s)", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Boolean-domain PySR evolution POC")
    ap.add_argument("--iwls-ids", nargs="*", default=None, help="IWLS ex ids (default: small subset)")
    ap.add_argument("--iwls-samples", type=int, default=2000, help="minterms per split per function")
    ap.add_argument("--niterations", type=int, default=60, help="PySR iterations for eval fits")
    ap.add_argument("--mutation-weight", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default="openai/gpt-5.4-mini")
    ap.add_argument("--effort", type=str, default="medium")
    # Evolution controls
    ap.add_argument("--evolve-generations", type=int, default=3)
    ap.add_argument("--evolve-population", type=int, default=6)
    ap.add_argument("--evolve-offspring", type=int, default=4)
    ap.add_argument("--evolve-niterations", type=int, default=40)
    ap.add_argument("--skip-evolve", action="store_true", help="load --evolved-operator instead")
    ap.add_argument("--evolved-operator", type=str, default="runs_local/boolean_poc/evolve/best_operator.json")
    ap.add_argument("--out", type=str, default="runs_local/boolean_poc")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.workers == 0:
        args.workers = default_n_workers()
    BooleanPOC(args).run()
